//! Operator lookup tables and Prefix suffix Read-raf checking protocols.
//!
//! This module provides infrastructure for proving operations using lookup tables,
//! particularly ReLU and comparison operations. The Prefix suffix Read-raf checking sum-check protocol
//! verifies correct reads from lookup tables and combines multiple claims using
//! gamma batching for efficiency.

use crate::utils::compute_lookup_indices_from_operands;
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait},
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::{
        ps_shout::{
            unary::{
                ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider, ReadRafClaims,
                UnaryReadRafSumcheckProver, UnaryReadRafSumcheckVerifier,
            },
            RafShoutProvider,
        },
        shout::RaOneHotEncoding,
    },
    transcripts::Transcript,
    utils::lookup_bits::LookupBits,
};

/// Provider for proving correct execution of operations using prefix-suffix structured lookup tables.
///
/// This provider implements the [`PrefixSuffixShoutProvider`] trait to enable efficient
/// lookups for operations that can be expressed via reads into prefix-suffix structured lookup tables (e.g., ReLU, comparison operations).
///
/// # Architecture
///
/// The provider operates on a single computation node from the ONNX trace and:
/// - Extracts operand data from the node's inputs
/// - Computes lookup table indices based on operand values
/// - Generates RAF claims for both prover and verifier
/// - Manages the sumcheck protocol for lookup verification
///
/// # Usage
///
/// ```ignore
/// let provider = OpLookupProvider::new(computation_node);
///
/// // Prover side
/// let (prover, indices) = provider.read_raf_prove::<F, _, ReLULookup>(
///     &trace,
///     &mut accumulator,
///     &mut transcript
/// );
///
/// // Verifier side
/// let verifier = provider.read_raf_verify::<F, _, ReLULookup>(
///     &mut accumulator,
///     &mut transcript
/// );
/// ```
///
/// # See Also
///
/// - [`PrefixSuffixShoutProvider`] - The trait this struct implements
/// - [`OpLookupEncoding`] - The struct that provides one-hot encoding parameters for one-hot checks related to these lookups
/// - [`ps_read_raf_prover`] and [`ps_read_raf_verifier`] - Underlying read-raf protocol
pub struct OpLookupProvider<Helper = DefaultLookupOperands>
where
    Helper: LookupOperandsTrait,
{
    /// The computation node being proven, containing operation type, inputs, and dimensionality.
    computation_node: ComputationNode,
    /// Helper providing operation-specific range-checking logic for the node's operands.
    helper: Helper,
}

/// Trait for custom lookup operation handling.
/// Allows to specify the link between model operands the the lookup table operands.
pub trait LookupOperandsTrait {
    /// log₂ of this lookup's address width.
    const LOG_K: usize;

    /// The lookup's "read value" claim (`rv_claim`): what the lookup's output is asserted to equal.
    fn rv_claim<F: JoltField>(node: &ComputationNode, accumulator: &dyn OpeningAccumulator<F>)
        -> F;

    /// Transforms the operand claims, accounting for lookup-specific adjustments (e.g., offsetting).
    /// Identity by default; override only if the lookup needs a claim-space transform.
    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        (claims[0], claims[1])
    }

    /// Transforms the output claim, accounting for lookup-specific adjustments (e.g., offsetting).
    /// Identity by default; see [`Self::transform_operand_claims`].
    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F {
        claim
    }

    /// Virtual polynomial identifying this op's one-hot read-address ("ra") polynomial.
    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly;

    /// Committed polynomial for the `d`-th chunk of this op's one-hot read-address
    /// decomposition. See [`Self::ra_virtual_poly`].
    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly;

    /// `OpeningId` this helper's lookup witness claim is registered/read under.
    fn witness_opening_id(node: &ComputationNode) -> OpeningId;

    /// The point (`r_cycle`) at which this helper's witness/operand is evaluated to produce
    /// the "raf" claim of the read-raf sumcheck. Every existing helper (`Clamp`,
    /// saturating-arithmetic clamps) uses the node's own output opening, since their pre-clamp
    /// witness naturally lives at the same point as the node's committed output. A helper whose
    /// witness is an internal, mid-pipeline value already evaluated at some other established
    /// point (e.g. softmax's saturating-clamp witness) overrides this to read that point instead.
    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    /// `OpeningId` whose accumulator entry's opening point equals [`Self::r_cycle`] — consumed
    /// by the one-hot `ra` checks (`OpLookupEncoding`'s `RaOneHotEncoding::r_cycle_source`) so
    /// they bind the same cycle point as the main lookup. Must resolve to the identical point
    /// as [`Self::r_cycle`].
    fn r_cycle_source(node_idx: usize) -> OpeningId;

    /// The polynomial that is evaluated at `r_cycle` to produce the "raf" claim of the read-raf sumcheck.
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64>;

    /// Computes the `LookupBits` used for the read-raf sumcheck + one-hot checks, from
    /// `witness` (the same value [`Self::witness`] already computed).
    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits>;
}

#[derive(Default)]
/// Default helper for operator lookup.
pub struct DefaultLookupOperands;

impl LookupOperandsTrait for DefaultLookupOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> F {
        accumulator.get_node_output_opening(node.idx).1
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::NodeOutputRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::NodeOutputRaD(node_idx, d)
    }

    /// The node's own first input, at this node's execution binding.
    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(node.idx),
        )
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator.get_node_output_opening(node.idx).0
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }

    /// Reads the node's first operand tensor directly from the trace.
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(trace, node);
        operands[0].padded_next_power_of_two().map(|v| v as i64)
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        let operand = witness.map(|v| v as i32);
        compute_lookup_indices_from_operands(&[&operand], false)
    }
}

impl<H: LookupOperandsTrait + Default> OpLookupProvider<H> {
    /// Creates a new lookup provider for the specified computation node.
    ///
    /// # Parameters
    ///
    /// - `computation_node`: The ONNX computation node whose execution will be proven
    ///   via lookup table queries. The node must support lookup-based operations
    ///   (e.g., ReLU, ULessThan).
    ///
    /// # Returns
    ///
    /// A new [`OpLookupProvider`] instance configured for the given node.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let node = trace.get_computation_node(node_idx);
    /// let provider = OpLookupProvider::new(node);
    /// ```
    pub fn new(computation_node: ComputationNode) -> Self {
        Self {
            computation_node,
            helper: H::default(),
        }
    }

    /// Creates a new lookup provider from an already-constructed `helper`, for helpers
    /// that carry per-call state (e.g. a precomputed witness) rather than being purely
    /// `Default`-constructed.
    pub fn with_helper(computation_node: ComputationNode, helper: H) -> Self {
        Self {
            computation_node,
            helper,
        }
    }

    /// Builds the [`OpLookupEncoding`] for this provider's node, using the same `Helper`
    /// so the Ra/RaD polynomial identifiers it exposes always match this provider's.
    pub fn encoding(&self) -> OpLookupEncoding<H> {
        OpLookupEncoding::new(&self.computation_node)
    }

    /// Combined prover flow: appends RAF claims + computes lookup indices + creates sumcheck prover.
    ///
    /// Returns `(sumcheck_prover, lookup_indices)` where `lookup_indices` can be reused
    /// for the one-hot encoding checks.
    pub fn read_raf_prove<F, T, LUT, const LOG_K: usize>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> (UnaryReadRafSumcheckProver<F, LUT, LOG_K>, Vec<usize>)
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    {
        let witness = self.helper.witness(&self.computation_node, trace);
        append_raf_claims_prover(self, &witness, accumulator, transcript);
        let lookup_bits = H::lookup_bits(&witness);
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }

    /// First half of [`Self::read_raf_prove`]: evaluate the lookup witness at
    /// `r_cycle`, append its opening claim, and return the lookup bits so the
    /// read-raf sumcheck itself can be built later (see
    /// [`Self::read_raf_prover_from_bits`] and `deferred_lookups`).
    pub fn append_witness_claim<F, T>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Vec<LookupBits>
    where
        F: JoltField,
        T: Transcript,
    {
        let witness = self.helper.witness(&self.computation_node, trace);
        append_raf_claims_prover(self, &witness, accumulator, transcript);
        H::lookup_bits(&witness)
    }

    /// Second half of [`Self::read_raf_prove`]: build the read-raf sumcheck
    /// prover from previously computed lookup bits (the witness claim must
    /// already be in the accumulator).
    pub fn read_raf_prover_from_bits<F, T, LUT, const LOG_K: usize>(
        &self,
        lookup_bits: Vec<LookupBits>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UnaryReadRafSumcheckProver<F, LUT, LOG_K>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    {
        ps_read_raf_prover(self, lookup_bits, accumulator, transcript)
    }

    /// Verifier counterpart of [`Self::append_witness_claim`].
    pub fn append_witness_claim_verifier<F, T>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) where
        F: JoltField,
        T: Transcript,
    {
        append_raf_claims_verifier(self, accumulator, transcript);
    }

    /// Verifier counterpart of [`Self::read_raf_prover_from_bits`].
    pub fn read_raf_verifier_only<F, T, LUT, const LOG_K: usize>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UnaryReadRafSumcheckVerifier<F, LUT, LOG_K>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    {
        ps_read_raf_verifier(self, accumulator, transcript)
    }

    /// Combined verifier flow: appends RAF claims + creates sumcheck verifier.
    pub fn read_raf_verify<F, T, LUT, const LOG_K: usize>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UnaryReadRafSumcheckVerifier<F, LUT, LOG_K>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    {
        append_raf_claims_verifier(self, accumulator, transcript);
        ps_read_raf_verifier(self, accumulator, transcript)
    }
}

impl<F, H> RafShoutProvider<F> for OpLookupProvider<H>
where
    F: JoltField,
    H: LookupOperandsTrait,
{
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        H::r_cycle(&self.computation_node, accumulator)
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            H::ra_virtual_poly(self.computation_node.idx),
            SumcheckId::NodeExecution(self.computation_node.idx),
        )
    }
}

impl<F, LUT, H, const LOG_K: usize> PrefixSuffixShoutProvider<F, LUT, LOG_K> for OpLookupProvider<H>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    H: LookupOperandsTrait,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
        let rv_claim = H::rv_claim(&self.computation_node, accumulator);
        let operand_id = H::witness_opening_id(&self.computation_node);
        let (_, operand_claim) = accumulator.get_virtual_polynomial_opening(operand_id);

        let rv_claim = self.helper.transform_output_claim(rv_claim);
        let (_, operand_claim) = self
            .helper
            .transform_operand_claims(vec![F::zero(), operand_claim]);

        ReadRafClaims {
            rv_claim,
            operand_claim,
        }
    }
}

// ---------------------------------------------------------------------------
// OpLookupEncoding — implements RaOneHotEncoding for op_lookups (ReLU, ULessThan, etc..)
// ---------------------------------------------------------------------------

/// Encoding for proving reads into prefix-suffix operator lookup tables.
///
/// Implements the [`RaOneHotEncoding`] trait to provide ra one-hot checks for
/// prefix-suffix lookups in the ONNX proof system.
///
/// Generic over the same `Helper` used by [`OpLookupProvider`], so that the ra/RaD
/// polynomial identifiers it exposes stay in sync with whichever virtual/committed
/// polynomial the helper's [`LookupOperandsTrait::ra_virtual_poly`] /
/// [`LookupOperandsTrait::ra_committed_poly`] designate.
pub struct OpLookupEncoding<Helper = DefaultLookupOperands> {
    /// Index of the computation node using this lookup encoding.
    pub node_idx: usize,
    /// log₂(T): number of output elements in the node.
    pub log_t: usize,
    _helper: std::marker::PhantomData<Helper>,
}

impl<Helper> OpLookupEncoding<Helper> {
    /// Creates a new operation lookup encoding for the given computation node.
    pub fn new(computation_node: &ComputationNode) -> Self {
        use joltworks::utils::math::Math;
        Self {
            node_idx: computation_node.idx,
            log_t: computation_node.pow2_padded_num_output_elements().log_2(),
            _helper: std::marker::PhantomData,
        }
    }
}

impl<Helper: LookupOperandsTrait> RaOneHotEncoding for OpLookupEncoding<Helper> {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        Helper::ra_committed_poly(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        Helper::r_cycle_source(self.node_idx)
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            Helper::ra_virtual_poly(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        Helper::LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t, self.log_k())
    }
}

/// Trait for determining if a computation node uses interleaved operand bits.
///
/// Some operations store operands with interleaved bits for prefix-suffix decomposition
/// . This trait provides a method to check if a node uses this representation.
pub trait InterleavedBitsMarker {
    /// Returns `true` if the operands are stored with interleaved bits.
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for ComputationNode {
    fn is_interleaved_operands(&self) -> bool {
        match self.operator {
            Operator::ReLU(_) => false,
            Operator::Clamp(_) => false,
            _ => unimplemented!(),
        }
    }
}

/// Appends `witness`'s opening claim (evaluated at this provider's `r_cycle`) under
/// `H::witness_opening_id`.
fn append_raf_claims_prover<F, H>(
    provider: &OpLookupProvider<H>,
    witness: &Tensor<i64>,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    F: JoltField,
    H: LookupOperandsTrait,
{
    let r_cycle =
        <OpLookupProvider<H> as RafShoutProvider<F>>::r_cycle(provider, opening_accumulator);
    let claim = MultilinearPolynomial::from(witness.data().to_vec()).evaluate(&r_cycle.r);
    let exec_id = H::witness_opening_id(&provider.computation_node);
    opening_accumulator.append_virtual(transcript, exec_id, r_cycle, claim);
}

/// Verifier counterpart of [`append_raf_claims_prover`]: appends the same opening
/// (claim loaded from the proof).
fn append_raf_claims_verifier<F, H>(
    provider: &OpLookupProvider<H>,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    F: JoltField,
    H: LookupOperandsTrait,
{
    let r_cycle =
        <OpLookupProvider<H> as RafShoutProvider<F>>::r_cycle(provider, opening_accumulator);
    let exec_id = H::witness_opening_id(&provider.computation_node);
    opening_accumulator.append_virtual(transcript, exec_id, r_cycle);
}
