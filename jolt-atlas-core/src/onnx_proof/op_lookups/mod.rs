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
    /// Transforms the operand claims, accounting for lookup-specific adjustments (e.g., offsetting).
    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F);

    /// Transforms the output claim, accounting for lookup-specific adjustments (e.g., offsetting).
    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F;

    /// Builds the encoding operands for the lookup table from the model's operand tensors.
    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>>;
}

#[derive(Default)]
/// Default helper for operator lookup.
pub struct DefaultLookupOperands;

impl LookupOperandsTrait for DefaultLookupOperands {
    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        (claims[0], claims[1])
    }

    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F {
        claim
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        operand_tensors.to_vec()
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

    /// Combined prover flow: appends RAF claims + computes lookup indices + creates sumcheck prover.
    ///
    /// Returns `(sumcheck_prover, lookup_indices)` where `lookup_indices` can be reused
    /// for the one-hot encoding checks.
    pub fn read_raf_prove<F, T, LUT>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> (UnaryReadRafSumcheckProver<F, LUT, XLEN>, Vec<usize>)
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        append_raf_claims_prover(self, trace, accumulator, transcript);
        let padded_operands = trace.padded_operand_tensors(&self.computation_node);
        let lookup_operands = self.helper.build_lookup_operands(&padded_operands);
        let operand_refs: Vec<_> = lookup_operands.iter().collect();
        let lookup_bits = compute_lookup_indices_from_operands(
            &operand_refs,
            self.computation_node.is_interleaved_operands(),
        );
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }

    /// Combined verifier flow: appends RAF claims + creates sumcheck verifier.
    pub fn read_raf_verify<F, T, LUT>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UnaryReadRafSumcheckVerifier<F, LUT, XLEN>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
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
        let (r_node_output, _) = accumulator.get_node_output_opening(self.computation_node.idx);
        r_node_output
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            VirtualPoly::NodeOutputRa(self.computation_node.idx),
            SumcheckId::NodeExecution(self.computation_node.idx),
        )
    }
}

impl<F, LUT, H> PrefixSuffixShoutProvider<F, LUT, XLEN> for OpLookupProvider<H>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    H: LookupOperandsTrait,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
        let (_, rv_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        let operand_id = OpeningId::new(
            VirtualPoly::NodeOutput(self.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.computation_node.idx),
        );
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
pub struct OpLookupEncoding {
    /// Index of the computation node using this lookup encoding.
    pub node_idx: usize,
    /// log₂(T): number of output elements in the node.
    pub log_t: usize,
}

impl OpLookupEncoding {
    /// Creates a new operation lookup encoding for the given computation node.
    pub fn new(computation_node: &ComputationNode) -> Self {
        use joltworks::utils::math::Math;
        Self {
            node_idx: computation_node.idx,
            log_t: computation_node.pow2_padded_num_output_elements().log_2(),
        }
    }
}

impl RaOneHotEncoding for OpLookupEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::NodeOutputRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutputRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        XLEN
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

fn append_raf_claims_prover<F, H>(
    provider: &OpLookupProvider<H>,
    trace: &Trace,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    F: JoltField,
    H: LookupOperandsTrait,
{
    let r_cycle =
        <OpLookupProvider<H> as RafShoutProvider<F>>::r_cycle(provider, opening_accumulator);
    let LayerData {
        output: _,
        operands,
    } = Trace::layer_data(trace, &provider.computation_node);
    let operand_tensor = operands[0].padded_next_power_of_two();
    let operand_claim = MultilinearPolynomial::from(operand_tensor.clone()).evaluate(&r_cycle.r);
    let exec_id = OpeningId::new(
        VirtualPoly::NodeOutput(provider.computation_node.inputs[0]),
        SumcheckId::NodeExecution(provider.computation_node.idx),
    );
    opening_accumulator.append_virtual(transcript, exec_id, r_cycle, operand_claim);
}

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
    let node = &provider.computation_node;
    opening_accumulator.get_node_output_opening(node.idx);
    let exec_id = SumcheckId::NodeExecution(node.idx);
    let input_opening = OpeningId::new(VirtualPoly::NodeOutput(node.inputs[0]), exec_id);
    opening_accumulator.append_virtual(transcript, input_opening, r_cycle);
}
