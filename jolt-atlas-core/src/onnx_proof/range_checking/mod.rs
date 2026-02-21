//! Range-checking protocol for verifying bounds on remainder values.
//!
//! This module implements range-checking for operations that produce remainders (Div, Rsqrt, Tanh).
//! It uses the prefix-suffix read-checking sumcheck protocol from the Twist and Shout paper to
//! efficiently verify that remainder values are within valid bounds using lookup tables.

use crate::onnx_proof::range_checking::range_check_operands::RangeCheckingOperandsTrait;
use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode};
use common::{
    consts::{LOG_K, XLEN},
    CommittedPolynomial, VirtualPolynomial,
};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait},
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::{
        ps_shout::{
            ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider, ReadRafClaims,
            ReadRafSumcheckProver, ReadRafSumcheckVerifier,
        },
        shout::RaOneHotEncoding,
    },
    transcripts::Transcript,
    utils::math::Math,
};

/// Range-checking operands definitions for operations requiring bounds-checking.
///
/// Defines the [`RangeCheckingOperandsTrait`] trait and implementations for different operations
/// (Div, Rsqrt, Tanh) that need range-checking in a trait-friendly manner.
pub mod range_check_operands;

/// Provider for proving correct bounds on remainder values using range-checking.
///
/// This provider implements the [`PrefixSuffixShoutProvider`] trait to enable lookups
/// into the UnsignedLT table for operations that produce remainders (Div, Rsqrt, Tanh). It verifies that remainder
/// values fall within valid bounds by using the Read-RAF checking protocol with lookup tables.
///
/// # Type Parameters
///
/// - `H`: A [`RangeCheckingOperandsTrait`] implementation that provides operation-specific logic
///   for computing lookup indices and managing operand claims.
///
/// # Architecture
///
/// The range-checking provider:
/// - Uses a helper to extract operation-specific operands (e.g., numerator/denominator for Div)
/// - Computes lookup indices based on remainder bounds
/// - Generates RAF claims for both left and right operands
/// - Integrates with the prefix-suffix read-checking sumcheck protocol
///
/// Unlike [`OpLookupProvider`](crate::onnx_proof::op_lookups::OpLookupProvider), this provider
/// is designed specifically for operations where we need to prove bounds on computed remainders
/// rather than direct lookup results.
///
/// # Usage
///
/// ```ignore
/// use crate::onnx_proof::range_checking::{RangeCheckProvider, range_check_operands::DivHelper};
///
/// let provider = RangeCheckProvider::<DivHelper>::new(&div_node);
///
/// // Prover side
/// let (prover, indices) = provider.read_raf_prove::<F, _, RangeLookup>(
///     &trace,
///     &mut accumulator,
///     &mut transcript
/// );
///
/// // Verifier side
/// let verifier = provider.read_raf_verify::<F, _, RangeLookup>(
///     &mut accumulator,
///     &mut transcript
/// );
/// ```
///
/// # See Also
///
/// - [`RangeCheckingOperandsTrait`] - Trait for operation-specific range-checking logic
/// - [`RangeCheckEncoding`] - The encoding struct for one-hot checks related to range-checking operations
/// - [`PrefixSuffixShoutProvider`] - The trait this struct implements
pub struct RangeCheckProvider<H: RangeCheckingOperandsTrait> {
    /// Helper providing operation-specific range-checking logic (e.g., Div, Rsqrt, Tanh).
    pub helper: H,
    /// The computation node being proven, containing operation type, inputs, and dimensionality.
    pub computation_node: ComputationNode,
}

impl<H: RangeCheckingOperandsTrait> RangeCheckProvider<H> {
    /// Creates a new range-checking provider for the specified computation node.
    ///
    /// This initializes both the helper (with operation-specific logic) and stores
    /// the computation node for later access during the proving/verification protocol.
    ///
    /// # Parameters
    ///
    /// - `computation_node`: The ONNX computation node whose remainder values will be
    ///   range-checked. Must be an operation that produces remainders (Div, Rsqrt, Tanh).
    ///
    /// # Returns
    ///
    /// A new [`RangeCheckProvider`] instance configured for the given node with the
    /// appropriate helper implementation.
    ///
    /// # Type Parameters
    ///
    /// The generic `H` determines which operation is being proven:
    /// - `DivHelper` for division operations
    /// - `RsqrtHelper` for reciprocal square root operations
    /// - `TanhHelper` for hyperbolic tangent operations
    ///
    /// # Example
    ///
    /// ```ignore
    /// use crate::onnx_proof::range_checking::{RangeCheckProvider, range_check_operands::DivHelper};
    ///
    /// let div_node = trace.get_computation_node(node_idx);
    /// let provider = RangeCheckProvider::<DivHelper>::new(&div_node);
    /// ```
    pub fn new(computation_node: &ComputationNode) -> Self {
        Self {
            helper: H::new(computation_node),
            computation_node: computation_node.clone(),
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
    ) -> (ReadRafSumcheckProver<F, LUT>, Vec<usize>)
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        append_raf_claims_prover::<F, LUT, H>(
            &self.computation_node,
            self,
            trace,
            accumulator,
            transcript,
        );
        let (left, right) = H::get_operands_tensors(trace, &self.computation_node);
        let lookup_bits = H::compute_lookup_indices(&left, &right);
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }

    /// Combined verifier flow: appends RAF claims + creates sumcheck verifier.
    pub fn read_raf_verify<F, T, LUT>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> ReadRafSumcheckVerifier<F, LUT>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        append_raf_claims_verifier::<F, LUT, H>(self, accumulator, transcript);
        ps_read_raf_verifier(self, accumulator, transcript)
    }
}

impl<F, LUT, H> PrefixSuffixShoutProvider<F, LUT> for RangeCheckProvider<H>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
    H: RangeCheckingOperandsTrait,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
        let (left_operand_claim, right_operand_claim) = self.helper.operand_claims(accumulator);
        ReadRafClaims {
            rv_claim: F::one(),
            left_operand_claim,
            right_operand_claim,
        }
    }

    fn is_interleaved_operands(&self) -> bool {
        true
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_node_output, _) = accumulator.get_virtual_polynomial_opening(
            self.helper.get_input_operands()[0],
            SumcheckId::Execution,
        );
        r_node_output
    }

    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId) {
        (self.helper.get_output_operand(), SumcheckId::Raf)
    }

    fn raf_claim_specs(&self) -> Vec<(VirtualPolynomial, SumcheckId)> {
        self.helper
            .get_input_operands()
            .into_iter()
            .map(|p| (p, SumcheckId::Raf))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RangeCheckEncoding — implements RaOneHotEncoding for range-checking ops
// (Div, Rsqrt, Tanh)
// ---------------------------------------------------------------------------

/// Encoding for range-checking read-address one-hot checking.
///
/// This struct encapsulates the information needed to perform one-hot-encoding checks for
/// operations that produce remainders (Div, Rsqrt, Tanh). It implements the
/// [`RaOneHotEncoding`] trait to integrate with the one-hot checking protocol.
pub struct RangeCheckEncoding<H: RangeCheckingOperandsTrait> {
    /// Helper providing operation-specific range-checking logic.
    pub helper: H,
    /// log2 of the number of elements in the computation (T).
    pub log_t: usize,
}

impl<H: RangeCheckingOperandsTrait> RangeCheckEncoding<H> {
    /// Create a new range-check encoding from a computation node.
    pub fn new(computation_node: &ComputationNode) -> Self {
        Self {
            helper: H::new(computation_node),
            log_t: computation_node.num_output_elements().log_2(),
        }
    }
}

impl<H: RangeCheckingOperandsTrait> RaOneHotEncoding for RangeCheckEncoding<H> {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        self.helper.rad_poly(d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutput(self.helper.node_idx()),
            SumcheckId::Execution,
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (self.helper.get_output_operand(), SumcheckId::Raf)
    }

    fn log_k(&self) -> usize {
        LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t)
    }
}

fn append_raf_claims_prover<F: JoltField, LUT, H>(
    computation_node: &ComputationNode,
    provider: &RangeCheckProvider<H>,
    trace: &Trace,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    H: RangeCheckingOperandsTrait,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    let (left_operand, right_operand) = H::get_operands_tensors(trace, computation_node);
    let r_cycle = <RangeCheckProvider<H> as PrefixSuffixShoutProvider<F, LUT>>::r_cycle(
        provider,
        opening_accumulator,
    );

    // Cache left/right operand claims. (In our case they have been cached in Execution sumcheck)
    let operand_tensors = [left_operand, right_operand];
    let raf_specs =
        <RangeCheckProvider<H> as PrefixSuffixShoutProvider<F, LUT>>::raf_claim_specs(provider);
    for (tensor, (poly, sumcheck_id)) in operand_tensors.into_iter().zip(raf_specs) {
        let claim = MultilinearPolynomial::from(tensor.into_container_data()) // TODO: make this work with from_i32
            .evaluate(&r_cycle.r);
        opening_accumulator.append_virtual(transcript, poly, sumcheck_id, r_cycle.clone(), claim);
    }
}

fn append_raf_claims_verifier<F: JoltField, LUT, H>(
    provider: &RangeCheckProvider<H>,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    H: RangeCheckingOperandsTrait,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    let r_cycle = <RangeCheckProvider<H> as PrefixSuffixShoutProvider<F, LUT>>::r_cycle(
        provider,
        opening_accumulator,
    );
    for (poly, sumcheck_id) in
        <RangeCheckProvider<H> as PrefixSuffixShoutProvider<F, LUT>>::raf_claim_specs(provider)
    {
        opening_accumulator.append_virtual(transcript, poly, sumcheck_id, r_cycle.clone());
    }
}
