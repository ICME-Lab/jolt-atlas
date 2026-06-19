//! Range-checking protocol for verifying bounds on remainder values.
//!
//! This module implements range-checking for operations that produce remainders (Div, Rsqrt, Tanh).
//! It uses the prefix-suffix read-checking sumcheck protocol from the Twist and Shout paper to
//! efficiently verify that remainder values are within valid bounds using lookup tables.

use crate::onnx_proof::range_checking::range_check_operands::{
    RangeCheckOperands, RangeCheckingOperandsTrait,
};
use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode};
use common::{
    consts::{LOG_K, XLEN},
    CommittedPoly, VirtualPoly,
};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait},
    poly::opening_proof::{
        OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
        VerifierOpeningAccumulator, BIG_ENDIAN,
    },
    subprotocols::{
        ps_shout::{
            binary::{
                ps_read_raf_prover, ps_read_raf_verifier, BinaryReadRafSumcheckProver,
                BinaryReadRafSumcheckVerifier, PrefixSuffixShoutProvider, ReadRafClaims,
            },
            RafShoutProvider,
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
/// - Uses the operands to extract operation-specific values (e.g., numerator/denominator for Div)
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
/// use crate::onnx_proof::range_checking::{RangeCheckProvider, range_check_operands::DivRangeCheckOperands};
///
/// let provider = RangeCheckProvider::<DivRangeCheckOperands>::new(&div_node);
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
    /// Operands providing operation-specific range-checking logic (e.g., Div, Rsqrt, Tanh).
    pub operands: RangeCheckOperands<H>,
    /// The computation node being proven, containing operation type, inputs, and dimensionality.
    pub computation_node: ComputationNode,
}

impl<H: RangeCheckingOperandsTrait> RangeCheckProvider<H> {
    /// Creates a new range-checking provider for the specified computation node.
    ///
    /// This initializes the operands (with operation-specific logic) and stores
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
    /// appropriate operands implementation.
    ///
    /// # Type Parameters
    ///
    /// The generic `H` determines which operation is being proven:
    /// - [`DivRangeCheckOperands`](range_check_operands::DivRangeCheckOperands) for division operations
    /// - [`RiRangeCheckOperands`](range_check_operands::RiRangeCheckOperands) / [`RsRangeCheckOperands`](range_check_operands::RsRangeCheckOperands) for reciprocal square root operations
    /// - [`TeleportRangeCheckOperands`](range_check_operands::TeleportRangeCheckOperands) for tanh operations
    ///
    /// # Example
    ///
    /// ```ignore
    /// use crate::onnx_proof::range_checking::{RangeCheckProvider, range_check_operands::DivRangeCheckOperands};
    ///
    /// let div_node = trace.get_computation_node(node_idx);
    /// let provider = RangeCheckProvider::<DivRangeCheckOperands>::new(&div_node);
    /// ```
    pub fn new(computation_node: &ComputationNode) -> Self {
        Self {
            operands: RangeCheckOperands::new(computation_node),
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
    ) -> (BinaryReadRafSumcheckProver<F, LUT>, Vec<usize>)
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        let padded_operands = trace.padded_operand_tensors(&self.computation_node);
        let lookup_operands = self.operands.build_lookup_operands(&padded_operands);

        let lookup_bits = self
            .operands
            .compute_lookup_indices(&lookup_operands[0], &lookup_operands[1]);
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }

    /// Combined verifier flow: appends RAF claims + creates sumcheck verifier.
    pub fn read_raf_verify<F, T, LUT>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> BinaryReadRafSumcheckVerifier<F, LUT>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        ps_read_raf_verifier(self, accumulator, transcript)
    }
}

impl<F, H> RafShoutProvider<F> for RangeCheckProvider<H>
where
    F: JoltField,
    H: RangeCheckingOperandsTrait,
{
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        let id = OpeningId::new(
            self.operands.get_input_operands_id()[0],
            SumcheckId::NodeExecution(self.operands.node_idx()),
        );
        let (r_node_output, _) = accumulator.get_virtual_polynomial_opening(id);
        r_node_output
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            self.operands.get_output_operand_id(),
            SumcheckId::NodeExecution(self.operands.node_idx()),
        )
    }
}

impl<F, LUT, H> PrefixSuffixShoutProvider<F, LUT> for RangeCheckProvider<H>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    H: RangeCheckingOperandsTrait,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
        let (left_operand_claim, right_operand_claim) = self.operands.operand_claims(accumulator);
        ReadRafClaims {
            rv_claim: F::one(),
            left_operand_claim,
            right_operand_claim,
        }
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
pub struct RangeCheckEncoding {
    /// Computation node index
    pub node_idx: usize,
    /// Virtual polynomial identifier, mapping to the one-hot encoding of the range-check output claim.
    pub one_hot_poly: VirtualPoly,
    /// Committed polynomial function for the one-hot encoding, representing `one_hot_poly`.
    pub one_hot_committed_fn: fn(usize, usize) -> CommittedPoly,
    /// log2 of the number of elements in the computation (T).
    pub log_t: usize,
}

impl RangeCheckEncoding {
    /// Create a new range-check encoding from a computation node.
    pub fn new<H: RangeCheckingOperandsTrait>(
        computation_node: &ComputationNode,
        rc_op: &RangeCheckOperands<H>,
    ) -> Self {
        Self {
            node_idx: computation_node.idx,
            one_hot_poly: rc_op.get_output_operand_id(),
            one_hot_committed_fn: H::rad_poly,
            log_t: computation_node.pow2_padded_num_output_elements().log_2(),
        }
    }
}

impl RaOneHotEncoding for RangeCheckEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        (self.one_hot_committed_fn)(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(self.one_hot_poly, SumcheckId::NodeExecution(self.node_idx))
    }

    fn log_k(&self) -> usize {
        LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t, self.log_k())
    }
}
