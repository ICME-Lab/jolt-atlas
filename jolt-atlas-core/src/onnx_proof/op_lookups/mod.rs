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
};
use common::{
    consts::{LOG_K, XLEN},
    CommittedPoly, VirtualPoly,
};
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
            ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider, ReadRafClaims,
            ReadRafSumcheckProver, ReadRafSumcheckVerifier,
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
pub struct OpLookupProvider {
    /// The computation node being proven, containing operation type, inputs, and dimensionality.
    computation_node: ComputationNode,
}

impl OpLookupProvider {
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
        Self { computation_node }
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
        append_raf_claims_prover::<F, LUT>(self, trace, accumulator, transcript);
        let padded_operands: Vec<_> = trace
            .layer_data(&self.computation_node)
            .operands
            .iter()
            .map(|tensor| tensor.padded_next_power_of_two())
            .collect();
        let operand_refs: Vec<_> = padded_operands.iter().collect();
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
    ) -> ReadRafSumcheckVerifier<F, LUT>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        append_raf_claims_verifier::<F, LUT>(self, accumulator, transcript);
        ps_read_raf_verifier(self, accumulator, transcript)
    }
}

impl<F, LUT> PrefixSuffixShoutProvider<F, LUT> for OpLookupProvider
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
        let (_, rv_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        let (left_operand_claim, right_operand_claim) =
            if self.computation_node.is_interleaved_operands() {
                let left_operand_id = OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::Raf,
                );
                let right_operand_id = OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[1]),
                    SumcheckId::Raf,
                );
                let (_, left_operand_claim) =
                    accumulator.get_virtual_polynomial_opening(left_operand_id);
                let (_, right_operand_claim) =
                    accumulator.get_virtual_polynomial_opening(right_operand_id);
                (left_operand_claim, right_operand_claim)
            } else {
                let right_operand_id = OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::Raf,
                );
                let (_, right_operand_claim) =
                    accumulator.get_virtual_polynomial_opening(right_operand_id);
                (F::zero(), right_operand_claim)
            };

        ReadRafClaims {
            rv_claim,
            left_operand_claim,
            right_operand_claim,
        }
    }

    fn is_interleaved_operands(&self) -> bool {
        self.computation_node.is_interleaved_operands()
    }

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

    fn raf_claim_specs(&self) -> Vec<OpeningId> {
        if self.computation_node.is_interleaved_operands() {
            vec![
                OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::Raf,
                ),
                OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[1]),
                    SumcheckId::Raf,
                ),
            ]
        } else {
            vec![
                OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::Raf,
                ),
                OpeningId::new(
                    VirtualPoly::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.computation_node.idx),
                ),
            ]
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
        LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t)
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
            _ => unimplemented!(),
        }
    }
}

fn append_raf_claims_prover<F: JoltField, LUT>(
    provider: &OpLookupProvider,
    trace: &Trace,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    let r_cycle = <OpLookupProvider as PrefixSuffixShoutProvider<F, LUT>>::r_cycle(
        provider,
        opening_accumulator,
    );
    let LayerData {
        output: _,
        operands,
    } = Trace::layer_data(trace, &provider.computation_node);
    let is_interleaved_operands = provider.computation_node.is_interleaved_operands();
    if is_interleaved_operands {
        let [left_operand_tensor, right_operand_tensor] = operands[..] else {
            panic!("Expected exactly two input tensors")
        };

        let left_operand_tensor = left_operand_tensor.padded_next_power_of_two();
        let right_operand_tensor = right_operand_tensor.padded_next_power_of_two();

        // Cache left/right operand claims.
        let left_operand_claim =
            MultilinearPolynomial::from(left_operand_tensor.into_container_data()) // TODO: make this work with from_i32
                .evaluate(&r_cycle.r);
        let left_raf_id = OpeningId::new(
            VirtualPoly::NodeOutput(provider.computation_node.inputs[0]),
            SumcheckId::Raf,
        );
        opening_accumulator.append_virtual(
            transcript,
            left_raf_id,
            r_cycle.clone(),
            left_operand_claim,
        );
        let right_operand_claim =
            MultilinearPolynomial::from(right_operand_tensor.into_container_data())
                .evaluate(&r_cycle.r);
        let right_raf_id = OpeningId::new(
            VirtualPoly::NodeOutput(provider.computation_node.inputs[1]),
            SumcheckId::Raf,
        );
        opening_accumulator.append_virtual(
            transcript,
            right_raf_id,
            r_cycle.clone(),
            right_operand_claim,
        );

        // HACK: we should modify RAF operand polynomials for proving these claims
        let left_operand_claim = MultilinearPolynomial::from(left_operand_tensor.clone()) // TODO: make this work with from_i32
            .evaluate(&r_cycle.r);
        let left_exec_id = OpeningId::new(
            VirtualPoly::NodeOutput(provider.computation_node.inputs[0]),
            SumcheckId::NodeExecution(provider.computation_node.idx),
        );
        opening_accumulator.append_virtual(
            transcript,
            left_exec_id,
            r_cycle.clone(),
            left_operand_claim,
        );
        let right_operand_claim =
            MultilinearPolynomial::from(right_operand_tensor.clone()).evaluate(&r_cycle.r);
        let right_exec_id = OpeningId::new(
            VirtualPoly::NodeOutput(provider.computation_node.inputs[1]),
            SumcheckId::NodeExecution(provider.computation_node.idx),
        );
        opening_accumulator.append_virtual(
            transcript,
            right_exec_id,
            r_cycle.clone(),
            right_operand_claim,
        );
    } else {
        let right_operand_tensor = operands[0].padded_next_power_of_two();
        let right_operand_claim =
            MultilinearPolynomial::from(right_operand_tensor.into_container_data())
                .evaluate(&r_cycle.r);
        let right_raf_id = OpeningId::new(
            VirtualPoly::NodeOutput(provider.computation_node.inputs[0]),
            SumcheckId::Raf,
        );
        opening_accumulator.append_virtual(
            transcript,
            right_raf_id,
            r_cycle.clone(),
            right_operand_claim,
        );
        let right_operand_claim =
            MultilinearPolynomial::from(right_operand_tensor.clone()).evaluate(&r_cycle.r);
        let right_exec_id = OpeningId::new(
            VirtualPoly::NodeOutput(provider.computation_node.inputs[0]),
            SumcheckId::NodeExecution(provider.computation_node.idx),
        );
        opening_accumulator.append_virtual(
            transcript,
            right_exec_id, // TODO: Add specialized sumcheck that relates raf and execution claims (signed vs unsigned claims)
            r_cycle.clone(),
            right_operand_claim,
        );
    };
}

fn append_raf_claims_verifier<F: JoltField, LUT>(
    provider: &OpLookupProvider,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) where
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    let r_cycle = <OpLookupProvider as PrefixSuffixShoutProvider<F, LUT>>::r_cycle(
        provider,
        opening_accumulator,
    );
    opening_accumulator.get_node_output_opening(provider.computation_node.idx);
    for opening_id in
        <OpLookupProvider as PrefixSuffixShoutProvider<F, LUT>>::raf_claim_specs(provider)
    {
        opening_accumulator.append_virtual(transcript, opening_id, r_cycle.clone());
    }
}
