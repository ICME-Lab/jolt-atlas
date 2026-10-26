//! Range-checking protocol for verifying bounds on remainder values.
//!
//! This module implements range-checking for operations that produce remainders (Div, Rsqrt, Tanh).
//! It uses the prefix-suffix read-checking sumcheck protocol from the Twist and Shout paper to
//! efficiently verify that remainder values are within valid bounds using lookup tables.

use atlas_trace::Trace;
pub use jolt_atlas_verifier::onnx_proof::range_checking::*;

use crate::onnx_proof::range_checking::range_check_operands::RangeCheckingOperandsTrait;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::field::JoltField;
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_sumcheck::{
    opening_proof::ProverOpeningAccumulator,
    ps_shout::binary::{ps_read_raf_prover, BinaryReadRafSumcheckProver},
};
use atlas_transcript::Transcript;
use common::consts::XLEN;

/// Range-checking operands definitions for operations requiring bounds-checking.
///
/// Defines the [`RangeCheckingOperandsTrait`] trait and implementations for different operations
/// (Div, Rsqrt, Tanh) that need range-checking in a trait-friendly manner.
pub mod range_check_operands;

/// Prover-side methods of [`RangeCheckProvider`].
pub trait RangeCheckProve {
    /// Combined prover flow: appends RAF claims + computes lookup indices + creates sumcheck prover.
    ///
    /// Returns `(sumcheck_prover, lookup_indices)` where `lookup_indices` can be reused
    /// for the one-hot encoding checks.
    fn read_raf_prove<F, T, LUT>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> (BinaryReadRafSumcheckProver<F, LUT>, Vec<usize>)
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default;
}

impl<H: RangeCheckingOperandsTrait> RangeCheckProve for RangeCheckProvider<H> {
    /// Combined prover flow: appends RAF claims + computes lookup indices + creates sumcheck prover.
    ///
    /// Returns `(sumcheck_prover, lookup_indices)` where `lookup_indices` can be reused
    /// for the one-hot encoding checks.
    fn read_raf_prove<F, T, LUT>(
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
        let padded_operands = TraceView::padded_operand_tensors(trace, &self.computation_node);
        let lookup_operands = self.operands.build_lookup_operands(trace, &padded_operands);

        let lookup_bits = self
            .operands
            .compute_lookup_indices(&lookup_operands[0], &lookup_operands[1]);
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }
}
