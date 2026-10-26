//! Op lookup tables and Prefix suffix Read-raf checking protocols.
//!
//! This module provides infrastructure for proving operations using lookup tables,
//! particularly ReLU and comparison operations. The Prefix suffix Read-raf checking sum-check protocol
//! verifies correct reads from lookup tables and combines multiple claims using
//! gamma batching for efficiency.

use atlas_trace::{Tensor, Trace};
pub use jolt_atlas_verifier::onnx_proof::op_lookups::*;

use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::EvaluateMle, utils::lookup_bits::LookupBits,
};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_sumcheck::{
    opening_proof::ProverOpeningAccumulator,
    ps_shout::{
        unary::{ps_read_raf_prover, UnaryReadRafSumcheckProver},
        RafShoutProvider,
    },
};
use atlas_transcript::Transcript;

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
    let claim = witness.evaluate_mle(&r_cycle.r);
    let exec_id = H::witness_opening_id(&provider.computation_node);
    opening_accumulator.append_virtual(transcript, exec_id, r_cycle, claim);
}

/// Prover-side methods of [`OpLookupProvider`].
pub trait OpLookupProve {
    /// Combined prover flow: appends RAF claims + computes lookup indices + creates sumcheck prover.
    ///
    /// Returns `(sumcheck_prover, lookup_indices)` where `lookup_indices` can be reused
    /// for the one-hot encoding checks.
    fn read_raf_prove<F, T, LUT, const LOG_K: usize>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> (UnaryReadRafSumcheckProver<F, LUT, LOG_K>, Vec<usize>)
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default;

    /// First half of [`Self::read_raf_prove`]: evaluate the lookup witness at
    /// `r_cycle`, append its opening claim, and return the lookup bits so the
    /// read-raf sumcheck itself can be built later (see
    /// [`Self::read_raf_prover_from_bits`] and `deferred_lookups`).
    fn append_witness_claim<F, T>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Vec<LookupBits>
    where
        F: JoltField,
        T: Transcript;

    /// [`Self::append_witness_claim`] for a witness the caller already holds
    /// (returns nothing; the caller keeps the witness).
    fn append_witness_claim_for<F, T>(
        &self,
        witness: &Tensor<i64>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) where
        F: JoltField,
        T: Transcript;

    /// Second half of [`Self::read_raf_prove`]: build the read-raf sumcheck
    /// prover from previously computed lookup bits (the witness claim must
    /// already be in the accumulator).
    fn read_raf_prover_from_bits<F, T, LUT, const LOG_K: usize>(
        &self,
        lookup_bits: Vec<LookupBits>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UnaryReadRafSumcheckProver<F, LUT, LOG_K>
    where
        F: JoltField,
        T: Transcript,
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default;
}

impl<H: LookupOperandsTrait> OpLookupProve for OpLookupProvider<H> {
    /// Combined prover flow: appends RAF claims + computes lookup indices + creates sumcheck prover.
    ///
    /// Returns `(sumcheck_prover, lookup_indices)` where `lookup_indices` can be reused
    /// for the one-hot encoding checks.
    fn read_raf_prove<F, T, LUT, const LOG_K: usize>(
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
        let lookup_bits = self.helper.lookup_bits(&witness);
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }

    /// First half of [`Self::read_raf_prove`]: evaluate the lookup witness at
    /// `r_cycle`, append its opening claim, and return the lookup bits so the
    /// read-raf sumcheck itself can be built later (see
    /// [`Self::read_raf_prover_from_bits`] and `deferred_lookups`).
    fn append_witness_claim<F, T>(
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
        self.helper.lookup_bits(&witness)
    }

    /// [`Self::append_witness_claim`] for a witness the caller already holds
    /// (returns nothing; the caller keeps the witness).
    fn append_witness_claim_for<F, T>(
        &self,
        witness: &Tensor<i64>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) where
        F: JoltField,
        T: Transcript,
    {
        append_raf_claims_prover(self, witness, accumulator, transcript);
    }

    /// Second half of [`Self::read_raf_prove`]: build the read-raf sumcheck
    /// prover from previously computed lookup bits (the witness claim must
    /// already be in the accumulator).
    fn read_raf_prover_from_bits<F, T, LUT, const LOG_K: usize>(
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
}
