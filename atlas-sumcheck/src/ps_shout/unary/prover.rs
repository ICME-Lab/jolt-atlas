//! The prover side of [`super`].

use super::*;
use crate::{
    opening_proof::prover::ProverOpeningAccumulator,
    ps_shout::{
        phase_schedule, prover::ReadRafSumcheckProver, RafProverState, ReadRafSumcheckParams,
    },
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        signed_identity::SignedIdentityPoly,
    },
    utils::lookup_bits::LookupBits,
};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
use num_traits::Zero;

impl<F: JoltField> RafProverState<F> for UnaryRafPS<F> {
    fn init_Q(&mut self, u_evals: &[F], lookup_indices: &[LookupBits]) {
        self.identity_ps.init_Q(u_evals, lookup_indices);
    }

    fn init_P(&mut self, registry: &mut PrefixRegistry<F>) {
        self.identity_ps.init_P(registry);
    }

    fn prover_msg(&self, gamma: F) -> [F; 2] {
        let len = self.identity_ps.Q_len();
        let [operand_0, operand_2] = (0..len / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|b| {
                let (o0, o2) = self.identity_ps.sumcheck_evals(b);
                [o0.to_unreduced(), o2.to_unreduced()]
            })
            .fold_with([F::UnreducedMulU64::zero(); 2], |running, new| {
                [running[0] + new[0], running[1] + new[1]]
            })
            .reduce(
                || [F::UnreducedMulU64::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        [
            F::reduce_mul_u64(operand_0) * gamma,
            F::reduce_mul_u64(operand_2) * gamma,
        ]
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.identity_ps.bind(r_j);
    }

    fn raf_val(&self, registry: &PrefixRegistry<F>, gamma: F) -> F {
        let effective_identity = registry.checkpoints[Prefix::SignedIdentity].unwrap();
        gamma * effective_identity
    }
}

pub fn ps_read_raf_prover<
    F: JoltField,
    T: Transcript,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    const LOG_K: usize,
>(
    provider: &impl PrefixSuffixShoutProvider<F, LUT, LOG_K>,
    lookup_indices: Vec<LookupBits>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> UnaryReadRafSumcheckProver<F, LUT, LOG_K> {
    let gamma = transcript.challenge_scalar();
    let claims = provider
        .read_raf_claims(accumulator)
        .expect("the prover holds the lookup's claims");
    let raf = UnaryRafVD {
        operand_claim: claims.operand_claim,
        log_k: LOG_K,
    };
    let r_node_output = provider
        .r_cycle(accumulator)
        .expect("the prover holds the lookup's cycle point");
    let log_T = r_node_output.len();
    let (polynomial_type, sumcheck_id) = provider.ra_poly();
    let params = ReadRafSumcheckParams::from_parts(
        gamma,
        log_T,
        r_node_output,
        LUT::default(),
        polynomial_type,
        sumcheck_id,
        claims.rv_claim,
        raf,
    );
    let (_phases, log_m) = phase_schedule(LOG_K);
    let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
    let _guard = span.enter();
    let identity_ps =
        PrefixSuffixDecomposition::new(Box::new(SignedIdentityPoly::<F>::new(LOG_K)), log_m, LOG_K);
    drop(_guard);
    let raf_state = UnaryRafPS { identity_ps };
    ReadRafSumcheckProver::new_inner(params, lookup_indices, raf_state)
}

pub type UnaryReadRafSumcheckProver<F, LUT, const LOG_K: usize> =
    ReadRafSumcheckProver<F, LUT, LOG_K, LOG_K, UnaryRafVD<F>, UnaryRafPS<F>>;
