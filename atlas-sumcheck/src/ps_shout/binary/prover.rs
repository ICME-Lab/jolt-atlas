//! The prover side of [`super`].

use super::*;
use crate::{
    opening_proof::prover::ProverOpeningAccumulator,
    ps_shout::{
        phase_schedule, prover::ReadRafSumcheckProver, RafProverState, ReadRafSumcheckParams,
    },
};
use ark_std::Zero;
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity_poly::OperandSide,
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        signed_identity_poly::SignedOperandPoly,
    },
    utils::lookup_bits::LookupBits,
};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_transcript::Transcript;
use common::{
    consts::{LOG_K, XLEN},
    parallel::par_enabled,
};

impl<F: JoltField> RafProverState<F> for BinaryRafPS<F> {
    fn init_Q(&mut self, u_evals: &[F], lookup_indices: &[LookupBits]) {
        PrefixSuffixDecomposition::init_Q_dual(
            &mut self.left_operand_ps,
            &mut self.right_operand_ps,
            u_evals,
            lookup_indices,
        );
    }

    fn init_P(&mut self, registry: &mut PrefixRegistry<F>) {
        self.right_operand_ps.init_P(registry);
        self.left_operand_ps.init_P(registry);
    }

    fn prover_msg(&self, gamma: F) -> [F; 2] {
        let gamma_sqr = gamma * gamma;
        let len = self.right_operand_ps.Q_len();
        let [left_0, left_2, right_0, right_2] = (0..len / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|b| {
                let (r0, r2) = self.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = self.left_operand_ps.sumcheck_evals(b);
                [
                    l0.to_unreduced(),
                    l2.to_unreduced(),
                    r0.to_unreduced(),
                    r2.to_unreduced(),
                ]
            })
            .fold_with([F::UnreducedMulU64::zero(); 4], |running, new| {
                [
                    running[0] + new[0],
                    running[1] + new[1],
                    running[2] + new[2],
                    running[3] + new[3],
                ]
            })
            .reduce(
                || [F::UnreducedMulU64::zero(); 4],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                    ]
                },
            );
        [
            F::reduce_mul_u64(left_0) * gamma + F::reduce_mul_u64(right_0) * gamma_sqr,
            F::reduce_mul_u64(left_2) * gamma + F::reduce_mul_u64(right_2) * gamma_sqr,
        ]
    }

    fn bind(&mut self, r_j: F::Challenge) {
        rayon::join(
            || self.left_operand_ps.bind(r_j),
            || self.right_operand_ps.bind(r_j),
        );
    }

    fn raf_val(&self, registry: &PrefixRegistry<F>, gamma: F) -> F {
        let gamma_sqr = gamma * gamma;
        let cp = &registry.checkpoints;
        let effective_left = cp[Prefix::LeftOperand].unwrap() + cp[Prefix::LeftOperandMSB].unwrap();
        let effective_right =
            cp[Prefix::RightOperand].unwrap() + cp[Prefix::RightOperandMSB].unwrap();
        gamma * effective_left + gamma_sqr * effective_right
    }
}

pub fn ps_read_raf_prover<
    F: JoltField,
    T: Transcript,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
>(
    provider: &impl PrefixSuffixShoutProvider<F, LUT>,
    lookup_indices: Vec<LookupBits>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> BinaryReadRafSumcheckProver<F, LUT> {
    let gamma = transcript.challenge_scalar();
    let claims = provider.read_raf_claims(accumulator);
    let raf = BinaryRafVD {
        left_operand_claim: claims.left_operand_claim,
        right_operand_claim: claims.right_operand_claim,
    };
    let r_node_output = provider.r_cycle(accumulator);
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
    let left_operand_ps = PrefixSuffixDecomposition::new(
        Box::new(SignedOperandPoly::<F, XLEN>::new(OperandSide::Left)),
        log_m,
        LOG_K,
    );
    let right_operand_ps = PrefixSuffixDecomposition::new(
        Box::new(SignedOperandPoly::<F, XLEN>::new(OperandSide::Right)),
        log_m,
        LOG_K,
    );
    drop(_guard);
    let raf_state = BinaryRafPS {
        left_operand_ps,
        right_operand_ps,
    };
    ReadRafSumcheckProver::new_inner(params, lookup_indices, raf_state)
}

pub type BinaryReadRafSumcheckProver<F, LUT> =
    ReadRafSumcheckProver<F, LUT, LOG_K, XLEN, BinaryRafVD<F>, BinaryRafPS<F>>;
