use crate::{
    field::{JoltField, MulTrunc},
    lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait},
    poly::{
        identity_poly::OperandSide,
        multilinear_polynomial::PolynomialEvaluation,
        opening_proof::{OpeningAccumulator, ProverOpeningAccumulator, VerifierOpeningAccumulator},
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        signed_identity_poly::SignedOperandPoly,
    },
    subprotocols::ps_shout::{
        RafProverState, RafVerifierData, ReadRafSumcheckParams, ReadRafSumcheckProver,
        ReadRafSumcheckVerifier, NUM_PHASES,
    },
    transcripts::Transcript,
    utils::lookup_bits::LookupBits,
};
use ark_std::Zero;
use common::{
    consts::{LOG_K, XLEN},
    parallel::par_enabled,
};
use rayon::prelude::*;

/// Verifier-side RAF data for binary ops (AND, OR, XOR, LTU).
///
/// `raf_input_claim(γ) = left_claim + γ * right_claim`
/// `raf_claim_at(r, γ) = left(r) + γ * right(r)`
pub struct BinaryRafVD<F: JoltField> {
    pub left_operand_claim: F,
    pub right_operand_claim: F,
}

impl<F: JoltField> RafVerifierData<F> for BinaryRafVD<F> {
    fn raf_input_claim(&self, gamma: F) -> F {
        self.left_operand_claim + gamma * self.right_operand_claim
    }

    fn raf_claim_at(&self, r_address: &[F], gamma: F) -> F {
        let left = SignedOperandPoly::<F, XLEN>::new(OperandSide::Left).evaluate(r_address);
        let right = SignedOperandPoly::<F, XLEN>::new(OperandSide::Right).evaluate(r_address);
        left + gamma * right
    }
}

/// Prover-side RAF state for binary ops: left and right operand PS decompositions.
pub struct BinaryRafPS<F: JoltField> {
    pub left_operand_ps: PrefixSuffixDecomposition<F, 3, true>,
    pub right_operand_ps: PrefixSuffixDecomposition<F, 3, true>,
}

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
                    *l0.as_unreduced_ref(),
                    *l2.as_unreduced_ref(),
                    *r0.as_unreduced_ref(),
                    *r2.as_unreduced_ref(),
                ]
            })
            .fold_with([F::Unreduced::<5>::zero(); 4], |running, new| {
                [
                    running[0] + new[0],
                    running[1] + new[1],
                    running[2] + new[2],
                    running[3] + new[3],
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 4],
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
            F::from_montgomery_reduce(
                left_0.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
                    + right_0.mul_trunc::<4, 9>(gamma_sqr.as_unreduced_ref()),
            ),
            F::from_montgomery_reduce(
                left_2.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
                    + right_2.mul_trunc::<4, 9>(gamma_sqr.as_unreduced_ref()),
            ),
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

#[derive(Debug, Clone, Copy)]
pub struct ReadRafClaims<F: JoltField> {
    pub rv_claim: F,
    pub left_operand_claim: F,
    pub right_operand_claim: F,
}

pub trait PrefixSuffixShoutProvider<F, LUT>: super::RafShoutProvider<F>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F>;
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
    let log_m = LOG_K / NUM_PHASES;
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

pub fn ps_read_raf_verifier<
    F: JoltField,
    T: Transcript,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
>(
    provider: &impl PrefixSuffixShoutProvider<F, LUT>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> BinaryReadRafSumcheckVerifier<F, LUT> {
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
    ReadRafSumcheckVerifier::new(params)
}

pub type BinaryReadRafSumcheckProver<F, LUT> =
    ReadRafSumcheckProver<F, LUT, LOG_K, XLEN, BinaryRafVD<F>, BinaryRafPS<F>>;
pub type BinaryReadRafSumcheckVerifier<F, LUT> =
    ReadRafSumcheckVerifier<F, LUT, LOG_K, XLEN, BinaryRafVD<F>>;

#[cfg(test)]
pub mod tests {
    use crate::{
        lookup_tables::{
            and::AndTable, or::OrTable, unsigned_less_than::UnsignedLessThanTable, xor::XorTable,
            JoltLookupTable, PrefixSuffixDecompositionTrait,
        },
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
                VerifierOpeningAccumulator, BIG_ENDIAN,
            },
        },
        subprotocols::{
            ps_shout::{
                binary::{
                    ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider,
                    ReadRafClaims,
                },
                RafShoutProvider,
            },
            sumcheck::Sumcheck,
        },
        transcripts::{Blake2bTranscript, Transcript},
        utils::{lookup_bits::LookupBits, uninterleave_bits},
    };
    use ark_bn254::Fr;
    use common::{
        consts::{LOG_K, XLEN},
        VirtualPoly,
    };
    use itertools::Itertools;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::marker::PhantomData;

    const LOG_NUM_LOOKUPS: usize = 10;
    const NUM_LOOKUPS: usize = 1 << LOG_NUM_LOOKUPS;

    #[test]
    fn test_and() {
        test_read_raf_sumcheck::<AndTable<XLEN>>();
    }

    #[test]
    fn test_or() {
        test_read_raf_sumcheck::<OrTable<XLEN>>();
    }

    #[test]
    fn test_ltu() {
        test_read_raf_sumcheck::<UnsignedLessThanTable<XLEN>>();
    }

    #[test]
    fn test_xor() {
        test_read_raf_sumcheck::<XorTable<XLEN>>();
    }

    pub fn test_read_raf_sumcheck<LUT>()
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        let trace = generate_trace::<LUT>();

        let (mut prover_transcript, r_cycle) = new_test_transcript();
        let mut prover_accumulator = ProverOpeningAccumulator::new();

        let rv_claim = trace.rv.evaluate(&r_cycle);
        let (left_operand_claim, right_operand_claim) =
            compute_operand_claims(&trace.lookup_indices, &r_cycle);

        let provider =
            TestProvider::<LUT>::new(rv_claim, left_operand_claim, right_operand_claim, r_cycle);

        let mut prover = ps_read_raf_prover(
            &provider,
            trace.lookup_bits,
            &mut prover_accumulator,
            &mut prover_transcript,
        );
        let (proof, prover_challenges) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let (mut verifier_transcript, _) = new_test_transcript();
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        transfer_openings(&prover_accumulator, &mut verifier_accumulator);

        let verifier = ps_read_raf_verifier(
            &provider,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        );
        let verifier_challenges = Sumcheck::verify(
            &proof,
            &verifier,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    fn compute_operand_claims(lookup_indices: &[u64], r_cycle: &[Fr]) -> (Fr, Fr) {
        let (left_indices, right_indices): (Vec<_>, Vec<_>) = lookup_indices
            .iter()
            .map(|&i| {
                let (l, r) = uninterleave_bits(i);
                (l as i32, r as i32)
            })
            .unzip();
        (
            MultilinearPolynomial::from(left_indices).evaluate(r_cycle),
            MultilinearPolynomial::from(right_indices).evaluate(r_cycle),
        )
    }

    fn transfer_openings(
        prover_acc: &ProverOpeningAccumulator<Fr>,
        verifier_acc: &mut VerifierOpeningAccumulator<Fr>,
    ) {
        for (key, (_, claim)) in &prover_acc.openings {
            verifier_acc
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }
    }

    struct TestTrace {
        lookup_indices: Vec<u64>,
        lookup_bits: Vec<LookupBits>,
        rv: MultilinearPolynomial<Fr>,
    }

    fn generate_trace<LUT: JoltLookupTable + Default>() -> TestTrace {
        let mut rng = StdRng::seed_from_u64(0x1109);
        let table = LUT::default();
        let lookup_indices: Vec<u64> = (0..NUM_LOOKUPS).map(|_| rng.gen()).collect();
        let lookup_bits = lookup_indices
            .iter()
            .map(|&i| LookupBits::new(i, LOG_K))
            .collect();
        let rv = MultilinearPolynomial::from(
            lookup_indices
                .iter()
                .map(|&i| table.materialize_entry(i))
                .collect_vec(),
        );
        TestTrace {
            lookup_indices,
            lookup_bits,
            rv,
        }
    }

    fn new_test_transcript() -> (Blake2bTranscript, Vec<Fr>) {
        let mut transcript = Blake2bTranscript::new(b"test");
        let r_cycle: Vec<Fr> = transcript.challenge_vector(LOG_NUM_LOOKUPS);
        (transcript, r_cycle)
    }

    struct TestProvider<LUT>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        rv_claim: Fr,
        left_operand_claim: Fr,
        right_operand_claim: Fr,
        r_cycle: Vec<Fr>,
        _phantom: PhantomData<LUT>,
    }

    impl<LUT> TestProvider<LUT>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        fn new(
            rv_claim: Fr,
            left_operand_claim: Fr,
            right_operand_claim: Fr,
            r_cycle: Vec<Fr>,
        ) -> Self {
            Self {
                rv_claim,
                left_operand_claim,
                right_operand_claim,
                r_cycle,
                _phantom: PhantomData,
            }
        }
    }

    impl<LUT> RafShoutProvider<Fr> for TestProvider<LUT>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
            (VirtualPoly::NodeOutputRa(0), SumcheckId::NodeExecution(0))
        }

        fn r_cycle(
            &self,
            _accumulator: &dyn OpeningAccumulator<Fr>,
        ) -> OpeningPoint<BIG_ENDIAN, Fr> {
            OpeningPoint::new(self.r_cycle.clone())
        }
    }

    impl<LUT> PrefixSuffixShoutProvider<Fr, LUT> for TestProvider<LUT>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        fn read_raf_claims(&self, _accumulator: &dyn OpeningAccumulator<Fr>) -> ReadRafClaims<Fr> {
            ReadRafClaims {
                rv_claim: self.rv_claim,
                left_operand_claim: self.left_operand_claim,
                right_operand_claim: self.right_operand_claim,
            }
        }
    }
}
