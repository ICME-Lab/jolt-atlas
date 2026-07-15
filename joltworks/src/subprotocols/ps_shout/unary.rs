use crate::{
    field::{JoltField, MulTrunc},
    lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait},
    poly::{
        multilinear_polynomial::PolynomialEvaluation,
        opening_proof::{OpeningAccumulator, ProverOpeningAccumulator, VerifierOpeningAccumulator},
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        signed_identity_poly::SignedIdentityPoly,
    },
    subprotocols::ps_shout::{
        RafProverState, RafVerifierData, ReadRafSumcheckParams, ReadRafSumcheckProver,
        ReadRafSumcheckVerifier, NUM_PHASES,
    },
    transcripts::Transcript,
    utils::lookup_bits::LookupBits,
};
use ark_std::Zero;
use common::parallel::par_enabled;
use rayon::prelude::*;

/// Verifier-side RAF data for unary ops (ReLU, UnsignedAbs).
///
/// `raf_input_claim(γ) = operand_claim` (no inner γ factor)
/// `raf_claim_at(r, γ) = identity(r)` (γ applied by the caller)
pub struct UnaryRafVD<F: JoltField> {
    pub operand_claim: F,
    pub log_k: usize,
}

impl<F: JoltField> RafVerifierData<F> for UnaryRafVD<F> {
    fn raf_input_claim(&self, _gamma: F) -> F {
        self.operand_claim
    }

    fn raf_claim_at(&self, r_address: &[F], _gamma: F) -> F {
        SignedIdentityPoly::<F>::new(self.log_k).evaluate(r_address)
    }
}

/// Prover-side RAF state for unary ops: identity PS decomposition.
pub struct UnaryRafPS<F: JoltField> {
    pub identity_ps: PrefixSuffixDecomposition<F, 2, true>,
}

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
                [*o0.as_unreduced_ref(), *o2.as_unreduced_ref()]
            })
            .fold_with([F::Unreduced::<5>::zero(); 2], |running, new| {
                [running[0] + new[0], running[1] + new[1]]
            })
            .reduce(
                || [F::Unreduced::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        [
            F::from_montgomery_reduce(operand_0.mul_trunc::<4, 9>(gamma.as_unreduced_ref())),
            F::from_montgomery_reduce(operand_2.mul_trunc::<4, 9>(gamma.as_unreduced_ref())),
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

#[derive(Debug, Clone, Copy)]
pub struct ReadRafClaims<F: JoltField> {
    pub rv_claim: F,
    pub operand_claim: F,
}

pub trait PrefixSuffixShoutProvider<F, LUT, const LOG_K: usize>:
    super::RafShoutProvider<F>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F>;
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
    let claims = provider.read_raf_claims(accumulator);
    let raf = UnaryRafVD {
        operand_claim: claims.operand_claim,
        log_k: LOG_K,
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
    let identity_ps =
        PrefixSuffixDecomposition::new(Box::new(SignedIdentityPoly::<F>::new(LOG_K)), log_m, LOG_K);
    drop(_guard);
    let raf_state = UnaryRafPS { identity_ps };
    ReadRafSumcheckProver::new_inner(params, lookup_indices, raf_state)
}

pub fn ps_read_raf_verifier<
    F: JoltField,
    T: Transcript,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    const LOG_K: usize,
>(
    provider: &impl PrefixSuffixShoutProvider<F, LUT, LOG_K>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> UnaryReadRafSumcheckVerifier<F, LUT, LOG_K> {
    let gamma = transcript.challenge_scalar();
    let claims = provider.read_raf_claims(accumulator);
    let raf = UnaryRafVD {
        operand_claim: claims.operand_claim,
        log_k: LOG_K,
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

pub type UnaryReadRafSumcheckProver<F, LUT, const LOG_K: usize> =
    ReadRafSumcheckProver<F, LUT, LOG_K, LOG_K, UnaryRafVD<F>, UnaryRafPS<F>>;
pub type UnaryReadRafSumcheckVerifier<F, LUT, const LOG_K: usize> =
    ReadRafSumcheckVerifier<F, LUT, LOG_K, LOG_K, UnaryRafVD<F>>;

#[cfg(test)]
pub mod tests {
    use crate::{
        lookup_tables::{
            relu, sat_clamp::SatClampTable, unsigned_abs::UnsignedAbsTable, JoltLookupTable,
            PrefixSuffixDecompositionTrait,
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
                unary::{
                    ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider,
                    ReadRafClaims,
                },
                RafShoutProvider,
            },
            sumcheck::Sumcheck,
        },
        transcripts::{Blake2bTranscript, Transcript},
        utils::lookup_bits::LookupBits,
    };
    use ark_bn254::Fr;
    use common::{consts::XLEN, VirtualPoly};
    use itertools::Itertools;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::marker::PhantomData;

    const LOG_NUM_LOOKUPS: usize = 10;
    const NUM_LOOKUPS: usize = 1 << LOG_NUM_LOOKUPS;

    #[test]
    fn test_sat_clamp() {
        test_read_raf_sumcheck::<SatClampTable<64>, 64>();
    }

    #[test]
    fn test_unsigned_abs() {
        test_read_raf_sumcheck::<UnsignedAbsTable<XLEN>, XLEN>();
    }

    #[test]
    fn test_relu() {
        test_read_raf_sumcheck::<relu::ReluTable<XLEN>, XLEN>();
    }

    pub fn test_read_raf_sumcheck<LUT, const XLEN: usize>()
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        let trace = generate_trace::<LUT, XLEN>();

        let (mut prover_transcript, r_cycle) = new_test_transcript();
        let mut prover_accumulator = ProverOpeningAccumulator::new();

        let rv_claim = trace.rv.evaluate(&r_cycle);
        let operand_claim = compute_operand_claim::<XLEN>(&trace.lookup_indices, &r_cycle);

        let provider = TestProvider::<LUT, XLEN>::new(rv_claim, operand_claim, r_cycle);

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

    fn compute_operand_claim<const XLEN: usize>(lookup_indices: &[u64], r_cycle: &[Fr]) -> Fr {
        MultilinearPolynomial::from(
            lookup_indices
                .iter()
                .map(|&i| match XLEN {
                    32 => i as u32 as i32 as i64,
                    64 => i as i64,
                    _ => unimplemented!(),
                })
                .collect::<Vec<_>>(),
        )
        .evaluate(r_cycle)
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

    fn generate_trace<LUT: JoltLookupTable + Default, const XLEN: usize>() -> TestTrace {
        let mut rng = StdRng::seed_from_u64(0x1109);
        let table = LUT::default();
        let lookup_indices: Vec<u64> = (0..NUM_LOOKUPS).map(|_| rng.gen()).collect();
        let lookup_bits = lookup_indices
            .iter()
            .map(|&i| LookupBits::new(i, XLEN))
            .collect();
        let rv = MultilinearPolynomial::from(
            lookup_indices
                .iter()
                .map(|&i| table.materialize_entry(i) as i64)
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

    struct TestProvider<LUT, const XLEN: usize>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        rv_claim: Fr,
        operand_claim: Fr,
        r_cycle: Vec<Fr>,
        _phantom: PhantomData<LUT>,
    }

    impl<LUT, const XLEN: usize> TestProvider<LUT, XLEN>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        fn new(rv_claim: Fr, operand_claim: Fr, r_cycle: Vec<Fr>) -> Self {
            Self {
                rv_claim,
                operand_claim,
                r_cycle,
                _phantom: PhantomData,
            }
        }
    }

    impl<LUT, const XLEN: usize> RafShoutProvider<Fr> for TestProvider<LUT, XLEN>
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

    impl<LUT, const XLEN: usize> PrefixSuffixShoutProvider<Fr, LUT, XLEN> for TestProvider<LUT, XLEN>
    where
        LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
    {
        fn read_raf_claims(&self, _accumulator: &dyn OpeningAccumulator<Fr>) -> ReadRafClaims<Fr> {
            ReadRafClaims {
                rv_claim: self.rv_claim,
                operand_claim: self.operand_claim,
            }
        }
    }
}
