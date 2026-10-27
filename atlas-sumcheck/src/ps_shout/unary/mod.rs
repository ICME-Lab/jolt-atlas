use crate::{
    opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    ps_shout::{CycleWeight, RafVerifierData, ReadRafSumcheckParams, ReadRafSumcheckVerifier},
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::PolynomialEvaluation, prefix_suffix::PrefixSuffixDecomposition,
        signed_identity_poly::SignedIdentityPoly,
    },
};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_transcript::Transcript;
use common::VirtualPoly;
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

/// Verifier counterpart of [`ps_read_raf_prover_with_cycle`].
pub fn ps_read_raf_verifier_with_cycle<
    F: JoltField,
    T: Transcript,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    const LOG_K: usize,
>(
    claims: ReadRafClaims<F>,
    cycle: CycleWeight<F>,
    ra_poly: (VirtualPoly, SumcheckId),
    transcript: &mut T,
) -> UnaryReadRafSumcheckVerifier<F, LUT, LOG_K> {
    let gamma = transcript.challenge_scalar();
    ps_read_raf_verifier_with_gamma::<F, LUT, LOG_K>(gamma, claims, cycle, ra_poly)
}

/// Verifier counterpart of [`ps_read_raf_prover_with_gamma`].
pub fn ps_read_raf_verifier_with_gamma<
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    const LOG_K: usize,
>(
    gamma: F,
    claims: ReadRafClaims<F>,
    cycle: CycleWeight<F>,
    ra_poly: (VirtualPoly, SumcheckId),
) -> UnaryReadRafSumcheckVerifier<F, LUT, LOG_K> {
    let raf = UnaryRafVD {
        operand_claim: claims.operand_claim,
        log_k: LOG_K,
    };
    let params = ReadRafSumcheckParams::from_parts_with_cycle(
        gamma,
        cycle,
        LUT::default(),
        ra_poly.0,
        ra_poly.1,
        claims.rv_claim,
        raf,
    );
    params.into_verifier()
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
    params.into_verifier()
}

pub type UnaryReadRafSumcheckVerifier<F, LUT, const LOG_K: usize> =
    ReadRafSumcheckVerifier<F, LUT, LOG_K, LOG_K, UnaryRafVD<F>>;

#[cfg(test)]
pub mod tests {
    use crate::relation::{prover::prove_relation, verify_relation};
    use crate::{
        opening_proof::{
            prover::ProverOpeningAccumulator, OpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
        ps_shout::{
            unary::{
                prover::ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider,
                ReadRafClaims,
            },
            RafShoutProvider,
        },
    };
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_algebra::{
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_point::{OpeningPoint, BIG_ENDIAN},
        },
        utils::lookup_bits::LookupBits,
    };
    use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
    use atlas_transcript::{Blake2bProver, ProvingTranscript, Transcript};
    use common::VirtualPoly;
    use itertools::Itertools;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::marker::PhantomData;

    const LOG_NUM_LOOKUPS: usize = 10;
    const NUM_LOOKUPS: usize = 1 << LOG_NUM_LOOKUPS;

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
            prove_relation(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let (mut verifier_transcript, _) = new_test_transcript();
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        transfer_openings(&prover_accumulator, &mut verifier_accumulator);

        let verifier = ps_read_raf_verifier(
            &provider,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        );
        let verifier_challenges = verify_relation(
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

    fn new_test_transcript() -> (Blake2bProver, Vec<Fr>) {
        let mut transcript = Blake2bProver::new(b"test");
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

pub mod prover;
