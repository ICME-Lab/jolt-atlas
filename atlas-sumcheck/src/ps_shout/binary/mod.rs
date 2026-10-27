use crate::{
    opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
    ps_shout::{RafVerifierData, ReadRafSumcheckParams, ReadRafSumcheckVerifier},
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity_poly::OperandSide, multilinear_polynomial::PolynomialEvaluation,
        prefix_suffix::PrefixSuffixDecomposition, signed_identity_poly::SignedOperandPoly,
    },
};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_transcript::Transcript;
use common::consts::{LOG_K, XLEN};
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
    params.into_verifier()
}

pub type BinaryReadRafSumcheckVerifier<F, LUT> =
    ReadRafSumcheckVerifier<F, LUT, LOG_K, XLEN, BinaryRafVD<F>>;

#[cfg(test)]
pub mod tests {
    use crate::{
        opening_proof::{
            prover::ProverOpeningAccumulator, OpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
        ps_shout::{
            binary::{
                prover::ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider,
                ReadRafClaims,
            },
            RafShoutProvider,
        },
        sumcheck::Sumcheck,
    };
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_algebra::{
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_point::{OpeningPoint, BIG_ENDIAN},
        },
        utils::{lookup_bits::LookupBits, uninterleave_bits},
    };
    use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
    use atlas_transcript::{Blake2bProver, ProvingTranscript, Transcript};
    use common::{
        consts::{LOG_K, XLEN},
        VirtualPoly,
    };
    use itertools::Itertools;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::marker::PhantomData;

    const LOG_NUM_LOOKUPS: usize = 10;
    const NUM_LOOKUPS: usize = 1 << LOG_NUM_LOOKUPS;

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

    fn new_test_transcript() -> (Blake2bProver, Vec<Fr>) {
        let mut transcript = Blake2bProver::new(b"test");
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

pub mod prover;
