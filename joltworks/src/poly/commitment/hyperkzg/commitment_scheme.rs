use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            hyperkzg::{
                HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGSRS,
                HyperKZGVerifierKey, KZGVerifierKey, UnivariateKZG, SRS,
            },
        },
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use ark_ec::CurveGroup;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{borrow::Borrow, sync::Arc};

impl CommitmentScheme for HyperKZG<ark_bn254::Bn254> {
    type Field = ark_bn254::Fr;
    type ProverSetup = HyperKZGProverKey<ark_bn254::Bn254>;
    type VerifierSetup = HyperKZGVerifierKey<ark_bn254::Bn254>;

    type Commitment = HyperKZGCommitment<ark_bn254::Bn254>;
    type Proof = HyperKZGProof<ark_bn254::Bn254>;
    type BatchedProof = HyperKZGProof<ark_bn254::Bn254>;
    type OpeningProofHint = ();

    const REQUIRES_MATERIALIZED_POLYS: bool = true;

    #[tracing::instrument(skip_all)]
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        HyperKZGSRS(Arc::new(SRS::setup(
            &mut ChaCha20Rng::from_seed(*b"HyperKZG_POLY_COMMITMENT_SCHEMEE"),
            1 << max_num_vars,
            2,
        )))
        .trim(1 << max_num_vars)
        .0
    }

    #[tracing::instrument(skip_all)]
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        HyperKZGVerifierKey {
            kzg_vk: KZGVerifierKey::from(&setup.kzg_pk),
        }
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::commit")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        assert!(
            setup.kzg_pk.g1_powers().len() >= poly.len(),
            "COMMIT KEY LENGTH ERROR {}, {}",
            setup.kzg_pk.g1_powers().len(),
            poly.len()
        );

        // Handle OneHot variant specially for efficiency
        let commitment = match poly {
            MultilinearPolynomial::OneHot(one_hot) => Self::commit_one_hot(setup, one_hot).unwrap(),
            _ => HyperKZGCommitment(
                UnivariateKZG::commit_as_univariate(&setup.kzg_pk, poly).unwrap(),
            ),
        };
        (commitment, ())
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::batch_commit")]
    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        UnivariateKZG::commit_batch(&gens.kzg_pk, polys)
            .unwrap()
            .into_par_iter()
            .map(|c| (HyperKZGCommitment(c), ()))
            .collect()
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let combined_commitment: ark_bn254::G1Projective = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.borrow().0 * coeff)
            .sum();
        HyperKZGCommitment(combined_commitment.into_affine())
    }

    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        // HyperKZG doesn't use hints, so combining is trivial
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::prove")]
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        HyperKZG::<ark_bn254::Bn254>::open(setup, poly, opening_point, transcript).unwrap()
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::verify")]
    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge], // point at which the polynomial is evaluated
        opening: &Self::Field,                                   // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        HyperKZG::<ark_bn254::Bn254>::verify_inner(
            setup,
            commitment,
            opening_point,
            opening,
            proof,
            transcript,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"hyperkzg"
    }
}
