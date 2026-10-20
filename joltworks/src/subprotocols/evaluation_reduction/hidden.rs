//! Hidden line reduction with proofs of every boundary evaluation.
//!
//! The registered graph supplies the points and scalar commitments. A shared
//! Schnorr argument proves knowledge of one bounded-degree coefficient vector
//! whose evaluations open every input commitment and the reduced commitment.
//! The following operator proof must use that exact reduced commitment.
//! This reduction alone does not prove that the polynomial is a model output.

use super::{compute_h, eval_on_l, group_by_variable};
use crate::{
    curve::{Bn254Curve, Bn254G1, JoltGroupElement},
    field::JoltField,
    poly::{
        commitment::pedersen::PedersenGenerators,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HiddenReductionStatement {
    pub context: Vec<u8>,
    pub points: Vec<Vec<Fr>>,
    pub claims: Vec<Bn254G1>,
}

/// Values returned by verification are public and derived from the transcript.
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenReducedClaim {
    pub point: Vec<Fr>,
    pub commitment: Bn254G1,
}

/// Prover-only state for the next operator; deliberately not serializable.
pub struct HiddenReducedWitness {
    pub value: Fr,
    pub blind: Fr,
}

/// Coefficients and evaluations are masked by fresh Schnorr randomness.
/// The verifier derives every vector length from the registered statement.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HiddenReductionProof {
    pub coefficient_commitments: Vec<Bn254G1>,
    pub reduced_commitment: Bn254G1,
    pub coefficient_masks: Vec<Bn254G1>,
    pub evaluation_masks: Vec<Bn254G1>,
    pub coefficient_responses: Vec<Fr>,
    pub coefficient_blind_responses: Vec<Fr>,
    pub evaluation_blind_responses: Vec<Fr>,
}

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(message.to_owned())
}

fn evaluate(coefficients: &[Fr], point: Fr) -> Fr {
    coefficients
        .iter()
        .rev()
        .fold(Fr::zero(), |a, b| a * point + b)
}

impl HiddenReductionStatement {
    fn coefficient_count(&self) -> Result<usize, ProofVerifyError> {
        let Some(first) = self.points.first() else {
            return Err(invalid("Empty hidden evaluation reduction"));
        };
        if self.context.is_empty()
            || self.claims.len() != self.points.len()
            || self.points.iter().any(|p| p.len() != first.len())
        {
            return Err(invalid("Malformed hidden evaluation reduction"));
        }
        first
            .len()
            .checked_mul(self.points.len() - 1)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("Hidden reduction degree overflow"))
    }

    fn absorb<T: Transcript>(&self, gens: &PedersenGenerators<Bn254Curve>, t: &mut T) {
        t.append_message(b"Atlas/hidden-line-reduction/v1");
        t.append_serializable(gens);
        t.append_serializable(self);
    }
}

impl HiddenReductionProof {
    pub fn prove<T: Transcript>(
        statement: &HiddenReductionStatement,
        polynomial: &MultilinearPolynomial<Fr>,
        claim_blinds: &[Fr],
        gens: &PedersenGenerators<Bn254Curve>,
        transcript: &mut T,
    ) -> Result<(Self, HiddenReducedClaim, HiddenReducedWitness), ProofVerifyError> {
        let count = statement.coefficient_count()?;
        if claim_blinds.len() != statement.claims.len()
            || polynomial.get_num_vars() != statement.points[0].len()
        {
            return Err(invalid("Hidden reduction witness shape mismatch"));
        }
        let values: Vec<_> = statement
            .points
            .iter()
            .map(|p| polynomial.evaluate(p))
            .collect();
        for ((value, blind), commitment) in values.iter().zip(claim_blinds).zip(&statement.claims) {
            if gens.commit(&[*value], blind) != *commitment {
                return Err(invalid("Hidden reduction input commitment mismatch"));
            }
        }
        statement.absorb(gens, transcript);
        if statement.points.len() == 1 {
            // No degree assertion or new claim is needed. Reuse the exact
            // commitment, including its original blind, and reject extra data.
            let reduced = HiddenReducedClaim {
                point: statement.points[0].clone(),
                commitment: statement.claims[0],
            };
            return Ok((
                Self {
                    coefficient_commitments: vec![],
                    reduced_commitment: reduced.commitment,
                    coefficient_masks: vec![],
                    evaluation_masks: vec![],
                    coefficient_responses: vec![],
                    coefficient_blind_responses: vec![],
                    evaluation_blind_responses: vec![],
                },
                reduced,
                HiddenReducedWitness {
                    value: values[0],
                    blind: claim_blinds[0],
                },
            ));
        }
        let line = group_by_variable(&statement.points);
        let mut coefficients = compute_h(polynomial, &line).coeffs;
        if coefficients.len() > count {
            return Err(invalid("Hidden restriction exceeds its degree bound"));
        }
        coefficients.resize(count, Fr::zero());
        let width = gens.message_generators.len();
        let mut rng = rand::thread_rng();
        let (coefficient_commitments, coefficient_blinds): (Vec<_>, Vec<_>) = gens
            .commit_chunked(&coefficients, &mut rng)
            .into_iter()
            .unzip();
        transcript.append_serializable(&coefficient_commitments);
        let x: Fr = transcript.challenge_scalar();
        let value = evaluate(&coefficients, x);
        let blind = Fr::random(&mut rng);
        let reduced_commitment = gens.commit(&[value], &blind);
        transcript.append_serializable(&reduced_commitment);

        let masks: Vec<_> = (0..count).map(|_| Fr::random(&mut rng)).collect();
        let (coefficient_masks, mask_blinds): (Vec<_>, Vec<_>) =
            gens.commit_chunked(&masks, &mut rng).into_iter().unzip();
        let evaluation_points: Vec<_> = (0..values.len())
            .map(|i| Fr::from(i as u64))
            .chain([x])
            .collect();
        let evaluation_mask_blinds: Vec<_> = evaluation_points
            .iter()
            .map(|_| Fr::random(&mut rng))
            .collect();
        let evaluation_masks: Vec<_> = evaluation_points
            .iter()
            .zip(&evaluation_mask_blinds)
            .map(|(x, b)| gens.commit(&[evaluate(&masks, *x)], b))
            .collect();
        transcript.append_serializable(&coefficient_masks);
        transcript.append_serializable(&evaluation_masks);
        let challenge: Fr = transcript.challenge_scalar();
        let coefficient_responses = masks
            .iter()
            .zip(&coefficients)
            .map(|(m, a)| *m + challenge * a)
            .collect();
        let coefficient_blind_responses = mask_blinds
            .iter()
            .zip(&coefficient_blinds)
            .map(|(m, b)| *m + challenge * b)
            .collect();
        let evaluation_blind_responses = evaluation_mask_blinds
            .iter()
            .zip(claim_blinds.iter().copied().chain([blind]))
            .map(|(m, b)| *m + challenge * b)
            .collect();
        debug_assert_eq!(coefficient_commitments.len(), count.div_ceil(width));
        let proof = Self {
            coefficient_commitments,
            reduced_commitment,
            coefficient_masks,
            evaluation_masks,
            coefficient_responses,
            coefficient_blind_responses,
            evaluation_blind_responses,
        };
        // Bind responses before any subsequent operator challenges are drawn.
        proof.absorb_responses(transcript);
        Ok((
            proof,
            HiddenReducedClaim {
                point: eval_on_l(&line, x),
                commitment: reduced_commitment,
            },
            HiddenReducedWitness { value, blind },
        ))
    }

    pub fn verify<T: Transcript>(
        &self,
        statement: &HiddenReductionStatement,
        gens: &PedersenGenerators<Bn254Curve>,
        transcript: &mut T,
    ) -> Result<HiddenReducedClaim, ProofVerifyError> {
        let count = statement.coefficient_count()?;
        let width = gens.message_generators.len();
        if width == 0 {
            return Err(invalid("Empty hidden reduction generators"));
        }
        let rows = count.div_ceil(width);
        let n = statement.points.len();
        statement.absorb(gens, transcript);
        if n == 1 {
            if self.reduced_commitment != statement.claims[0]
                || !self.coefficient_commitments.is_empty()
                || !self.coefficient_masks.is_empty()
                || !self.evaluation_masks.is_empty()
                || !self.coefficient_responses.is_empty()
                || !self.coefficient_blind_responses.is_empty()
                || !self.evaluation_blind_responses.is_empty()
            {
                return Err(invalid("Single hidden claim must retain its commitment"));
            }
            return Ok(HiddenReducedClaim {
                point: statement.points[0].clone(),
                commitment: self.reduced_commitment,
            });
        }
        if self.coefficient_commitments.len() != rows
            || self.coefficient_masks.len() != rows
            || self.coefficient_responses.len() != count
            || self.coefficient_blind_responses.len() != rows
            || self.evaluation_masks.len() != n + 1
            || self.evaluation_blind_responses.len() != n + 1
        {
            return Err(invalid(
                "Hidden reduction proof has an incorrect degree or shape",
            ));
        }
        transcript.append_serializable(&self.coefficient_commitments);
        let x: Fr = transcript.challenge_scalar();
        transcript.append_serializable(&self.reduced_commitment);
        transcript.append_serializable(&self.coefficient_masks);
        transcript.append_serializable(&self.evaluation_masks);
        let challenge: Fr = transcript.challenge_scalar();
        for (i, chunk) in self.coefficient_responses.chunks(width).enumerate() {
            let lhs = gens.commit(chunk, &self.coefficient_blind_responses[i]);
            let rhs =
                self.coefficient_masks[i] + self.coefficient_commitments[i].scalar_mul(&challenge);
            if lhs != rhs {
                return Err(invalid("Hidden coefficient relation rejected"));
            }
        }
        for (i, point) in (0..n).map(|i| Fr::from(i as u64)).chain([x]).enumerate() {
            let commitment = if i < n {
                statement.claims[i]
            } else {
                self.reduced_commitment
            };
            let lhs = gens.commit(
                &[evaluate(&self.coefficient_responses, point)],
                &self.evaluation_blind_responses[i],
            );
            let rhs = self.evaluation_masks[i] + commitment.scalar_mul(&challenge);
            if lhs != rhs {
                return Err(invalid("Hidden evaluation relation rejected"));
            }
        }
        self.absorb_responses(transcript);
        Ok(HiddenReducedClaim {
            point: eval_on_l(&group_by_variable(&statement.points), x),
            commitment: self.reduced_commitment,
        })
    }

    fn absorb_responses<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_serializable(&self.coefficient_responses);
        transcript.append_serializable(&self.coefficient_blind_responses);
        transcript.append_serializable(&self.evaluation_blind_responses);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::commitment::{commitment_scheme::CommitmentScheme, dory::DoryScheme},
        transcripts::Blake2bTranscript,
    };

    #[test]
    fn hidden_reduction_binds_all_evaluations_and_degree_after_serialization() {
        let pp = DoryScheme::setup_prover(8);
        let gens = DoryScheme::pedersen_generators(&pp, 4);
        let p = MultilinearPolynomial::from(vec![3i64, -7, 12, 0, 22, -8, 10, 5]);
        let points: Vec<Vec<Fr>> = vec![vec![1, 5, 9], vec![3, 8, 11], vec![19, 3, 22]]
            .into_iter()
            .map(|v| v.into_iter().map(Fr::from).collect())
            .collect();
        let blinds: Vec<_> = (0..points.len())
            .map(|_| Fr::random(&mut rand::thread_rng()))
            .collect();
        let claims = points
            .iter()
            .zip(&blinds)
            .map(|(r, b)| gens.commit(&[p.evaluate(r)], b))
            .collect();
        let statement = HiddenReductionStatement {
            context: b"graph/node/17".to_vec(),
            points,
            claims,
        };
        let mut pt = Blake2bTranscript::new(b"hidden-reduction-test");
        let (proof, reduced, witness) =
            HiddenReductionProof::prove(&statement, &p, &blinds, &gens, &mut pt).unwrap();
        assert_eq!(witness.value, p.evaluate(&reduced.point));
        assert_eq!(
            gens.commit(&[witness.value], &witness.blind),
            reduced.commitment
        );
        let mut bytes = vec![];
        proof.serialize_compressed(&mut bytes).unwrap();
        let decoded = HiddenReductionProof::deserialize_compressed(bytes.as_slice()).unwrap();
        let mut vt = Blake2bTranscript::new(b"hidden-reduction-test");
        assert_eq!(decoded.verify(&statement, &gens, &mut vt).unwrap(), reduced);
        assert_eq!(pt.challenge_scalar::<Fr>(), vt.challenge_scalar::<Fr>());
        let verify = |s: &HiddenReductionStatement, p: &HiddenReductionProof| {
            p.verify(
                s,
                &gens,
                &mut Blake2bTranscript::new(b"hidden-reduction-test"),
            )
        };
        for i in 0..statement.claims.len() {
            let mut wrong = statement.clone();
            wrong.claims[i] += gens.message_generators[0];
            assert!(verify(&wrong, &decoded).is_err());
            assert!(HiddenReductionProof::prove(
                &wrong,
                &p,
                &blinds,
                &gens,
                &mut Blake2bTranscript::new(b"hidden-reduction-test")
            )
            .is_err());
        }
        let mut wrong = statement.clone();
        wrong.points[1][0] += Fr::from(1u64);
        assert!(verify(&wrong, &decoded).is_err());
        let mut wrong = statement.clone();
        wrong.context.push(0);
        assert!(verify(&wrong, &decoded).is_err());
        let mut wrong = decoded.clone();
        wrong.reduced_commitment += gens.message_generators[0];
        assert!(verify(&statement, &wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.coefficient_responses.push(Fr::zero());
        assert!(verify(&statement, &wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.coefficient_responses[4] += Fr::from(1u64);
        assert!(verify(&statement, &wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.evaluation_blind_responses.pop();
        assert!(verify(&statement, &wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.coefficient_commitments.pop();
        assert!(verify(&statement, &wrong).is_err());
        assert!(HiddenReductionProof::deserialize_compressed(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn single_hidden_claim_keeps_the_same_commitment() {
        let pp = DoryScheme::setup_prover(8);
        let gens = DoryScheme::pedersen_generators(&pp, 4);
        let p = MultilinearPolynomial::from(vec![4i64, 11]);
        let point = vec![Fr::from(9u64)];
        let blind = Fr::random(&mut rand::thread_rng());
        let s = HiddenReductionStatement {
            context: b"graph/node/1".to_vec(),
            claims: vec![gens.commit(&[p.evaluate(&point)], &blind)],
            points: vec![point],
        };
        let mut pt = Blake2bTranscript::new(b"single-hidden-reduction");
        let (mut proof, reduced, witness) =
            HiddenReductionProof::prove(&s, &p, &[blind], &gens, &mut pt).unwrap();
        let mut vt = Blake2bTranscript::new(b"single-hidden-reduction");
        assert_eq!(proof.verify(&s, &gens, &mut vt).unwrap(), reduced);
        assert_eq!(reduced.commitment, s.claims[0]);
        assert_eq!(witness.blind, blind);
        assert_eq!(pt.challenge_scalar::<Fr>(), vt.challenge_scalar::<Fr>());
        proof.coefficient_responses.push(Fr::zero());
        assert!(proof
            .verify(
                &s,
                &gens,
                &mut Blake2bTranscript::new(b"single-hidden-reduction")
            )
            .is_err());
    }
}
