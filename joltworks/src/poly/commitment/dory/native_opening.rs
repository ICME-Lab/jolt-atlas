//! Native BlindFold proof that a required set of hidden claims opens the
//! registered Dory commitments. This is an opening subprotocol, not an ONNX
//! execution proof. The computation verifier must derive the expected statement
//! and bind its own claims to these exact scalar commitments.

use super::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use crate::{
    curve::{Bn254Curve, Bn254G1},
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, pedersen::PedersenGenerators},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningId, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    subprotocols::{
        blindfold::{
            assembly::NativeBlindFold,
            protocol::{BlindFoldProof, BlindFoldVerifierInput},
            witness::ExtraConstraintWitness,
            BlindFoldAccumulator, OutputClaimConstraint,
        },
        sumcheck::ZkSumcheckProof,
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use common::CommittedPoly;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct OpeningGroup {
    pub polynomials: Vec<CommittedPoly>,
    pub source: SumcheckId,
    pub point: Vec<Fr>,
    /// Dense groups have one polynomial. Sparse groups share an address/cycle
    /// split and may batch several one-hot polynomials.
    pub address_variables: Option<usize>,
}

/// Public input, supplied by the caller's registered relation. Proof bytes do
/// not select which claims are required, their points, or their commitments.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeOpeningStatement {
    pub context: Vec<u8>,
    pub commitments: BTreeMap<CommittedPoly, DoryCommitment>,
    pub groups: Vec<OpeningGroup>,
    pub claim_commitments: BTreeMap<OpeningId, Bn254G1>,
}

/// All three arguments are mandatory. No private field evaluations, opening
/// points, constraint descriptions, or commitment blinds are serialized here.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeOpeningProof {
    pub sumcheck: ZkSumcheckProof<Fr, Bn254Curve, Blake2bTranscript>,
    pub pcs: DoryProof,
    pub blindfold: BlindFoldProof<Fr, Bn254Curve>,
}

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(message.to_owned())
}

impl NativeOpeningStatement {
    fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
        let mut ids = BTreeSet::new();
        let mut polynomials = BTreeSet::new();
        if self.groups.is_empty() || self.groups.iter().all(|g| g.point.is_empty()) {
            return Err(invalid("Native opening statement has no sumcheck rounds"));
        }
        for g in &self.groups {
            if g.polynomials.is_empty()
                || g.point.len() > max_vars
                || g.address_variables.is_some_and(|n| n > g.point.len())
                || (g.address_variables.is_none() && g.polynomials.len() != 1)
                || matches!(
                    g.source,
                    SumcheckId::BlindFoldBatchOpening | SumcheckId::BlindFoldOpeningGroup(_)
                )
            {
                return Err(invalid("Malformed native opening group"));
            }
            for p in &g.polynomials {
                if !ids.insert(OpeningId::new(*p, g.source)) {
                    return Err(invalid("Repeated source opening in native statement"));
                }
                polynomials.insert(*p);
            }
        }
        if !polynomials.iter().eq(self.commitments.keys())
            || !ids.iter().eq(self.claim_commitments.keys())
        {
            return Err(invalid(
                "Native statement commitment set does not match its required openings",
            ));
        }
        Ok(())
    }

    fn transcript(&self) -> Blake2bTranscript {
        let mut transcript = Blake2bTranscript::new(b"Atlas/native-hidden-openings/v1");
        transcript.append_serializable(self);
        transcript
    }

    fn initial_constraints(&self) -> Vec<OutputClaimConstraint> {
        self.claim_commitments
            .keys()
            .map(|id| OutputClaimConstraint::direct(*id))
            .collect()
    }
}

impl NativeOpeningProof {
    pub fn prove(
        statement: &NativeOpeningStatement,
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        hints: BTreeMap<CommittedPoly, DoryHint>,
        claim_blinds: &BTreeMap<OpeningId, Fr>,
        setup: &DoryProverSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<Self, ProofVerifyError> {
        statement.validate(setup.verifier.max_log_n)?;
        Self::check_generators(&DoryVerifierSetup(setup.verifier.clone()), gens)?;
        if !polynomials.keys().eq(statement.commitments.keys())
            || !hints.keys().eq(polynomials.keys())
            || !claim_blinds.keys().eq(statement.claim_commitments.keys())
        {
            return Err(invalid("Native opening witness set mismatch"));
        }
        let mut transcript = statement.transcript();
        let mut accumulator = ProverOpeningAccumulator::new();
        accumulator.zk_mode = true;
        for g in &statement.groups {
            let mut claims = Vec::new();
            for p in &g.polynomials {
                let poly = &polynomials[p];
                if poly.get_num_vars() != g.point.len()
                    || (g.address_variables.is_some()
                        != matches!(poly, MultilinearPolynomial::OneHot(_)))
                {
                    return Err(invalid(
                        "Polynomial does not match its registered opening group",
                    ));
                }
                if let (Some(address), MultilinearPolynomial::OneHot(sparse)) =
                    (g.address_variables, poly)
                {
                    if sparse.K != 1usize << address {
                        return Err(invalid("Sparse address shape mismatch"));
                    }
                }
                let value = poly.evaluate(&g.point);
                let id = OpeningId::new(*p, g.source);
                if gens.commit(&[value], &claim_blinds[&id]) != statement.claim_commitments[&id] {
                    return Err(invalid(
                        "Private claim does not open its required commitment",
                    ));
                }
                claims.push(value);
            }
            if let Some(address) = g.address_variables {
                accumulator.append_sparse(
                    &mut transcript,
                    g.polynomials.clone(),
                    g.source,
                    g.point[..address].to_vec(),
                    g.point[address..].to_vec(),
                    claims,
                );
            } else {
                accumulator.append_dense(
                    &mut transcript,
                    OpeningId::new(g.polynomials[0], g.source),
                    g.point.clone(),
                    claims[0],
                );
            }
        }
        let mut extra = statement
            .claim_commitments
            .keys()
            .map(|id| {
                let value = accumulator.openings[id].1;
                ExtraConstraintWitness {
                    output_value: value,
                    blinding: claim_blinds[id],
                    challenge_values: vec![],
                    opening_values: vec![value],
                }
            })
            .collect::<Vec<_>>();
        let mut bf = BlindFoldAccumulator::new();
        accumulator.prepare_for_sumcheck(polynomials, &mut transcript);
        let (sumcheck, point) = accumulator.prove_batch_opening_sumcheck_zk::<_, Bn254Curve, _>(
            &mut bf,
            &mut vec![],
            gens,
            &mut rand::thread_rng(),
            &mut transcript,
        );
        let state = accumulator.finalize_batch_opening_sumcheck(point, &mut transcript);
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let joint_value = coefficients
            .iter()
            .zip(&state.sumcheck_claims)
            .map(|(a, b)| *a * *b)
            .sum();
        let (pcs, eval, blind) = DoryScheme::prove_rlc_zk(
            setup,
            polynomials,
            &state.poly_coeffs,
            hints.into_values().collect(),
            &state.r_sumcheck,
            &mut transcript,
        )?;
        if eval != gens.commit(&[joint_value], &blind) {
            return Err(invalid("PCS and opening reduction evaluations differ"));
        }
        extra.push(ExtraConstraintWitness {
            output_value: joint_value,
            blinding: blind,
            challenge_values: coefficients.clone(),
            opening_values: state.sumcheck_claims,
        });
        let mut constraints = statement.initial_constraints();
        constraints.push(relation);
        let mut evaluations: Vec<_> = statement.claim_commitments.values().copied().collect();
        evaluations.push(eval);
        let data = bf.take_stage_data();
        let native = NativeBlindFold::new(
            NativeBlindFold::prover_relations(&data),
            &constraints,
            &coefficients,
            gens.message_generators.len(),
        )?;
        let openings = accumulator
            .openings
            .iter()
            .map(|(id, (_, v))| (*id, *v))
            .collect();
        let blindfold =
            native.prove(&data, &openings, extra, evaluations, gens, &mut transcript)?;
        Ok(Self {
            sumcheck,
            pcs,
            blindfold,
        })
    }

    pub fn verify(
        &self,
        statement: &NativeOpeningStatement,
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        statement.validate(setup.0.max_log_n)?;
        Self::check_generators(setup, gens)?;
        let mut transcript = statement.transcript();
        let mut accumulator = VerifierOpeningAccumulator::new_zk();
        for g in &statement.groups {
            if g.address_variables.is_some() {
                accumulator.append_sparse(
                    &mut transcript,
                    g.polynomials.clone(),
                    g.source,
                    g.point.clone(),
                );
            } else {
                accumulator.append_dense(
                    &mut transcript,
                    OpeningId::new(g.polynomials[0], g.source),
                    g.point.clone(),
                );
            }
        }
        let point = accumulator.verify_batch_opening_sumcheck_zk(
            &self.sumcheck,
            &mut transcript,
            gens.message_generators.len(),
        )?;
        let claims = vec![Fr::zero(); accumulator.num_sumchecks()];
        let state = accumulator.finalize_batch_opening_sumcheck(point, &claims, &mut transcript);
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let commitments: Vec<_> = state
            .polynomials
            .iter()
            .map(|p| statement.commitments[p])
            .collect();
        let joint = DoryScheme::combine_commitments(&commitments, &state.poly_coeffs);
        let eval = self
            .pcs
            .0
            .y_com
            .map(|g| Bn254G1(g.0))
            .ok_or_else(|| invalid("Missing PCS evaluation commitment"))?;
        DoryScheme::verify_zk(
            &self.pcs,
            setup,
            &mut transcript,
            &state.r_sumcheck,
            &eval,
            &joint,
        )?;
        let mut constraints = statement.initial_constraints();
        constraints.push(relation);
        let native = NativeBlindFold::new(
            accumulator.zk_stages,
            &constraints,
            &coefficients,
            gens.message_generators.len(),
        )?;
        let mut evaluations: Vec<_> = statement.claim_commitments.values().copied().collect();
        evaluations.push(eval);
        native.verify(
            &self.blindfold,
            &BlindFoldVerifierInput {
                round_commitments: self.sumcheck.round_commitments.clone(),
                output_claims_row_commitments: self.sumcheck.output_claims_commitments.clone(),
                eval_commitments: evaluations,
            },
            gens,
            &mut transcript,
        )
    }

    fn check_generators(
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        if gens.message_generators.len() <= 2
            || !gens.message_generators.len().is_power_of_two()
            || gens.message_generators[0] != Bn254G1(setup.0.g1_0.0)
            || gens.blinding_generator != Bn254G1(setup.0.h1.0)
        {
            return Err(invalid(
                "BlindFold and Dory must use the same registered evaluation bases",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{field::JoltField, poly::one_hot_polynomial::OneHotPolynomial};

    #[test]
    fn native_opening_serialized_proof_requires_every_claim_and_argument() {
        let setup = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&setup);
        let gens = DoryScheme::pedersen_generators(&setup, 16);
        let ids: Vec<_> = (0..3).map(CommittedPoly::DivNodeQuotient).collect();
        let polynomials = BTreeMap::from([
            (ids[0], MultilinearPolynomial::from(vec![1i64, 5, -7, 9])),
            (
                ids[1],
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    vec![Some(0), None, Some(3), Some(1)],
                    4,
                )),
            ),
            (
                ids[2],
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    vec![Some(1), Some(2), Some(0), Some(2)],
                    4,
                )),
            ),
        ]);
        let mut commitments = BTreeMap::new();
        let mut hints = BTreeMap::new();
        for (id, poly) in &polynomials {
            let (c, h) = DoryScheme::commit_zk(poly, &setup);
            commitments.insert(*id, c);
            hints.insert(*id, h);
        }
        let groups = vec![
            OpeningGroup {
                polynomials: vec![ids[0]],
                source: SumcheckId::NodeExecution(0),
                point: vec![Fr::from(2u64), Fr::from(3u64)],
                address_variables: None,
            },
            OpeningGroup {
                polynomials: vec![ids[0]],
                source: SumcheckId::NodeExecution(1),
                point: vec![Fr::from(5u64), Fr::from(7u64)],
                address_variables: None,
            },
            OpeningGroup {
                polynomials: vec![ids[1], ids[2]],
                source: SumcheckId::NodeExecution(2),
                point: [11u64, 13, 17, 19].into_iter().map(Fr::from).collect(),
                address_variables: Some(2),
            },
        ];
        let mut claim_blinds = BTreeMap::new();
        let mut claim_commitments = BTreeMap::new();
        for g in &groups {
            for p in &g.polynomials {
                let id = OpeningId::new(*p, g.source);
                let value = polynomials[p].evaluate(&g.point);
                let blind = Fr::random(&mut rand::thread_rng());
                claim_blinds.insert(id, blind);
                claim_commitments.insert(id, gens.commit(&[value], &blind));
            }
        }
        let statement = NativeOpeningStatement {
            context: b"registered-model/stage/session".to_vec(),
            commitments,
            groups,
            claim_commitments,
        };
        let proof = NativeOpeningProof::prove(
            &statement,
            &polynomials,
            hints,
            &claim_blinds,
            &setup,
            &gens,
        )
        .unwrap();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).unwrap();
        let decoded = NativeOpeningProof::deserialize_compressed(bytes.as_slice()).unwrap();
        decoded.verify(&statement, &vp, &gens).unwrap();
        assert!(NativeOpeningProof::deserialize_compressed(&bytes[..bytes.len() - 1]).is_err());
        let mut changed = statement.clone();
        changed.context.push(0);
        assert!(decoded.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.claim_commitments.pop_first();
        assert!(decoded.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.groups.pop();
        assert!(decoded.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.groups[0].point[0] += Fr::from(1u64);
        assert!(decoded.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        *changed.claim_commitments.values_mut().next().unwrap() += gens.message_generators[0];
        assert!(decoded.verify(&changed, &vp, &gens).is_err());
        let mut changed = decoded.clone();
        changed.pcs.0.y_com = None;
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = decoded.clone();
        changed.sumcheck.round_commitments.pop();
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = decoded.clone();
        changed.sumcheck.poly_degrees[0] += 1;
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = decoded.clone();
        changed.sumcheck.output_claims_commitments.clear();
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = decoded.clone();
        changed.blindfold.az_r += Fr::from(1u64);
        assert!(changed.verify(&statement, &vp, &gens).is_err());
    }
}
