//! Assemble BlindFold from verifier-derived sumcheck relations. No stage
//! descriptions, challenges, or private initial claims belong in the proof.

use super::{
    protocol::{BlindFoldProof, BlindFoldProver, BlindFoldVerifier, BlindFoldVerifierInput},
    witness::{
        BlindFoldWitness, ExtraConstraintWitness, FinalOutputWitness, RoundWitness, StageWitness,
    },
    BakedPublicInputs, InputClaimConstraint, OutputClaimConstraint, RelaxedR1CSInstance,
    StageConfig, VerifierR1CS, VerifierR1CSBuilder, ZkStageData, ZkVerifierStage,
};
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::{commitment::pedersen::PedersenGenerators, opening_proof::OpeningId},
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::collections::BTreeMap;

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(message.to_owned())
}

/// The caller obtains these relations by replaying the registered computation,
/// not by reading fields supplied by the prover. Extra relations bind external
/// evaluation commitments, including the hiding PCS evaluation.
pub struct NativeBlindFold<F: JoltField> {
    r1cs: VerifierR1CS<F>,
    relations: Vec<ZkVerifierStage<F>>,
    inputs: Vec<InputClaimConstraint>,
    outputs: Vec<OutputClaimConstraint>,
    input_values: Vec<Vec<F>>,
    output_values: Vec<Vec<F>>,
}

/// Verification needs the assembled relation, but no witness-construction data.
pub struct NativeBlindFoldVerifier<F: JoltField> {
    inner: NativeBlindFold<F>,
}

impl<F: JoltField> NativeBlindFoldVerifier<F> {
    pub fn verify<C: JoltCurve<F = F>, T: Transcript>(
        &self,
        proof: &BlindFoldProof<F, C>,
        input: &BlindFoldVerifierInput<C>,
        gens: &PedersenGenerators<C>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        self.inner.verify(proof, input, gens, transcript)
    }
}

impl<F: JoltField> NativeBlindFold<F> {
    #[tracing::instrument(skip_all, name = "NativeBlindFold::new", fields(stages = relations.len()))]
    pub fn new(
        relations: Vec<ZkVerifierStage<F>>,
        extra_constraints: &[OutputClaimConstraint],
        extra_challenges: &[F],
        width: usize,
    ) -> Result<Self, ProofVerifyError> {
        Self::assemble(relations, extra_constraints, extra_challenges, width, true)
    }

    /// Assemble the same matrices while releasing stage data after batching.
    pub fn new_verifier(
        relations: Vec<ZkVerifierStage<F>>,
        extra_constraints: &[OutputClaimConstraint],
        extra_challenges: &[F],
        width: usize,
    ) -> Result<NativeBlindFoldVerifier<F>, ProofVerifyError> {
        Ok(NativeBlindFoldVerifier {
            inner: Self::assemble(relations, extra_constraints, extra_challenges, width, false)?,
        })
    }

    fn assemble(
        relations: Vec<ZkVerifierStage<F>>,
        extra_constraints: &[OutputClaimConstraint],
        extra_challenges: &[F],
        width: usize,
        retain_witness: bool,
    ) -> Result<Self, ProofVerifyError> {
        if relations.is_empty() || !width.is_power_of_two() {
            return Err(invalid("Empty native relation or invalid commitment width"));
        }
        let mut configs = Vec::new();
        let mut baked = BakedPublicInputs::default();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut input_values = Vec::new();
        let mut output_values = Vec::new();
        let mut retained_relations = Vec::new();
        let mut blocks = Vec::new();
        for s in relations {
            let n = s.batching_coefficients.len();
            if s.num_rounds == 0
                || s.num_rounds != s.challenges.len()
                || s.degree >= width
                || n == 0
                || s.input_constraints.len() != n
                || s.output_constraints.len() != n
                || s.input_claim_scaling_exponents.len() != n
                || s.input_constraint_challenge_values.len() != n
                || s.constraint_challenge_values.len() != n
                || s.input_constraints.iter().any(|c| c.terms.is_empty())
            {
                return Err(invalid("Incomplete native sumcheck relation"));
            }
            let input = InputClaimConstraint::batch_required(&s.input_constraints, n);
            // Output constraints already carry their sumcheck batching factor.
            let output = OutputClaimConstraint::batch(&s.output_constraints)
                .ok_or_else(|| invalid("Missing native output relation"))?;
            let mut iv: Vec<F> = s
                .batching_coefficients
                .iter()
                .zip(&s.input_claim_scaling_exponents)
                .map(|(coefficient, shift)| coefficient.mul_pow_2(*shift))
                .collect();
            iv.extend(
                s.input_constraint_challenge_values
                    .iter()
                    .flatten()
                    .copied(),
            );
            let mut ov = vec![F::one(); n];
            ov.extend(s.constraint_challenge_values.iter().flatten().copied());
            if iv.len() != input.num_challenges || ov.len() != output.num_challenges {
                return Err(invalid("Native relation challenge count mismatch"));
            }
            if retain_witness {
                inputs.push(input.clone());
                outputs.push(output.clone());
            }
            configs.push(
                StageConfig::new_chain(s.num_rounds, s.degree)
                    .with_input_constraint(input)
                    .with_constraint(output),
            );
            baked
                .challenges
                .extend(s.challenges.iter().map(|r| (*r).into()));
            // Every initial claim is a constrained witness variable.
            baked.initial_claims.push(F::zero());
            baked.input_constraint_challenges.extend_from_slice(&iv);
            baked.output_constraint_challenges.extend_from_slice(&ov);
            if retain_witness {
                input_values.push(iv);
                output_values.push(ov);
                blocks.push(s.output_claim_ids.clone());
                retained_relations.push(s);
            } else {
                blocks.push(s.output_claim_ids);
            }
        }
        if extra_constraints
            .iter()
            .map(|c| c.num_challenges)
            .sum::<usize>()
            != extra_challenges.len()
        {
            return Err(invalid("Native extra relation challenge count mismatch"));
        }
        baked.extra_constraint_challenges = extra_challenges.to_vec();
        let r1cs = VerifierR1CSBuilder::new_with_extra_owned(
            configs,
            extra_constraints.to_vec(),
            baked,
            blocks,
            BTreeMap::new(),
        )
        .with_row_width(width)
        .build();
        Ok(Self {
            r1cs,
            relations: retained_relations,
            inputs,
            outputs,
            input_values,
            output_values,
        })
    }

    /// This helper is for the prover only. The verifier uses its replayed stages.
    pub fn prover_relations<C: JoltCurve<F = F>>(
        stages: &[ZkStageData<F, C>],
    ) -> Vec<ZkVerifierStage<F>> {
        stages
            .iter()
            .map(|s| ZkVerifierStage {
                num_rounds: s.challenges.len(),
                degree: s.poly_coeffs.iter().map(|c| c.len() - 1).max().unwrap_or(0),
                challenges: s.challenges.clone(),
                batching_coefficients: s.batching_coefficients.clone(),
                input_constraints: s.input_constraints.clone(),
                input_constraint_challenge_values: s.input_constraint_challenge_values.clone(),
                input_claim_scaling_exponents: s.input_claim_scaling_exponents.clone(),
                output_constraints: s.output_constraints.clone(),
                constraint_challenge_values: s.constraint_challenge_values.clone(),
                output_claim_ids: s.output_claims.iter().map(|(id, _)| *id).collect(),
            })
            .collect()
    }

    pub fn prove<C: JoltCurve<F = F>, T: Transcript>(
        &self,
        stages: &[ZkStageData<F, C>],
        openings: &BTreeMap<OpeningId, F>,
        extra_witnesses: Vec<ExtraConstraintWitness<F>>,
        eval_commitments: Vec<C::G1>,
        gens: &PedersenGenerators<C>,
        transcript: &mut T,
    ) -> Result<BlindFoldProof<F, C>, ProofVerifyError> {
        let width = self.r1cs.hyrax.C;
        if stages.len() != self.relations.len()
            || gens.message_generators.len() != width
            || extra_witnesses.len() != self.r1cs.extra_constraints.len()
            || eval_commitments.len() != extra_witnesses.len()
        {
            return Err(invalid("Native BlindFold witness shape mismatch"));
        }
        let resolve = |c: &OutputClaimConstraint| -> Result<Vec<F>, ProofVerifyError> {
            c.required_openings
                .iter()
                .map(|id| {
                    openings
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("Missing native witness opening"))
                })
                .collect()
        };
        let mut witnesses = Vec::new();
        let mut oc_values = Vec::new();
        for (i, s) in stages.iter().enumerate() {
            let r = &self.relations[i];
            if s.poly_coeffs.len() != r.num_rounds
                || s.blinding_factors.len() != r.num_rounds
                || s.round_commitments.len() != r.num_rounds
                || s.output_claims.len() != r.output_claim_ids.len()
                || s.output_claims_commitments.len() != r.output_claim_ids.len().div_ceil(width)
                || s.output_claims_blindings.len() != s.output_claims_commitments.len()
            {
                return Err(invalid("Native stage witness shape mismatch"));
            }
            let rounds = s
                .poly_coeffs
                .iter()
                .zip(&r.challenges)
                .map(|(c, r)| RoundWitness::new(c.clone(), (*r).into()))
                .collect();
            witnesses.push(StageWitness::with_both(
                rounds,
                FinalOutputWitness::general(
                    self.input_values[i].clone(),
                    resolve(&self.inputs[i])?,
                ),
                FinalOutputWitness::general(
                    self.output_values[i].clone(),
                    resolve(&self.outputs[i])?,
                ),
            ));
            let start = oc_values.len();
            oc_values.extend(s.output_claims.iter().map(|(_, v)| *v));
            oc_values.resize(
                start + s.output_claims.len().div_ceil(width) * width,
                F::zero(),
            );
        }
        let witness = BlindFoldWitness::with_output_claims(
            stages.iter().map(|s| s.initial_claim).collect(),
            witnesses,
            extra_witnesses,
            oc_values,
        );
        let z = witness.assign(&self.r1cs);
        self.r1cs
            .check_satisfaction(&z)
            .map_err(|_| invalid("Unsatisfied native BlindFold relation"))?;
        let h = &self.r1cs.hyrax;
        let mut blindings = vec![F::zero(); h.R_prime];
        let round_commitments: Vec<_> = stages
            .iter()
            .flat_map(|s| s.round_commitments.iter().copied())
            .collect();
        for (i, blind) in stages.iter().flat_map(|s| &s.blinding_factors).enumerate() {
            blindings[i] = *blind;
            if gens.commit(&z[1 + i * width..1 + (i + 1) * width], blind) != round_commitments[i] {
                return Err(invalid(
                    "Sumcheck transcript commitment does not match BlindFold witness",
                ));
            }
        }
        let oc_commitments: Vec<_> = stages
            .iter()
            .flat_map(|s| s.output_claims_commitments.iter().copied())
            .collect();
        for (i, blind) in stages
            .iter()
            .flat_map(|s| &s.output_claims_blindings)
            .enumerate()
        {
            let row = h.R_coeff + i;
            blindings[row] = *blind;
            if gens.commit(&z[1 + row * width..1 + (row + 1) * width], blind) != oc_commitments[i] {
                return Err(invalid(
                    "Opening transcript commitment does not match BlindFold witness",
                ));
            }
        }
        let mut rng = rand::thread_rng();
        let noncoeff = (0..h.regular_noncoeff_rows())
            .map(|i| {
                let row = h.R_coeff + h.output_claims_rows + i;
                blindings[row] = F::random(&mut rng);
                gens.commit(&z[1 + row * width..1 + (row + 1) * width], &blindings[row])
            })
            .collect();
        let (instance, rw) = RelaxedR1CSInstance::<F, C>::new_non_relaxed(
            &z[1..],
            self.r1cs.num_constraints,
            width,
            round_commitments,
            oc_commitments,
            noncoeff,
            eval_commitments,
            blindings,
        );
        let bases = Some((gens.message_generators[0], gens.blinding_generator));
        Ok(BlindFoldProver::new(gens, &self.r1cs, bases).prove(&instance, &rw, &z, transcript))
    }

    /// `input` uses commitments already absorbed by sumcheck and the PCS, never
    /// a separate set of commitments supplied for BlindFold alone.
    pub fn verify<C: JoltCurve<F = F>, T: Transcript>(
        &self,
        proof: &BlindFoldProof<F, C>,
        input: &BlindFoldVerifierInput<C>,
        gens: &PedersenGenerators<C>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        if gens.message_generators.len() != self.r1cs.hyrax.C
            || input.round_commitments.len() != self.r1cs.hyrax.total_rounds
            || input.output_claims_row_commitments.len() != self.r1cs.hyrax.output_claims_rows
            || input.eval_commitments.len() != self.r1cs.extra_constraints.len()
        {
            return Err(invalid("Native BlindFold verifier shape mismatch"));
        }
        BlindFoldVerifier::new(
            gens,
            &self.r1cs,
            Some((gens.message_generators[0], gens.blinding_generator)),
        )
        .verify(proof, input, transcript)
        .map_err(|e| invalid(&format!("Native BlindFold rejected: {e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curve::Bn254Curve,
        poly::{
            commitment::commitment_scheme::CommitmentScheme, commitment::dory::DoryScheme,
            opening_proof::SumcheckId,
        },
    };
    use ark_bn254::Fr;
    use ark_std::{One, Zero};
    use common::CommittedPoly;

    #[test]
    fn repeated_claim_slots_are_constrained_and_all_blocks_are_masked() {
        let a = OpeningId::new(
            CommittedPoly::DivNodeQuotient(0),
            SumcheckId::NodeExecution(0),
        );
        let b = OpeningId::new(
            CommittedPoly::DivNodeQuotient(1),
            SumcheckId::NodeExecution(0),
        );
        let r1cs = VerifierR1CSBuilder::<Fr>::new_with_extra(
            &[],
            &[],
            &BakedPublicInputs::default(),
            vec![vec![a], vec![b, a]],
            BTreeMap::new(),
        )
        .with_row_width(4)
        .build();
        let mut z = vec![Fr::zero(); r1cs.num_vars];
        z[0] = Fr::one();
        let start = 1 + r1cs.hyrax.R_coeff * r1cs.hyrax.C;
        z[start] = Fr::from(7u64);
        z[start + 4] = Fr::from(9u64);
        z[start + 5] = Fr::from(7u64);
        r1cs.check_satisfaction(&z).unwrap();
        z[start + 5] += Fr::one();
        assert!(r1cs.check_satisfaction(&z).is_err());
        let setup = DoryScheme::setup_prover(8);
        let gens = DoryScheme::pedersen_generators(&setup, 4);
        let (_, w, _) = super::super::sample_random_satisfying_pair::<Fr, Bn254Curve, _>(
            &gens,
            &r1cs,
            None,
            &mut rand::thread_rng(),
        );
        let offset = r1cs.hyrax.R_coeff * r1cs.hyrax.C;
        assert!(w.W[offset..offset + 8].iter().all(|v| !v.is_zero()));
    }
}

#[cfg(test)]
#[path = "storage_tests.rs"]
mod storage_tests;
