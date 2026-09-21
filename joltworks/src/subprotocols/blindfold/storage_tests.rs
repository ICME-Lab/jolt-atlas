use super::*;
use crate::poly::opening_proof::SumcheckId;
use crate::subprotocols::blindfold::{ProductTerm, ValueSource};
use ark_bn254::Fr;
use common::CommittedPoly;

fn fixture(count: usize) -> Vec<ZkVerifierStage<Fr>> {
    let ids: Vec<_> = (0..count)
        .map(|i| {
            OpeningId::new(
                CommittedPoly::DivNodeQuotient(i),
                SumcheckId::NodeExecution(i),
            )
        })
        .collect();
    (0..3)
        .map(|stage| {
            let constraints: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    OutputClaimConstraint::sum_of_products(vec![
                        ProductTerm::scaled(
                            ValueSource::Challenge(0),
                            vec![ValueSource::Opening(*id)],
                        ),
                        ProductTerm::product(vec![
                            ValueSource::Opening(*id),
                            ValueSource::Opening(ids[(i + 1) % count]),
                        ]),
                        ProductTerm::single(ValueSource::Constant(-3)),
                    ])
                })
                .collect();
            ZkVerifierStage {
                num_rounds: stage + 1,
                degree: 2,
                challenges: vec![2u128.into(); stage + 1],
                batching_coefficients: vec![Fr::from(3u64); count],
                input_constraints: constraints.clone(),
                input_constraint_challenge_values: vec![vec![Fr::from(5u64)]; count],
                input_claim_scaling_exponents: (0..count).map(|i| i % 3).collect(),
                output_constraints: constraints.into_iter().map(Some).collect(),
                constraint_challenge_values: vec![vec![Fr::from(7u64)]; count],
                output_claim_ids: ids.iter().chain(ids.iter().take(2)).copied().collect(),
            }
        })
        .collect()
}

#[test]
fn storage_all_matrices_and_layouts_match_original_builder() {
    for count in [1, 2, 7, 17, 257] {
        let stages = fixture(count);
        let extra = OutputClaimConstraint::all_weighted_openings(&stages[0].output_claim_ids);
        let challenges = vec![Fr::from(11u64); extra.num_challenges];
        for width in [4, 8, 16] {
            let expected = NativeBlindFold::reference_assembly(
                stages.clone(),
                &[extra.clone()],
                &challenges,
                width,
            )
            .unwrap();
            let prover =
                NativeBlindFold::new(stages.clone(), &[extra.clone()], &challenges, width).unwrap();
            let verifier =
                NativeBlindFold::new_verifier(stages.clone(), &[extra.clone()], &challenges, width)
                    .unwrap();
            assert_eq!(format!("{:?}", prover.r1cs), expected);
            assert_eq!(format!("{:?}", verifier.inner.r1cs), expected);
            assert_eq!(prover.relations.len(), 3);
            assert!(verifier.inner.relations.is_empty());
            assert!(verifier.inner.inputs.is_empty());
            assert!(verifier.inner.outputs.is_empty());
            assert!(verifier.inner.input_values.is_empty());
            assert!(verifier.inner.output_values.is_empty());
        }
    }
}

#[test]
fn storage_keeps_relation_shape_rejections() {
    let check = |stages: Vec<ZkVerifierStage<Fr>>,
                 extra: &[OutputClaimConstraint],
                 challenges: &[Fr],
                 width| {
        assert!(
            NativeBlindFold::reference_assembly(stages.clone(), extra, challenges, width).is_err()
        );
        assert!(NativeBlindFold::new_verifier(stages, extra, challenges, width).is_err());
    };
    check(Vec::new(), &[], &[], 4);
    check(fixture(2), &[], &[], 3);
    let mut stages = fixture(2);
    stages[0].challenges.clear();
    check(stages, &[], &[], 4);
    let mut stages = fixture(2);
    stages[1].input_constraints.pop();
    check(stages, &[], &[], 4);
    let mut stages = fixture(2);
    stages[2].output_constraints[0] = None;
    check(stages, &[], &[], 4);
    let mut stages = fixture(2);
    stages[0].input_constraint_challenge_values[0].clear();
    check(stages, &[], &[], 4);
    let stages = fixture(2);
    let extra = OutputClaimConstraint::all_weighted_openings(&stages[0].output_claim_ids);
    check(stages, &[extra], &[], 4);
}

#[test]
fn storage_builder_preserves_aliases_empty_and_uniskip_layouts() {
    use super::super::storage_reference_r1cs::VerifierR1CSBuilder as Reference;
    let id = |i| {
        OpeningId::new(
            CommittedPoly::DivNodeQuotient(i),
            SumcheckId::NodeExecution(0),
        )
    };
    for width in [4, 8, 16] {
        for configs in [
            Vec::new(),
            vec![StageConfig::new_uniskip(3, vec![4, 0, 10, 0])],
            vec![StageConfig::new(2, 2), StageConfig::new_chain(1, 2)],
        ] {
            let rounds = configs.iter().map(|c| c.num_rounds).sum();
            let baked = BakedPublicInputs {
                challenges: vec![Fr::from(5u64); rounds],
                initial_claims: vec![Fr::from(13u64); 2],
                extra_constraint_challenges: vec![Fr::from(7u64); 3],
                ..Default::default()
            };
            let extra = vec![OutputClaimConstraint::all_weighted_openings(&[
                id(0),
                id(2),
                id(1),
            ])];
            let blocks = vec![vec![id(0), id(1)], vec![id(1), id(0)]];
            let aliases = BTreeMap::from([(id(2), id(0))]);
            let reference = Reference::new_with_extra(
                &configs,
                &extra,
                &baked,
                blocks.clone(),
                aliases.clone(),
            )
            .with_row_width(width)
            .build();
            let borrowed = VerifierR1CSBuilder::new_with_extra(
                &configs,
                &extra,
                &baked,
                blocks.clone(),
                aliases.clone(),
            )
            .with_row_width(width)
            .build();
            let owned =
                VerifierR1CSBuilder::new_with_extra_owned(configs, extra, baked, blocks, aliases)
                    .with_row_width(width)
                    .build();
            assert_eq!(format!("{reference:?}"), format!("{borrowed:?}"));
            assert_eq!(format!("{reference:?}"), format!("{owned:?}"));
        }
    }
}

impl<F: JoltField> NativeBlindFold<F> {
    fn reference_assembly(
        relations: Vec<ZkVerifierStage<F>>,
        extra_constraints: &[OutputClaimConstraint],
        extra_challenges: &[F],
        width: usize,
    ) -> Result<String, ProofVerifyError> {
        if relations.is_empty() || !width.is_power_of_two() {
            return Err(invalid("Empty native relation or invalid commitment width"));
        }
        let mut configs = Vec::new();
        let mut baked = BakedPublicInputs::default();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut input_values = Vec::new();
        let mut output_values = Vec::new();
        for s in &relations {
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
            configs.push(
                StageConfig::new_chain(s.num_rounds, s.degree)
                    .with_input_constraint(input.clone())
                    .with_constraint(output.clone()),
            );
            baked
                .challenges
                .extend(s.challenges.iter().map(|r| (*r).into()));
            // Every initial claim is a constrained witness variable.
            baked.initial_claims.push(F::zero());
            baked.input_constraint_challenges.extend_from_slice(&iv);
            baked.output_constraint_challenges.extend_from_slice(&ov);
            inputs.push(input);
            outputs.push(output);
            input_values.push(iv);
            output_values.push(ov);
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
        let blocks = relations
            .iter()
            .map(|s| s.output_claim_ids.clone())
            .collect();
        let r1cs = super::super::storage_reference_r1cs::VerifierR1CSBuilder::new_with_extra(
            &configs,
            extra_constraints,
            &baked,
            blocks,
            BTreeMap::new(),
        )
        .with_row_width(width)
        .build();
        Ok(format!("{r1cs:?}"))
    }
}
