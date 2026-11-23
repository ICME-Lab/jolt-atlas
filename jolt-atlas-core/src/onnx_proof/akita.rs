//! Plan Akita's commitment groups for a model execution.

use super::coefficient_checks::{BooleanityProver, DigitRangeProver};
use super::{
    prover::{iop, output_claim_at, polynomial_map},
    AtlasSharedPreprocessing, ONNXProof, Prover,
};
use atlas_algebra::{
    field::akita::AkitaField,
    poly::multilinear::{MultilinearPolynomial, PolynomialEvaluation},
    utils::errors::ProofVerifyError,
};
use atlas_pcs::akita::{
    protocol_tag, prover_transcript, AkitaProverSetup, AkitaScheme, CommitmentScheme, Group,
    GroupLayout, Keccak,
};
use atlas_sumcheck::relation::prover::prove_relation;
use atlas_trace::{trace::ModelExecutionIO, Model, Tensor};
use atlas_transcript::{Blake2bProver, ProvingTranscript};
use common::CommittedPoly;
use jolt_atlas_verifier::onnx_proof::akita::{
    opening_error, opening_seed, AkitaMessage, CommitmentGroup, WitnessShape, OPENING_TAG,
    OUTPUT_SUMCHECK,
};
pub use jolt_atlas_verifier::onnx_proof::akita::{verify, Preprocessing as VerifierPreprocessing};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

fn witness_shapes(
    polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<AkitaField>>,
) -> Result<BTreeMap<CommittedPoly, WitnessShape>, ProofVerifyError> {
    polynomials
        .iter()
        .map(|(&id, polynomial)| {
            let layout = GroupLayout::for_polynomial(polynomial).map_err(opening_error)?;
            let k = match polynomial {
                MultilinearPolynomial::OneHot(poly) => Some(poly.K),
                _ => None,
            };
            Ok((id, WitnessShape::new(layout.num_vars(), k)?))
        })
        .collect()
}

type SetupKey = (GroupLayout, usize, [u8; 32]);

/// Model preprocessing for repeated Akita proofs on the CPU.
pub struct ProverPreprocessing<'a> {
    verifier: VerifierPreprocessing<'a>,
    setups: Mutex<BTreeMap<SetupKey, Arc<AkitaProverSetup>>>,
}

impl<'a> ProverPreprocessing<'a> {
    /// Load the public preprocessing for this model once.
    pub fn new(model: &'a Model) -> Result<Self, ProofVerifyError> {
        Ok(Self {
            verifier: VerifierPreprocessing::new(model)?,
            setups: Mutex::default(),
        })
    }

    /// Prove with the cached plan and setups for these witness shapes.
    pub fn prove(
        &self,
        inputs: &[Tensor<i32>],
    ) -> Result<(ONNXProof, ModelExecutionIO), ProofVerifyError> {
        prove_preprocessed(
            self,
            inputs,
            #[cfg(test)]
            None,
        )
    }

    /// The public preprocessing used by this prover.
    pub fn verifier(&self) -> &VerifierPreprocessing<'a> {
        &self.verifier
    }

    fn setup(&self, group: &CommitmentGroup) -> Result<Arc<AkitaProverSetup>, ProofVerifyError> {
        let key = (group.layout, group.polynomials.len(), group.digest);
        let mut setups = self.setups.lock().map_err(opening_error)?;
        if let Some(setup) = setups.get(&key) {
            return Ok(setup.clone());
        }
        let (setup, _) = AkitaScheme::setup(group.setup.clone()).map_err(opening_error)?;
        let setup = Arc::new(setup);
        setups.insert(key, setup.clone());
        Ok(setup)
    }
}

/// Prove a model execution using Akita on the CPU.
pub fn prove(
    model: &Model,
    inputs: &[Tensor<i32>],
) -> Result<(ONNXProof, ModelExecutionIO), ProofVerifyError> {
    ProverPreprocessing::new(model)?.prove(inputs)
}

#[cfg(test)]
#[derive(Default)]
struct ProofSections {
    commitments: std::ops::Range<usize>,
    openings: Vec<OpeningSections>,
}

#[cfg(test)]
struct OpeningSections {
    evaluations: std::ops::Range<usize>,
    proof: std::ops::Range<usize>,
}

fn prove_preprocessed(
    preprocessing: &ProverPreprocessing<'_>,
    inputs: &[Tensor<i32>],
    #[cfg(test)] mut sections: Option<&mut ProofSections>,
) -> Result<(ONNXProof, ModelExecutionIO), ProofVerifyError> {
    let model = preprocessing.verifier.model();
    let schedule = preprocessing.verifier.schedule();
    let trace = model
        .trace(inputs)
        .map_err(|err| ProofVerifyError::Malformed(err.to_string()))?;
    let mut prover = Prover::<AkitaField, Blake2bProver>::new(
        AtlasSharedPreprocessing::preprocess(model.clone()),
        trace,
    );
    preprocessing
        .verifier
        .bind_statement(&prover.io, &mut prover.transcript);
    let originals = polynomial_map(schedule, model, &prover.trace, &prover.buckets);
    let plan = preprocessing
        .verifier
        .commitment_plan(witness_shapes(&originals)?)?;

    let mut padded = BTreeMap::new();
    let mut commitments = Vec::with_capacity(plan.groups.len());
    let mut commitment_bytes = Vec::new();
    for group in &plan.groups {
        let polynomials = group
            .polynomials
            .iter()
            .map(|id| {
                group
                    .layout
                    .pad_polynomial(&originals[id])
                    .map(|poly| (*id, poly.into_owned()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(opening_error)?;
        let columns = polynomials.iter().map(|(_, poly)| poly).collect::<Vec<_>>();
        let witness = Group::new(&columns).map_err(opening_error)?;
        let setup = preprocessing.setup(group)?;
        let (commitment, hint) = witness
            .commit(&setup, group.digest)
            .map_err(opening_error)?;
        bincode::serde::encode_into_std_write(
            &commitment,
            &mut commitment_bytes,
            bincode::config::standard(),
        )
        .map_err(opening_error)?;
        padded.extend(polynomials);
        commitments.push((witness, setup, commitment, hint));
    }
    #[cfg(test)]
    let commitment_start = prover.transcript.narg().len();
    prover
        .transcript
        .send_serializable(&AkitaMessage(commitment_bytes));
    #[cfg(test)]
    if let Some(sections) = sections.as_deref_mut() {
        sections.commitments = commitment_start..prover.transcript.narg().len();
    }

    let mut claimed = BTreeSet::new();
    for &node in model.outputs() {
        if claimed.insert(node) {
            output_claim_at(&mut prover, node.index(), OUTPUT_SUMCHECK);
        }
    }
    iop(schedule, &mut prover);
    for check in plan.coefficient_checks()? {
        let setup = check.draw(&mut prover.transcript);
        if check.one_hot_address_vars().is_some() {
            let witnesses = check
                .polynomials()
                .iter()
                .map(|id| match &originals[id] {
                    MultilinearPolynomial::OneHot(poly) => Ok(poly),
                    _ => Err(opening_error("missing original one-hot witness")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut check = BooleanityProver::new(check, setup, &witnesses)?;
            prove_relation(&mut check, &mut prover.accumulator, &mut prover.transcript);
        } else {
            let digits = check
                .polynomials()
                .iter()
                .map(|id| originals[id].clone())
                .collect();
            let mut check = DigitRangeProver::new(check, setup, digits)?;
            prove_relation(&mut check, &mut prover.accumulator, &mut prover.transcript);
        }
    }

    if !plan.groups.is_empty() {
        prover.accumulator.prepare_for_akita_sumcheck(
            &padded,
            &plan.padding,
            &mut prover.transcript,
        )?;
        let point = prover
            .accumulator
            .prove_batch_opening_sumcheck(&mut prover.transcript);
        let state = prover
            .accumulator
            .finalize_batch_opening_sumcheck(point, &mut prover.transcript);
        let tag = protocol_tag(OPENING_TAG);
        for (group, (witness, setup, commitment, hint)) in plan.groups.iter().zip(commitments) {
            let point = group.point(&state.r_sumcheck)?;
            let values: Vec<_> = group
                .polynomials
                .iter()
                .map(|id| padded[id].evaluate(point))
                .collect();
            #[cfg(test)]
            let evaluations_start = prover.transcript.narg().len();
            prover.transcript.send_scalars(&values);
            #[cfg(test)]
            let evaluations_end = prover.transcript.narg().len();
            let seed = opening_seed(&mut prover.transcript);
            let mut opening = prover_transcript::<Keccak>(&tag, &seed);
            let proof = witness
                .prove_opening(&setup, &commitment, hint, point, &values, &mut opening)
                .map_err(opening_error)?;
            let bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
                .map_err(opening_error)?;
            #[cfg(test)]
            let proof_start = prover.transcript.narg().len();
            prover.transcript.send_serializable(&AkitaMessage(bytes));
            #[cfg(test)]
            if let Some(sections) = sections.as_deref_mut() {
                sections.openings.push(OpeningSections {
                    evaluations: evaluations_start..evaluations_end,
                    proof: proof_start..prover.transcript.narg().len(),
                });
            }
        }
    }
    Ok((ONNXProof(prover.transcript.into_narg()), prover.io))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::testing::ModelBuilder;
    use crate::onnx_proof::{range_bucket::BucketWitnesses, schedule::Schedule};
    use atlas_pcs::akita::schedule_artifacts;
    use atlas_sumcheck::opening_proof::AkitaOpeningPadding;
    use jolt_atlas_verifier::onnx_proof::akita::CommitmentPlan;

    /// A commitment layout and the witness polynomials assigned to it.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct WitnessGroup {
        /// The shared polynomial representation and arity.
        layout: GroupLayout,
        /// Witness identifiers in commitment order.
        polynomials: Vec<CommittedPoly>,
        /// The original and committed domains for witnesses that need padding.
        padding: BTreeMap<CommittedPoly, AkitaOpeningPadding>,
    }

    /// Execute a model and plan groups for the witness used by Atlas's proof schedule.
    fn plan_witness_groups(
        model: &Model,
        inputs: &[Tensor<i32>],
    ) -> Result<Vec<WitnessGroup>, ProofVerifyError> {
        let schedule = Schedule::<AkitaField>::compile(model)?;
        let trace = model
            .trace(inputs)
            .map_err(|error| ProofVerifyError::Malformed(error.to_string()))?;
        let buckets = BucketWitnesses::build(model, &trace);
        let polynomials = polynomial_map(&schedule, model, &trace, &buckets);
        let plan = CommitmentPlan::new(
            &witness_shapes(&polynomials)?,
            schedule_artifacts().map_err(opening_error)?,
        )?;
        Ok(plan
            .groups
            .iter()
            .map(|group| WitnessGroup {
                layout: group.layout,
                padding: group
                    .polynomials
                    .iter()
                    .filter_map(|id| plan.padding.get(id).map(|&pad| (*id, pad)))
                    .collect(),
                polynomials: group.polynomials.clone(),
            })
            .collect())
    }

    #[test]
    fn proves_and_verifies_a_model_with_padded_one_hot_groups() {
        let mut builder = ModelBuilder::new();
        let dictionary = builder.constant(Tensor::construct((0..16).collect(), vec![16, 1]));
        let short = builder.input(vec![512]);
        let long = builder.input(vec![1024]);
        let first = builder.gather(dictionary, short, 0, vec![512, 1]);
        let second = builder.gather(dictionary, long, 0, vec![1024, 1]);
        let tail = builder.constant(Tensor::construct(vec![0; 512], vec![512, 1]));
        let output = builder.concat(&[second, first, tail], 0);
        builder.mark_output(output);
        let model = builder.build();
        let inputs = [512, 1024].map(|rows| {
            Tensor::construct((0..rows).map(|i| (i % 16) as i32).collect(), vec![rows])
        });
        let planned = plan_witness_groups(&model, &inputs).unwrap();
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].padding.len(), 1);
        let preprocessing = ProverPreprocessing::new(&model).unwrap();
        let mut sections = ProofSections::default();
        let (proof, io) = prove_preprocessed(&preprocessing, &inputs, Some(&mut sections)).unwrap();
        preprocessing.verifier().verify(&proof, &io).unwrap();
        let first_setup = preprocessing
            .setups
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .clone();
        let second_inputs = [512, 1024].map(|rows| {
            Tensor::construct(
                (0..rows).map(|i| ((i + 3) % 16) as i32).collect(),
                vec![rows],
            )
        });
        let mut second_sections = ProofSections::default();
        let (second_proof, second_io) =
            prove_preprocessed(&preprocessing, &second_inputs, Some(&mut second_sections)).unwrap();
        let verifier = preprocessing.verifier();
        verifier.verify(&second_proof, &second_io).unwrap();

        let [opening] = sections.openings.as_slice() else {
            panic!("expected one padded commitment group");
        };
        let [second_opening] = second_sections.openings.as_slice() else {
            panic!("expected one padded commitment group in the second proof");
        };
        for (label, first_range, second_range) in [
            (
                "commitments",
                sections.commitments,
                second_sections.commitments,
            ),
            (
                "opening proof",
                opening.proof.clone(),
                second_opening.proof.clone(),
            ),
            (
                "evaluations",
                opening.evaluations.clone(),
                second_opening.evaluations.clone(),
            ),
        ] {
            let replacement = &second_proof.0[second_range];
            assert_ne!(&proof.0[first_range.clone()], replacement, "{label}");
            let mut mismatched = proof.clone();
            mismatched
                .0
                .splice(first_range, replacement.iter().copied());
            assert!(
                verifier.verify(&mismatched, &io).is_err(),
                "accepted different {label}"
            );
        }

        let setups = preprocessing.setups.lock().unwrap();
        assert_eq!(setups.len(), 1);
        assert!(Arc::ptr_eq(setups.values().next().unwrap(), &first_setup));
        let mut malformed = io.clone();
        let mut serialized = serde_json::to_value(&malformed.outputs[0]).unwrap();
        serialized["data"].as_array_mut().unwrap().pop();
        malformed.outputs[0] = serde_json::from_value(serialized).unwrap();
        assert!(matches!(
            verify(&proof, &model, &malformed),
            Err(ProofVerifyError::Malformed(_))
        ));

        let mut truncated = proof;
        truncated.0.pop();
        assert!(verify(&truncated, &model, &io).is_err());
    }

    #[test]
    fn proves_and_verifies_a_model_with_packed_range_digits() {
        let mut builder = ModelBuilder::new();
        let rows = 1 << 14;
        let left = builder.input(vec![rows, 1]);
        let right = builder.input(vec![rows, 1]);
        let indices = builder.input(vec![512]);
        let dictionary = builder.constant(Tensor::construct((0..16).collect(), vec![16, 1]));
        let gathered = builder.gather(dictionary, indices, 0, vec![512, 1]);
        let product = builder.mul(left, right);
        let mut members = vec![product];
        for size in [8192, 4096, 2048, 1024, 512] {
            members.push(builder.constant(Tensor::construct(vec![0; size], vec![size, 1])));
        }
        members.push(gathered);
        let output = builder.concat(&members, 0);
        builder.mark_output(output);
        let model = builder.build();
        let inputs = [
            Tensor::construct(vec![3; rows], vec![rows, 1]),
            Tensor::construct(vec![9; rows], vec![rows, 1]),
            Tensor::construct((0..512).map(|i| i % 16).collect(), vec![512]),
        ];
        let planned = plan_witness_groups(&model, &inputs).unwrap();
        assert!(planned
            .iter()
            .flat_map(|group| &group.polynomials)
            .any(|poly| matches!(poly, CommittedPoly::GlobalRemainderDigit(..))));
        let (proof, io) = prove(&model, &inputs).unwrap();
        verify(&proof, &model, &io).unwrap();
    }

    #[test]
    fn proves_and_verifies_a_model_with_padded_dense_groups() {
        let mut builder = ModelBuilder::new();
        let dictionary = builder.constant(Tensor::construct((1..=8).collect(), vec![8, 1]));
        let indices = builder.input(vec![8]);
        let factors = builder.input(vec![8, 1]);
        let gathered = builder.gather(dictionary, indices, 0, vec![8, 1]);
        let product = builder.mul(gathered, factors);
        builder.mark_output(product);
        let model = builder.build();
        let inputs = [
            Tensor::construct((0..8).collect(), vec![8]),
            Tensor::construct((1..=8).collect(), vec![8, 1]),
        ];

        let groups = plan_witness_groups(&model, &inputs).unwrap();
        let padded: BTreeMap<_, _> = groups
            .iter()
            .flat_map(|group| group.padding.keys().map(|&id| (id, group.layout)))
            .collect();
        let layout = GroupLayout::new(11, None).unwrap();
        assert_eq!(
            padded.get(&CommittedPoly::GatherRa(gathered)),
            Some(&layout)
        );
        assert!(padded.iter().any(|(id, &group_layout)| {
            matches!(id, CommittedPoly::GlobalRemainderDigit(..)) && group_layout == layout
        }));

        let (proof, io) = prove(&model, &inputs).unwrap();
        verify(&proof, &model, &io).unwrap();
    }

    #[test]
    fn proves_multiple_outputs_with_short_one_hot_witnesses() {
        let mut builder = ModelBuilder::new();
        let dictionary = builder.constant(Tensor::construct((0..16).collect(), vec![16, 1]));
        let indices = builder.input(vec![16]);
        let gathered = builder.gather(dictionary, indices, 0, vec![16, 1]);
        let doubled = builder.add(gathered, gathered);
        builder.mark_output(gathered);
        builder.mark_output(doubled);
        builder.mark_output(gathered);
        let model = builder.build();
        let inputs = [Tensor::construct((0..16).collect(), vec![16])];

        let groups = plan_witness_groups(&model, &inputs).unwrap();
        let gather = CommittedPoly::GatherRa(gathered);
        assert!(groups.iter().any(|group| {
            group.layout.one_hot_width() == Some(16)
                && group.layout.num_vars() > 8
                && group.padding.contains_key(&gather)
        }));

        let (proof, io) = prove(&model, &inputs).unwrap();
        verify(&proof, &model, &io).unwrap();
        let mut mismatched = io.clone();
        mismatched.outputs[1] = io.outputs[0].clone();
        assert!(verify(&proof, &model, &mismatched).is_err());
        let mut repeated = io;
        repeated.outputs[2] = repeated.outputs[1].clone();
        assert!(verify(&proof, &model, &repeated).is_err());
    }

    #[test]
    fn groups_the_witnesses_of_a_model_with_multiple_gathers() {
        let mut builder = ModelBuilder::new();
        let dictionary = builder.constant(Tensor::construct((0..16).collect(), vec![16, 1]));
        let short = builder.input(vec![512]);
        let long = builder.input(vec![1024]);
        let mut expected = Vec::new();
        for index in 0..18 {
            let (indices, rows) = if index < 8 {
                (short, 512)
            } else {
                (long, 1024)
            };
            let gather = builder.gather(dictionary, indices, 0, vec![rows, 1]);
            builder.mark_output(gather);
            expected.push(CommittedPoly::GatherRa(gather));
        }
        let model = builder.build();
        let inputs = [512, 1024].map(|rows| {
            Tensor::construct(
                (0..rows).map(|index| (index % 16) as i32).collect(),
                vec![rows],
            )
        });
        let groups = plan_witness_groups(&model, &inputs).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].layout, GroupLayout::new(14, Some(16)).unwrap());
        assert_eq!(groups[0].polynomials, expected[..16]);
        assert_eq!(groups[1].polynomials, expected[16..]);
        assert_eq!(
            groups[0].padding.keys().copied().collect::<Vec<_>>(),
            expected[..8]
        );
        assert!(groups[1].padding.is_empty());
    }
}
