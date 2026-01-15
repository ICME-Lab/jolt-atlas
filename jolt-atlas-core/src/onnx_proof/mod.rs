use crate::onnx_proof::{
    lookup_tables::relu::ReluTable,
    op_lookups::{
        ra_virtual::{
            InstructionRaSumcheckParams, InstructionRaSumcheckProver, RaSumcheckVerifier,
        },
        read_raf_checking::{
            ReadRafSumcheckParams, ReadRafSumcheckProver, ReadRafSumcheckVerifier,
        },
    },
    ops::{
        add::{AddParams, AddProver, AddVerifier},
        broadcast::{BroadcastParams, BroadcastProver, BroadcastVerifier},
        cube::{CubeParams, CubeProver, CubeVerifier},
        div::{DivParams, DivProver, DivVerifier},
        einsum::{EinsumProver, EinsumVerifier},
        iff::{IffParams, IffProver, IffVerifier},
        moveaxis::{MoveAxisParams, MoveAxisProver, MoveAxisVerifier},
        mul::{MulParams, MulProver, MulVerifier},
        reshape::{ReshapeParams, ReshapeProver, ReshapeVerifier},
        square::{SquareParams, SquareProver, SquareVerifier},
        sub::{SubParams, SubProver, SubVerifier},
    },
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    ops::Operator,
    tensor::Tensor,
};
use common::{consts::XLEN, VirtualPolynomial};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, Openings, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    subprotocols::{
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod lookup_tables;
pub mod op_lookups;
pub mod ops;
pub mod witness;

/* ---------- Prover Logic ---------- */

#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    pub opening_claims: Claims<F>,
    pub proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    pub commitments: Vec<PCS::Commitment>,
}

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    pub fn prove(pp: &AtlasSharedPreprocessing, input: &Tensor<i32>) -> (Self, ModelExecutionIO) {
        // Initialize prover state
        let transcript = &mut T::new(b"ONNXProof");
        let mut opening_accumulator = ProverOpeningAccumulator::new(pp.model.max_T());
        let mut proofs = BTreeMap::new();

        // Generate trace and io
        let trace = pp.model.trace(&[input.clone()]); // TODO: Allow for multiple inputs
        let io = Trace::io(&trace, &pp.model);

        // Evaluate output MLE at random point τ
        let output_index = pp.model.outputs()[0];
        let output_computation_node = &pp.model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, output_computation_node);
        let r_node_output = transcript.challenge_vector_optimized::<F>(output.len().log_2());
        let output_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            output_claim,
        );

        // Iterate over computation graph in reverse topological order
        // Prove each operation using sum-check and virtual polynomials
        for (&node_idx, computation_node) in pp.model.graph.nodes.iter().rev() {
            let node_poly = VirtualPolynomial::NodeOutput(node_idx);
            match &computation_node.operator {
                // TODO: attatch this operation processing logic i.e. create parameters, initialize prover, run sumcheck, and store proof,
                //       onto the inner operators struct & move to dedicated operator module. Also refactor duplicate logic
                Operator::Add(_) => {
                    let params: AddParams<F> =
                        AddParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = AddProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Sub(_) => {
                    let params: SubParams<F> =
                        SubParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = SubProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Mul(_) => {
                    let params: MulParams<F> =
                        MulParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = MulProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Div(_) => {
                    let params: DivParams<F> =
                        DivParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = DivProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Square(_) => {
                    let params: SquareParams<F> =
                        SquareParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = SquareProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Cube(_) => {
                    let params: CubeParams<F> =
                        CubeParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = CubeProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Iff(_) => {
                    let params: IffParams<F> =
                        IffParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = IffProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Broadcast(_) => {
                    let params =
                        BroadcastParams::new(computation_node.clone(), &opening_accumulator);
                    let broadcast_prover = BroadcastProver::initialize(&trace, params);
                    broadcast_prover.prove(&mut opening_accumulator, transcript);
                }
                Operator::Reshape(_) => {
                    let params =
                        ReshapeParams::<F>::new(computation_node.clone(), &opening_accumulator);
                    let reshape_prover = ReshapeProver::initialize(params);
                    reshape_prover.prove(&mut opening_accumulator, transcript);
                }
                Operator::MoveAxis(_) => {
                    let params =
                        MoveAxisParams::<F>::new(computation_node.clone(), &opening_accumulator);
                    let moveaxis_prover = MoveAxisProver::initialize(params);
                    moveaxis_prover.prove(&mut opening_accumulator, transcript);
                }
                Operator::Einsum(_) => {
                    let mut prover_sumcheck = EinsumProver::sumcheck(
                        &pp.model,
                        &trace,
                        computation_node.clone(),
                        &opening_accumulator,
                    );
                    let (proof, _) = Sumcheck::prove(
                        &mut *prover_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    );
                    proofs.insert(ProofId(node_idx, ProofType::Execution), proof);
                }
                Operator::Input(_) => {
                    // Noop

                    // Assert! claim is already cached
                    let opening = opening_accumulator
                        .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);
                    assert!(opening.is_some())
                }
                Operator::ReLU(_) => {
                    let params = ReadRafSumcheckParams::<F, ReluTable<XLEN>>::new(
                        computation_node.clone(),
                        &opening_accumulator,
                        transcript,
                    );
                    let mut execution_sumcheck = ReadRafSumcheckProver::initialize(
                        params,
                        &trace,
                        &mut opening_accumulator,
                        transcript,
                    );
                    let (execution_proof, _) = Sumcheck::prove(
                        &mut execution_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    );
                    proofs.insert(ProofId(node_idx, ProofType::Execution), execution_proof);

                    let log_T = computation_node.num_output_elements().log_2();
                    let one_hot_params = OneHotParams::new(log_T);
                    let ra_params = InstructionRaSumcheckParams::new(
                        computation_node.clone(),
                        &OneHotParams::new(log_T),
                        &opening_accumulator,
                    );
                    let ra_prover_sumcheck =
                        InstructionRaSumcheckProver::initialize(ra_params, &trace);

                    let lookups_hamming_weight_params = op_lookups::ra_hamming_weight_params(
                        computation_node,
                        &one_hot_params,
                        &opening_accumulator,
                        transcript,
                    );
                    let lookups_booleanity_params = op_lookups::ra_booleanity_params(
                        computation_node,
                        &one_hot_params,
                        &opening_accumulator,
                        transcript,
                    );

                    let (lookups_ra_booleanity, lookups_ra_hamming_weight) =
                        op_lookups::gen_ra_one_hot_provers(
                            lookups_hamming_weight_params,
                            lookups_booleanity_params,
                            &trace,
                            computation_node,
                            &one_hot_params,
                        );

                    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
                        Box::new(ra_prover_sumcheck),
                        Box::new(lookups_ra_booleanity),
                        Box::new(lookups_ra_hamming_weight),
                    ];
                    let (ra_one_hot_proof, _r_stage6) = BatchedSumcheck::prove(
                        instances.iter_mut().map(|v| &mut **v as _).collect(),
                        &mut opening_accumulator,
                        transcript,
                    );
                    proofs.insert(
                        ProofId(node_idx, ProofType::RaOneHotChecks),
                        ra_one_hot_proof,
                    );
                }
                Operator::Constant(c) => {
                    // 1. v send `r`
                    // 2. p sends π := virtual_const(r)
                    // 3. v assert_eq!(π, const(r))

                    // Assert! claim is already cached
                    let opening = opening_accumulator
                        .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);

                    if opening.is_none() {
                        // Handle un-needed Relu operand const
                        assert!(c.0.len() == 1);
                        opening_accumulator.append_virtual(
                            transcript,
                            node_poly,
                            SumcheckId::Execution,
                            OpeningPoint::new(vec![F::Challenge::default()]),
                            F::zero(),
                        );
                    }
                }

                _ => println!("Unhandled operator in graph: {computation_node:#?}"),
            }
        }

        (
            Self {
                proofs,
                opening_claims: Claims(opening_accumulator.take()),
                commitments: vec![],
            },
            io,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProofId(pub usize, pub ProofType);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProofType {
    Execution,
    RaOneHotChecks,
}

#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

/* ---------- Verifier Logic ---------- */

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    pub fn verify(
        &self,
        pp: &AtlasSharedPreprocessing,
        io: &ModelExecutionIO,
    ) -> Result<(), ProofVerifyError> {
        // Initialize verifier state
        let transcript = &mut T::new(b"ONNXProof");
        let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new(pp.model.max_T());
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &self.opening_claims.0 {
            opening_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        // Evaluate output MLE at random point τ
        let output_index = pp.model.outputs()[0];
        let output_computation_node = &pp.model[output_index];
        let r_node_output = transcript
            .challenge_vector_optimized::<F>(output_computation_node.num_output_elements().log_2());
        let expected_output_claim =
            MultilinearPolynomial::from(io.outputs[0].clone()).evaluate(&r_node_output);
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            SumcheckId::Execution,
            r_node_output.clone().into(),
        );
        let output_claim = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(output_computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        if expected_output_claim != output_claim {
            return Err(ProofVerifyError::InvalidOpeningProof);
        }

        for (&node_idx, computation_node) in pp.model.graph.nodes.iter().rev() {
            match &computation_node.operator {
                Operator::Add(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        AddVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )
                    .unwrap();
                    Ok(())
                }
                Operator::Sub(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        SubVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Mul(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        MulVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Div(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        DivVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Square(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        SquareVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Cube(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        CubeVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Iff(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        IffVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Broadcast(_) => {
                    let broadcast_verifier = BroadcastVerifier::new(
                        computation_node.clone(),
                        &opening_accumulator,
                        &pp.model.graph,
                    );
                    broadcast_verifier.verify(&mut opening_accumulator, transcript)?;
                    Ok(())
                }
                Operator::Reshape(_) => {
                    let reshape_verifier =
                        ReshapeVerifier::new(computation_node.clone(), &opening_accumulator);
                    reshape_verifier.verify(&mut opening_accumulator, transcript)?;
                    Ok(())
                }
                Operator::MoveAxis(_) => {
                    let moveaxis_verifier =
                        MoveAxisVerifier::new(computation_node.clone(), &opening_accumulator);
                    moveaxis_verifier.verify(&mut opening_accumulator, transcript)?;
                    Ok(())
                }
                Operator::Einsum(_) => {
                    let proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck = EinsumVerifier::sumcheck(
                        &pp.model,
                        computation_node.clone(),
                        &opening_accumulator,
                    );
                    let _ = Sumcheck::verify(
                        proof,
                        &*verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;

                    Ok(())
                }
                Operator::ReLU(_) => {
                    let verifier_sumcheck = ReadRafSumcheckVerifier::<F, ReluTable<XLEN>>::new(
                        computation_node.clone(),
                        &mut opening_accumulator,
                        transcript,
                    );
                    let execution_proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::Execution))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let _ = Sumcheck::verify(
                        execution_proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;

                    let log_T = computation_node.num_output_elements().log_2();
                    let one_hot_params = OneHotParams::new(log_T);
                    let ra_verifier_sumcheck = RaSumcheckVerifier::new(
                        computation_node.clone(),
                        &one_hot_params,
                        &opening_accumulator,
                    );
                    let (lookups_ra_booleanity, lookups_rs_hamming_weight) =
                        op_lookups::new_ra_one_hot_verifiers(
                            computation_node,
                            &one_hot_params,
                            &opening_accumulator,
                            transcript,
                        );
                    let ra_one_hot_proof = self
                        .proofs
                        .get(&ProofId(node_idx, ProofType::RaOneHotChecks))
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let _ = BatchedSumcheck::verify(
                        ra_one_hot_proof,
                        vec![
                            &ra_verifier_sumcheck,
                            &lookups_ra_booleanity,
                            &lookups_rs_hamming_weight,
                        ],
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Input(_) => {
                    // Check input_claim == IO.evaluate_input(r_input)
                    let (r_node_input, input_claim) = opening_accumulator
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::NodeOutput(computation_node.idx),
                            SumcheckId::Execution,
                        );
                    let expected_claim =
                        MultilinearPolynomial::from(io.inputs[0].clone()).evaluate(&r_node_input.r);
                    if expected_claim != input_claim {
                        return Err(ProofVerifyError::InvalidOpeningProof);
                    }
                    Ok(())
                }
                Operator::Constant(inner) => {
                    if inner.0.len() == 1 {
                        // Handle un-needed Relu operand const
                        opening_accumulator.append_virtual(
                            transcript,
                            VirtualPolynomial::NodeOutput(computation_node.idx),
                            SumcheckId::Execution,
                            OpeningPoint::new(vec![F::Challenge::default()]),
                        );
                        let (_, const_claim) = opening_accumulator.get_virtual_polynomial_opening(
                            VirtualPolynomial::NodeOutput(computation_node.idx),
                            SumcheckId::Execution,
                        );
                        if F::zero() != const_claim {
                            return Err(ProofVerifyError::InvalidOpeningProof);
                        }
                    } else {
                        let (r_node_const, const_claim) = opening_accumulator
                            .get_virtual_polynomial_opening(
                                VirtualPolynomial::NodeOutput(computation_node.idx),
                                SumcheckId::Execution,
                            );
                        let expected_claim =
                            MultilinearPolynomial::from(inner.0.clone()).evaluate(&r_node_const.r);
                        if expected_claim != const_claim {
                            return Err(ProofVerifyError::InvalidOpeningProof);
                        }
                    }
                    Ok(())
                }
                _ => {
                    tracing::warn!("Unhandled operator in graph: {computation_node:#?}"); // TODO: return error
                    Ok(())
                }
            }?;
        }
        Ok(())
    }
}

/* ---------- Preprocessing ---------- */

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasSharedPreprocessing {
    pub model: Model,
}

impl AtlasSharedPreprocessing {
    pub fn preprocess(model: Model) -> Self {
        Self { model }
    }
}

#[derive(Clone)]
pub struct AtlasProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub shared: AtlasSharedPreprocessing,
}

impl<F, PCS> AtlasProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "AtlasProverPreprocessing::gen")]
    pub fn new(shared: AtlasSharedPreprocessing) -> AtlasProverPreprocessing<F, PCS> {
        use common::consts::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let max_T: usize = shared.model.max_T();
        let max_log_T = max_T.log_2();
        // Use the maximum possible log_k_chunk for generator setup
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let generators = PCS::setup_prover(max_log_k_chunk + max_log_T);
        AtlasProverPreprocessing { generators, shared }
    }
}

#[derive(Debug, Clone)]
pub struct AtlasVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: AtlasSharedPreprocessing,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&AtlasProverPreprocessing<F, PCS>>
    for AtlasVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &AtlasProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{model::Model, tensor::Tensor};
    use joltworks::{poly::commitment::dory::DoryCommitmentScheme, transcripts::Blake2bTranscript};
    use rand::{rngs::StdRng, SeedableRng};
    use serde_json::Value;
    use std::{collections::HashMap, fs::File, io::Read, time::Instant};

    use crate::onnx_proof::{AtlasSharedPreprocessing, ONNXProof};

    #[test]
    fn test_article_classification() {
        let working_dir = "../atlas-onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json");
        let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");
        // Input text string to classify
        let input_text = "The government plans new trade policies.";

        // Build input vector from the input text (512 features for small MLP)
        let input_vector = build_input_vector(input_text, &vocab);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) = ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(
            &pp,
            &Tensor::construct(input_vector, vec![1, 512]),
        );
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        /// Load vocab.json into HashMap<String, (usize, i32)>
        fn load_vocab(
            path: &str,
        ) -> Result<HashMap<String, (usize, i32)>, Box<dyn std::error::Error>> {
            let mut file = File::open(path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;

            let json_value: Value = serde_json::from_str(&contents)?;
            let mut vocab = HashMap::new();

            if let Value::Object(map) = json_value {
                for (word, data) in map {
                    if let (Some(index), Some(idf)) = (
                        data.get("index").and_then(|v| v.as_u64()),
                        data.get("idf").and_then(|v| v.as_f64()),
                    ) {
                        vocab.insert(word, (index as usize, (idf * 1000.0) as i32));
                        // Scale IDF and convert to i32
                    }
                }
            }

            Ok(vocab)
        }

        fn build_input_vector(text: &str, vocab: &HashMap<String, (usize, i32)>) -> Vec<i32> {
            let mut vec = vec![0; 512];

            // Split text into tokens (preserve punctuation as tokens)
            let re = regex::Regex::new(r"\w+|[^\w\s]").unwrap();
            for cap in re.captures_iter(text) {
                let token = cap.get(0).unwrap().as_str().to_lowercase();
                if let Some(&(index, idf)) = vocab.get(&token) {
                    if index < 512 {
                        vec[index] += idf; // accumulate idf value
                    }
                }
            }

            vec
        }
    }

    #[test]
    fn test_add_sub_mul() {
        let working_dir = "../atlas-onnx-tracer/models/test_add_sub_mul/";

        // Create test input vector of size 65536
        // Using small values to avoid overflow
        let mut rng = StdRng::seed_from_u64(0x100);
        // Create tensor with shape [65536]
        let input = Tensor::random_small(&mut rng, &[1 << 16]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }

    #[test]
    fn test_broadcast() {
        let working_dir = "../atlas-onnx-tracer/models/broadcast/";

        // Create test input vector of size [4]
        // Using simple values to test broadcasting
        let input_vector = vec![1, 2, 3, 4];

        let input = Tensor::construct(input_vector, vec![4]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
        println!("Expected: input [4] broadcasted through operations to shape [2, 5, 4]");
    }

    #[test]
    fn test_reshape() {
        let working_dir = "../atlas-onnx-tracer/models/reshape/";

        // Create test input vector of size [4]
        // Using simple values to test reshaping
        let input_vector = vec![1, 2, 3, 4];

        let input = Tensor::construct(input_vector, vec![4]);

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
    }

    #[test]
    fn test_moveaxis() {
        let working_dir = "../atlas-onnx-tracer/models/moveaxis/";

        // Create test input vector of size [2, 4, 8]
        // Using simple values to test moveaxis
        let input_vector: Vec<i32> = (1..=64).collect();

        let input = Tensor::construct(input_vector, vec![2, 4, 8]);
        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let timing = Instant::now();
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);
        println!("Proof generation took {:?}", timing.elapsed());

        // verify proof
        proof.verify(&pp, &io).unwrap();

        // Print output for verification
        println!("Output shape: {:?}", io.outputs[0].dims());
    }

    #[test]
    fn test_mlp_square() {
        // Fixed-point scale factor: 2^7 = 128
        const SCALE: i32 = 128;

        // Create test input vector [1, 4]
        // Using simple values for testing
        let input_vector = vec![
            (70.0 * SCALE as f32) as i32,
            (71.0 * SCALE as f32) as i32,
            (72.0 * SCALE as f32) as i32,
            (73.0 * SCALE as f32) as i32,
        ];
        let working_dir = "../atlas-onnx-tracer/models/mlp_square/";

        let input = Tensor::new(Some(&input_vector), &[1, 4]).unwrap();

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }

    #[test]
    fn test_mlp_square_4layer() {
        // Fixed-point scale factor: 2^7 = 128
        const SCALE: i32 = 128;

        // Create test input vector [1, 4]
        // Using simple values for testing
        let input_vector = vec![
            (1.0 * SCALE as f32) as i32,
            (2.0 * SCALE as f32) as i32,
            (3.0 * SCALE as f32) as i32,
            (4.0 * SCALE as f32) as i32,
        ];
        let working_dir = "../atlas-onnx-tracer/models/mlp_square_4layer/";

        let input = Tensor::new(Some(&input_vector), &[1, 4]).unwrap();

        // Load the model
        let model = Model::load(&format!("{working_dir}network.onnx"), &Default::default());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) =
            ONNXProof::<Fr, Blake2bTranscript, DoryCommitmentScheme>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }
}
