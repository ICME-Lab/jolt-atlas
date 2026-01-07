use crate::onnx_proof::ops::{
    add::{AddParams, AddProver, AddVerifier},
    cube::{CubeParams, CubeProver, CubeVerifier},
    div::{DivParams, DivProver, DivVerifier},
    einsum::{EinsumProver, EinsumVerifier},
    iff::{IffParams, IffProver, IffVerifier},
    mul::{MulParams, MulProver, MulVerifier},
    square::{SquareParams, SquareProver, SquareVerifier},
    sub::{SubParams, SubProver, SubVerifier},
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, Openings, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator,
        },
    },
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod lookup_tables;
pub mod op_lookups;
pub mod ops;

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

/* ---------- Prover Logic ---------- */

#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, T: Transcript> {
    pub opening_claims: Claims<F>,
    pub proofs: BTreeMap<usize, SumcheckInstanceProof<F, T>>,
}

impl<F: JoltField, T: Transcript> ONNXProof<F, T> {
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
            match computation_node.operator {
                // TODO: attatch this operation processing logic i.e. create parameters, initialize prover, run sumcheck, and store proof,
                //       onto the inner operators struct & move to dedicated operator module. Also refactor duplicate logic
                Operator::Add(_) => {
                    let params: AddParams<F> =
                        AddParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = AddProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
                }
                Operator::Sub(_) => {
                    let params: SubParams<F> =
                        SubParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = SubProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
                }
                Operator::Mul(_) => {
                    let params: MulParams<F> =
                        MulParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = MulProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
                }
                Operator::Div(_) => {
                    let params: DivParams<F> =
                        DivParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = DivProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
                }
                Operator::Square(_) => {
                    let params: SquareParams<F> =
                        SquareParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = SquareProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
                }
                Operator::Cube(_) => {
                    let params: CubeParams<F> =
                        CubeParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = CubeProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
                }
                Operator::Iff(_) => {
                    let params: IffParams<F> =
                        IffParams::new(computation_node.clone(), &opening_accumulator);
                    let mut prover_sumcheck = IffProver::initialize(&trace, params);
                    let (proof, _) =
                        Sumcheck::prove(&mut prover_sumcheck, &mut opening_accumulator, transcript);
                    proofs.insert(node_idx, proof);
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
                    proofs.insert(node_idx, proof);
                }
                Operator::Input(_) => {
                    // Noop

                    // Assert! claim is already cached
                    opening_accumulator
                        .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);
                }
                Operator::Constant(_) => {
                    // 1. v send `r`
                    // 2. p sends π := virtual_const(r)
                    // 3. v assert_eq!(π, const(r))

                    // Assert! claim is already cached
                    opening_accumulator
                        .assert_virtual_polynomial_opening_exists(node_poly, SumcheckId::Execution);
                }
                _ => println!("Unhandled operator in graph: {computation_node:#?}"),
            }
        }

        (
            Self {
                proofs,
                opening_claims: Claims(opening_accumulator.take()),
            },
            io,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

/* ---------- Verifier Logic ---------- */

impl<F: JoltField, T: Transcript> ONNXProof<F, T> {
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
                        .get(&node_idx)
                        .ok_or(ProofVerifyError::MissingProof(node_idx))?;
                    let verifier_sumcheck =
                        AddVerifier::new(computation_node.clone(), &opening_accumulator);
                    let _ = Sumcheck::verify(
                        proof,
                        &verifier_sumcheck,
                        &mut opening_accumulator,
                        transcript,
                    )?;
                    Ok(())
                }
                Operator::Sub(_) => {
                    let proof = self
                        .proofs
                        .get(&node_idx)
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
                        .get(&node_idx)
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
                        .get(&node_idx)
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
                        .get(&node_idx)
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
                        .get(&node_idx)
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
                        .get(&node_idx)
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
                Operator::Einsum(_) => {
                    let proof = self
                        .proofs
                        .get(&node_idx)
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

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{model::Model, tensor::Tensor};
    use joltworks::transcripts::Blake2bTranscript;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::onnx_proof::{AtlasSharedPreprocessing, ONNXProof};

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
        let (proof, io) = ONNXProof::<Fr, Blake2bTranscript>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
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
        let (proof, io) = ONNXProof::<Fr, Blake2bTranscript>::prove(&pp, &input);

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
        println!("model: {}", model.pretty_print());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let (proof, io) = ONNXProof::<Fr, Blake2bTranscript>::prove(&pp, &input);

        // verify proof
        proof.verify(&pp, &io).unwrap();
    }
}
