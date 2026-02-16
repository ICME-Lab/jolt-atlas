use crate::onnx_proof::{
    op_lookups::{
        read_raf_checking::{
            compute_lookup_indices_from_operands, ReadRafSumcheckParams, ReadRafSumcheckProver,
            ReadRafSumcheckVerifier,
        },
        InterleavedBitsMarker, OpLookupEncoding,
    },
    ops::OperatorProofTrait,
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::ReLU,
};
use common::CommittedPolynomial;
use joltworks::{
    self,
    field::JoltField,
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use rayon::prelude::*;

use crate::onnx_proof::lookup_tables::relu::ReluTable;
use common::consts::XLEN;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ReLU {
    #[tracing::instrument(skip_all, name = "ReLU::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Execution proof
        let params = ReadRafSumcheckParams::<F, ReluTable<XLEN>>::new(
            node.clone(),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut execution_sumcheck = ReadRafSumcheckProver::initialize(
            params,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (execution_proof, _) = Sumcheck::prove(
            &mut execution_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), execution_proof));

        // RaOneHotChecks proof
        let encoding = OpLookupEncoding::new(node);
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(&prover.trace, node);
        let is_interleaved = node.is_interleaved_operands();
        let lookup_indices_bits =
            compute_lookup_indices_from_operands(&operands, is_interleaved);
        let lookup_indices: Vec<usize> =
            lookup_indices_bits.par_iter().map(|&x| x.into()).collect();

        let [ra_prover, hw_prover, bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_prover, hw_prover, bool_prover];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ));

        results
    }

    #[tracing::instrument(skip_all, name = "ReLU::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Verify execution proof
        let verifier_sumcheck = ReadRafSumcheckVerifier::<F, ReluTable<XLEN>>::new(
            node.clone(),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        let execution_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        Sumcheck::verify(
            execution_proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Verify RaOneHotChecks
        let encoding = OpLookupEncoding::new(node);
        let [ra_verifier, hw_verifier, bool_verifier] = shout::ra_onehot_verifiers(
            &encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![&*ra_verifier, &*hw_verifier, &*bool_verifier],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let encoding = OpLookupEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPolynomial::NodeOutputRaD(node.idx, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::onnx_proof::AtlasSharedPreprocessing;

    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_relu() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        println!("Input: {input:?}");
        let model = model::test::relu_model(T);
        let trace = model.trace(&[input]);

        let prover_transcript = Blake2bTranscript::new(&[]);
        let preprocessing: AtlasSharedPreprocessing =
            AtlasSharedPreprocessing::preprocess(model.clone());
        let prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();
        let mut prover = Prover {
            trace: trace.clone(),
            accumulator: prover_opening_accumulator,
            preprocessing,
            transcript: prover_transcript,
        };

        let r_node_output: Vec<<Fr as JoltField>::Challenge> =
            prover.transcript.challenge_vector_optimized::<Fr>(log_T);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let relu_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            relu_claim,
        );

        let proofs = ReLU.prove(computation_node, &mut prover);
        let proofs = BTreeMap::from_iter(proofs);

        let verifier_transcript = Blake2bTranscript::new(&[]);
        let verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();

        let io = Trace::io(&trace, &model);

        let mut verifier = Verifier {
            proofs: &proofs,
            accumulator: verifier_opening_accumulator,
            preprocessing: &prover.preprocessing.clone(),
            io: &io,
            transcript: verifier_transcript,
        };
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
            verifier.transcript.challenge_vector_optimized::<Fr>(log_T);

        // Take claims
        for (key, (_, value)) in &prover.accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier
                .accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }
        verifier.accumulator.virtual_operand_claims =
            prover.accumulator.virtual_operand_claims.clone();

        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let res = ReLU.verify(computation_node, &mut verifier);

        prover.transcript.compare_to(verifier.transcript.clone());
        res.unwrap();
    }
}
