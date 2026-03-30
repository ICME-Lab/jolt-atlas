/// Exponentiation phase of softmax computation.
pub mod exponentiation;
/// Maximum value computation for softmax.
pub mod max;
/// Scalar division for softmax normalization.
pub mod scalar_div;
/// Sum computation for softmax.
pub mod sum;

/// Index identifying a specific feature vector in a softmax operation.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub struct SoftmaxIndex {
    /// Node index in the computation graph.
    pub node_idx: usize,
    /// Feature vector index within the softmax operation.
    pub feature_idx: usize,
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::softmax_axes::softmax::{
        exponentiation::{ReadRafParams, ReadRafProver, ReadRafVerifier, SoftmaxExpRaEncoding},
        max,
        scalar_div::{DivParams, DivProver, DivVerifier},
        sum::{SumParams, SumProver, SumVerifier},
        SoftmaxIndex,
    };
    use ark_bn254::Fr;
    use atlas_onnx_tracer::tensor::{ops::nonlinearities::softmax_fixed_128, Tensor};
    use common::VirtualPolynomial;
    use joltworks::{
        field::{IntoOpening, JoltField},
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
                VerifierOpeningAccumulator, VirtualOpeningId, BIG_ENDIAN,
            },
        },
        subprotocols::{shout, sumcheck::BatchedSumcheck, sumcheck_prover::SumcheckInstanceProver},
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_softmax() {
        // softmax i/o sizes
        let n: usize = 8;
        let N: usize = 1 << n;

        // Generate random input and trace
        let mut rng = StdRng::seed_from_u64(0x100081);
        let input = Tensor::random_small(&mut rng, &[N]);
        let (_, trace) = softmax_fixed_128::<true>(&input);
        let trace = trace.unwrap();

        // Setup prover and verifier data
        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        let _r_feature_output: Vec<Fr> = prover_transcript
            .challenge_vector_optimized::<Fr>(n)
            .into_opening();
        let r_feature_output: Vec<Fr> = verifier_transcript
            .challenge_vector_optimized::<Fr>(n)
            .into_opening();

        // Setup index for sumcheck claims
        let softmax_index = SoftmaxIndex {
            node_idx: 0,
            feature_idx: 0,
        };

        // compute output claim for sum-check (which we check as div claim)
        let div_claim =
            MultilinearPolynomial::from(trace.softmax_q.clone()).evaluate(&r_feature_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxFeatureOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::new(r_feature_output.clone()),
            div_claim,
        );

        // Send verifier sum claim
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxSumOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new()),
            Fr::from_i32(trace.exp_sum_q),
        );

        // Send max value and index claims
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxMaxOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new()),
            Fr::from_i32(trace.max_logit),
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxMaxIndex(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new()),
            Fr::from_u32(trace.max_index as u32),
        );

        // Construct div/sum/max prover and proof
        let div_params: DivParams<Fr> = DivParams::new(softmax_index, &prover_opening_accumulator);
        let div_prover_sumcheck = DivProver::initialize(&trace, div_params);
        let sum_params = SumParams::new(softmax_index, &prover_opening_accumulator);
        let sum_prover_sumcheck = SumProver::initialize(&trace, sum_params);
        let max_params = max::IndicatorParams::new(softmax_index, &prover_opening_accumulator);
        let max_prover_sumcheck = max::IndicatorProver::initialize(&trace, max_params);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(div_prover_sumcheck),
            Box::new(sum_prover_sumcheck),
            Box::new(max_prover_sumcheck),
        ];
        let (div_sum_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        // Construct exponentiation & booleanity prover and proof
        let exponentiation_read_raf_params: ReadRafParams<Fr> = ReadRafParams::new(
            softmax_index,
            &prover_opening_accumulator,
            prover_transcript,
        );
        let exponentiation_read_raf_prover_sumcheck = ReadRafProver::initialize(
            &trace,
            exponentiation_read_raf_params,
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(exponentiation_read_raf_prover_sumcheck)];
        let (exponentiation_read_raf_proof, _r_sc) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        let encoding = SoftmaxExpRaEncoding { softmax_index };
        let lookup_indices: Vec<usize> = trace
            .abs_centered_logits
            .data()
            .iter()
            .map(|v| *v as usize)
            .collect();
        let mut ra_onehot_sumchecks = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover_opening_accumulator,
            prover_transcript,
        );
        let (exponentiation_ra_onehot_proof, _r_sc) = BatchedSumcheck::prove(
            ra_onehot_sumchecks
                .iter_mut()
                .map(|v| &mut **v as _)
                .collect(),
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new());
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        // Add verifier opening points (and implicitly append claims to transcript)
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxFeatureOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::new(r_feature_output),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxSumOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxMaxOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new()),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxMaxIndex(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(Vec::<Fr>::new()),
        );

        // Verify div/sum/max sumcheck
        let div_verifier_sumcheck = DivVerifier::new(softmax_index, &verifier_opening_accumulator);
        let sum_verifier_sumcheck = SumVerifier::new(softmax_index, &verifier_opening_accumulator);
        let max_verifier_sumcheck =
            max::IndicatorVerifier::new(softmax_index, &verifier_opening_accumulator);

        BatchedSumcheck::verify(
            &div_sum_proof,
            vec![
                &div_verifier_sumcheck,
                &sum_verifier_sumcheck,
                &max_verifier_sumcheck,
            ],
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        // Verify exponentiation sumcheck
        let exponentiation_read_raf_verifier_sumcheck = ReadRafVerifier::new(
            softmax_index,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );

        let _ = BatchedSumcheck::verify(
            &exponentiation_read_raf_proof,
            vec![&exponentiation_read_raf_verifier_sumcheck],
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        let encoding = SoftmaxExpRaEncoding { softmax_index };
        let ra_onehot_verifiers = shout::ra_onehot_verifiers(
            &encoding,
            &verifier_opening_accumulator,
            verifier_transcript,
        );
        let _ = BatchedSumcheck::verify(
            &exponentiation_ra_onehot_proof,
            ra_onehot_verifiers.iter().map(|v| &**v as _).collect(),
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        //  Check input from raf
        let (_, abs_centered_logits_claim) = verifier_opening_accumulator.get_virtual_polynomial_opening(
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
        );
        let (_, softmax_operand_claim) = verifier_opening_accumulator.get_virtual_polynomial_opening(
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxInputLogitsOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
        );
        assert_eq!(
            softmax_operand_claim,
            (-abs_centered_logits_claim) + Fr::from_i32(trace.max_logit)
        );
    }

    #[test]
    fn test_onnx_softmax() {
        let n: usize = 4;
        let N: usize = 1 << n;

        let mut rng = StdRng::seed_from_u64(0x100081);
        let input = Tensor::random_small(&mut rng, &[N]);
        println!("input: {input:?}");

        let (softmax_q, _) = softmax_fixed_128::<false>(&input);
        println!("output: {softmax_q:?}");

        // Verify sum is approximately equal to scale (128)
        let sum: i32 = softmax_q.iter().sum();
        println!("sum: {sum} (expected ~128)");

        // Verify all outputs are non-negative
        assert!(
            softmax_q.iter().all(|&x| x >= 0),
            "All softmax outputs should be non-negative"
        );

        // Sum should be close to 128 (within ~10% due to integer division truncation)
        assert!(
            (sum - 128).abs() <= 16,
            "Sum should be approximately 128, got {sum}"
        );
    }

    #[test]
    #[ignore = "TODO: non-power-of-two softmax subprotocol path not fully validated yet"]
    fn test_softmax_non_power_of_two_input_len() {
        let n_elems: usize = 1000;
        let n: usize = n_elems.next_power_of_two().ilog2() as usize;

        let mut rng = StdRng::seed_from_u64(0x100082);
        let input = Tensor::random_small(&mut rng, &[n_elems]);
        let (_, trace) = softmax_fixed_128::<true>(&input);
        let trace = trace.unwrap();

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let _r_feature_output: Vec<Fr> = prover_transcript
            .challenge_vector_optimized::<Fr>(n)
            .into_opening();
        let r_feature_output: Vec<Fr> = verifier_transcript
            .challenge_vector_optimized::<Fr>(n)
            .into_opening();

        let softmax_index = SoftmaxIndex {
            node_idx: 0,
            feature_idx: 0,
        };

        let div_claim =
            MultilinearPolynomial::from(trace.softmax_q.clone()).evaluate(&r_feature_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualOpeningId::new(
                VirtualPolynomial::SoftmaxFeatureOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::NodeExecution(softmax_index.node_idx),
            ),
            OpeningPoint::<BIG_ENDIAN, Fr>::new(r_feature_output.clone()),
            div_claim,
        );
    }
}
