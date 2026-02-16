use crate::onnx_proof::{
    ops::OperatorProofTrait,
    range_checking::{
        read_raf_checking::{RangecheckRafSumcheckProver, RangecheckRafSumcheckVerifier},
        sumcheck_instance::{ReadRafSumcheckHelper, RiRangeCheckOperands, RsRangeCheckOperands},
        RangeCheckEncoding,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Rsqrt,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use rayon::prelude::*;

// TODO: Handle global scale
pub const Q: i32 = 128;
pub const Q_SQUARE: i32 = Q * Q;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Rsqrt {
    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();
        // Execution proof
        let params = RsqrtParams::new(node.clone(), &prover.accumulator);
        let mut prover_sumcheck =
            RsqrtProver::initialize(&prover.trace, &mut prover.transcript, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        // Rangecheck Raf proof
        let div_rc_prover =
            RangecheckRafSumcheckProver::<_, RiRangeCheckOperands>::new_from_prover(node, prover);
        let sqrt_rc_prover =
            RangecheckRafSumcheckProver::<_, RsRangeCheckOperands>::new_from_prover(node, prover);
        let mut rc_instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(div_rc_prover), Box::new(sqrt_rc_prover)];
        let (rangecheck_proof, _) = BatchedSumcheck::prove(
            rc_instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

        // RaOneHotChecks proof
        let div_encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
        let (div_left, div_right) =
            RiRangeCheckOperands::get_operands_tensors(&prover.trace, node);
        let div_lookup_bits =
            RiRangeCheckOperands::compute_lookup_indices(&div_left, &div_right);
        let div_lookup_indices: Vec<usize> =
            div_lookup_bits.par_iter().map(|&x| x.into()).collect();
        let [div_ra, div_hw, div_bool] = shout::ra_onehot_provers(
            &div_encoding,
            &div_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let sqrt_encoding = RangeCheckEncoding::<RsRangeCheckOperands>::new(node);
        let (sqrt_left, sqrt_right) =
            RsRangeCheckOperands::get_operands_tensors(&prover.trace, node);
        let sqrt_lookup_bits =
            RsRangeCheckOperands::compute_lookup_indices(&sqrt_left, &sqrt_right);
        let sqrt_lookup_indices: Vec<usize> =
            sqrt_lookup_bits.par_iter().map(|&x| x.into()).collect();
        let [sqrt_ra, sqrt_hw, sqrt_bool] = shout::ra_onehot_provers(
            &sqrt_encoding,
            &sqrt_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut ra_one_hot_instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            div_ra, div_hw, div_bool,
            sqrt_ra, sqrt_hw, sqrt_bool,
        ];

        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            ra_one_hot_instances
                .iter_mut()
                .map(|v| &mut **v as _)
                .collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ));

        results
    }

    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution proof
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = RsqrtVerifier::new(
            node.clone(),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let rangecheck_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RangeCheck))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        // Rangecheck Raf proof
        let div_rc_verifier =
            RangecheckRafSumcheckVerifier::<_, RiRangeCheckOperands>::new_from_verifier(
                node, verifier,
            );
        let sqrt_rc_verifier =
            RangecheckRafSumcheckVerifier::<_, RsRangeCheckOperands>::new_from_verifier(
                node, verifier,
            );
        let rc_instances: Vec<&dyn SumcheckInstanceVerifier<_, _>> =
            vec![&div_rc_verifier, &sqrt_rc_verifier];
        BatchedSumcheck::verify(
            rangecheck_proof,
            rc_instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // RaOneHotChecks proof
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let div_encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
        let [div_ra, div_hw, div_bool] = shout::ra_onehot_verifiers(
            &div_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        let sqrt_encoding = RangeCheckEncoding::<RsRangeCheckOperands>::new(node);
        let [sqrt_ra, sqrt_hw, sqrt_bool] = shout::ra_onehot_verifiers(
            &sqrt_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        );

        let ra_one_hot_instances: Vec<&dyn SumcheckInstanceVerifier<_, _>> = vec![
            &*div_ra, &*div_hw, &*div_bool,
            &*sqrt_ra, &*sqrt_hw, &*sqrt_bool,
        ];
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            ra_one_hot_instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let mut polys = vec![
            CommittedPolynomial::RsqrtNodeInv(node.idx),
            CommittedPolynomial::RsqrtNodeRsqrt(node.idx),
        ];
        let encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        for i in 0..d {
            polys.push(CommittedPolynomial::SqrtDivRangeCheckRaD(node.idx, i));
            polys.push(CommittedPolynomial::SqrtRangeCheckRaD(node.idx, i));
        }
        polys
    }
}

// Decomposes rsqrt into an inverse and a square root where the inverse `inv` of x is such that:
// 1 = x * inv + r_i  where 0 <= r_i < x
// and the square root `sqrt` of inv is such that:
// inv = sqrt * sqrt + r_s  where 0 <= r_s < 2 * sqrt + 1

// HANDLING SCALE:
// Any input to the model is quantized by a scale factor of S
// Therefore, for an input x, the quantized representation is x̂ = x * S
// The inverse inv of x is given by:
// inv = 1 / x = S / x̂
// Therefore, the quantized representation of inv is:
// in̂v = S * inv = S^2 / x̂
// Similarly, for the square root sqt of in̂v:
// sqt = sqrt(in̂v) = sqrt(S / x̂)
// Therefore, the quantized representation of sqrt is:
// sq̂t = S * sqt = S * sqrt(S / x̂) = sqrt(S^3 / x̂) = sqrt(S * in̂v)
// The two relations that we will batch together in a sumcheck instance are:
// - 0 = x̂ * in̂v + r_i - S^2
// - 0 = S * in̂v - sq̂t * sqt - r_s

// Possible optimization is to only commit to the result and a remainder,
// and find the associated range check for the unique remainder.

const DEGREE_BOUND: usize = 3;

#[derive(Clone)]
pub struct RsqrtParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> RsqrtParams<F> {
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RsqrtParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
    }
}

pub struct RsqrtProver<F: JoltField> {
    params: RsqrtParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    inv: MultilinearPolynomial<F>,
    rsqrt: MultilinearPolynomial<F>,
    r_i: MultilinearPolynomial<F>,
    r_s: MultilinearPolynomial<F>,
    // folding challenge
    gamma: F,
}

impl<F: JoltField> RsqrtProver<F> {
    pub fn initialize<T: Transcript>(
        trace: &Trace,
        transcript: &mut T,
        params: RsqrtParams<F>,
    ) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand] = operands[..] else {
            panic!("Expected one operand for Rsqrt operation")
        };
        let inv_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE / x).collect();
        let ri_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE % x).collect();
        let rs_data: Vec<i32> = inv_data
            .iter()
            .zip(output.iter())
            .map(|(&inv, &sqrt)| Q * inv - sqrt * sqrt)
            .collect();

        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let inv = MultilinearPolynomial::from(inv_data.clone());
        let r_i = MultilinearPolynomial::from(ri_data.clone());
        let rsqrt = MultilinearPolynomial::from(output.clone());
        let r_s = MultilinearPolynomial::from(rs_data.clone());
        #[cfg(test)]
        {
            let claim_inv = (0..left_operand.len())
                .map(|i| {
                    let a: F = left_operand.get_bound_coeff(i);
                    let inv = inv.get_bound_coeff(i);
                    let r_i: F = r_i.get_bound_coeff(i);
                    // range checking
                    assert!(r_i.to_u64().unwrap() < a.to_u64().unwrap());

                    a * inv + r_i - F::from_i32(Q_SQUARE)
                })
                .sum();
            assert_eq!(F::zero(), claim_inv);

            let claim_sqrt = (0..left_operand.len())
                .map(|i| {
                    let inv = inv.get_bound_coeff(i);
                    let sqrt: F = rsqrt.get_bound_coeff(i);
                    let r_s: F = r_s.get_bound_coeff(i);
                    // range checking
                    assert!(r_s.to_u64().unwrap() <= 2 * sqrt.to_u64().unwrap());

                    sqrt * sqrt + r_s - F::from_i32(Q) * inv
                })
                .sum();
            assert_eq!(F::zero(), claim_sqrt)
        }

        let gamma = transcript.challenge_scalar();
        Self {
            params,
            eq_r_node_output,
            left_operand,
            inv,
            r_i,
            rsqrt,
            r_s,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RsqrtProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            inv,
            rsqrt,
            r_i,
            r_s,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let lo1 = left_operand.get_bound_coeff(2 * g + 1);
            let inv0 = inv.get_bound_coeff(2 * g);
            let inv1 = inv.get_bound_coeff(2 * g + 1);
            let r_i0 = r_i.get_bound_coeff(2 * g);

            let rsqrt0 = rsqrt.get_bound_coeff(2 * g);
            let rsqrt1 = rsqrt.get_bound_coeff(2 * g + 1);
            let r_s0 = r_s.get_bound_coeff(2 * g);

            let c0 = lo0 * inv0 + r_i0 - F::from_i32(Q_SQUARE);

            let c1 = rsqrt0 * rsqrt0 + r_s0 - F::from_i32(Q) * inv0;

            let e0 = (lo1 - lo0) * (inv1 - inv0);
            let e1 = (rsqrt1 - rsqrt0) * (rsqrt1 - rsqrt0);
            [c0 + self.gamma * c1, e0 + self.gamma * e1]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.inv.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rsqrt.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.r_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.r_s.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.inv.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRsqrt(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.rsqrt.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
            self.r_i.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SqrtRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
            self.r_s.final_sumcheck_claim(),
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

pub struct RsqrtVerifier<F: JoltField> {
    params: RsqrtParams<F>,
    gamma: F,
}

impl<F: JoltField> RsqrtVerifier<F> {
    pub fn new<T: Transcript>(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let params = RsqrtParams::new(computation_node, accumulator);
        let gamma = transcript.challenge_scalar();
        Self { params, gamma }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RsqrtVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let inv_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let r_i_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let rsqrt_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RsqrtNodeRsqrt(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let r_s_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SqrtRemainder(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let [left_operand_claim] =
            accumulator.get_operand_claims::<1>(self.params.computation_node.idx);
        eq_eval
            * (left_operand_claim * inv_claim + r_i_claim - F::from_i32(Q_SQUARE)
                + self.gamma * (rsqrt_claim * rsqrt_claim + r_s_claim - F::from_i32(Q) * inv_claim))
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRsqrt(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SqrtRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::AtlasSharedPreprocessing;
    use std::collections::BTreeMap;

    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
        utils::f32::F32,
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
    fn test_rsqrt() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_range(&mut rng, &[T], 1..Q_SQUARE);

        let model = model::test::rsqrt_model(T);
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

        let rsqrt_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            rsqrt_claim,
        );

        let verifier_transcript = Blake2bTranscript::new(&[]);
        let verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();

        let proofs = Rsqrt { scale: F32(0.0) }.prove(computation_node, &mut prover);
        let proofs = BTreeMap::from_iter(proofs);

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

        let res = Rsqrt { scale: F32(0.0) }.verify(computation_node, &mut verifier);

        let r_prover: Fr = prover.transcript.challenge_scalar();
        let r_verifier: Fr = verifier.transcript.challenge_scalar();
        assert_eq!(r_prover, r_verifier);

        verifier.transcript.compare_to(prover.transcript);
        res.unwrap();
    }
}
