use std::array;

use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use common::VirtualPolynomial;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;

use crate::utils::dims::EinsumDims;

const DEGREE_BOUND_DOT: usize = 2;
const DEGREE_BOUND_EQ: usize = 3;

#[derive(Clone, Copy)]
enum RbmkRbnkBmnVariant {
    AbmkAbnkAbmn {
        log_a: usize,
        log_b: usize,
        log_m: usize,
        log_n: usize,
        log_k: usize,
    },
    AcbmkKcnCbmn {
        log_a: usize,
        log_c: usize,
        log_b: usize,
        log_m: usize,
        log_n: usize,
        log_k: usize,
    },
    CbmkCbknAmn {
        log_cb: usize,
        log_m: usize,
        log_n: usize,
        log_k: usize,
    },
}

impl RbmkRbnkBmnVariant {
    fn from_node(computation_node: &ComputationNode, einsum_dims: &EinsumDims) -> Self {
        let atlas_onnx_tracer::ops::Operator::Einsum(einsum) = &computation_node.operator else {
            panic!("Expected Einsum operator")
        };
        match einsum.equation.as_str() {
            "abmk,abnk->abmn" => Self::AbmkAbnkAbmn {
                log_a: {
                    assert!(
                        computation_node.output_dims.len() == 4,
                        "abmk,abnk->abmn expects a rank-4 output"
                    );
                    assert_eq!(
                        einsum_dims.left_operand()[0],
                        computation_node.output_dims[0] * computation_node.output_dims[1],
                        "abmk,abnk->abmn requires flattening contiguous a,b axes into the retained batch pack"
                    );
                    assert_eq!(
                        einsum_dims.right_operand()[0],
                        einsum_dims.left_operand()[0],
                        "abmk,abnk->abmn requires both operands to share the same retained batch pack"
                    );
                    computation_node.output_dims[0].log_2()
                },
                log_b: computation_node.output_dims[1].log_2(),
                log_m: computation_node.output_dims[2].log_2(),
                log_n: computation_node.output_dims[3].log_2(),
                log_k: einsum_dims.left_operand()[2].log_2(),
            },
            "acbmk,kcn->cbmn" => Self::AcbmkKcnCbmn {
                log_a: {
                    assert!(
                        computation_node.output_dims.len() == 4,
                        "acbmk,kcn->cbmn expects a rank-4 output"
                    );
                    assert_eq!(
                        einsum_dims.left_operand()[1],
                        computation_node.output_dims[0] * computation_node.output_dims[1],
                        "acbmk,kcn->cbmn requires flattening contiguous c,b axes into the retained batch pack"
                    );
                    assert_eq!(
                        einsum_dims.right_operand()[1],
                        computation_node.output_dims[0],
                        "acbmk,kcn->cbmn requires the right operand c axis to match the output c axis"
                    );
                    einsum_dims.left_operand()[0].log_2()
                },
                log_c: computation_node.output_dims[0].log_2(),
                log_b: computation_node.output_dims[1].log_2(),
                log_m: computation_node.output_dims[2].log_2(),
                log_n: computation_node.output_dims[3].log_2(),
                log_k: einsum_dims.left_operand()[3].log_2(),
            },
            "cbmk,cbkn->amn" => Self::CbmkCbknAmn {
                log_cb: {
                    assert!(
                        computation_node.output_dims.len() == 3,
                        "cbmk,cbkn->amn expects a rank-3 output"
                    );
                    assert!(
                        computation_node.output_dims[0] == 1,
                        "cbmk,cbkn->amn is interpreted as cbmk,cbkn->1mn in the current tracer"
                    );
                    assert_eq!(
                        einsum_dims.left_operand()[0],
                        einsum_dims.right_operand()[0],
                        "cbmk,cbkn->amn requires both operands to share the same reduced batch pack"
                    );
                    einsum_dims.left_operand()[0].log_2()
                },
                log_m: computation_node.output_dims[computation_node.output_dims.len() - 2].log_2(),
                log_n: computation_node.output_dims[computation_node.output_dims.len() - 1].log_2(),
                log_k: einsum_dims.left_operand()[2].log_2(),
            },
            other => panic!("unexpected rbmk_rbnk_bmn equation: {other}"),
        }
    }

    fn degree(self) -> usize {
        match self {
            Self::CbmkCbknAmn { .. } => DEGREE_BOUND_DOT,
            Self::AbmkAbnkAbmn { .. } | Self::AcbmkKcnCbmn { .. } => DEGREE_BOUND_EQ,
        }
    }

    fn num_rounds(self) -> usize {
        match self {
            Self::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_k,
                ..
            } => log_a + log_b + log_k,
            Self::AcbmkKcnCbmn {
                log_a,
                log_c,
                log_b,
                log_k,
                ..
            } => log_c + log_b + log_a + log_k,
            Self::CbmkCbknAmn { log_cb, log_k, .. } => log_cb + log_k,
        }
    }
}

/// Parameters for the shared rbmk_rbnk_bmn einsum family.
#[derive(Clone)]
pub struct RbmkRbnkBmnParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    variant: RbmkRbnkBmnVariant,
}

impl<F: JoltField> RbmkRbnkBmnParams<F> {
    /// Creates params for the shared rbmk_rbnk_bmn family.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r_node_output = accumulator.get_node_output_opening(computation_node.idx).0;
        let variant = RbmkRbnkBmnVariant::from_node(&computation_node, &einsum_dims);
        Self {
            r_node_output,
            computation_node,
            variant,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RbmkRbnkBmnParams<F> {
    fn degree(&self) -> usize {
        self.variant.degree()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, einsum_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        einsum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.variant.num_rounds()
    }
}

/// Prover for the shared rbmk_rbnk_bmn einsum family.
pub struct RbmkRbnkBmnProver<F: JoltField> {
    params: RbmkRbnkBmnParams<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    eq_output_batches: Option<MultilinearPolynomial<F>>,
    eq_bound_claim: Option<F>,
    #[cfg(debug_assertions)]
    original_left: Vec<i32>,
    #[cfg(debug_assertions)]
    original_right: Vec<i32>,
}

impl<F: JoltField> RbmkRbnkBmnProver<F> {
    #[allow(clippy::too_many_arguments)]
    fn build_abmk_abnk_abmn(
        left_operand: &[i32],
        right_operand: &[i32],
        params: &RbmkRbnkBmnParams<F>,
        log_a: usize,
        log_b: usize,
        log_m: usize,
        log_n: usize,
        log_k: usize,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let a = 1usize << log_a;
        let b = 1usize << log_b;
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        let k = 1usize << log_k;

        let (r_a, r_bmn) = params.r_node_output.split_at(log_a);
        let (r_b, r_mn) = r_bmn.split_at(log_b);
        let (r_m, r_n) = r_mn.split_at(log_m);
        let eq_r_ab = EqPolynomial::evals(&[r_a.r.as_slice(), r_b.r.as_slice()].concat());
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);

        let batch = a * b;
        let left: Vec<F> = (0..batch * k)
            .into_par_iter()
            .map(|hj| {
                let h = hj / k;
                let j = hj % k;
                let mut sum = F::zero();
                for i in 0..m {
                    let idx = (h * m + i) * k + j;
                    sum += F::from_i32(left_operand[idx]) * eq_r_m[i];
                }
                sum
            })
            .collect();
        let right: Vec<F> = (0..batch * k)
            .into_par_iter()
            .map(|hj| {
                let h = hj / k;
                let j = hj % k;
                let mut sum = F::zero();
                for l in 0..n {
                    let idx = (h * n + l) * k + j;
                    sum += F::from_i32(right_operand[idx]) * eq_r_n[l];
                }
                sum
            })
            .collect();
        (left, right, eq_r_ab)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_acbmk_kcn_cbmn(
        left_operand: &[i32],
        right_operand: &[i32],
        params: &RbmkRbnkBmnParams<F>,
        log_a: usize,
        log_c: usize,
        log_b: usize,
        log_m: usize,
        log_n: usize,
        log_k: usize,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let a = 1usize << log_a;
        let c = 1usize << log_c;
        let b = 1usize << log_b;
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        let k = 1usize << log_k;

        let (r_c, r_bmn) = params.r_node_output.split_at(log_c);
        let (r_b, r_mn) = r_bmn.split_at(log_b);
        let (r_m, r_n) = r_mn.split_at(log_m);
        let eq_r_cb = EqPolynomial::evals(&[r_c.r.as_slice(), r_b.r.as_slice()].concat());
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);

        let cb = c * b;
        let left: Vec<F> = (0..cb * a * k)
            .into_par_iter()
            .map(|hak| {
                let h = hak / (a * k);
                let rem = hak % (a * k);
                let a_idx = rem / k;
                let k_idx = rem % k;
                let c_idx = h / b;
                let b_idx = h % b;
                let mut sum = F::zero();
                for i in 0..m {
                    let idx = ((((a_idx * c + c_idx) * b + b_idx) * m + i) * k) + k_idx;
                    sum += F::from_i32(left_operand[idx]) * eq_r_m[i];
                }
                sum
            })
            .collect();
        let right_base: Vec<F> = (0..c * k)
            .into_par_iter()
            .map(|ck| {
                let c_idx = ck / k;
                let k_idx = ck % k;
                let mut sum = F::zero();
                for n_idx in 0..n {
                    let idx = (k_idx * c + c_idx) * n + n_idx;
                    sum += F::from_i32(right_operand[idx]) * eq_r_n[n_idx];
                }
                sum
            })
            .collect();
        let right: Vec<F> = (0..cb * a * k)
            .into_par_iter()
            .map(|hak| {
                let h = hak / (a * k);
                let rem = hak % (a * k);
                let _a_idx = rem / k;
                let k_idx = rem % k;
                let c_idx = h / b;
                right_base[c_idx * k + k_idx]
            })
            .collect();
        (left, right, eq_r_cb)
    }

    fn build_cbmk_cbkn_amn(
        left_operand: &[i32],
        right_operand: &[i32],
        params: &RbmkRbnkBmnParams<F>,
        log_cb: usize,
        log_m: usize,
        log_n: usize,
        log_k: usize,
    ) -> (Vec<F>, Vec<F>) {
        let cb = 1usize << log_cb;
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        let k = 1usize << log_k;

        let split_at_mn = params.r_node_output.r.len().saturating_sub(log_m + log_n);
        let (_, r_mn) = params.r_node_output.split_at(split_at_mn);
        let (r_m, r_n) = r_mn.split_at(log_m);
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);

        let left: Vec<F> = (0..cb * k)
            .into_par_iter()
            .map(|hk| {
                let h = hk / k;
                let k_idx = hk % k;
                let mut sum = F::zero();
                for i in 0..m {
                    let idx = (h * m + i) * k + k_idx;
                    sum += F::from_i32(left_operand[idx]) * eq_r_m[i];
                }
                sum
            })
            .collect();
        let right: Vec<F> = (0..cb * k)
            .into_par_iter()
            .map(|hk| {
                let h = hk / k;
                let k_idx = hk % k;
                let mut sum = F::zero();
                for l in 0..n {
                    let idx = (h * k + k_idx) * n + l;
                    sum += F::from_i32(right_operand[idx]) * eq_r_n[l];
                }
                sum
            })
            .collect();
        (left, right)
    }

    /// Initializes the prover.
    pub fn initialize(trace: &Trace, params: RbmkRbnkBmnParams<F>) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        #[cfg(not(debug_assertions))]
        let _ = &output;
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for rbmk_rbnk_bmn operation")
        };

        let (left_values, right_values, eq_values) = match params.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_m,
                log_n,
                log_k,
            } => {
                let (left, right, eq) = Self::build_abmk_abnk_abmn(
                    left_operand,
                    right_operand,
                    &params,
                    log_a,
                    log_b,
                    log_m,
                    log_n,
                    log_k,
                );
                (left, right, Some(eq))
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_a,
                log_c,
                log_b,
                log_m,
                log_n,
                log_k,
            } => {
                let (left, right, eq) = Self::build_acbmk_kcn_cbmn(
                    left_operand,
                    right_operand,
                    &params,
                    log_a,
                    log_c,
                    log_b,
                    log_m,
                    log_n,
                    log_k,
                );
                (left, right, Some(eq))
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn {
                log_cb,
                log_m,
                log_n,
                log_k,
            } => {
                let (left, right) = Self::build_cbmk_cbkn_amn(
                    left_operand,
                    right_operand,
                    &params,
                    log_cb,
                    log_m,
                    log_n,
                    log_k,
                );
                (left, right, None)
            }
        };

        #[cfg(debug_assertions)]
        {
            let packed_claim: F = match (&params.variant, &eq_values) {
                (RbmkRbnkBmnVariant::CbmkCbknAmn { .. }, None) => left_values
                    .iter()
                    .zip(right_values.iter())
                    .map(|(l, r)| *l * *r)
                    .sum(),
                (RbmkRbnkBmnVariant::AbmkAbnkAbmn { log_k, .. }, Some(eq))
                | (
                    RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                        log_a: _, log_k, ..
                    },
                    Some(eq),
                ) => {
                    let low_bits = match params.variant {
                        RbmkRbnkBmnVariant::AbmkAbnkAbmn { .. } => *log_k,
                        RbmkRbnkBmnVariant::AcbmkKcnCbmn { log_a, .. } => log_a + *log_k,
                        RbmkRbnkBmnVariant::CbmkCbknAmn { .. } => unreachable!(),
                    };
                    left_values
                        .iter()
                        .zip(right_values.iter())
                        .enumerate()
                        .map(|(idx, (l, r))| *l * *r * eq[idx >> low_bits])
                        .sum()
                }
                _ => unreachable!(),
            };
            let output_claim =
                MultilinearPolynomial::from(output.clone()).evaluate(&params.r_node_output.r);
            debug_assert_eq!(packed_claim, output_claim);
        }

        Self {
            params,
            left_operand: MultilinearPolynomial::from(left_values),
            right_operand: MultilinearPolynomial::from(right_values),
            eq_output_batches: eq_values.map(MultilinearPolynomial::from),
            eq_bound_claim: None,
            #[cfg(debug_assertions)]
            original_left: left_operand.data().to_vec(),
            #[cfg(debug_assertions)]
            original_right: right_operand.data().to_vec(),
        }
    }

    fn compute_message_with_eq(
        &mut self,
        round: usize,
        previous_claim: F,
        log_eq: usize,
        low_bits_after_eq: usize,
    ) -> UniPoly<F> {
        let eq_poly = self
            .eq_output_batches
            .as_ref()
            .expect("eq polynomial must exist");
        let half_poly_len = self.left_operand.len() / 2;
        let uni_poly_evals: [F; DEGREE_BOUND_EQ] = (0..half_poly_len)
            .into_par_iter()
            .map(|idx| {
                let l_evals = self
                    .left_operand
                    .sumcheck_evals_array::<DEGREE_BOUND_EQ>(idx, BindingOrder::HighToLow);
                let r_evals = self
                    .right_operand
                    .sumcheck_evals_array::<DEGREE_BOUND_EQ>(idx, BindingOrder::HighToLow);
                let eq_evals = if round < log_eq {
                    let h = idx >> low_bits_after_eq;
                    eq_poly.sumcheck_evals_array::<DEGREE_BOUND_EQ>(h, BindingOrder::HighToLow)
                } else {
                    let eq_bound = self.eq_bound_claim.expect("eq claim should be cached");
                    [eq_bound; DEGREE_BOUND_EQ]
                };
                [
                    l_evals[0] * r_evals[0] * eq_evals[0],
                    l_evals[1] * r_evals[1] * eq_evals[1],
                    l_evals[2] * r_evals[2] * eq_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND_EQ],
                |running, new| array::from_fn(|idx| running[idx] + new[idx]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RbmkRbnkBmnProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        match self.params.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_k,
                ..
            } => self.compute_message_with_eq(round, previous_claim, log_a + log_b, log_k),
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_c,
                log_b,
                log_a,
                log_k,
                ..
            } => self.compute_message_with_eq(round, previous_claim, log_c + log_b, log_a + log_k),
            RbmkRbnkBmnVariant::CbmkCbknAmn { .. } => {
                let half_poly_len = self.left_operand.len() / 2;
                let uni_poly_evals: [F; DEGREE_BOUND_DOT] = (0..half_poly_len)
                    .into_par_iter()
                    .map(|idx| {
                        let l_evals = self.left_operand.sumcheck_evals(
                            idx,
                            DEGREE_BOUND_DOT,
                            BindingOrder::HighToLow,
                        );
                        let r_evals = self.right_operand.sumcheck_evals(
                            idx,
                            DEGREE_BOUND_DOT,
                            BindingOrder::HighToLow,
                        );
                        [l_evals[0] * r_evals[0], l_evals[1] * r_evals[1]]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND_DOT],
                        |running, new| array::from_fn(|idx| running[idx] + new[idx]),
                    );
                UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
            }
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.left_operand
            .bind_parallel(r_j, BindingOrder::HighToLow);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::HighToLow);
        if let Some(eq_poly) = &mut self.eq_output_batches {
            let log_eq = match self.params.variant {
                RbmkRbnkBmnVariant::AbmkAbnkAbmn { log_a, log_b, .. } => log_a + log_b,
                RbmkRbnkBmnVariant::AcbmkKcnCbmn { log_c, log_b, .. } => log_c + log_b,
                RbmkRbnkBmnVariant::CbmkCbknAmn { .. } => 0,
            };
            if round < log_eq {
                eq_poly.bind_parallel(r_j, BindingOrder::HighToLow);
                if round == log_eq - 1 {
                    self.eq_bound_claim = Some(eq_poly.final_sumcheck_claim());
                }
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        match self.params.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_m,
                ..
            } => {
                let (_, r_bmn) = self.params.r_node_output.split_at(log_a);
                let (_, r_mn) = r_bmn.split_at(log_b);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_ab, r_k) = sumcheck_challenges.split_at(log_a + log_b);
                let (r_a, r_b) = r_ab.split_at(log_a);
                let left_point = [r_a, r_b, &r_m.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&left_point),
                    self.left_operand.final_sumcheck_claim(),
                );
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    MultilinearPolynomial::from(self.original_left.clone()).evaluate(&left_point),
                    self.left_operand.final_sumcheck_claim()
                );

                let right_point = [r_a, r_b, &r_n.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&right_point),
                    self.right_operand.final_sumcheck_claim(),
                );
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    MultilinearPolynomial::from(self.original_right.clone()).evaluate(&right_point),
                    self.right_operand.final_sumcheck_claim()
                );
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_c,
                log_b,
                log_m,
                log_a,
                ..
            } => {
                let (_, r_bmn) = self.params.r_node_output.split_at(log_c);
                let (_, r_mn) = r_bmn.split_at(log_b);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_cb, r_ak) = sumcheck_challenges.split_at(log_c + log_b);
                let (r_c, r_b) = r_cb.split_at(log_c);
                let (r_a, r_k) = r_ak.split_at(log_a);
                let left_point = [r_a, r_c, r_b, &r_m.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&left_point),
                    self.left_operand.final_sumcheck_claim(),
                );
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    MultilinearPolynomial::from(self.original_left.clone()).evaluate(&left_point),
                    self.left_operand.final_sumcheck_claim()
                );

                let right_point = [r_k, r_c, &r_n.r].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&right_point),
                    self.right_operand.final_sumcheck_claim(),
                );
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    MultilinearPolynomial::from(self.original_right.clone()).evaluate(&right_point),
                    self.right_operand.final_sumcheck_claim()
                );
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn {
                log_cb,
                log_m,
                log_n,
                ..
            } => {
                let split_at_mn = self
                    .params
                    .r_node_output
                    .r
                    .len()
                    .saturating_sub(log_m + log_n);
                let (_, r_mn) = self.params.r_node_output.split_at(split_at_mn);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_cb, r_k) = sumcheck_challenges.split_at(log_cb);
                let left_point = [r_cb, &r_m.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&left_point),
                    self.left_operand.final_sumcheck_claim(),
                );
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    MultilinearPolynomial::from(self.original_left.clone()).evaluate(&left_point),
                    self.left_operand.final_sumcheck_claim()
                );

                let right_point = [r_cb, r_k, &r_n.r].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&right_point),
                    self.right_operand.final_sumcheck_claim(),
                );
                #[cfg(debug_assertions)]
                debug_assert_eq!(
                    MultilinearPolynomial::from(self.original_right.clone()).evaluate(&right_point),
                    self.right_operand.final_sumcheck_claim()
                );
            }
        }
    }
}

/// Verifier for the shared rbmk_rbnk_bmn einsum family.
pub struct RbmkRbnkBmnVerifier<F: JoltField> {
    params: RbmkRbnkBmnParams<F>,
}

impl<F: JoltField> RbmkRbnkBmnVerifier<F> {
    /// Creates the verifier.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = RbmkRbnkBmnParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RbmkRbnkBmnVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let left_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        let right_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[1],
            self.params.computation_node.idx,
        );
        let eq_claim = match self.params.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn { log_a, log_b, .. } => {
                let (r_a, r_bmn) = self.params.r_node_output.split_at(log_a);
                let (r_b, _) = r_bmn.split_at(log_b);
                let sumcheck_opening = sumcheck_challenges.into_opening();
                let (r_ab, _) = sumcheck_opening.split_at(log_a + log_b);
                EqPolynomial::mle(&[r_a.r.as_slice(), r_b.r.as_slice()].concat(), r_ab)
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn { log_c, log_b, .. } => {
                let (r_c, r_bmn) = self.params.r_node_output.split_at(log_c);
                let (r_b, _) = r_bmn.split_at(log_b);
                let sumcheck_opening = sumcheck_challenges.into_opening();
                let (r_cb, _) = sumcheck_opening.split_at(log_c + log_b);
                EqPolynomial::mle(&[r_c.r.as_slice(), r_b.r.as_slice()].concat(), r_cb)
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn { .. } => F::one(),
        };
        left_claim * right_claim * eq_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        match self.params.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_m,
                ..
            } => {
                let (_, r_bmn) = self.params.r_node_output.split_at(log_a);
                let (_, r_mn) = r_bmn.split_at(log_b);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_ab, r_k) = sumcheck_challenges.split_at(log_a + log_b);
                let (r_a, r_b) = r_ab.split_at(log_a);
                let left_point = [r_a, r_b, &r_m.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&left_point),
                );
                let right_point = [r_a, r_b, &r_n.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&right_point),
                );
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_c,
                log_b,
                log_m,
                log_a,
                ..
            } => {
                let (_, r_bmn) = self.params.r_node_output.split_at(log_c);
                let (_, r_mn) = r_bmn.split_at(log_b);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_cb, r_ak) = sumcheck_challenges.split_at(log_c + log_b);
                let (r_c, r_b) = r_cb.split_at(log_c);
                let (r_a, r_k) = r_ak.split_at(log_a);
                let left_point = [r_a, r_c, r_b, &r_m.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&left_point),
                );
                let right_point = [r_k, r_c, &r_n.r].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&right_point),
                );
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn {
                log_cb,
                log_m,
                log_n,
                ..
            } => {
                let split_at_mn = self
                    .params
                    .r_node_output
                    .r
                    .len()
                    .saturating_sub(log_m + log_n);
                let (_, r_mn) = self.params.r_node_output.split_at(split_at_mn);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_cb, r_k) = sumcheck_challenges.split_at(log_cb);
                let left_point = [r_cb, &r_m.r, r_k].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&left_point),
                );
                let right_point = [r_cb, r_k, &r_n.r].concat();
                accumulator.append_virtual(
                    transcript,
                    VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                    SumcheckId::NodeExecution(self.params.computation_node.idx),
                    self.params.normalize_opening_point(&right_point),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn abmk_abnk_abmn_model(
        rng: &mut StdRng,
        a: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Model {
        let mut builder = ModelBuilder::new();
        let left = builder.input(vec![a, b, m, k]);
        let right = builder.constant(Tensor::random_small(rng, &[a, b, n, k]));
        let out = builder.einsum("abmk,abnk->abmn", vec![left, right], vec![a, b, m, n]);
        builder.mark_output(out);
        builder.build()
    }

    fn acbmk_kcn_cbmn_model(
        rng: &mut StdRng,
        a: usize,
        c: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Model {
        let mut builder = ModelBuilder::new();
        let left = builder.input(vec![a, c, b, m, k]);
        let right = builder.constant(Tensor::random_small(rng, &[k, c, n]));
        let out = builder.einsum("acbmk,kcn->cbmn", vec![left, right], vec![c, b, m, n]);
        builder.mark_output(out);
        builder.build()
    }

    fn cbmk_cbkn_amn_model(
        rng: &mut StdRng,
        c: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Model {
        let mut builder = ModelBuilder::new();
        let left = builder.input(vec![c, b, m, k]);
        let right = builder.constant(Tensor::random_small(rng, &[c, b, k, n]));
        let out = builder.einsum("cbmk,cbkn->amn", vec![left, right], vec![1, m, n]);
        builder.mark_output(out);
        builder.build()
    }

    #[test]
    fn test_abmk_abnk_abmn() {
        let mut rng = StdRng::seed_from_u64(0);
        let model = abmk_abnk_abmn_model(&mut rng, 2, 2, 4, 4, 8);
        let input = Tensor::<i32>::random_small(&mut rng, &[2, 2, 4, 4]);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_acbmk_kcn_cbmn() {
        let mut rng = StdRng::seed_from_u64(1);
        let model = acbmk_kcn_cbmn_model(&mut rng, 2, 2, 2, 4, 4, 8);
        let input = Tensor::<i32>::random_small(&mut rng, &[2, 2, 2, 4, 4]);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_cbmk_cbkn_amn() {
        let mut rng = StdRng::seed_from_u64(2);
        let model = cbmk_cbkn_amn_model(&mut rng, 2, 2, 4, 4, 8);
        let input = Tensor::<i32>::random_small(&mut rng, &[2, 2, 4, 4]);
        unit_test_op(model, &[input]);
    }
}
