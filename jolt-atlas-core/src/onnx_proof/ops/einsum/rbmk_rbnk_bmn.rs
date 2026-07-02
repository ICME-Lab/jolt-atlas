//! `rbmk,rbnk->bmn`-family layouts for the einsum [`dot`](super::dot) engine.
//!
//! Covers three Qwen patterns that share a retained/reduced batch-pack structure:
//! `abmk,abnk->abmn`, `acbmk,kcn->cbmn`, and `cbmk,cbkn->amn`. The first two carry
//! a batch eq-poly (bound first, [`EqSchedule::High`]); the last is a plain dot
//! product ([`EqSchedule::None`]).

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumLayout, EqSchedule, Folded},
    utils::dims::EinsumDims,
};
use atlas_onnx_tracer::{node::ComputationNode, tensor::Tensor};
use common::parallel::par_enabled;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
    utils::math::Math,
};
use rayon::prelude::*;

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

/// `rbmk,rbnk->bmn`-family layout for the [`dot`](super::dot) engine.
pub struct RbmkRbnkBmnLayout {
    variant: RbmkRbnkBmnVariant,
}

impl RbmkRbnkBmnLayout {
    /// Build the layout from the node and canonical einsum dims.
    pub fn new(computation_node: &ComputationNode, einsum_dims: &EinsumDims) -> Self {
        Self {
            variant: RbmkRbnkBmnVariant::from_node(computation_node, einsum_dims),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_abmk_abnk_abmn<F: JoltField>(
    left_operand: &[i32],
    right_operand: &[i32],
    r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
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

    let (r_a, r_bmn) = r_node_output.split_at(log_a);
    let (r_b, r_mn) = r_bmn.split_at(log_b);
    let (r_m, r_n) = r_mn.split_at(log_m);
    let eq_r_ab = EqPolynomial::evals(&[r_a.r.as_slice(), r_b.r.as_slice()].concat());
    let eq_r_m = EqPolynomial::evals(&r_m.r);
    let eq_r_n = EqPolynomial::evals(&r_n.r);

    let batch = a * b;
    let left: Vec<F> = (0..batch * k)
        .into_par_iter()
        .with_min_len(par_enabled())
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
        .with_min_len(par_enabled())
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
fn build_acbmk_kcn_cbmn<F: JoltField>(
    left_operand: &[i32],
    right_operand: &[i32],
    r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
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

    let (r_c, r_bmn) = r_node_output.split_at(log_c);
    let (r_b, r_mn) = r_bmn.split_at(log_b);
    let (r_m, r_n) = r_mn.split_at(log_m);
    let eq_r_cb = EqPolynomial::evals(&[r_c.r.as_slice(), r_b.r.as_slice()].concat());
    let eq_r_m = EqPolynomial::evals(&r_m.r);
    let eq_r_n = EqPolynomial::evals(&r_n.r);

    let cb = c * b;
    let left: Vec<F> = (0..cb * a * k)
        .into_par_iter()
        .with_min_len(par_enabled())
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
        .with_min_len(par_enabled())
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
        .with_min_len(par_enabled())
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

fn build_cbmk_cbkn_amn<F: JoltField>(
    left_operand: &[i32],
    right_operand: &[i32],
    r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    log_cb: usize,
    log_m: usize,
    log_n: usize,
    log_k: usize,
) -> (Vec<F>, Vec<F>) {
    let cb = 1usize << log_cb;
    let m = 1usize << log_m;
    let n = 1usize << log_n;
    let k = 1usize << log_k;

    let split_at_mn = r_node_output.r.len().saturating_sub(log_m + log_n);
    let (_, r_mn) = r_node_output.split_at(split_at_mn);
    let (r_m, r_n) = r_mn.split_at(log_m);
    let eq_r_m = EqPolynomial::evals(&r_m.r);
    let eq_r_n = EqPolynomial::evals(&r_n.r);

    let left: Vec<F> = (0..cb * k)
        .into_par_iter()
        .with_min_len(par_enabled())
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
        .with_min_len(par_enabled())
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

impl<F: JoltField> EinsumLayout<F> for RbmkRbnkBmnLayout {
    fn num_rounds(&self) -> usize {
        self.variant.num_rounds()
    }

    fn schedule(&self) -> EqSchedule {
        match self.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_k,
                ..
            } => EqSchedule::High {
                log_eq: log_a + log_b,
                low_bits: log_k,
            },
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_c,
                log_b,
                log_a,
                log_k,
                ..
            } => EqSchedule::High {
                log_eq: log_c + log_b,
                low_bits: log_a + log_k,
            },
            RbmkRbnkBmnVariant::CbmkCbknAmn { .. } => EqSchedule::None,
        }
    }

    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        match self.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_m,
                log_n,
                log_k,
            } => {
                let (left, right, eq) = build_abmk_abnk_abmn(
                    left_operand,
                    right_operand,
                    r_node_output,
                    log_a,
                    log_b,
                    log_m,
                    log_n,
                    log_k,
                );
                Folded {
                    left: MultilinearPolynomial::from(left),
                    right: MultilinearPolynomial::from(right),
                    eq: Some(MultilinearPolynomial::from(eq)),
                }
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_a,
                log_c,
                log_b,
                log_m,
                log_n,
                log_k,
            } => {
                let (left, right, eq) = build_acbmk_kcn_cbmn(
                    left_operand,
                    right_operand,
                    r_node_output,
                    log_a,
                    log_c,
                    log_b,
                    log_m,
                    log_n,
                    log_k,
                );
                Folded {
                    left: MultilinearPolynomial::from(left),
                    right: MultilinearPolynomial::from(right),
                    eq: Some(MultilinearPolynomial::from(eq)),
                }
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn {
                log_cb,
                log_m,
                log_n,
                log_k,
            } => {
                let (left, right) = build_cbmk_cbkn_amn(
                    left_operand,
                    right_operand,
                    r_node_output,
                    log_cb,
                    log_m,
                    log_n,
                    log_k,
                );
                Folded {
                    left: MultilinearPolynomial::from(left),
                    right: MultilinearPolynomial::from(right),
                    eq: None,
                }
            }
        }
    }

    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        match self.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn {
                log_a,
                log_b,
                log_m,
                ..
            } => {
                let (_, r_bmn) = r_node_output.split_at(log_a);
                let (_, r_mn) = r_bmn.split_at(log_b);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_ab, r_k) = sumcheck_challenges.split_at(log_a + log_b);
                let (r_a, r_b) = r_ab.split_at(log_a);
                let left = [r_a, r_b, r_m.r.as_slice(), r_k].concat();
                let right = [r_a, r_b, r_n.r.as_slice(), r_k].concat();
                (left, right)
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn {
                log_c,
                log_b,
                log_m,
                log_a,
                ..
            } => {
                let (_, r_bmn) = r_node_output.split_at(log_c);
                let (_, r_mn) = r_bmn.split_at(log_b);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_cb, r_ak) = sumcheck_challenges.split_at(log_c + log_b);
                let (r_c, r_b) = r_cb.split_at(log_c);
                let (r_a, r_k) = r_ak.split_at(log_a);
                let left = [r_a, r_c, r_b, r_m.r.as_slice(), r_k].concat();
                let right = [r_k, r_c, r_n.r.as_slice()].concat();
                (left, right)
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn {
                log_cb,
                log_m,
                log_n,
                ..
            } => {
                let split_at_mn = r_node_output.r.len().saturating_sub(log_m + log_n);
                let (_, r_mn) = r_node_output.split_at(split_at_mn);
                let (r_m, r_n) = r_mn.split_at(log_m);
                let (r_cb, r_k) = sumcheck_challenges.split_at(log_cb);
                let left = [r_cb, r_m.r.as_slice(), r_k].concat();
                let right = [r_cb, r_k, r_n.r.as_slice()].concat();
                (left, right)
            }
        }
    }

    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        match self.variant {
            RbmkRbnkBmnVariant::AbmkAbnkAbmn { log_a, log_b, .. } => {
                let (r_a, r_bmn) = r_node_output.split_at(log_a);
                let (r_b, _) = r_bmn.split_at(log_b);
                let sumcheck_opening = sumcheck_challenges.into_opening();
                let (r_ab, _) = sumcheck_opening.split_at(log_a + log_b);
                EqPolynomial::mle(&[r_a.r.as_slice(), r_b.r.as_slice()].concat(), r_ab)
            }
            RbmkRbnkBmnVariant::AcbmkKcnCbmn { log_c, log_b, .. } => {
                let (r_c, r_bmn) = r_node_output.split_at(log_c);
                let (r_b, _) = r_bmn.split_at(log_b);
                let sumcheck_opening = sumcheck_challenges.into_opening();
                let (r_cb, _) = sumcheck_opening.split_at(log_c + log_b);
                EqPolynomial::mle(&[r_c.r.as_slice(), r_b.r.as_slice()].concat(), r_cb)
            }
            RbmkRbnkBmnVariant::CbmkCbknAmn { .. } => F::one(),
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
