//! The einsum contraction sumcheck: one protocol for every two-operand einsum,
//! driven by the role of each letter.
//!
//! ## Roles
//!
//! Every letter has exactly one role:
//! - **batch**: in both operands and the output;
//! - **contracted**: in both operands, not in the output;
//! - **free**: in one operand and the output.
//!
//! ## Points
//!
//! Both points are `NamedPoint`s: the axis of letter `c` owns `log2(size of c)`
//! coordinates.
//! - `r_output`, the output opening point, has one axis per output letter (batch
//!   and free letters).
//! - `r_sumcheck`, the sumcheck challenges, has the axes `[batch…, contracted…]`,
//!   each role in left-operand letter order.
//!
//! ## Claim
//!
//! With `acc` the raw `i64` accumulation of the einsum, and `L'`, `R'` the
//! operands folded over their free axes,
//!
//! ```text
//! acc(r_output) = Σ_{s ∈ [batch, contracted]} eq(r_output.batch, s.batch) · L'(s) · R'(s)
//! ```
//!
//! The `eq` factor makes the sumcheck degree 3 when the equation has batch
//! letters; otherwise it is a degree-2 dot product. The batch axes come first,
//! so the `eq` table only has `2^log_batch` entries and is a scalar once they
//! are bound. The initial claim is recovered from the fused rescale's advice by
//! [`fused_rebase::fused_input_claim`].
//!
//! ## Fold
//!
//! For an operand `X`,
//!
//! ```text
//! X'(s) = Σ_f eq(r_output[free axes of X], f) · X[index(s, f)]
//! ```
//!
//! Tensors are row-major, so a flat index is `Σ_axis coordinate · stride`. The
//! sumcheck axes and the free axes of `X` are disjoint, hence
//! `index(s, f) = offset_sumcheck[s] + offset_free[f]`.
//!
//! ## Opening points
//!
//! After the last round the verifier checks
//! `eq(r_output.batch, r_sumcheck.batch) · L(p_L) · R(p_R)`. Axis by axis in the
//! operand's order, `p_X` takes the coordinates of `r_output` for a free letter
//! and those of `r_sumcheck` for a batch or contracted letter.

use crate::onnx_proof::trace_view::TraceView;
use std::{array, collections::BTreeMap, sync::Arc};

use atlas_trace::{op::EinsumEq, trace::LayerData, Tensor, Trace};

use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        fused_rebase,
        ops::einsum::named_point::{axis_of, Letter, NamedPoint},
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use common::parallel::par_enabled;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
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

#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};

/// The eq-poly binding schedule within the (HighToLow) sumcheck round order.
///
/// The contraction cube is bound high-to-low. The batch `EQ` cube, when present,
/// occupies a contiguous range of rounds; the two layouts in use differ in
/// whether that range is at the top or the bottom of the round order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqSchedule {
    /// No `EQ` factor: a plain degree-2 dot product over the contraction cube.
    None,
    /// `EQ` cube occupies the **first** `log_eq` rounds (high bits) and is bound
    /// first; afterwards the cached scalar is reused.
    High {
        /// Number of rounds the `EQ` cube spans (its bit-width).
        log_eq: usize,
        /// Number of contraction bits below the `EQ` cube, used to index `EQ`.
        low_bits: usize,
    },
    /// `EQ` cube occupies the **last** `log_b` rounds (low bits) and is bound
    /// last; during the first `log_k` rounds `EQ` is read by coefficient.
    Low {
        /// Number of contraction rounds above the `EQ` cube (bound before `EQ`).
        log_k: usize,
        /// Number of rounds the `EQ` cube spans (its bit-width).
        log_b: usize,
    },
}

impl EqSchedule {
    /// Whether this schedule carries an `EQ` factor (degree 3 vs degree 2).
    pub fn has_eq(&self) -> bool {
        !matches!(self, EqSchedule::None)
    }

    /// The sumcheck degree implied by this schedule.
    pub fn degree(&self) -> usize {
        if self.has_eq() {
            3
        } else {
            2
        }
    }
}

/// Operands folded at the output randomness into MLEs over the sumcheck cube.
pub struct Folded<F: JoltField> {
    /// Left operand folded over the contraction(+batch) cube.
    pub left: MultilinearPolynomial<F>,
    /// Right operand folded over the contraction(+batch) cube.
    pub right: MultilinearPolynomial<F>,
    /// Batch eq-poly, present iff the schedule is not [`EqSchedule::None`].
    pub eq: Option<MultilinearPolynomial<F>>,
}

/// Per-pattern description of an einsum contraction sumcheck.
pub trait EinsumLayout<F: JoltField>: Send + Sync {
    /// Number of sumcheck rounds (`log` of the contraction(+batch) cube size).
    fn num_rounds(&self) -> usize;

    /// The eq-poly binding schedule for this pattern.
    fn schedule(&self) -> EqSchedule;

    /// Fold the two operand tensors at `r_node_output` into the sumcheck MLEs.
    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F>;

    /// Map the final sumcheck point back to the `(left, right)` operand opening
    /// points. `sumcheck_challenges` is the post-protocol challenge vector.
    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>);

    /// The `EQ` factor multiplying `left·right` in the output claim
    /// (`F::one()` when the schedule carries no `EQ`).
    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F;
}

/// The letter roles of a two-operand einsum, and the sumcheck they define. See
/// the [module docs](self).
pub struct Contraction {
    left: Vec<Letter>,
    right: Vec<Letter>,
    output: Vec<Letter>,
    batch: Vec<Letter>,
    contracted: Vec<Letter>,
    /// Physical size of each letter.
    sizes: BTreeMap<Letter, usize>,
}

impl Contraction {
    /// Derive the roles of `equation` over operands of physical dims `left_dims` and `right_dims`.
    pub fn new(equation: &EinsumEq, left_dims: &[usize], right_dims: &[usize]) -> Self {
        let letters = |term: &[u8]| term.iter().map(|&c| char::from(c)).collect::<Vec<Letter>>();
        let [left, right] = equation.operands() else {
            panic!("einsum {equation} does not have two operands");
        };
        let (left, right) = (letters(left), letters(right));
        let output = letters(equation.output());
        let sizes = left
            .iter()
            .copied()
            .zip(left_dims.iter().copied())
            .chain(right.iter().copied().zip(right_dims.iter().copied()))
            .collect();
        let shared = || left.iter().copied().filter(|c| right.contains(c));
        let batch = shared().filter(|c| output.contains(c)).collect();
        let contracted = shared().filter(|c| !output.contains(c)).collect();
        Self {
            left,
            right,
            output,
            batch,
            contracted,
            sizes,
        }
    }

    /// The axes of `r_sumcheck`.
    fn sumcheck_letters(&self) -> Vec<Letter> {
        [self.batch.as_slice(), self.contracted.as_slice()].concat()
    }

    /// The free letters of an operand, in its axis order.
    fn free_letters(&self, operand: &[Letter]) -> Vec<Letter> {
        let sumcheck_letters = self.sumcheck_letters();
        operand
            .iter()
            .copied()
            .filter(|c| !sumcheck_letters.contains(c))
            .collect()
    }

    fn dims(&self, letters: &[Letter]) -> Vec<usize> {
        letters.iter().map(|c| self.sizes[c]).collect()
    }

    fn num_vars(&self, letters: &[Letter]) -> usize {
        letters.iter().map(|c| self.sizes[c].log_2()).sum()
    }

    fn r_output<F>(&self, r_node_output: OpeningPoint<BIG_ENDIAN, F>) -> NamedPoint<F>
    where
        F: JoltField,
    {
        NamedPoint::new(
            r_node_output.r,
            self.output.clone(),
            &self.dims(&self.output),
        )
    }

    fn r_sumcheck<F: JoltField>(&self, sumcheck_challenges: &[F]) -> NamedPoint<F> {
        let letters = self.sumcheck_letters();
        let dims = self.dims(&letters);
        NamedPoint::new(sumcheck_challenges.to_vec(), letters, &dims)
    }

    /// For every coordinate of the axes `letters`, enumerated in row-major
    /// order, the flat offset of that coordinate in `operand`.
    fn offsets(&self, operand: &[Letter], letters: &[Letter]) -> Vec<usize> {
        letters.iter().fold(vec![0], |table, c| {
            let stride: usize = self
                .dims(&operand[axis_of(operand, *c) + 1..])
                .iter()
                .product();
            table
                .iter()
                .flat_map(|base| (0..self.sizes[c]).map(move |coord| base + coord * stride))
                .collect()
        })
    }

    /// `X'(s) = Σ_f eq(r_output[free axes of X], f) · X[offset_sumcheck[s] + offset_free[f]]`.
    fn fold_operand<F: JoltField>(
        &self,
        tensor: &Tensor<i32>,
        operand: &[Letter],
        r_output: &NamedPoint<F>,
    ) -> MultilinearPolynomial<F> {
        let free = self.free_letters(operand);
        let offset_sumcheck = self.offsets(operand, &self.sumcheck_letters());
        let offset_free = self.offsets(operand, &free);
        let eq_free = EqPolynomial::<F>::evals(&r_output.coords(&free));
        let folded: Vec<F> = offset_sumcheck
            .par_iter()
            .with_min_len(par_enabled())
            .map(|base| {
                offset_free
                    .iter()
                    .zip(&eq_free)
                    .map(|(offset, eq)| eq.mul_i64(tensor[base + offset] as i64))
                    .sum()
            })
            .collect();
        MultilinearPolynomial::from(folded)
    }

    /// The opening point `p_X` of an operand.
    fn operand_point<F: JoltField>(
        &self,
        operand: &[Letter],
        r_output: &NamedPoint<F>,
        r_sumcheck: &NamedPoint<F>,
    ) -> Vec<F> {
        let sumcheck_letters = self.sumcheck_letters();
        operand
            .iter()
            .flat_map(|c| {
                if sumcheck_letters.contains(c) {
                    r_sumcheck.axis(*c)
                } else {
                    r_output.axis(*c)
                }
            })
            .copied()
            .collect()
    }

    /// `eq(r_output.batch, r_sumcheck.batch)`.
    fn batch_eq<F: JoltField>(&self, r_output: &NamedPoint<F>, r_sumcheck: &NamedPoint<F>) -> F {
        EqPolynomial::mle(
            &r_output.coords(&self.batch),
            &r_sumcheck.coords(&self.batch),
        )
    }
}

impl<F: JoltField> EinsumLayout<F> for Contraction {
    fn num_rounds(&self) -> usize {
        self.num_vars(&self.sumcheck_letters())
    }

    fn schedule(&self) -> EqSchedule {
        if self.batch.is_empty() {
            EqSchedule::None
        } else {
            EqSchedule::High {
                log_eq: self.num_vars(&self.batch),
                low_bits: self.num_vars(&self.contracted),
            }
        }
    }

    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        let r_output = self.r_output(r_node_output.clone());
        Folded {
            left: self.fold_operand(left_operand, &self.left, &r_output),
            right: self.fold_operand(right_operand, &self.right, &r_output),
            eq: (!self.batch.is_empty()).then(|| {
                MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_output.coords(&self.batch)))
            }),
        }
    }

    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        let r_output = self.r_output(r_node_output.clone());
        let r_sumcheck = self.r_sumcheck(sumcheck_challenges);
        (
            self.operand_point(&self.left, &r_output, &r_sumcheck),
            self.operand_point(&self.right, &r_output, &r_sumcheck),
        )
    }

    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_output = self.r_output(r_node_output.clone());
        let r_sumcheck = self.r_sumcheck(&sumcheck_challenges.into_opening());
        self.batch_eq(&r_output, &r_sumcheck)
    }
}

/// Parameters of the contraction sumcheck.
#[derive(Clone)]
pub struct ContractionParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    layout: Arc<dyn EinsumLayout<F>>,
}

impl<F: JoltField> ContractionParams<F> {
    /// Create params for the given pattern `layout`.
    pub fn new(
        computation_node: ComputationNode,
        layout: Arc<dyn EinsumLayout<F>>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            r_node_output,
            computation_node,
            layout,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ContractionParams<F> {
    fn degree(&self) -> usize {
        self.layout.schedule().degree()
    }

    /// The sumcheck's initial claim: the raw accumulation `acc(r_output)`.
    ///
    /// The einsum fuses an i64 accumulate + rescaling division + saturating
    /// clamp; [`fused_rebase::fused_input_claim`] recovers `acc(r_output)` from
    /// the pre-clamp `rescaled` and remainder advice.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        fused_rebase::fused_input_claim(accumulator, &self.computation_node)
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.layout.num_rounds()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        use crate::utils::opening_access::OpeningIdBuilder;
        let builder = OpeningIdBuilder::new(&self.computation_node);
        let left = ValueSource::Opening(builder.nodeio(Target::Input(0)));
        let right = ValueSource::Opening(builder.nodeio(Target::Input(1)));
        // With batch letters the claim is scaled by the batch eq value; a plain
        // dot product is just the operand product.
        let term = if self.layout.schedule().has_eq() {
            ProductTerm::scaled(ValueSource::Challenge(0), vec![left, right])
        } else {
            ProductTerm::product(vec![left, right])
        };
        Some(OutputClaimConstraint::sum_of_products(vec![term]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        if self.layout.schedule().has_eq() {
            vec![self
                .layout
                .output_eq(&self.r_node_output, sumcheck_challenges)]
        } else {
            vec![]
        }
    }
}

/// Prover of the contraction sumcheck.
pub struct ContractionProver<F: JoltField> {
    params: ContractionParams<F>,
    left: MultilinearPolynomial<F>,
    right: MultilinearPolynomial<F>,
    /// `eq(r_output.batch, ·)`, present when the equation has batch letters.
    eq: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> ContractionProver<F> {
    /// Initialize the prover by folding the operands at `r_output`.
    #[tracing::instrument(skip_all, name = "ContractionProver::initialize")]
    pub fn initialize(trace: &Trace, params: ContractionParams<F>) -> Self {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for einsum operation")
        };
        let Folded { left, right, eq } =
            params
                .layout
                .fold(left_operand, right_operand, &params.r_node_output);

        Self {
            params,
            left,
            right,
            eq,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ContractionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.left.len() / 2;
        match self.params.layout.schedule() {
            EqSchedule::None => {
                let evals: [F; 2] = (0..half)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|i| {
                        let l = self.left.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                        let r = self.right.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                        [l[0] * r[0], l[1] * r[1]]
                    })
                    .reduce(|| [F::zero(); 2], |a, b| array::from_fn(|i| a[i] + b[i]));
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
            EqSchedule::High { log_eq, low_bits } => {
                let eq = self.eq.as_ref().expect("eq poly must exist");
                // `eq` is a single value once the batch axes are bound.
                let eq_bound = (round >= log_eq).then(|| eq.final_claim());
                let evals: [F; 3] = (0..half)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|i| {
                        let l = self
                            .left
                            .sumcheck_evals_array::<3>(i, BindingOrder::HighToLow);
                        let r = self
                            .right
                            .sumcheck_evals_array::<3>(i, BindingOrder::HighToLow);
                        let e = match eq_bound {
                            Some(value) => [value; 3],
                            None => {
                                eq.sumcheck_evals_array::<3>(i >> low_bits, BindingOrder::HighToLow)
                            }
                        };
                        [l[0] * r[0] * e[0], l[1] * r[1] * e[1], l[2] * r[2] * e[2]]
                    })
                    .reduce(|| [F::zero(); 3], |a, b| array::from_fn(|i| a[i] + b[i]));
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
            EqSchedule::Low { log_k, log_b } => {
                let eq = self.eq.as_ref().expect("eq poly must exist");
                let evals: [F; 3] = (0..half)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|jh| {
                        let l = self
                            .left
                            .sumcheck_evals_array::<3>(jh, BindingOrder::HighToLow);
                        let r = self
                            .right
                            .sumcheck_evals_array::<3>(jh, BindingOrder::HighToLow);
                        let e = if round < log_k {
                            [eq.get_bound_coeff(jh & ((1 << log_b) - 1)); 3]
                        } else {
                            eq.sumcheck_evals_array::<3>(jh, BindingOrder::HighToLow)
                        };
                        [l[0] * r[0] * e[0], l[1] * r[1] * e[1], l[2] * r[2] * e[2]]
                    })
                    .reduce(|| [F::zero(); 3], |a, b| array::from_fn(|i| a[i] + b[i]));
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.left.bind_parallel(r_j, BindingOrder::HighToLow);
        self.right.bind_parallel(r_j, BindingOrder::HighToLow);
        match self.params.layout.schedule() {
            EqSchedule::None => {}
            EqSchedule::High { log_eq, .. } => {
                if round < log_eq {
                    let eq = self.eq.as_mut().expect("eq poly must exist");
                    eq.bind_parallel(r_j, BindingOrder::HighToLow);
                }
            }
            EqSchedule::Low { log_k, .. } => {
                if round >= log_k {
                    self.eq
                        .as_mut()
                        .expect("eq poly must exist")
                        .bind_parallel(r_j, BindingOrder::HighToLow);
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
        let (left_point, right_point) = self
            .params
            .layout
            .operand_points(&self.params.r_node_output, &sumcheck_challenges);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default());
        provider.append_nodeio_at(
            Target::Input(0),
            self.params.normalize_opening_point(&left_point),
            self.left.final_claim(),
        );
        provider.append_nodeio_at(
            Target::Input(1),
            self.params.normalize_opening_point(&right_point),
            self.right.final_claim(),
        );
    }
}

/// Verifier of the contraction sumcheck.
pub struct ContractionVerifier<F: JoltField> {
    params: ContractionParams<F>,
}

impl<F: JoltField> ContractionVerifier<F> {
    /// Create a verifier for the given pattern `layout`.
    pub fn new(
        computation_node: ComputationNode,
        layout: Arc<dyn EinsumLayout<F>>,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ContractionParams::new(computation_node, layout, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ContractionVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let left = accessor.get_nodeio(Target::Input(0)).1;
        let right = accessor.get_nodeio(Target::Input(1)).1;
        let eq = self
            .params
            .layout
            .output_eq(&self.params.r_node_output, sumcheck_challenges);
        left * right * eq
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let (left_point, right_point) = self
            .params
            .layout
            .operand_points(&self.params.r_node_output, &sumcheck_challenges);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default());
        provider.append_nodeio_at(
            Target::Input(0),
            self.params.normalize_opening_point(&left_point),
        );
        provider.append_nodeio_at(
            Target::Input(1),
            self.params.normalize_opening_point(&right_point),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use joltworks::poly::multilinear_polynomial::EvaluateMle;
    use rand::{rngs::StdRng, SeedableRng};

    /// Each folded operand, evaluated at `r_sumcheck`, must equal the operand's
    /// own MLE at the opening point the letter roles assign to it.
    #[test]
    fn folded_operands_match_their_mle_at_the_opening_point() {
        let cases: [(&str, &[usize], &[usize]); 10] = [
            ("mk,kn->mn", &[4, 8], &[8, 2]),
            ("mk,nk->mn", &[4, 8], &[2, 8]),
            ("km,kn->mn", &[8, 4], &[8, 2]),
            ("k,nk->n", &[8], &[4, 8]),
            ("amk,ank->amn", &[1, 4, 1], &[1, 2, 1]),
            ("abmk,abnk->abmn", &[2, 2, 4, 4], &[2, 2, 8, 4]),
            ("bmk,kbn->nbm", &[2, 4, 8], &[8, 2, 4]),
            ("ab,ab->a", &[4, 8], &[4, 8]),
            ("ab,b->a", &[4, 8], &[8]),
            ("k,k->", &[8], &[8]),
        ];
        let mut rng = StdRng::seed_from_u64(0xE1_0001);
        for (equation, left_dims, right_dims) in cases {
            let contraction = Contraction::new(&equation.parse().unwrap(), left_dims, right_dims);
            let left = Tensor::<i32>::random_small(&mut rng, left_dims);
            let right = Tensor::<i32>::random_small(&mut rng, right_dims);
            let mut random_point =
                |len: usize| -> Vec<Fr> { (0..len).map(|_| Fr::rand(&mut rng)).collect() };
            let r_node_output =
                OpeningPoint::new(random_point(contraction.num_vars(&contraction.output)));
            let r_sumcheck = random_point(EinsumLayout::<Fr>::num_rounds(&contraction));

            let folded = contraction.fold(&left, &right, &r_node_output);
            let (left_point, right_point) = contraction.operand_points(&r_node_output, &r_sumcheck);

            let left_claim: Fr = left.evaluate_mle(&left_point);
            let right_claim: Fr = right.evaluate_mle(&right_point);
            assert_eq!(
                folded.left.evaluate(&r_sumcheck),
                left_claim,
                "{equation} left"
            );
            assert_eq!(
                folded.right.evaluate(&r_sumcheck),
                right_claim,
                "{equation} right"
            );
        }
    }
}
