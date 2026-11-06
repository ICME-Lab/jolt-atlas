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
//! bits; otherwise it is a degree-2 dot product. The batch axes come first,
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
use std::{array, collections::BTreeMap};

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
    log_batch: usize,
    log_contracted: usize,
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
        let sizes: BTreeMap<Letter, usize> = left
            .iter()
            .copied()
            .zip(left_dims.iter().copied())
            .chain(right.iter().copied().zip(right_dims.iter().copied()))
            .collect();
        let shared = || left.iter().copied().filter(|c| right.contains(c));
        let batch: Vec<Letter> = shared().filter(|c| output.contains(c)).collect();
        let contracted: Vec<Letter> = shared().filter(|c| !output.contains(c)).collect();
        let num_vars = |term: &[Letter]| -> usize { term.iter().map(|c| sizes[c].log_2()).sum() };
        let (log_batch, log_contracted) = (num_vars(&batch), num_vars(&contracted));
        Self {
            left,
            right,
            output,
            batch,
            contracted,
            sizes,
            log_batch,
            log_contracted,
        }
    }

    fn num_rounds(&self) -> usize {
        self.log_batch + self.log_contracted
    }

    /// Whether a batch axis adds sumcheck variables: a size-1 batch letter has none.
    fn has_batch_bits(&self) -> bool {
        self.log_batch > 0
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

/// Parameters of the contraction sumcheck.
pub struct ContractionParams<F: JoltField> {
    r_output: NamedPoint<F>,
    computation_node: ComputationNode,
    contraction: Contraction,
}

impl<F: JoltField> ContractionParams<F> {
    /// Create the params of an einsum node opened at `r_node_output`.
    pub fn new(
        computation_node: ComputationNode,
        contraction: Contraction,
        r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Self {
        Self {
            r_output: contraction.r_output(r_node_output),
            computation_node,
            contraction,
        }
    }

    /// Create the params of an einsum node at its opening in `accumulator`.
    pub fn from_accumulator(
        computation_node: ComputationNode,
        contraction: Contraction,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        Self::new(computation_node, contraction, r_node_output)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ContractionParams<F> {
    fn degree(&self) -> usize {
        if self.contraction.has_batch_bits() {
            3
        } else {
            2
        }
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
        self.contraction.num_rounds()
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
        // With batch bits the claim is scaled by the batch eq value; a plain dot
        // product is just the operand product.
        let term = if self.contraction.has_batch_bits() {
            ProductTerm::scaled(ValueSource::Challenge(0), vec![left, right])
        } else {
            ProductTerm::product(vec![left, right])
        };
        Some(OutputClaimConstraint::sum_of_products(vec![term]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        if self.contraction.has_batch_bits() {
            let r_sumcheck = self
                .contraction
                .r_sumcheck(&sumcheck_challenges.into_opening());
            vec![self.contraction.batch_eq(&self.r_output, &r_sumcheck)]
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
    /// `eq(r_output.batch, ·)`, present when the equation has batch bits.
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
        let contraction = &params.contraction;
        let r_output = &params.r_output;
        let left = contraction.fold_operand(left_operand, &contraction.left, r_output);
        let right = contraction.fold_operand(right_operand, &contraction.right, r_output);
        let eq = contraction.has_batch_bits().then(|| {
            MultilinearPolynomial::from(EqPolynomial::<F>::evals(
                &r_output.coords(&contraction.batch),
            ))
        });

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
        match &self.eq {
            None => {
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
            Some(eq) => {
                let log_batch = self.params.contraction.log_batch;
                let log_contracted = self.params.contraction.log_contracted;
                // `eq` is a single value once the batch axes are bound.
                let eq_bound = (round >= log_batch).then(|| eq.final_claim());
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
                            None => eq.sumcheck_evals_array::<3>(
                                i >> log_contracted,
                                BindingOrder::HighToLow,
                            ),
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
        if round < self.params.contraction.log_batch {
            let eq = self.eq.as_mut().expect("eq poly must exist");
            eq.bind_parallel(r_j, BindingOrder::HighToLow);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let contraction = &self.params.contraction;
        let r_output = &self.params.r_output;
        let r_sumcheck = contraction.r_sumcheck(&sumcheck_challenges.into_opening());
        let left_point = contraction.operand_point(&contraction.left, r_output, &r_sumcheck);
        let right_point = contraction.operand_point(&contraction.right, r_output, &r_sumcheck);
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
    /// Create the verifier of an einsum node.
    pub fn new(
        computation_node: ComputationNode,
        contraction: Contraction,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            ContractionParams::from_accumulator(computation_node, contraction, accumulator);
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
        let contraction = &self.params.contraction;
        let r_sumcheck = contraction.r_sumcheck(&sumcheck_challenges.into_opening());
        left * right * contraction.batch_eq(&self.params.r_output, &r_sumcheck)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let contraction = &self.params.contraction;
        let r_output = &self.params.r_output;
        let r_sumcheck = contraction.r_sumcheck(&sumcheck_challenges.into_opening());
        let left_point = contraction.operand_point(&contraction.left, r_output, &r_sumcheck);
        let right_point = contraction.operand_point(&contraction.right, r_output, &r_sumcheck);
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
    use crate::{node_view::ModelView, onnx_proof::testing::ModelBuilder};
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use joltworks::poly::multilinear_polynomial::EvaluateMle;
    use rand::{rngs::StdRng, SeedableRng};

    /// A size-1 batch letter adds no sumcheck variable, so no `eq` factor.
    #[test]
    fn degree_is_3_only_with_batch_bits() {
        let equation = "amk,ank->amn";
        for (a, degree) in [(1, 2), (2, 3)] {
            let (left_dims, right_dims) = ([a, 4, 8], [a, 2, 8]);
            let mut b = ModelBuilder::new();
            let left = b.input(left_dims.to_vec());
            let right = b.input(right_dims.to_vec());
            let node = b.einsum(equation, vec![left, right], vec![a, 4, 2]);
            b.mark_output(node);
            let model = b.build();

            let contraction = Contraction::new(&equation.parse().unwrap(), &left_dims, &right_dims);
            let r_node_output =
                OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![Fr::from(1u64); a.log_2() + 3]);
            let params = ContractionParams::new(model.view(node), contraction, r_node_output);
            assert_eq!(params.degree(), degree, "a = {a}");
        }
    }

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
            let num_output_vars = contraction
                .output
                .iter()
                .map(|c| contraction.sizes[c].log_2())
                .sum();
            let r_output = contraction.r_output(OpeningPoint::new(random_point(num_output_vars)));
            let z = random_point(contraction.num_rounds());
            let r_sumcheck = contraction.r_sumcheck(&z);

            let left_folded = contraction.fold_operand(&left, &contraction.left, &r_output);
            let right_folded = contraction.fold_operand(&right, &contraction.right, &r_output);
            let left_point = contraction.operand_point(&contraction.left, &r_output, &r_sumcheck);
            let right_point = contraction.operand_point(&contraction.right, &r_output, &r_sumcheck);

            let left_claim: Fr = left.evaluate_mle(&left_point);
            let right_claim: Fr = right.evaluate_mle(&right_point);
            assert_eq!(left_folded.evaluate(&z), left_claim, "{equation} left");
            assert_eq!(right_folded.evaluate(&z), right_claim, "{equation} right");
        }
    }
}
