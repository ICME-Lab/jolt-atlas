use common::parallel::par_enabled;
use std::array;

use crate::utils::opening_access::{AccOpeningAccessor, Target};
use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPoly;
#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::drop_in_background_thread},
};
use rayon::prelude::*;

const DEGREE_BOUND: usize = 3;

/// Evaluate a sparse indicator with one selected column per row. Reuse the
/// column equality table when it costs no more than visiting each index bit;
/// otherwise select the bit factors directly without allocating bit vectors.
fn evaluate_max_indicator<F: JoltField>(r_rows: &[F], r_columns: &[F], indices: &[usize]) -> F {
    let rows = 1usize
        .checked_shl(r_rows.len() as u32)
        .expect("row domain too large");
    let columns = 1usize
        .checked_shl(r_columns.len() as u32)
        .expect("column domain too large");
    assert_eq!(indices.len(), rows);
    assert!(indices.iter().all(|index| *index < columns));
    let row_weights = EqPolynomial::evals(r_rows);
    if columns <= rows.saturating_mul(r_columns.len().max(1)) {
        let column_weights = EqPolynomial::evals(r_columns);
        row_weights
            .iter()
            .zip(indices)
            .map(|(row, index)| *row * column_weights[*index])
            .sum()
    } else {
        let complements: Vec<F> = r_columns.iter().map(|value| F::one() - value).collect();
        row_weights
            .iter()
            .zip(indices)
            .map(|(row, index)| {
                let column: F = r_columns
                    .iter()
                    .enumerate()
                    .map(|(bit, value)| {
                        if (index >> (r_columns.len() - bit - 1)) & 1 == 1 {
                            *value
                        } else {
                            complements[bit]
                        }
                    })
                    .product();
                *row * column
            })
            .sum()
    }
}

/// Shared parameters for the max indicator sumcheck instance.
#[derive(Clone)]
pub struct MaxIndicatorParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r1_k: Vec<F>,
    /// max_k(r1_k)
    pub input_claim: F,
    /// Softmax dims
    pub F_N: [usize; 2],
    /// Max indicator
    pub e: Vec<u32>,
    /// Argmax index per feature (length F)
    pub argmax_k: Vec<usize>,
}

impl<F: JoltField> MaxIndicatorParams<F> {
    /// Create a new instance of max indicator sumcheck parameters.
    pub fn new(
        node: ComputationNode,
        F_N: [usize; 2],
        argmax_k: Vec<usize>,
        input_claim: F,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let [F, N] = F_N;
        let log_f = F.log_2();

        // Get r1 (the point from Stage 1 reciprocal multiplication output)
        let accessor = AccOpeningAccessor::new(accumulator, &node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_k = r1.split_at(log_f).0.r;

        let mut e: Vec<u32> = vec![0; F * N];
        // Only set the argmax positions — everything else is already 0
        for k in 0..F {
            e[k * N + argmax_k[k]] = 1;
        }
        Self {
            r1_k: r1_k.to_vec(),
            input_claim,
            F_N,
            e,
            argmax_k,
            node,
        }
    }

    /// Returns log2 of the leading-dimension product `F`.
    pub fn log_F(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.F_N[1].log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MaxIndicatorParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.F_N.iter().product::<usize>().log_2()
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

    // output = eq(r1_k, r_k) * e_claim * X_claim = Challenge(0) * Opening(X)
    // e_claim is verifier-computable from argmax_k, folded into Challenge(0)
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let builder = crate::utils::opening_access::OpeningIdBuilder::new(&self.node);
        let x_id = builder.nodeio(Target::Input(0));
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(ValueSource::Challenge(0), vec![ValueSource::Opening(x_id)]),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_sc: Vec<F> = self
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let (r_k, r_j) = r_sc.split_at(self.log_F());
        let e_claim = evaluate_max_indicator(r_k, r_j, &self.argmax_k);
        let eq_eval = EqPolynomial::mle(&self.r1_k, r_k);
        vec![eq_eval * e_claim]
    }
}

/// Prover for the max indicator sumcheck instance.
pub struct MaxIndicatorProver<F: JoltField> {
    params: MaxIndicatorParams<F>,
    eq: Vec<F>,
    gs_eq: Option<GruenSplitEqPolynomial<F>>,
    // log(F * N) variables
    X: MultilinearPolynomial<F>,
    // log(F * N) variables
    e: MultilinearPolynomial<F>,
}

impl<F: JoltField> MaxIndicatorProver<F> {
    /// Constructor for softmax exp multiplication prover.
    pub fn initialize(X: Vec<i32>, mut params: MaxIndicatorParams<F>) -> Self {
        let eq = EqPolynomial::evals(&params.r1_k);
        let X = MultilinearPolynomial::from(X);
        // Take e out of params — the prover never reads params.e again
        let e = MultilinearPolynomial::from(std::mem::take(&mut params.e));
        Self {
            params,
            eq,
            gs_eq: None,
            X,
            e,
        }
    }

    fn compute_phase_1_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let Self { X, e, eq, .. } = self;
        let half_poly_len = X.len() / 2;
        let evals: [F; DEGREE_BOUND] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|kj| {
                let k = kj >> (self.params.log_N() - m);
                let eq_val = eq[k];
                let e_vals = e.sumcheck_evals(kj, DEGREE_BOUND, BindingOrder::LowToHigh);
                let X_vals = X.sumcheck_evals(kj, DEGREE_BOUND, BindingOrder::LowToHigh);
                [
                    eq_val * X_vals[0] * e_vals[0],
                    eq_val * X_vals[1] * e_vals[1],
                    eq_val * X_vals[2] * e_vals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn compute_phase_2_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            gs_eq,
            X,
            e: indicator,
            ..
        } = self;
        let gs_eq = gs_eq.as_ref().unwrap();
        let [q_constant, q_quadratic] = gs_eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let i_0 = indicator.get_bound_coeff(2 * g);
            let i_inf = indicator.get_bound_coeff(2 * g + 1) - i_0;

            let X_0 = X.get_bound_coeff(2 * g);
            let X_inf = X.get_bound_coeff(2 * g + 1) - X_0;

            let c0 = i_0 * X_0;
            let e = i_inf * X_inf;
            [c0, e]
        });
        gs_eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MaxIndicatorProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(name = "MaxIndicatorProver::compute_message", skip_all)]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_N() {
            self.compute_phase_1_message(round, previous_claim)
        } else {
            self.compute_phase_2_message(previous_claim)
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.X.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.e.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == self.params.log_N() - 1 {
            self.gs_eq = Some(GruenSplitEqPolynomial::new(
                &self.params.r1_k,
                BindingOrder::LowToHigh,
            ));
            drop_in_background_thread(std::mem::take(&mut self.eq));
        }
        if round >= self.params.log_N() {
            self.gs_eq.as_mut().unwrap().bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_nodeio(Target::Input(0), self.X.final_claim());
    }
}

/// Verifier for the max indicator sumcheck instance.
pub struct MaxIndicatorVerifier<F: JoltField> {
    params: MaxIndicatorParams<F>,
}

impl<F: JoltField> MaxIndicatorVerifier<F> {
    /// Create new verifier for max indicator sumcheck.
    pub fn new(
        node: ComputationNode,
        F_N: [usize; 2],
        argmax_k: Vec<usize>,
        input_claim: F,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let params = MaxIndicatorParams::new(node, F_N, argmax_k, input_claim, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MaxIndicatorVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_nodeio(Target::Input(0));
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::expected_output_claim", skip_all)]
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_sc = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.node);
        let X_claim = accessor.get_nodeio(Target::Input(0)).1;

        let (r_k, r_j) = r_sc.split_at(self.params.log_F());
        let e_claim = evaluate_max_indicator(r_k, r_j, &self.params.argmax_k);
        EqPolynomial::mle(&self.params.r1_k, r_k) * e_claim * X_claim
    }
}

#[cfg(test)]
mod tests {
    use super::evaluate_max_indicator;
    use ark_bn254::Fr;
    use joltworks::{
        field::JoltField,
        poly::{
            eq_poly::EqPolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        utils::{index_to_field_bitvector, math::Math},
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn indicator_evaluation_matches_dense_and_sparse_definitions() {
        let mut rng = StdRng::seed_from_u64(0x494e4449434154);
        for (rows, columns) in [
            (1usize, 1usize),
            (1, 32),
            (2, 64),
            (32, 2),
            (16, 16),
            (64, 128),
        ] {
            for repeated in [false, true] {
                let indices: Vec<usize> = (0..rows)
                    .map(|_| {
                        if repeated {
                            columns - 1
                        } else {
                            rng.gen_range(0..columns)
                        }
                    })
                    .collect();
                let row_point: Vec<Fr> = (0..rows.log_2()).map(|_| Fr::random(&mut rng)).collect();
                let column_point: Vec<Fr> =
                    (0..columns.log_2()).map(|_| Fr::random(&mut rng)).collect();
                let mut dense = vec![Fr::from(0u64); rows * columns];
                for (row, index) in indices.iter().enumerate() {
                    dense[row * columns + index] = Fr::from(1u64);
                }
                let point: Vec<Fr> = row_point.iter().chain(&column_point).copied().collect();
                let expected = MultilinearPolynomial::from(dense).evaluate(&point);
                let sparse: Fr = EqPolynomial::<Fr>::evals(&row_point)
                    .iter()
                    .zip(&indices)
                    .map(|(weight, index)| {
                        let bits = index_to_field_bitvector::<Fr>(*index as u64, columns.log_2());
                        *weight * EqPolynomial::mle(&column_point, &bits)
                    })
                    .sum();
                assert_eq!(
                    evaluate_max_indicator(&row_point, &column_point, &indices),
                    expected
                );
                assert_eq!(sparse, expected);
                for row in [0, rows - 1] {
                    let r = index_to_field_bitvector::<Fr>(row as u64, rows.log_2());
                    for column in [0, columns - 1] {
                        let c = index_to_field_bitvector::<Fr>(column as u64, columns.log_2());
                        assert_eq!(
                            evaluate_max_indicator(&r, &c, &indices),
                            Fr::from((indices[row] == column) as u64)
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn indicator_rejects_out_of_range_index() {
        evaluate_max_indicator::<Fr>(&[], &[Fr::from(0u64)], &[2]);
    }

    #[test]
    #[should_panic]
    fn indicator_rejects_wrong_row_count() {
        evaluate_max_indicator::<Fr>(&[Fr::from(0u64)], &[], &[0]);
    }
}
