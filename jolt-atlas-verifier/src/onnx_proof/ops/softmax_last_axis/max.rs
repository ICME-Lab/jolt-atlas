use crate::node_view::ComputationNode;
use crate::utils::opening_access::{AccOpeningAccessor, Target};
use common::VirtualPoly;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{
            OpeningAccumulator, OpeningPoint, VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
    },
    subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    transcripts::Transcript,
    utils::{index_to_field_bitvector, math::Math},
};

/// Degree of the max-indicator sumcheck.
pub const DEGREE_BOUND: usize = 3;

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

    // output = eq(r1_k, r_k) * e_claim * X_claim = Challenge(0) * Opening(X)
    // e_claim is verifier-computable from argmax_k, folded into Challenge(0)
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

        // Evaluate e(r_sc) in O(F · log N) by exploiting sparsity:
        // e has exactly F nonzero entries at positions k·N + argmax_k[k],
        // so ẽ(r_k, r_j) = Σ_k eq(r_k, bits(k)) · eq(r_j, bits(argmax_k[k]))
        let (r_k, r_j) = r_sc.split_at(self.params.log_F());
        let eq_k_evals = EqPolynomial::evals(r_k);
        let log_n = r_j.len();
        let e_claim: F = eq_k_evals
            .iter()
            .zip(self.params.argmax_k.iter())
            .map(|(&eq_k, &argmax_j)| {
                let y = index_to_field_bitvector::<F>(argmax_j as u64, log_n);
                eq_k * EqPolynomial::mle(r_j, &y)
            })
            .sum();
        EqPolynomial::mle(&self.params.r1_k, r_k) * e_claim * X_claim
    }
}
