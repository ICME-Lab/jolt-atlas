use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::{
    onnx_proof::{ProofId, ProofType, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::PolynomialEvaluation,
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_trace::{Model, Op};
use atlas_transcript::Transcript;
use common::VirtualPoly;

/// Gather over a large dictionary (read-raf lookup).
pub mod large;
/// Gather over a small dictionary (one-hot reads).
pub mod small;

/// Degree of the execution sumcheck.
pub const DEGREE_BOUND: usize = 2;

/// Parameters for proving gather (indexed lookup) operations.
///
/// Gather selects elements from an input dictionary tensor using an index tensor.
/// Uses one-hot encoding to prove correct lookups with a folding challenge gamma.
#[derive(Clone)]
pub struct GatherParams<F: JoltField> {
    /// Batching challenge.
    pub gamma: F,
    /// The node output's reduced opening point.
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    /// The node being proven.
    pub computation_node: ComputationNode,
    /// Variables of the dictionary address.
    pub lookup_vars: usize,
    /// Number of dictionary rows.
    pub num_words: usize,
}

impl<F: JoltField> GatherParams<F> {
    /// Create new gather parameters from a computation node, model, accumulator, and transcript.
    pub fn new(
        computation_node: ComputationNode,
        model: &Model,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let gamma = transcript.challenge_scalar();

        let dict_len = match &computation_node.operator {
            Op::GatherSmall(gather_op) => gather_op.dict_len,
            Op::GatherLarge(gather_op) => gather_op.dict_len,
            _ => panic!("Expected Gather operator"),
        };

        let input_indices = &model.view(computation_node.inputs[1]);
        let num_words = dict_len.next_power_of_two();
        let lookup_vars = input_indices.pow2_padded_num_output_elements().log_2();

        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            gamma,
            r_node_output,
            computation_node,
            num_words,
            lookup_vars,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for GatherParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let rv_claim = accessor.get_reduced_opening().1;
        let index_claim = accessor.get_nodeio(Target::Input(1)).1;

        rv_claim + self.gamma * index_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_words.log_2()
    }

    // output = ra_claim * (dict_claim + gamma * int_eval)
    //        = Challenge(0) * Opening(ra) * Opening(dict)
    //          + Challenge(1) * Opening(ra)
}

/// Verifier for gather sumcheck protocol.
///
/// Verifies that the prover's claims are consistent with the gather (indexed lookup) operation.
pub struct GatherVerifier<F: JoltField> {
    params: GatherParams<F>,
}

impl<F: JoltField> GatherVerifier<F> {
    /// Create a new verifier for the gather operation, caching index operand openings.
    pub fn new(
        computation_node: ComputationNode,
        model: &Model,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = GatherParams::new(computation_node, model, accumulator, transcript);

        let (r_index, _) = params.r_node_output.r.split_at(params.lookup_vars);
        let mut provider = AccOpeningAccessor::new(accumulator, &params.computation_node)
            .into_provider(transcript, OpeningPoint::new(r_index.to_vec()));
        provider.append_nodeio(Target::Input(1));

        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for GatherVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);

        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let ra_claim = accessor.get_advice(VirtualPoly::NodeOutputRa).1;
        let int_eval =
            IdentityPolynomial::new(self.params.num_words.log_2()).evaluate(&opening_point.r);
        let dict_claim = accessor.get_nodeio(Target::Input(0)).1;

        ra_claim * (dict_claim + self.params.gamma * int_eval)
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

        let (r_index, r_word) = self
            .params
            .r_node_output
            .r
            .split_at(self.params.lookup_vars);
        let r_idx_onehot = [&opening_point.r, r_index].concat().into();
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default());
        provider.append_advice_at(VirtualPoly::NodeOutputRa, r_idx_onehot);

        let r_dict = [&opening_point.r, r_word].concat().into();
        provider.append_nodeio_at(Target::Input(0), r_dict);
    }
}
