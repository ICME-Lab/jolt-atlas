use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationChallenge, RelationDerived, RelationExpr,
    RelationId, RelationOpening, RelationParams, RelationSetup, RelationVerifier, SetupRelation,
};
use crate::{
    onnx_proof::{ProofId, ProofType, Verifier},
    utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        identity_poly::IdentityPolynomial, multilinear_polynomial::PolynomialEvaluation,
        opening_point::OpeningPoint,
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId};
use atlas_trace::{Model, Op};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Gather over a large dictionary (read-raf lookup).
pub mod large;
/// Gather over a small dictionary (one-hot reads).
pub mod small;

/// Degree of the execution sumcheck.
pub const DEGREE_BOUND: usize = 2;

/// The dictionary shape of a gather node.
pub struct GatherShape {
    node: ComputationNode,
    lookup_vars: usize,
    num_words: usize,
}

/// The read of a gather's indices from its dictionary:
///
/// ```text
/// input claim   out(r_out) + γ·index(r_index)           (r_out = r_index ‖ r_word)
/// output claim  ra(ρ ‖ r_index) · (dict(ρ ‖ r_word) + γ·id(ρ))
/// ```
///
/// **S ⊆ R.** Each output row is the dictionary row its index names, so an honest one-hot `ra`
/// reads the output rows and, through the identity, the indices. The setup draws `γ` and takes
/// the reduced output point as its incoming point; the index opening at `r_index` is appended
/// by [`gather_verifier`] between the setup and the sumcheck.
pub struct GatherRelation {
    shape: GatherShape,
}

impl GatherRelation {
    /// The relation of a gather `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let dict_len = match &node.operator {
            Op::GatherSmall(gather_op) => gather_op.dict_len,
            Op::GatherLarge(gather_op) => gather_op.dict_len,
            _ => panic!("Expected Gather operator"),
        };
        let lookup_vars = model
            .view(node.inputs[1])
            .pow2_padded_num_output_elements()
            .log_2();
        Self::new(GatherShape {
            node,
            lookup_vars,
            num_words: dict_len.next_power_of_two(),
        })
    }

    /// The gather node.
    pub fn node(&self) -> &ComputationNode {
        &self.shape.node
    }

    /// Variables of the index tensor: the leading coordinates of the output point.
    pub fn lookup_vars(&self) -> usize {
        self.shape.lookup_vars
    }

    /// Dictionary rows, padded to a power of two.
    pub fn num_words(&self) -> usize {
        self.shape.num_words
    }

    fn builder(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.shape.node)
    }
}

impl SymbolicSumcheck for GatherRelation {
    crate::atlas_relation_types!();
    type Shape = GatherShape;

    fn new(shape: GatherShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Gather
    }

    fn rounds(&self) -> usize {
        self.shape.num_words.log_2()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        opening::<G, _, _, _>(RelationOpening::ReducedNodeOutput(self.shape.node.idx))
            + challenge::<G, _, _, _>(RelationChallenge::Gamma)
                * opening(RelationOpening::Id(self.builder().nodeio(Target::Input(1))))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let [ra, dict] = [0, 1].map(|i| RelationOpening::Id(self.produced()[i]));
        opening::<G, _, _, _>(ra)
            * (opening::<G, _, _, _>(dict)
                + challenge::<G, _, _, _>(RelationChallenge::Gamma)
                    * derived(RelationDerived::Identity))
    }
}

impl Produces for GatherRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.builder().advice(VirtualPoly::NodeOutputRa),
            self.builder().nodeio(Target::Input(0)),
        ]
    }
}

impl<F: JoltField> Relation<F> for GatherRelation {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let rho = low_to_high_point(challenges);
        let (r_index, r_word) = setup.incoming_point.split_at(self.shape.lookup_vars);
        vec![[&rho, r_index].concat(), [&rho, r_word].concat()]
    }

    fn derive(&self, id: &RelationDerived, _setup: &RelationSetup<F>, challenges: &[F]) -> F {
        match id {
            RelationDerived::Identity => IdentityPolynomial::new(self.shape.num_words.log_2())
                .evaluate(&low_to_high_point(challenges)),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

impl<F: JoltField> SetupRelation<F> for GatherRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> RelationSetup<F> {
        let gamma = transcript.challenge_scalar();
        let (r_node_output, _) =
            AccOpeningAccessor::new(accumulator, &self.shape.node).get_reduced_opening();
        RelationSetup {
            incoming_point: r_node_output.r,
            challenges: vec![(RelationChallenge::Gamma, gamma)],
        }
    }
}

/// The execution verifier of a gather `node`: the relation set up, then the index opening
/// appended at the output point's index coordinates.
pub fn gather_verifier<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> RelationVerifier<F, GatherRelation> {
    let relation = GatherRelation::for_node(&verifier.preprocessing.model, node.clone());
    let (relation, setup) =
        RelationParams::new(relation, &verifier.accumulator, &mut verifier.transcript).into_parts();
    let r_index = setup.incoming_point[..relation.shape.lookup_vars].to_vec();
    AccOpeningAccessor::new(&mut verifier.accumulator, node)
        .into_provider(&mut verifier.transcript, OpeningPoint::new(r_index))
        .append_nodeio(Target::Input(1));
    RelationVerifier::at(relation, setup)
}
