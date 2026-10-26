use crate::{
    node_view::ComputationNode,
    onnx_proof::{
        ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
        range_checking::{
            range_check_operands::{DivRangeCheckOperands, RangeCheckOperands},
            RangeCheckProvider,
        },
        relation::{
            prove_tie, verify_tie, AtlasRelation, EngineRelation, RelationChallenge,
            RelationDerived, RelationExpr, RelationId, RelationOpening, RelationParams,
            RelationProver, RelationSetup, RelationVerifier, TieRelation,
        },
        trace_view::TraceView,
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::Div, semantics::fixed::rem_floor, trace::LayerData, Trace};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use jolt_claims::{derived, opening, Expr, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::{
        evaluation_reduction::EvalReductionProof,
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Div {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    #[tracing::instrument(skip_all, name = "Div::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let mut results = Vec::new();

        // Execution proof
        let params = RelationParams::<F, _>::new(
            DivRelation::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        results
    }

    fn prove_with_reduction(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (
        EvalReductionProof<F>,
        Vec<(ProofId, SumcheckInstanceProof<F>)>,
    ) {
        let mut proofs = self.prove(node, prover);

        // Reduce node-output openings first, then bind the reduced output claim
        // as the committed Div quotient opening used by PCS opening verification.
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        prove_tie(
            &DivQuotientRelation::new(node.clone()),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        if node.is_scalar() {
            return (eval_reduction_proof, proofs);
        }

        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
    }

    #[tracing::instrument(skip_all, name = "Div::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution verification
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = RelationVerifier::<F, _>::new(
            DivRelation::new(node.clone()),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn verify_with_reduction(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
        eval_reduction_proof: &EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        self.verify(node, verifier)?;

        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        verify_tie(
            &DivQuotientRelation::new(node.clone()),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        if node.is_scalar() {
            return Ok(());
        }

        verify_range_and_onehot(node, verifier)?;
        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let mut polys = vec![CommittedPoly::DivNodeQuotient(node.idx)];
        if node.is_scalar() {
            return polys;
        }
        let rc_operands = RangeCheckOperands::<DivRangeCheckOperands>::new(node);
        let encoding = rc_operands.get_encoding(node);
        let d = encoding.one_hot_params().instruction_d;
        polys.extend((0..d).map(|i| CommittedPoly::DivRangeCheckRaD(node.idx, i)));
        polys
    }
}

crate::bind_relation!(Div, DivRelation);
crate::bind_relation!(Div, DivQuotientRelation);

/// Floor division `q = ⌊left / right⌋` as a relation, its point `r₀` drawn from the transcript:
///
/// ```text
/// input claim   0
/// output claim  eq(r₀, r′) · (right·q + rem − left)
/// ```
///
/// **S ⊆ R.** `atlas_trace::semantics::fixed::floor_div` and its remainder `rem_floor` (the
/// divisor's sign) satisfy `left = right·q + rem` at every element, so the summand vanishes on
/// the hypercube. The bound on `rem` that makes `q` the floor is the range check below,
/// outside the relation.
pub struct DivRelation {
    node: ComputationNode,
}

impl DivRelation {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for DivRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::Division
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        Expr::zero()
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let open = |id| opening(RelationOpening::Id(id));
        let left = open(self.ids().nodeio(Target::Input(0)));
        let right = open(self.ids().nodeio(Target::Input(1)));
        let quotient = open(self.ids().nodeio(Target::Current));
        let remainder = open(self.ids().advice(VirtualPoly::DivRemainder));
        derived(RelationDerived::Eq) * (right * quotient + remainder - left)
    }
}

impl AtlasRelation for DivRelation {
    type Spec = Div;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: transcript
                .challenge_vector_optimized::<F>(self.rounds())
                .into_opening(),
            challenges: Vec::new(),
        }
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().nodeio(Target::Input(1)),
            self.ids().nodeio(Target::Current),
            self.ids().advice(VirtualPoly::DivRemainder),
        ]
    }
}

impl EngineRelation for DivRelation {
    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, output } = TraceView::layer_data(trace, &self.node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for Div operation")
        };
        let remainder: Vec<i32> = left
            .padded_pow2()
            .iter()
            .zip(right.padded_pow2().iter())
            // Padding lanes divide 0 by 0: no remainder.
            .map(|(&a, &b)| if b == 0 { 0 } else { rem_floor(a, b) })
            .collect();
        vec![
            MultilinearPolynomial::from(left.padded_pow2()),
            MultilinearPolynomial::from(right.padded_pow2()),
            MultilinearPolynomial::from(output.padded_pow2()),
            MultilinearPolynomial::from(remainder),
        ]
    }
}

// TODO: Reduce two claims to 1 via 4.5.2 PAZK for Quotient polynomial openings

/// The committed quotient `q` (what the PCS opens) bound to the node's reduced output, a tie:
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  q(r)                   (the committed quotient, at the same point)
/// ```
///
/// **S ⊆ R.** The committed quotient is the node's output tensor.
pub struct DivQuotientRelation {
    node: ComputationNode,
}

impl SymbolicSumcheck for DivQuotientRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::DivisionQuotient
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl AtlasRelation for DivQuotientRelation {
    type Spec = Div;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: accumulator.get_node_output_opening(self.node.idx).0.r,
            challenges: Vec::new(),
        }
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.node).advice(CommittedPoly::DivNodeQuotient)]
    }
}

impl TieRelation for DivQuotientRelation {
    fn claims<F: JoltField>(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // The committed quotient is the node's output: its claim is the reduced one.
        vec![accumulator.get_node_output_opening(self.node.idx).1]
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    let mut results = Vec::new();

    let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
    let (mut rangecheck_sumcheck, lookup_indices) = rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let (rangecheck_proof, _) = Sumcheck::prove(
        &mut rangecheck_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    results.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

    let rc_operands = RangeCheckOperands::<DivRangeCheckOperands>::new(node);
    let encoding = rc_operands.get_encoding(node);

    let [ra_sumcheck, bool_sumcheck] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![ra_sumcheck, bool_sumcheck];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        ra_one_hot_proof,
    ));

    results
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let rangecheck_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RangeCheck))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;

    let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
    let rangecheck_verifier = rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    Sumcheck::verify(
        rangecheck_proof,
        &rangecheck_verifier,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let rc_operands = RangeCheckOperands::<DivRangeCheckOperands>::new(node);
    let encoding = rc_operands.get_encoding(node);
    let [ra_sumcheck, bool_sumcheck] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        vec![&*ra_sumcheck, &*bool_sumcheck],
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    Ok(())
}
