//! Shared plumbing for the clamped Erf/Sigmoid/Tanh variants.
//!
//! Each op fuses two lookup stages into one node:
//! 1. **Execution**: `output = SmallTable[clamped]`, where `clamped` is an internal
//!    advice claim (gamma-batched with the identity polynomial so the one-hot `ra`
//!    address is soundly tied to its value).
//! 2. **ActivationClamp**: proves that same `clamped` claim equals
//!    `ActivationClampTable[raw_input]` (natively symmetric clamp lookup, same shape as
//!    `Clamp`), which soundly ties `clamped` back to the real input rather than just
//!    asserting it.
//!
//! Both stages proceed sumcheck-then-sumcheck (Execution, then ActivationClamp), with
//! their one-hot (`ra`/`hw`/`bool`) checks batched together afterwards into a single
//! `BatchedSumcheck` call.

use crate::onnx_proof::neural_teleport::small_table::SmallTableRelation;
use crate::onnx_proof::op_lookups::OpLookupProve;
use crate::onnx_proof::relation::{
    RelationChallenge, RelationParams, RelationSetup, RelationWitness,
};
use atlas_trace::{trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::activation_clamped::*;
use jolt_claims::SymbolicSumcheck;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::{
    deferred_lookups::{DeferredBatch, DeferredOneHot},
    neural_teleport::{n_bits_to_usize, utils::compute_ra_evals_nbits_2comp},
    op_lookups::OpLookupProvider,
    ProofId, Prover,
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_point::OpeningPoint,
        signed_identity_poly::SignedIdentityPoly,
    },
};
use atlas_lookup_tables::clamp::ActivationClampTable;
use atlas_sumcheck::{
    opening_proof::prover::ProverOpeningAccumulator, sumcheck::SumcheckInstanceProof,
};
use atlas_transcript::Transcript;
use common::{
    consts::{ACTIVATION_BOUND, ACTIVATION_TABLE_VARS, XLEN},
    parallel::par_enabled,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

struct SmallTableProver<F: JoltField, Table: SmallActivationTable> {
    relation: SmallTableRelation<Table>,
    setup: RelationSetup<F>,
    gamma: F,
    table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: SignedIdentityPoly<F>,
    /// Lookup indices for this stage's one-hot checks, so the caller can reuse them
    /// without recomputing from the trace.
    lookup_indices: Vec<usize>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableProver<F, Table> {
    /// Computes the clamped tensor and registers its advice claim at `r_node_output`;
    /// the `ActivationClamp` stage then ties that claim back to `raw_input`.
    fn initialize(
        params: RelationParams<F, SmallTableRelation<Table>>,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (relation, setup) = params.into_parts();
        let gamma = setup.challenge(&RelationChallenge::Gamma);
        let r_node_output = OpeningPoint::new(setup.incoming_point.clone());
        let LayerData { operands, .. } = TraceView::layer_data(trace, relation.node());
        let raw_input = operands[0];
        let clamp_bound = 1i32 << ACTIVATION_BOUND;
        let clamped_tensor = raw_input.map(|x| x.clamp(-clamp_bound, clamp_bound - 1));
        let clamped_tensor_padded = clamped_tensor.padded_pow2();

        let clamped_claim = MultilinearPolynomial::from(
            clamped_tensor_padded
                .data()
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<_>>(),
        )
        .evaluate(&r_node_output.r);
        accumulator.append_virtual(
            transcript,
            clamped_opening_id(relation.node().idx),
            r_node_output.clone(),
            clamped_claim,
        );

        let table = MultilinearPolynomial::from(Table::materialize());
        let input_onehot: Vec<F> =
            compute_ra_evals_nbits_2comp(&r_node_output.r, &clamped_tensor, ACTIVATION_TABLE_VARS);
        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), table.len());
        let identity = SignedIdentityPoly::new(ACTIVATION_TABLE_VARS);

        let lookup_indices: Vec<usize> = clamped_tensor_padded
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&x| n_bits_to_usize(x, ACTIVATION_TABLE_VARS))
            .collect();

        Self {
            relation,
            setup,
            gamma,
            table,
            input_onehot,
            identity,
            lookup_indices,
        }
    }
}

impl<F: JoltField, Table: SmallActivationTable> RelationWitness<F> for SmallTableProver<F, Table> {
    type Relation = SmallTableRelation<Table>;

    fn relation(&self) -> &SmallTableRelation<Table> {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.input_onehot.final_claim()]
    }
}

atlas_sumcheck::relation_witness_instance!(
    [F: JoltField, Table: SmallActivationTable] SmallTableProver<F, Table>
);

impl<F: JoltField, Table: SmallActivationTable> SmallTableProver<F, Table> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let Self {
            input_onehot,
            table,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra_evals =
                    input_onehot.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);
                let table_evals =
                    table.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);
                let id_evals =
                    identity.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (table_evals[0] + id_evals[0] * self.gamma),
                    ra_evals[1] * (table_evals[1] + id_evals[1] * self.gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UnivariatePoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.input_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField, Table: SmallActivationTable> ProveRounds<F> for SmallTableProver<F, Table> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public entry points, called by the thin per-op `ProveOp`/`VerifyOp` impls.
// ---------------------------------------------------------------------------

/// Full (Execution + ActivationClamp) proof for a clamped-activation node: Execution
/// establishes the clamped advice claim, ActivationClamp ties it back to the raw
/// input, then both stages' one-hot checks are batched into one `BatchedSumcheck`.
pub fn prove_clamped_activation<F, T, Table>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)>
where
    F: JoltField,
    T: Transcript,
    Table: SmallActivationTable,
{
    // Stage 1: Execution -- output = SmallTable[clamped]. Built here (its
    // challenge is drawn now, its `clamped` advice claim appended now), proven
    // after the node loop in the `ActivationExec` batch.
    let params = small_table_params::<F, Table>(node, &prover.accumulator, &mut prover.transcript);
    let exec_sumcheck = SmallTableProver::<F, Table>::initialize(
        params,
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let small_indices = exec_sumcheck.lookup_indices.clone();
    prover.defer(DeferredBatch::ActivationExec, Box::new(exec_sumcheck));

    // Stage 2: ActivationClamp -- clamped = ActivationClampTable[raw_input].
    // Same: built now, proven in the `ActivationClamp` batch.
    let clamp_provider: OpLookupProvider<ActivationClampOperands> =
        OpLookupProvider::new(node.clone());
    let (clamp_sumcheck, clamp_lookup_indices) = clamp_provider
        .read_raf_prove::<F, T, ActivationClampTable<XLEN>, XLEN>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::ActivationClamp, Box::new(clamp_sumcheck));

    // Stage 3: both stages' one-hot checks, built and proven after their batches.
    prover
        .deferred_onehots
        .push((DeferredOneHot::ActivationSmall(node.idx), small_indices));
    prover.deferred_onehots.push((
        DeferredOneHot::ActivationClamp(node.clone()),
        clamp_lookup_indices,
    ));

    Vec::new()
}
