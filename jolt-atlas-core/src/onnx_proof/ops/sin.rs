use crate::node_view::ComputationNode;
use crate::onnx_proof::neural_teleport::trig_table::{TrigTableProver, TrigTableRelation};
use crate::onnx_proof::op_lookups::OpLookupProve;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::range_checking::RangeCheckProve;
use crate::onnx_proof::relation::RelationParams;
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::{
    neural_teleport::{
        division::{
            cache_teleport_input_claim_prove, cache_teleport_quotient_prove, compute_division,
        },
        sin::SinTable,
        trig_downscale::{cache_downscaled_prove, TrigDownscaleOperands},
    },
    op_lookups::OpLookupProvider,
    range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckProvider},
    ProofId, ProofType, Prover,
};
use atlas_algebra::field::JoltField;
use atlas_lookup_tables::{
    right_shift::RightShiftTable, unsigned_less_than::UnsignedLessThanTable,
};
use atlas_sumcheck::{
    shout,
    sumcheck::{prover::SumcheckInstanceProver, BatchedSumcheck, SumcheckInstanceProof},
};
use atlas_trace::{op::Sin, trace::LayerData};
use atlas_transcript::Transcript;
use common::consts::{MODEL_SCALE, TRIG_PERIOD_MODULUS, XLEN};
pub use jolt_atlas_verifier::onnx_proof::ops::sin::*;
use jolt_claims::SymbolicSumcheck;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Sin {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        assert_eq!(
            node.scale, MODEL_SCALE as u8,
            "Sin teleportation proving is only calibrated for MODEL_SCALE={MODEL_SCALE} \
             (got {})",
            node.scale
        );
        let mut results = Vec::new();

        // Plumbing: cache the quotient's claim.
        cache_teleport_quotient_prove(node, prover, TRIG_PERIOD_MODULUS as i32);

        // Plumbing: downscale.
        let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
        let input = operands[0];
        let (_quotient, remainder) = compute_division(input, TRIG_PERIOD_MODULUS as i32);
        let downscaled = cache_downscaled_prove(node, prover, &remainder);

        // Stage 1: batch every read-raf "value" sumcheck together — downscale, the Sin
        // table lookup, and the remainder range-check.
        let dsc_provider: OpLookupProvider<TrigDownscaleOperands> =
            OpLookupProvider::new(node.clone());
        let (dsc_exec, dsc_lookup_indices) = dsc_provider
            .read_raf_prove::<F, T, RightShiftTable<XLEN>, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let params = RelationParams::new(
            TrigTableRelation::<SinTable>::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let (sin_exec, sin_lookup_indices) =
            TrigTableProver::initialize(&downscaled, params, &mut prover.accumulator);

        let rc_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let (rangecheck_exec, rc_lookup_indices) = rc_provider
            .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let mut exec_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![
            Box::new(dsc_exec),
            Box::new(sin_exec),
            Box::new(rangecheck_exec),
        ];
        let (exec_proof, _) = BatchedSumcheck::prove(
            exec_instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

        // Stage 2: batch every one-hot correctness triple together — downscale, Sin,
        // and the range-check.
        let dsc_encoding = dsc_provider.encoding();
        let sin_encoding = SinRaEncoding::new(node);
        let rc_encoding = rc_provider.encoding();

        let mut onehot_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = Vec::new();
        onehot_instances.extend(shout::prover::ra_onehot_provers(
            &dsc_encoding,
            &dsc_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        onehot_instances.extend(shout::prover::ra_onehot_provers(
            &sin_encoding,
            &sin_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        onehot_instances.extend(shout::prover::ra_onehot_provers(
            &rc_encoding,
            &rc_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        let (onehot_proof, _) = BatchedSumcheck::prove(
            onehot_instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), onehot_proof));

        // Plumbing: derive `a`'s input claim now that `q` and `r` both hold claims at the
        // node's reduced output-opening point.
        cache_teleport_input_claim_prove(node, prover, TRIG_PERIOD_MODULUS as i32);

        results
    }
}
