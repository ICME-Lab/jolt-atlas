use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::{
    neural_teleport::{
        division::TeleportInputRelation, sin::SIN_TABLE_VARS, trig_downscale::TrigDownscaleOperands,
    },
    op_lookups::{OpLookupEncoding, OpLookupProvider},
    plan::{OpLookup, OpOpen, OpPlan, OpRangeCheck, OpRelation, OpTie, OpTrig, Plan, Step},
    range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckProvider},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningId, SumcheckId},
    shout::RaOneHotEncoding,
};
use atlas_trace::{op::Sin, trace::ModelExecutionIO, Model};
use common::consts::{MODEL_SCALE, TRIG_PERIOD_MODULUS};
use common::{CommittedPoly, VirtualPoly};

impl OpLayout for Sin {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let sin_encoding = SinRaEncoding::new(node);
        let rc_encoding = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node)?.encoding();
        let dsc_encoding: OpLookupEncoding<TrigDownscaleOperands> =
            OpLookupProvider::<TrigDownscaleOperands>::new(node.clone()).encoding();
        let sin_d = sin_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let dsc_d = dsc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![CommittedPoly::TeleportNodeQuotient(node.idx)];
        polys.extend((0..dsc_d).map(|i| CommittedPoly::TrigDownscaleRaD(node.idx, i)));
        polys.extend((0..sin_d).map(|i| CommittedPoly::SinRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPoly::TeleportRangeCheckRaD(node.idx, i)));
        Ok(polys)
    }
}

/// The plan: the evaluation reduction; the teleport quotient and the downscaled remainder
/// opened at the reduced output point; one batch of the read-raf "value" sumchecks (downscale,
/// the Sin table, the remainder range check); one batch of their one-hot checks; then the
/// teleport identity `a = P·q + r` in the clear, once `q` and `r` both hold claims at the
/// reduced output point.
impl OpPlan for Sin {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        if node.scale != MODEL_SCALE as u8 {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "Sin teleportation verifying is only calibrated for MODEL_SCALE={MODEL_SCALE} \
                     (got {})",
                node.scale
            )));
        }
        Ok(Plan(vec![
            Step::Reduce,
            Step::Open(OpOpen::TeleportQuotient(node.clone())),
            Step::Open(OpOpen::TrigDownscaled(node.clone())),
            Step::Batch {
                proof: ProofType::Execution,
                relations: vec![
                    OpRelation::Lookup(OpLookup::TrigDownscale, node.clone()),
                    OpRelation::Trig(OpTrig::Sin, node.clone()),
                    OpRelation::RangeCheck(OpRangeCheck::Teleport, node.clone()),
                ],
            },
            Step::Batch {
                proof: ProofType::RaOneHotChecks,
                relations: vec![
                    OpRelation::LookupOneHot(OpLookup::TrigDownscale, node.clone()),
                    OpRelation::TrigOneHot(OpTrig::Sin, node.clone()),
                    OpRelation::RangeCheckOneHot(OpRangeCheck::Teleport, node.clone()),
                ],
            },
            Step::Tie {
                relation: OpTie::TeleportInput(TeleportInputRelation {
                    node: node.clone(),
                    tau: TRIG_PERIOD_MODULUS as i32,
                }),
            },
        ]))
    }
}

/// One-hot read-address encoding for the Sin table lookup.
pub struct SinRaEncoding {
    /// Index of the computation node in the trace.
    pub node_idx: usize,
}

impl SinRaEncoding {
    /// Create a new Sin table one-hot encoding for the given computation node.
    pub fn new(node: &ComputationNode) -> Self {
        Self { node_idx: node.idx }
    }
}

impl RaOneHotEncoding for SinRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SinRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TrigDownscaled(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SinRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        SIN_TABLE_VARS
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), SIN_TABLE_VARS)
    }
}
