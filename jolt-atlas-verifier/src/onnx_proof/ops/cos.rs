use crate::node_view::ComputationNode;
use crate::onnx_proof::neural_teleport::trig_table::TrigTableRelation;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::relation::RelationVerifier;
use crate::{
    onnx_proof::{
        neural_teleport::{
            cos::{CosTable, COS_TABLE_VARS},
            division::{cache_teleport_quotient_verify, verify_teleport_input_claim},
            trig_downscale::{cache_downscaled_verify, TrigDownscaleOperands},
        },
        op_lookups::{OpLookupEncoding, OpLookupProvider},
        range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckProvider},
        ProofId, ProofType, Verifier,
    },
    utils::opening_access::AccOpeningAccessor,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    utils::errors::ProofVerifyError,
};
use atlas_lookup_tables::{
    right_shift::RightShiftTable, unsigned_less_than::UnsignedLessThanTable,
};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    shout::{self, RaOneHotEncoding},
    sumcheck::BatchedSumcheck,
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_trace::{op::Cos, Model};
use atlas_transcript::Transcript;
use common::consts::{MODEL_SCALE, TRIG_PERIOD_MODULUS, XLEN};
use common::{CommittedPoly, VirtualPoly};
use jolt_claims::SymbolicSumcheck;

impl OpLayout for Cos {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let cos_encoding = CosRaEncoding::new(node);
        let rc_encoding = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node).encoding();
        let dsc_encoding: OpLookupEncoding<TrigDownscaleOperands> =
            OpLookupProvider::<TrigDownscaleOperands>::new(node.clone()).encoding();
        let cos_d = cos_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let dsc_d = dsc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![CommittedPoly::TeleportNodeQuotient(node.idx)];
        polys.extend((0..dsc_d).map(|i| CommittedPoly::TrigDownscaleRaD(node.idx, i)));
        polys.extend((0..cos_d).map(|i| CommittedPoly::CosRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPoly::TeleportRangeCheckRaD(node.idx, i)));
        polys
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Cos {
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        if node.scale != MODEL_SCALE as u8 {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "Cos teleportation verifying is only calibrated for MODEL_SCALE={MODEL_SCALE} \
                     (got {})",
                node.scale
            )));
        }
        // Plumbing: cache the quotient's claim.
        cache_teleport_quotient_verify(node, verifier);

        // Plumbing: downscale.
        cache_downscaled_verify(node, verifier);

        // Stage 1: batch every read-raf "value" sumcheck together — downscale, the Cos
        // table lookup, and the remainder range-check.
        let dsc_provider: OpLookupProvider<TrigDownscaleOperands> =
            OpLookupProvider::new(node.clone());
        let dsc_verifier = dsc_provider.read_raf_verify::<F, T, RightShiftTable<XLEN>, XLEN>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );

        let exec_sumcheck = RelationVerifier::<F, TrigTableRelation<CosTable>>::new(
            TrigTableRelation::new(node.clone()),
            &verifier.accumulator,
            &mut verifier.transcript,
        );

        let rc_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let rangecheck_verifier = rc_provider.read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );

        let exec_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        BatchedSumcheck::verify(
            exec_proof,
            vec![&dsc_verifier, &exec_sumcheck, &rangecheck_verifier],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 2: batch every one-hot correctness triple together — downscale, Cos,
        // and the range-check.
        let dsc_encoding = dsc_provider.encoding();
        let cos_encoding = CosRaEncoding::new(node);
        let rc_encoding = rc_provider.encoding();

        let onehot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let mut onehot_verifiers: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = Vec::new();
        onehot_verifiers.extend(shout::ra_onehot_verifiers(
            &dsc_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        ));
        onehot_verifiers.extend(shout::ra_onehot_verifiers(
            &cos_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        ));
        onehot_verifiers.extend(shout::ra_onehot_verifiers(
            &rc_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        ));
        BatchedSumcheck::verify(
            onehot_proof,
            onehot_verifiers.iter().map(|v| &**v as _).collect(),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Plumbing: derive `a`'s input claim now that `q` and `r` both hold claims at the
        // node's reduced output-opening point.
        verify_teleport_input_claim(node, verifier, TRIG_PERIOD_MODULUS as i32)?;

        Ok(())
    }
}

/// Degree of the execution sumcheck.
pub const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
/// Parameters for the cosine trigonometric function.
pub struct CosParams<F: JoltField> {
    /// Batching challenge.
    pub gamma: F,
    /// The node output's reduced opening point.
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    /// The node being proven.
    pub computation_node: ComputationNode,
}

impl<F: JoltField> CosParams<F> {
    /// Create a new CosParams instance for the given computation node.
    pub fn new(
        computation_node: ComputationNode,
        _model: &Model,
        accumulator: &impl OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);

        let gamma = transcript.challenge_scalar();
        let (r_node_output, _) = accessor.get_reduced_opening();

        Self {
            gamma,
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for CosParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let rv_claim = accessor.get_reduced_opening().1;

        let downscaled_claim = accessor.get_advice(VirtualPoly::TrigDownscaled).1;

        rv_claim + self.gamma * downscaled_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        COS_TABLE_VARS
    }

    // output = ra_claim * (table_claim + gamma * int_eval)
}

/// One-hot read-address encoding for the Cos table lookup.
pub struct CosRaEncoding {
    /// Index of the computation node in the trace.
    pub node_idx: usize,
}

impl CosRaEncoding {
    /// Create a new Cos table one-hot encoding for the given computation node.
    pub fn new(node: &ComputationNode) -> Self {
        Self { node_idx: node.idx }
    }
}

impl RaOneHotEncoding for CosRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::CosRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TrigDownscaled(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::CosRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        COS_TABLE_VARS
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), COS_TABLE_VARS)
    }
}
