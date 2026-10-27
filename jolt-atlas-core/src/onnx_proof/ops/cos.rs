use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::OpLookupProve;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::range_checking::RangeCheckProve;
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::IntoProverProvider;
use crate::{
    onnx_proof::{
        neural_teleport::{
            cos::{CosTable, COS_TABLE_VARS},
            division::{
                cache_teleport_input_claim_prove, cache_teleport_quotient_prove, compute_division,
            },
            trig_downscale::{cache_downscaled_prove, TrigDownscaleOperands},
            utils::compute_ra_evals_from_usize_indices,
        },
        op_lookups::OpLookupProvider,
        range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckProvider},
        ProofId, ProofType, Prover,
    },
    utils::opening_access::AccOpeningAccessor,
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_point::OpeningPoint,
    },
};
use atlas_lookup_tables::{
    right_shift::RightShiftTable, unsigned_less_than::UnsignedLessThanTable,
};
use atlas_sumcheck::{
    opening_proof::prover::ProverOpeningAccumulator,
    shout,
    sumcheck::{prover::SumcheckInstanceProver, BatchedSumcheck, SumcheckInstanceProof},
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_trace::{op::Cos, trace::LayerData, Tensor};
use atlas_transcript::Transcript;
use common::consts::{MODEL_SCALE, TRIG_PERIOD_MODULUS, XLEN};
use common::parallel::par_enabled;
use common::VirtualPoly;
pub use jolt_atlas_verifier::onnx_proof::ops::cos::*;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Cos {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        assert_eq!(
            node.scale, MODEL_SCALE as u8,
            "Cos teleportation proving is only calibrated for MODEL_SCALE={MODEL_SCALE} \
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

        // Stage 1: batch every read-raf "value" sumcheck together — downscale, the Cos
        // table lookup, and the remainder range-check.
        let dsc_provider: OpLookupProvider<TrigDownscaleOperands> =
            OpLookupProvider::new(node.clone());
        let (dsc_exec, dsc_lookup_indices) = dsc_provider
            .read_raf_prove::<F, T, RightShiftTable<XLEN>, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let params = CosParams::new(
            node.clone(),
            &prover.preprocessing.model,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let (cos_exec, cos_lookup_indices) =
            CosProver::initialize(&downscaled, params, &mut prover.accumulator);

        let rc_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let (rangecheck_exec, rc_lookup_indices) = rc_provider
            .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let mut exec_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![
            Box::new(dsc_exec),
            Box::new(cos_exec),
            Box::new(rangecheck_exec),
        ];
        let (exec_proof, _) = BatchedSumcheck::prove(
            exec_instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

        // Stage 2: batch every one-hot correctness triple together — downscale, Cos,
        // and the range-check.
        let dsc_encoding = dsc_provider.encoding();
        let cos_encoding = CosRaEncoding::new(node);
        let rc_encoding = rc_provider.encoding();

        let mut onehot_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = Vec::new();
        onehot_instances.extend(shout::prover::ra_onehot_provers(
            &dsc_encoding,
            &dsc_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        onehot_instances.extend(shout::prover::ra_onehot_provers(
            &cos_encoding,
            &cos_lookup_indices,
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

/// Read-Raf sumcheck prover for the Cos table lookup: `output[i] = CosTable[downscaled[i]]`.
pub struct CosProver<F: JoltField> {
    params: CosParams<F>,
    cos_table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> CosProver<F> {
    /// Initialize the prover state.
    pub fn initialize(
        downscaled: &Tensor<i32>,
        params: CosParams<F>,
        _accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> (Self, Vec<usize>) {
        assert!(downscaled
            .iter()
            .all(|&x| (0..CosTable::table_size() as i32).contains(&x)));

        let cos_table = MultilinearPolynomial::from(CosTable::materialize());
        let lookup_indices: Vec<usize> = downscaled
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&x| x as usize)
            .collect();
        let input_onehot: Vec<F> = compute_ra_evals_from_usize_indices(
            &params.r_node_output.r,
            &lookup_indices,
            1 << COS_TABLE_VARS,
        );

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), cos_table.len());
        let identity = IdentityPolynomial::new(COS_TABLE_VARS);

        #[cfg(test)]
        {
            let accessor = AccOpeningAccessor::new(_accumulator, &params.computation_node);
            let downscaled_claim = accessor.get_advice(VirtualPoly::TrigDownscaled).1;
            let rv_claim = accessor.get_reduced_opening().1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    let a = input_onehot.get_bound_coeff(i);
                    let b = cos_table.get_bound_coeff(i);
                    let int = F::from_u32(i as u32);
                    a * (b + params.gamma * int)
                })
                .sum();
            crate::onnx_proof::testing::probe::probe_assert_eq!(
                rv_claim + params.gamma * downscaled_claim,
                claim
            )
        }

        (
            Self {
                params,
                cos_table,
                input_onehot,
                identity,
            },
            lookup_indices,
        )
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for CosProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
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
        let r = [
            opening_point.r.as_slice(),
            self.params.r_node_output.r.as_slice(),
        ]
        .concat();
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::CosRa, self.input_onehot.final_claim());
    }
}

impl<F: JoltField> CosProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let Self {
            input_onehot,
            cos_table,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra_evals =
                    input_onehot.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let table_evals =
                    cos_table.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let id_evals = identity.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (table_evals[0] + id_evals[0] * self.params.gamma),
                    ra_evals[1] * (table_evals[1] + id_evals[1] * self.params.gamma),
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
        self.cos_table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for CosProver<F> {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceParams::num_rounds(&self.params)
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
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
    }
}
