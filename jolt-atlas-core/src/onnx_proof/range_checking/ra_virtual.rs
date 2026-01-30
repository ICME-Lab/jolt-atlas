use std::sync::Arc;

use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::compute_mles_product_sum,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};
use rayon::prelude::*;

use crate::onnx_proof::{
    range_checking::{sumcheck_instance::ReadRafSumcheckHelper, LOG_K},
    Prover, Verifier,
};

// Instruction read-access (RA) virtualization sumcheck
//
// Proves the relation:
//   Σ_j eq(r_cycle, j) ⋅ Π_{i=0}^{D-1} ra_i(r_{address,i}, j) = ra_claim
// where:
// - eq is the MLE of equality on bitstrings; evaluated at field points (r_cycle, j).
// - ra_i are MLEs of chunk-wise access indicators (1 on matching {0,1}-points).
// - ra_claim is the claimed evaluation of the virtual read-access polynomial from the read-raf sumcheck.

pub struct InstructionRaSumcheckParams<F: JoltField, Helper: ReadRafSumcheckHelper> {
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    pub one_hot_params: OneHotParams,
    computation_node: ComputationNode,
    helper: Helper,
}

impl<F: JoltField, Helper: ReadRafSumcheckHelper> InstructionRaSumcheckParams<F, Helper> {
    pub fn new(
        computation_node: ComputationNode,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let helper = Helper::new(&computation_node);

        let (r, _) = opening_accumulator
            .get_virtual_polynomial_opening(helper.get_output_operand(), SumcheckId::Raf);
        let (r_address, r_cycle) = r.split_at(LOG_K);
        Self {
            r_address,
            r_cycle,
            one_hot_params: one_hot_params.clone(),
            computation_node,
            helper,
        }
    }
}

impl<F: JoltField, Helper: ReadRafSumcheckHelper> SumcheckInstanceParams<F>
    for InstructionRaSumcheckParams<F, Helper>
{
    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, ra_claim) = accumulator
            .get_virtual_polynomial_opening(self.helper.get_output_operand(), SumcheckId::Raf);
        ra_claim
    }

    fn degree(&self) -> usize {
        self.one_hot_params.instruction_d + 1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

pub struct InstructionRaSumcheckProver<F: JoltField, Helper: ReadRafSumcheckHelper> {
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    params: InstructionRaSumcheckParams<F, Helper>,
}

impl<F: JoltField, Helper: ReadRafSumcheckHelper> InstructionRaSumcheckProver<F, Helper> {
    pub fn new_from_prover(
        computation_node: ComputationNode,
        one_hot_params: &OneHotParams,
        prover: &mut Prover<F, impl Transcript>,
    ) -> Self {
        let params =
            InstructionRaSumcheckParams::new(computation_node, one_hot_params, &prover.accumulator);
        Self::initialize(params, &prover.trace)
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::initialize")]
    pub fn initialize(params: InstructionRaSumcheckParams<F, Helper>, trace: &Trace) -> Self {
        // Compute r_address_chunks with proper padding
        let r_address_chunks = params
            .one_hot_params
            .compute_r_address_chunks::<F>(&params.r_address.r);

        let (left_operand, right_operand) =
            Helper::get_operands_tensors(trace, &params.computation_node);

        let lookup_indices = Helper::compute_lookup_indices(&left_operand, &right_operand);

        let H_indices: Vec<Vec<Option<u8>>> = (0..params.one_hot_params.instruction_d)
            .map(|i| {
                lookup_indices
                    .par_iter()
                    .map(|lookup_index| {
                        Some(
                            params
                                .one_hot_params
                                .lookup_index_chunk(lookup_index.into(), i),
                        )
                    })
                    .collect()
            })
            .collect();

        let ra_i_polys = H_indices
            .into_par_iter()
            .enumerate()
            .map(|(i, lookup_indices)| {
                let eq_evals = EqPolynomial::evals(&r_address_chunks[i]);
                RaPolynomial::new(Arc::new(lookup_indices), eq_evals)
            })
            .collect();

        Self {
            ra_i_polys,
            eq_poly: GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript, Helper: ReadRafSumcheckHelper + Sync + Send>
    SumcheckInstanceProver<F, T> for InstructionRaSumcheckProver<F, Helper>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let ra_i_polys = &self.ra_i_polys;
        let eq_poly = &self.eq_poly;
        compute_mles_product_sum(ra_i_polys, previous_claim, eq_poly)
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra_i_polys
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);
        let (r, _) = accumulator.get_virtual_polynomial_opening(
            self.params.helper.get_output_operand(),
            SumcheckId::Raf,
        );

        let (r_address, _) = r.split_at_r(LOG_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(r_address);

        for (i, r_address) in r_address_chunks.into_iter().enumerate() {
            let claim = self.ra_i_polys[i].final_sumcheck_claim();
            accumulator.append_sparse(
                transcript,
                vec![self.params.helper.rad_poly(i)],
                SumcheckId::RaVirtualization,
                r_address,
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }
}

pub struct InstructionRaSumcheckVerifier<F: JoltField, Helper: ReadRafSumcheckHelper> {
    params: InstructionRaSumcheckParams<F, Helper>,
}

impl<F: JoltField, Helper: ReadRafSumcheckHelper> InstructionRaSumcheckVerifier<F, Helper> {
    pub fn new_from_verifier(
        computation_node: ComputationNode,
        one_hot_params: &OneHotParams,
        verifier: &mut Verifier<F, impl Transcript>,
    ) -> Self {
        let params = InstructionRaSumcheckParams::new(
            computation_node,
            one_hot_params,
            &verifier.accumulator,
        );
        Self { params }
    }

    pub fn new(
        computation_node: ComputationNode,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            InstructionRaSumcheckParams::new(computation_node, one_hot_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript, Helper: ReadRafSumcheckHelper> SumcheckInstanceVerifier<F, T>
    for InstructionRaSumcheckVerifier<F, Helper>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        let eq_eval = EqPolynomial::mle_endian(&self.params.r_cycle, &r);
        let ra_claim_prod: F = (0..self.params.one_hot_params.instruction_d)
            .map(|i| {
                let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                    self.params.helper.rad_poly(i),
                    SumcheckId::RaVirtualization,
                );
                ra_i_claim
            })
            .product();

        eq_eval * ra_claim_prod
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);
        let (r, _) = accumulator.get_virtual_polynomial_opening(
            self.params.helper.get_output_operand(),
            SumcheckId::Raf,
        );

        let (r_address, _) = r.split_at_r(LOG_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(r_address);

        for (i, r_address) in r_address_chunks.iter().enumerate() {
            let opening_point = [r_address.as_slice(), r_cycle.r.as_slice()].concat();

            accumulator.append_sparse(
                transcript,
                vec![self.params.helper.rad_poly(i)],
                SumcheckId::RaVirtualization,
                opening_point,
            );
        }
    }
}
