use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            BIG_ENDIAN, LITTLE_ENDIAN, OpeningAccumulator, OpeningId, OpeningPoint,
            ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding, ReadRafProvider},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    claim::{Claim, CommittedOpeningClaim, Shape, TensorId},
    error::{ProverError, Result},
};

pub const ROUND_FRAC_BITS: usize = 8;

// Shout-backed fixed-point rounding.
//
// The low 8 bits are treated as a single lookup index:
//
//   rem       = x mod 2^8
//   round_bit = ROUND_LUT_Q8[rem]  // 0 for rem < 128, 1 otherwise
//   x + round_bit * 2^8 - rem = y * 2^8
//
// Shout proves `rem -> round_bit`, including the read-address onehot checks, so
// the public API no longer returns eight frac-bit claims.

pub(crate) const ROUND_LUT_LEN: usize = 1 << ROUND_FRAC_BITS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundParams {
    pub shape: Shape,
    pub input_tensor: TensorId,
    pub output_tensor: TensorId,
    pub frac_bit_tensors: [TensorId; ROUND_FRAC_BITS],
    pub frac_bits: usize,
    pub lookup_site: usize,
}

impl RoundParams {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        input_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
    ) -> Self {
        let input_tensor = input_tensor.into();
        Self {
            shape: Shape::new(shape),
            input_tensor: TensorId::new(input_tensor.clone()),
            output_tensor: TensorId::new(output_tensor),
            frac_bit_tensors: std::array::from_fn(|idx| {
                TensorId::new(format!("{input_tensor}_frac_bit_{idx}"))
            }),
            frac_bits: ROUND_FRAC_BITS,
            lookup_site: 0,
        }
    }

    pub fn with_frac_bit_tensors(
        shape: impl Into<Vec<usize>>,
        input_tensor: impl Into<String>,
        output_tensor: impl Into<String>,
        frac_bit_tensors: [String; ROUND_FRAC_BITS],
    ) -> Self {
        Self {
            shape: Shape::new(shape),
            input_tensor: TensorId::new(input_tensor),
            output_tensor: TensorId::new(output_tensor),
            frac_bit_tensors: frac_bit_tensors.map(TensorId::new),
            frac_bits: ROUND_FRAC_BITS,
            lookup_site: 0,
        }
    }

    pub fn with_lookup_site(mut self, lookup_site: usize) -> Self {
        self.lookup_site = lookup_site;
        self
    }
}

#[derive(Debug, Clone)]
pub struct RoundWitness<'a> {
    pub input: &'a [i64],
    pub output: &'a [i32],
    pub remainder: Vec<usize>,
    pub round_bit: Vec<i32>,
}

impl<'a> RoundWitness<'a> {
    pub fn from_input_output(input: &'a [i64], output: &'a [i32]) -> Self {
        let remainder = input
            .iter()
            .map(|value| value.rem_euclid(ROUND_LUT_LEN as i64) as usize)
            .collect::<Vec<_>>();
        let round_bit = remainder
            .iter()
            .map(|&rem| round_lut_q8(rem))
            .collect::<Vec<_>>();
        Self {
            input,
            output,
            remainder,
            round_bit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoundProof<F: JoltField, T: Transcript> {
    pub relation: SumcheckInstanceProof<F, T>,
    pub read_raf: SumcheckInstanceProof<F, T>,
    pub ra_onehot: SumcheckInstanceProof<F, T>,
    pub input_opening: F,
    pub remainder_opening: F,
    pub round_bit_opening: F,
    pub ra_opening: F,
    pub ra_committed_openings: RoundRaCommittedOpenings<F>,
}

impl<F: JoltField, T: Transcript> RoundProof<F, T> {
    pub fn committed_opening_claims(&self) -> Vec<CommittedOpeningClaim<F>> {
        self.ra_committed_openings.all().cloned().collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RoundRaCommittedOpenings<F: JoltField> {
    pub ra_virtual: Vec<CommittedOpeningClaim<F>>,
    pub hamming_weight: Vec<CommittedOpeningClaim<F>>,
    pub booleanity: Vec<CommittedOpeningClaim<F>>,
}

impl<F: JoltField> RoundRaCommittedOpenings<F> {
    pub fn all(&self) -> impl Iterator<Item = &CommittedOpeningClaim<F>> {
        self.ra_virtual
            .iter()
            .chain(self.hamming_weight.iter())
            .chain(self.booleanity.iter())
    }
}

pub fn prove_round<F, T>(
    output_claims: Vec<Claim<F>>,
    witness: &RoundWitness<'_>,
    params: &RoundParams,
    transcript: &mut T,
) -> Result<(RoundProof<F, T>, Claim<F>, Vec<CommittedOpeningClaim<F>>)>
where
    F: JoltField,
    T: Transcript,
{
    validate_inputs(&output_claims, witness, params)?;

    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let relation_claim = round_scale::<F>() * batched_input_claim(&output_claims, &alphas);
    let eq_batch = batched_eq_poly(&output_claims, &alphas);
    let input_poly = padded_i64_tensor(witness.input, &params.shape);
    let remainder_poly = padded_usize_tensor(&witness.remainder, &params.shape);
    let round_bit_poly = padded_i32_tensor(&witness.round_bit, &params.shape);

    let relation_params = RoundRelationParams::new(
        params.shape.padded_power_of_two().point_len(),
        relation_claim,
    );
    let mut relation_prover = RoundRelationProver::new(
        relation_params,
        eq_batch,
        input_poly,
        remainder_poly,
        round_bit_poly,
    );
    let mut accumulator = ProverOpeningAccumulator::new();
    let (relation, relation_challenges) =
        Sumcheck::prove(&mut relation_prover, &mut accumulator, transcript);
    let input_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
    )?;
    let remainder_opening = prover_opening(
        &accumulator,
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            round_relation_sumcheck_id(),
        ),
    )?;
    let round_bit_opening = prover_opening(
        &accumulator,
        OpeningId::new(VirtualPoly::QwenRoundLut, round_relation_sumcheck_id()),
    )?;
    let relation_point = normalize_sumcheck_point::<F>(&relation_challenges.into_opening());

    let lookup_indices = padded_lookup_indices(&witness.remainder, &params.shape);
    let table = round_lut_table();
    let shout = prove_round_lookup(
        params.lookup_site,
        relation_point.clone(),
        round_bit_opening,
        remainder_opening,
        lookup_indices,
        table,
        &mut accumulator,
        transcript,
    )?;

    let ra_claims = shout.committed_openings.all().cloned().collect();

    Ok((
        RoundProof {
            relation,
            read_raf: shout.read_raf,
            ra_onehot: shout.ra_onehot,
            input_opening,
            remainder_opening,
            round_bit_opening,
            ra_opening: shout.ra_opening,
            ra_committed_openings: shout.committed_openings,
        },
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: relation_point.clone(),
            value: input_opening,
        },
        ra_claims,
    ))
}

pub fn verify_round<F, T>(
    output_claims: Vec<Claim<F>>,
    proof: &RoundProof<F, T>,
    params: &RoundParams,
    transcript: &mut T,
) -> std::result::Result<(Claim<F>, Vec<CommittedOpeningClaim<F>>), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_inputs(&output_claims, params)?;

    let alphas = transcript.challenge_scalar_powers(output_claims.len());
    let relation_claim = round_scale::<F>() * batched_input_claim(&output_claims, &alphas);
    let y_points = output_claims
        .iter()
        .map(|claim| claim.point.clone())
        .collect::<Vec<_>>();
    let relation_verifier = RoundRelationVerifier {
        params: RoundRelationParams::new(
            params.shape.padded_power_of_two().point_len(),
            relation_claim,
        ),
        y_points,
        alphas,
    };
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.input_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            round_relation_sumcheck_id(),
        ),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.remainder_opening,
        ),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, round_relation_sumcheck_id()),
        (
            OpeningPoint::<BIG_ENDIAN, F>::default(),
            proof.round_bit_opening,
        ),
    );
    let relation_challenges = Sumcheck::verify(
        &proof.relation,
        &relation_verifier,
        &mut accumulator,
        transcript,
    )?;
    let relation_point = normalize_sumcheck_point::<F>(&relation_challenges.into_opening());

    let shout = verify_round_lookup(
        params.lookup_site,
        params.shape.padded_power_of_two().numel(),
        relation_point.clone(),
        proof.round_bit_opening,
        proof.remainder_opening,
        proof.ra_opening,
        &proof.ra_committed_openings,
        &proof.read_raf,
        &proof.ra_onehot,
        &mut accumulator,
        transcript,
    )?;

    Ok((
        Claim {
            tensor: params.input_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            point: relation_point,
            value: proof.input_opening,
        },
        shout.committed_opening_claims(),
    ))
}

struct RoundRelationParams<F: JoltField> {
    num_rounds: usize,
    input_claim: F,
}

impl<F: JoltField> RoundRelationParams<F> {
    fn new(num_rounds: usize, input_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RoundRelationParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

struct RoundRelationProver<F: JoltField> {
    eq_batch: MultilinearPolynomial<F>,
    input: MultilinearPolynomial<F>,
    remainder: MultilinearPolynomial<F>,
    round_bit: MultilinearPolynomial<F>,
    params: RoundRelationParams<F>,
}

impl<F: JoltField> RoundRelationProver<F> {
    fn new(
        params: RoundRelationParams<F>,
        eq_batch: Vec<F>,
        input: Vec<F>,
        remainder: Vec<F>,
        round_bit: Vec<F>,
    ) -> Self {
        Self {
            eq_batch: MultilinearPolynomial::from(eq_batch),
            input: MultilinearPolynomial::from(input),
            remainder: MultilinearPolynomial::from(remainder),
            round_bit: MultilinearPolynomial::from(round_bit),
            params,
        }
    }

    fn relation_at(&self, idx: usize) -> F {
        self.input.get_bound_coeff(idx) + self.round_bit.get_bound_coeff(idx) * round_scale::<F>()
            - self.remainder.get_bound_coeff(idx)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RoundRelationProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let mut eval_at_0 = F::zero();
        let mut eval_at_2 = F::zero();
        for g in 0..self.input.len() / 2 {
            let e0 = self.eq_batch.get_bound_coeff(2 * g);
            let e1 = self.eq_batch.get_bound_coeff(2 * g + 1);
            let l0 = self.relation_at(2 * g);
            let l1 = self.relation_at(2 * g + 1);
            eval_at_0 += e0 * l0;
            eval_at_2 += lerp(e0, e1, F::from_u64(2)) * lerp(l0, l1, F::from_u64(2));
        }
        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_batch.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.remainder.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round_bit.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
            point.clone(),
            self.input.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                round_relation_sumcheck_id(),
            ),
            point.clone(),
            self.remainder.final_claim(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, round_relation_sumcheck_id()),
            point,
            self.round_bit.final_claim(),
        );
    }
}

struct RoundRelationVerifier<F: JoltField> {
    params: RoundRelationParams<F>,
    y_points: Vec<Vec<F>>,
    alphas: Vec<F>,
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RoundRelationVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let input = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRebaseAcc,
                round_relation_sumcheck_id(),
            ))
            .1;
        let remainder = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                round_relation_sumcheck_id(),
            ))
            .1;
        let round_bit = accumulator
            .get_virtual_polynomial_opening(OpeningId::new(
                VirtualPoly::QwenRoundLut,
                round_relation_sumcheck_id(),
            ))
            .1;
        let point = normalize_sumcheck_point::<F>(&sumcheck_challenges.into_opening());
        let mut eq_batch = F::zero();
        for (alpha, y_point) in self.alphas.iter().zip(&self.y_points) {
            eq_batch += *alpha * EqPolynomial::mle(y_point, &point);
        }
        eq_batch * (input + round_bit * round_scale::<F>() - remainder)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRebaseAcc, round_relation_sumcheck_id()),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(
                VirtualPoly::QwenRoundRemainder,
                round_relation_sumcheck_id(),
            ),
            point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            OpeningId::new(VirtualPoly::QwenRoundLut, round_relation_sumcheck_id()),
            point,
        );
    }
}

#[derive(Clone)]
struct RoundReadRafProvider<F: JoltField> {
    log_k: usize,
    r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    rv_claim: F,
    raf_claim: F,
}

impl<F: JoltField> ReadRafProvider<F> for RoundReadRafProvider<F> {
    fn rv_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.rv_claim
    }

    fn raf_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.raf_claim
    }

    fn r(&self, _accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        self.r_cycle.clone()
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (VirtualPoly::QwenRoundRa, round_sumcheck_id())
    }

    fn log_K(&self) -> usize {
        self.log_k
    }
}

struct RoundRaEncoding {
    log_k: usize,
    lookup_site: usize,
}

impl RaOneHotEncoding for RoundRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::QwenRoundRaD(self.lookup_site, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(VirtualPoly::QwenRoundLut, round_relation_sumcheck_id())
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(VirtualPoly::QwenRoundRa, round_sumcheck_id())
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RoundLookupProof<F: JoltField, T: Transcript> {
    pub(crate) read_raf: SumcheckInstanceProof<F, T>,
    pub(crate) ra_onehot: SumcheckInstanceProof<F, T>,
    pub(crate) ra_opening: F,
    pub(crate) committed_openings: RoundRaCommittedOpenings<F>,
}

impl<F: JoltField, T: Transcript> RoundLookupProof<F, T> {
    pub(crate) fn committed_opening_claims(&self) -> Vec<CommittedOpeningClaim<F>> {
        self.committed_openings.all().cloned().collect()
    }
}

pub(crate) fn prove_round_lookup<F, T>(
    lookup_site: usize,
    tensor_point: Vec<F>,
    round_bit_opening: F,
    remainder_opening: F,
    lookup_indices: Vec<usize>,
    table: Vec<i32>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<RoundLookupProof<F, T>>
where
    F: JoltField,
    T: Transcript,
{
    let log_k = ROUND_LUT_LEN.trailing_zeros() as usize;
    let r_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(tensor_point.clone());
    let provider = RoundReadRafProvider {
        log_k,
        r_cycle,
        rv_claim: round_bit_opening,
        raf_claim: remainder_opening,
    };
    let mut read_prover =
        shout::read_raf_prover(&provider, &lookup_indices, &table, accumulator, transcript);
    let (read_raf, _) = Sumcheck::prove(&mut *read_prover, accumulator, transcript);

    let encoding = RoundRaEncoding { log_k, lookup_site };
    let [ra_prover, hw_prover, bool_prover] =
        shout::ra_onehot_provers(&encoding, &lookup_indices, accumulator, transcript);
    let use_ra_virtual = lookup_indices.len().next_power_of_two() >= 8;
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = if !use_ra_virtual {
        vec![hw_prover]
    } else {
        vec![ra_prover, hw_prover, bool_prover]
    };
    let (ra_onehot, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        accumulator,
        transcript,
    );
    let (_ra_point, ra_opening) = accumulator.get_virtual_polynomial_opening(OpeningId::new(
        VirtualPoly::QwenRoundRa,
        round_sumcheck_id(),
    ));
    let committed_openings =
        round_ra_committed_openings(lookup_site, log_k, use_ra_virtual, accumulator)?;
    Ok(RoundLookupProof {
        read_raf,
        ra_onehot,
        ra_opening,
        committed_openings,
    })
}

pub(crate) fn verify_round_lookup<F, T>(
    lookup_site: usize,
    logical_len: usize,
    tensor_point: Vec<F>,
    round_bit_opening: F,
    remainder_opening: F,
    ra_opening: F,
    committed_openings: &RoundRaCommittedOpenings<F>,
    read_raf: &SumcheckInstanceProof<F, T>,
    ra_onehot: &SumcheckInstanceProof<F, T>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> std::result::Result<RoundLookupProof<F, T>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let log_k = ROUND_LUT_LEN.trailing_zeros() as usize;
    let r_cycle = OpeningPoint::<BIG_ENDIAN, F>::new(tensor_point.clone());
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundLut, round_relation_sumcheck_id()),
        (r_cycle.clone(), round_bit_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(
            VirtualPoly::QwenRoundRemainder,
            round_relation_sumcheck_id(),
        ),
        (r_cycle.clone(), remainder_opening),
    );
    accumulator.openings.insert(
        OpeningId::new(VirtualPoly::QwenRoundRa, round_sumcheck_id()),
        (OpeningPoint::<BIG_ENDIAN, F>::default(), ra_opening),
    );
    let provider = RoundReadRafProvider {
        log_k,
        r_cycle,
        rv_claim: round_bit_opening,
        raf_claim: remainder_opening,
    };
    let read_verifier =
        shout::read_raf_verifier(&provider, round_lut_table(), accumulator, transcript);
    Sumcheck::verify(read_raf, &*read_verifier, accumulator, transcript)?;

    let encoding = RoundRaEncoding { log_k, lookup_site };
    let [ra_verifier, hw_verifier, bool_verifier] =
        shout::ra_onehot_verifiers(&encoding, accumulator, transcript);
    let use_ra_virtual = logical_len.next_power_of_two() >= 8;
    insert_round_ra_committed_openings(
        lookup_site,
        use_ra_virtual,
        committed_openings,
        accumulator,
    )?;
    let verifier_instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = if !use_ra_virtual {
        vec![&*hw_verifier]
    } else {
        vec![&*ra_verifier, &*hw_verifier, &*bool_verifier]
    };
    BatchedSumcheck::verify(ra_onehot, verifier_instances, accumulator, transcript)?;
    let (_ra_point, ra_opening) = accumulator.get_virtual_polynomial_opening(OpeningId::new(
        VirtualPoly::QwenRoundRa,
        round_sumcheck_id(),
    ));
    Ok(RoundLookupProof {
        read_raf: read_raf.clone(),
        ra_onehot: ra_onehot.clone(),
        ra_opening,
        committed_openings: committed_openings.clone(),
    })
}

fn round_ra_committed_openings<F: JoltField>(
    lookup_site: usize,
    log_k: usize,
    include_full_checks: bool,
    accumulator: &ProverOpeningAccumulator<F>,
) -> Result<RoundRaCommittedOpenings<F>> {
    let d = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
    let collect = |sumcheck| -> Vec<CommittedOpeningClaim<F>> {
        (0..d)
            .map(|idx| {
                let poly = CommittedPoly::QwenRoundRaD(lookup_site, idx);
                let (point, value) =
                    accumulator.get_committed_polynomial_opening(OpeningId::new(poly, sumcheck));
                CommittedOpeningClaim {
                    poly,
                    sumcheck,
                    point: point.r,
                    value,
                    sparse: true,
                }
            })
            .collect()
    };
    Ok(RoundRaCommittedOpenings {
        ra_virtual: if include_full_checks {
            collect(SumcheckId::RaVirtualization)
        } else {
            vec![]
        },
        hamming_weight: collect(SumcheckId::HammingWeight),
        booleanity: if include_full_checks {
            collect(SumcheckId::Booleanity)
        } else {
            vec![]
        },
    })
}

fn insert_round_ra_committed_openings<F: JoltField>(
    lookup_site: usize,
    include_full_checks: bool,
    openings: &RoundRaCommittedOpenings<F>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
) -> std::result::Result<(), ProofVerifyError> {
    let log_k = ROUND_LUT_LEN.trailing_zeros() as usize;
    let d = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
    if openings.hamming_weight.len() != d {
        return Err(ProofVerifyError::InvalidInputLength(
            d,
            openings.hamming_weight.len(),
        ));
    }
    if include_full_checks && (openings.ra_virtual.len() != d || openings.booleanity.len() != d) {
        return Err(ProofVerifyError::InvalidInputLength(
            d,
            openings.ra_virtual.len().min(openings.booleanity.len()),
        ));
    }
    let mut insert_group = |sumcheck, values: &[CommittedOpeningClaim<F>]| {
        for (idx, opening) in values.iter().enumerate() {
            accumulator.openings.insert(
                OpeningId::new(CommittedPoly::QwenRoundRaD(lookup_site, idx), sumcheck),
                (
                    OpeningPoint::<BIG_ENDIAN, F>::new(opening.point.clone()),
                    opening.value,
                ),
            );
        }
    };
    if include_full_checks {
        insert_group(SumcheckId::RaVirtualization, &openings.ra_virtual);
    }
    insert_group(SumcheckId::HammingWeight, &openings.hamming_weight);
    if include_full_checks {
        insert_group(SumcheckId::Booleanity, &openings.booleanity);
    }
    Ok(())
}

fn validate_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    witness: &RoundWitness<'_>,
    params: &RoundParams,
) -> Result<()> {
    if params.frac_bits != ROUND_FRAC_BITS {
        return Err(ProverError::InvalidSumcheckDomain(params.frac_bits));
    }
    if output_claims.is_empty() {
        return Err(ProverError::InvalidSumcheckDomain(0));
    }
    let expected = params.shape.numel();
    ensure_len("input", expected, witness.input.len())?;
    ensure_len("output", expected, witness.output.len())?;
    ensure_len("remainder", expected, witness.remainder.len())?;
    ensure_len("round_bit", expected, witness.round_bit.len())?;
    for idx in 0..expected {
        let rem = witness.input[idx].rem_euclid(ROUND_LUT_LEN as i64) as usize;
        if witness.remainder[idx] != rem {
            return Err(ProverError::InvalidSumcheckDomain(rem));
        }
        if witness.round_bit[idx] != round_lut_q8(rem) {
            return Err(ProverError::InvalidSumcheckDomain(rem));
        }
        let lhs =
            witness.input[idx] + i64::from(witness.round_bit[idx]) * round_scale_i64() - rem as i64;
        let rhs = i64::from(witness.output[idx]) * round_scale_i64();
        if lhs != rhs {
            return Err(ProverError::InvalidSumcheckDomain(idx));
        }
    }
    for claim in output_claims {
        if claim.logical_shape != params.shape {
            return Err(ProverError::InvalidSumcheckDomain(params.shape.numel()));
        }
    }
    Ok(())
}

fn verify_inputs<F: JoltField>(
    output_claims: &[Claim<F>],
    params: &RoundParams,
) -> std::result::Result<(), ProofVerifyError> {
    if params.frac_bits != ROUND_FRAC_BITS {
        return Err(ProofVerifyError::InvalidInputLength(
            ROUND_FRAC_BITS,
            params.frac_bits,
        ));
    }
    if output_claims.is_empty() {
        return Err(ProofVerifyError::InvalidInputLength(1, 0));
    }
    for claim in output_claims {
        if claim.logical_shape != params.shape {
            return Err(ProofVerifyError::InvalidInputLength(
                params.shape.numel(),
                claim.logical_shape.numel(),
            ));
        }
    }
    Ok(())
}

fn batched_input_claim<F: JoltField>(claims: &[Claim<F>], alphas: &[F]) -> F {
    claims
        .iter()
        .zip(alphas)
        .map(|(claim, alpha)| claim.value * *alpha)
        .sum()
}

fn batched_eq_poly<F: JoltField>(claims: &[Claim<F>], alphas: &[F]) -> Vec<F> {
    let len = claims[0].domain_shape.numel();
    let mut out = vec![F::zero(); len];
    for (claim, alpha) in claims.iter().zip(alphas) {
        let eq = EqPolynomial::<F>::evals(&claim.point);
        for (out, eq) in out.iter_mut().zip(eq) {
            *out += *alpha * eq;
        }
    }
    out
}

fn padded_i64_tensor<F: JoltField>(values: &[i64], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = field_from_i64(value);
    }
    out
}

fn padded_i32_tensor<F: JoltField>(values: &[i32], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = field_from_i64(i64::from(value));
    }
    out
}

fn padded_usize_tensor<F: JoltField>(values: &[usize], shape: &Shape) -> Vec<F> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![F::zero(); len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = F::from_u64(value as u64);
    }
    out
}

pub(crate) fn padded_lookup_indices(values: &[usize], shape: &Shape) -> Vec<usize> {
    let padded_dims = shape.padded_power_of_two().0;
    let len = padded_dims.iter().product();
    let mut out = vec![0; len];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = value;
    }
    out
}

pub(crate) fn round_lut_table() -> Vec<i32> {
    (0..ROUND_LUT_LEN).map(round_lut_q8).collect()
}

fn round_lut_q8(rem: usize) -> i32 {
    if rem >= ROUND_LUT_LEN / 2 { 1 } else { 0 }
}

fn ensure_len(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(ProverError::TensorLenMismatch {
            name,
            shape: vec![expected],
            expected,
            actual,
        });
    }
    Ok(())
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    let mut stride = 1;
    for (idx, &dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride *= dim;
    }
    strides
}

fn prover_opening<F: JoltField>(
    accumulator: &ProverOpeningAccumulator<F>,
    id: OpeningId,
) -> Result<F> {
    accumulator
        .openings
        .get(&id)
        .map(|(_, value)| *value)
        .ok_or(ProverError::MissingOpening)
}

fn normalize_sumcheck_point<F: JoltField>(challenges: &[F]) -> Vec<F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec())
        .match_endianness::<BIG_ENDIAN>()
        .r
}

fn field_from_i64<F: JoltField>(value: i64) -> F {
    if value >= 0 {
        F::from_u64(value as u64)
    } else {
        -F::from_u64(value.unsigned_abs())
    }
}

fn lerp<F: JoltField>(v0: F, v1: F, t: F) -> F {
    v0 + t * (v1 - v0)
}

fn round_scale<F: JoltField>() -> F {
    F::from_u64(round_scale_i64() as u64)
}

fn round_scale_i64() -> i64 {
    1_i64 << ROUND_FRAC_BITS
}

fn round_relation_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(0)
}

fn round_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(1)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::{
        field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript,
    };

    use super::*;

    #[test]
    fn proves_and_verifies_round() {
        let params = RoundParams::new(vec![2, 3], "round_acc", "round_y");
        let input = vec![-300_i64, -129, -128, -1, 0, 383];
        let output = input
            .iter()
            .map(|&x| ((x + ((x.rem_euclid(256) >> 7) * 256) - x.rem_euclid(256)) / 256) as i32)
            .collect::<Vec<_>>();
        let witness = RoundWitness::from_input_output(&input, &output);
        let point = vec![Fr::from(7_u64), Fr::from(11_u64), Fr::from(13_u64)];
        let output_claim = Claim {
            tensor: params.output_tensor.clone(),
            logical_shape: params.shape.clone(),
            domain_shape: params.shape.padded_power_of_two(),
            value: eval_i32(&output, &params.shape, &point),
            point,
        };

        let mut prover_transcript = Blake2bTranscript::default();
        let (proof, input_claim, _) = prove_round::<Fr, _>(
            vec![output_claim.clone()],
            &witness,
            &params,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let (verified_input, _) = verify_round::<Fr, _>(
            vec![output_claim],
            &proof,
            &params,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(verified_input, input_claim);
    }

    fn eval_i32<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
        let padded: Vec<F> = padded_i32_tensor(values, shape);
        let eq = EqPolynomial::<F>::evals(point);
        padded
            .iter()
            .zip(eq)
            .fold(F::zero(), |acc, (value, eq)| acc + *value * eq)
    }
}
