use std::array;

use crate::onnx_proof::{
    lookup_tables::{
        prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
        unsigned_less_than::UnsignedLessThanTable,
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
    op_lookups::LOG_K,
    range_checking::sumcheck_instance::ReadRafSumcheckHelper,
    Prover, Verifier,
};
use ark_std::Zero;
use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode, tensor::Tensor};
use common::consts::XLEN;
use itertools::Itertools;
use joltworks::{
    field::{JoltField, MulTrunc},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        interleave_bits,
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use rayon::prelude::*;
use strum::EnumCount;

const DEGREE_BOUND: usize = 2;

pub struct RangecheckRafSumcheckParams<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper,
{
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    pub gamma: F,
    pub gamma_sqr: F,
    /// log2(T): number of variables in the node output polynomial (last rounds bind input).
    pub log_T: usize,
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    /// Table for this node
    table: UnsignedLessThanTable<XLEN>,
    helper: Helper,
}

impl<F, Helper> RangecheckRafSumcheckParams<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper,
{
    pub fn new(
        computation_node: ComputationNode,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let helper = Helper::new(&computation_node);

        let (r_node_output, _) = opening_accumulator
            .get_virtual_polynomial_opening(helper.get_input_operands()[0], SumcheckId::Execution);
        let log_T = computation_node.num_output_elements().log_2();

        Self {
            gamma,
            gamma_sqr,
            log_T,
            r_node_output,
            computation_node,
            table: UnsignedLessThanTable::<XLEN>,
            helper,
        }
    }
}

impl<F, Helper> SumcheckInstanceParams<F> for RangecheckRafSumcheckParams<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper,
{
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let raw_claims = self
            .helper
            .get_input_operands()
            .iter()
            .map(|operand| {
                let (_, claim) =
                    accumulator.get_virtual_polynomial_opening(*operand, SumcheckId::Raf);
                claim
            })
            .collect::<Vec<_>>();

        // rv is the 1-polynomial: we expect the LessThan lookup to always return true (1)
        Helper::compute_input_claim(&raw_claims, self.gamma, self.gamma_sqr)
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_node_output_prime) = challenges.split_at(LOG_K);
        let r_node_output_prime = r_node_output_prime
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();

        OpeningPoint::new([r_address_prime.to_vec(), r_node_output_prime].concat())
    }
}

pub struct RangecheckRafSumcheckProver<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper,
{
    params: RangecheckRafSumcheckParams<F, Helper>,
    /// Materialized `ra(k, j)` MLE over (address, cycle) after the first log(K) rounds.
    /// Present only in the last log(T) rounds.
    ra: Option<MultilinearPolynomial<F>>,
    /// Running list of sumcheck challenges r_j (address then cycle) in binding order.
    r: Vec<F::Challenge>,
    /// u_evals for read-checking and RAF: eq(r_node_output, j).
    u_evals: Vec<F>,
    /// Prefix checkpoints for each registered `Prefix` variant, updated every two rounds.
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    /// For each lookup table, dense polynomials holding suffix contributions in the current phase.
    suffix_polys: Vec<DensePolynomial<F>>,
    /// Precomputed lookup keys k (bit-packed) per cycle j.
    lookup_indices: Vec<LookupBits>,
    /// Val(r_address)
    val: Option<F>,
    /// Val(r_address)
    raf_val: Option<F>,
    /// number of phases in the first log K rounds
    phases: usize,
    /// Expanding tables accumulating address-prefix products per phase.
    v: Vec<ExpandingTable<F>>,
    /// Gruen-split equality polynomial over cycle vars.
    eq_r_node_output: GruenSplitEqPolynomial<F>,

    // --- RAF stuff ---
    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for right operand identity polynomial family.
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for left operand identity polynomial family.
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
}

impl<F, Helper> RangecheckRafSumcheckProver<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper + Sync,
{
    pub fn new_from_prover(
        node: &ComputationNode,
        prover: &mut Prover<F, impl Transcript>,
    ) -> Self {
        let params = RangecheckRafSumcheckParams::new(
            node.clone(),
            &prover.accumulator,
            &mut prover.transcript,
        );
        Self::initialize(
            params,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
    }

    pub fn initialize(
        params: RangecheckRafSumcheckParams<F, Helper>,
        trace: &Trace,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (left_operand, right_operand) =
            Helper::get_operands_tensors(trace, &params.computation_node);

        let [left_operand_virtual_poly, right_operand_virtual_poly] =
            params.helper.get_input_operands()[..]
        else {
            panic!("Expected exactly two input operands for RAF range check");
        };

        // Cache left/right operand claims. (In our case they have been cached in Execution sumcheck)
        let left_claim = MultilinearPolynomial::from(left_operand.into_container_data()) // TODO: make this work with from_i32
            .evaluate(&params.r_node_output.r); // TODO: rm these clones
        opening_accumulator.append_virtual(
            transcript,
            left_operand_virtual_poly,
            SumcheckId::Raf,
            params.r_node_output.clone(),
            left_claim,
        );
        let right_claim = MultilinearPolynomial::from(right_operand.into_container_data())
            .evaluate(&params.r_node_output.r);
        opening_accumulator.append_virtual(
            transcript,
            right_operand_virtual_poly,
            SumcheckId::Raf,
            params.r_node_output.clone(),
            right_claim,
        );

        Self::new_prover(params, &left_operand, &right_operand)
    }

    /// Creates a new prover instance for the "LessThan" lookup table with RAF.
    fn new_prover(
        params: RangecheckRafSumcheckParams<F, Helper>,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
    ) -> Self {
        let phases = 8;
        let log_m = LOG_K / phases;
        let u_evals = EqPolynomial::evals(&params.r_node_output.r);
        let prefix_checkpoints = vec![None.into(); Prefixes::COUNT];
        let suffix_polys: Vec<DensePolynomial<F>> = params
            .table
            .suffixes()
            .iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|_| DensePolynomial::default())
            .collect();

        let lookup_indices = Helper::compute_lookup_indices(left_operand, right_operand);
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);

        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), log_m, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), log_m, LOG_K);
        drop(_guard);
        drop(span);
        let eq_r_node_output =
            GruenSplitEqPolynomial::<F>::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        // TODO: Adjust [PrefixSuffixDecomposition::init_Q_dual] and [PrefixSuffixDecomposition::init_Q] to be compatible with jolt-atlas usage

        let mut res = Self {
            r: Vec::with_capacity(params.log_T + LOG_K),
            params,
            phases,

            // Prefix-suffix state (first log(K) rounds)
            lookup_indices,
            prefix_checkpoints,
            suffix_polys,
            u_evals,
            v: (0..phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            // raf
            right_operand_ps,
            left_operand_ps,

            // State for last log(T) rounds
            eq_r_node_output,
            prefix_registry: PrefixRegistry::new(),
            val: None,
            ra: None,
            raf_val: None,
        };
        res.init_phase(0);
        res
    }

    fn init_phase(&mut self, phase: usize) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let T = 1 << self.params.log_T;

        // Condensation: update u evals
        if phase != 0 {
            self.lookup_indices
                .par_iter()
                .zip(&mut self.u_evals)
                .for_each(|(k, u_eval)| {
                    let (prefix, _) = k.split((self.phases - phase) * log_m);
                    let k_bound = prefix & m_mask;
                    *u_eval *= self.v[phase - 1][k_bound];
                });
        }

        rayon::scope(|s| {
            // Single pass over lookup_indices_uninterleave for both operands
            s.spawn(|_| {
                PrefixSuffixDecomposition::init_Q_dual(
                    &mut self.left_operand_ps,
                    &mut self.right_operand_ps,
                    &self.u_evals,
                    &(0..T).collect::<Vec<_>>(),
                    &self.lookup_indices,
                )
            });
        });

        self.init_suffix_polys(phase);

        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v[phase].reset(F::one());
    }

    /// Recomputes per-table suffix accumulators used by read-checking for the
    /// current phase. For each table's suffix family, bucket cycles by the
    /// current chunk value and aggregate weighted contributions into Dense MLEs
    /// of size M = 2^{log_m}.
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let suffix_len = (self.phases - 1 - phase) * log_m;
        let new_suffix_polys = self
            .params
            .table
            .suffixes()
            .par_iter()
            .map(|s| {
                let mut Q = unsafe_allocate_zero_vec(m);
                self.lookup_indices.iter().enumerate().for_each(|(j, &k)| {
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);
                    let y = prefix_bits & m_mask;
                    let t = s.suffix_mle::<XLEN>(suffix_bits);
                    if t != 0 {
                        Q[y] += self.u_evals[j] * F::from_u32(t)
                    };
                });
                Q
            })
            .collect::<Vec<_>>();

        // Replace existing suffix polynomials
        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(poly, mut coeffs)| {
                *poly = DensePolynomial::new(std::mem::take(&mut coeffs))
            });
    }

    /// Address-round prover message: sum of read-checking and RAF components.
    ///
    /// Each component is a degree-2 univariate evaluated at X∈{0,2} using
    /// prefix–suffix decomposition, then added to form the batched message.
    fn compute_prefix_suffix_prover_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || {
                read_checking = self.prover_msg_read_checking(round);
            },
            || {
                raf = self.prover_msg_raf();
            },
        );

        let eval_at_0 = read_checking[0] + raf[0];
        let eval_at_2 = read_checking[1] + raf[1];

        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let r_x = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };
        let half_poly_len = self.suffix_polys[0].len() / 2;
        let [eval_0, eval_2_low, eval_2_high] = (0..half_poly_len)
            .into_par_iter()
            .map(|i| {
                let b = LookupBits::new(i as u64, half_poly_len.log_2());
                let prefix_evals_0 = self
                    .params
                    .table
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect_vec();
                let prefix_evals_2 = self
                    .params
                    .table
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect_vec();
                let suffix_evals_low = self.suffix_polys.iter().map(|s| s[i]).collect_vec();
                let suffix_evals_high = self
                    .suffix_polys
                    .iter()
                    .map(|s| s[i + half_poly_len])
                    .collect_vec();
                [
                    self.params
                        .table
                        .combine(&prefix_evals_0, &suffix_evals_low),
                    self.params
                        .table
                        .combine(&prefix_evals_2, &suffix_evals_low),
                    self.params
                        .table
                        .combine(&prefix_evals_2, &suffix_evals_high),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        [eval_0, eval_2_high + eval_2_high - eval_2_low]
    }

    /// RAF part for address rounds.
    ///
    /// Builds two evaluations at X∈{0,2} for the batched
    /// (Left + γ·Right) vs Identity path, folding γ-weights into the result.
    fn prover_msg_raf(&self) -> [F; 2] {
        let len = self.right_operand_ps.Q_len();
        let [left_0, left_2, right_0, right_2] = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let (r0, r2) = self.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = self.left_operand_ps.sumcheck_evals(b);
                [
                    *l0.as_unreduced_ref(),
                    *l2.as_unreduced_ref(),
                    *r0.as_unreduced_ref(),
                    *r2.as_unreduced_ref(),
                ]
            })
            .fold_with([F::Unreduced::<5>::zero(); 4], |running, new| {
                [
                    running[0] + new[0],
                    running[1] + new[1],
                    running[2] + new[2],
                    running[3] + new[3],
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 4],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                    ]
                },
            );
        [
            F::from_montgomery_reduce(
                left_0.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                    + right_0.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
            ),
            F::from_montgomery_reduce(
                left_2.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                    + right_2.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
            ),
        ]
    }

    /// To be called before the last log(T) rounds
    /// Handoff between address and cycle rounds:
    /// - Materializes ra(k,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Drop stuff that's no longer needed
        drop_in_background_thread((std::mem::take(&mut self.u_evals),));

        // Materialize ra polynomial
        let ra = {
            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomial");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .map(|k| {
                    (0..self.phases)
                        .map(|phase| {
                            let (prefix, _) = k.split((self.phases - 1 - phase) * log_m);
                            let k_bound = prefix & m_mask;
                            self.v[phase][k_bound]
                        })
                        .product::<F>()
                })
                .collect::<Vec<_>>()
        };

        drop_in_background_thread(std::mem::take(&mut self.v));
        self.ra = Some(ra.into());
    }
}

impl<F: JoltField, FS: Transcript, Helper: ReadRafSumcheckHelper + Send + Sync>
    SumcheckInstanceProver<F, FS> for RangecheckRafSumcheckProver<F, Helper>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::compute_message")]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQ.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < LOG_K {
            // Phase 1: First log(K) rounds
            self.compute_prefix_suffix_prover_message(round, previous_claim)
        } else {
            let ra = self.ra.as_ref().unwrap();
            let val = self.val.unwrap();
            let raf_val = self.raf_val.unwrap();
            let [eval_at_0] = self
                .eq_r_node_output
                .par_fold_out_in(
                    || [F::Unreduced::<9>::zero(); 1],
                    |inner, j, _x_in, e_in| {
                        let ra_at_0_j = ra.get_bound_coeff(2 * j);
                        inner[0] += e_in.mul_unreduced::<9>(ra_at_0_j);
                    },
                    |_x_out, e_out, inner| {
                        array::from_fn(|i| {
                            let reduced = F::from_montgomery_reduce(inner[i]);
                            e_out.mul_unreduced::<9>(reduced)
                        })
                    },
                    |a, b| array::from_fn(|i| a[i] + b[i]),
                )
                .map(F::from_montgomery_reduce);
            self.eq_r_node_output
                .gruen_poly_deg_2(eval_at_0 * (val + raf_val), previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::ingest_challenge")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQ.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let log_m = LOG_K / self.phases;
        self.r.push(r_j);
        if round < LOG_K {
            let phase = round / log_m;

            // Bind suffix polynomials & update v
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.suffix_polys
                        .par_iter_mut()
                        .for_each(|s| s.bind(r_j, BindingOrder::HighToLow));
                });
                s.spawn(|_| self.right_operand_ps.bind(r_j));
                s.spawn(|_| self.left_operand_ps.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });

            // update checkpoints
            {
                if self.r.len().is_multiple_of(2) {
                    // Calculate suffix_len based on phases, using the same formula as original current_suffix_len
                    let suffix_len = LOG_K - (round / log_m + 1) * log_m;
                    Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                        &mut self.prefix_checkpoints,
                        self.r[self.r.len() - 2],
                        self.r[self.r.len() - 1],
                        round,
                        suffix_len,
                    );
                }
            }

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(log_m) {
                self.prefix_registry.update_checkpoints();
                // if not last phase, init next phase
                if phase != self.phases - 1 {
                    self.init_phase(phase + 1);
                };
            }

            if (round + 1) == LOG_K {
                let prefixes: Vec<PrefixEval<F>> = self
                    .params
                    .table
                    .prefixes()
                    .into_iter()
                    .map(|p| self.prefix_checkpoints[p as usize].unwrap())
                    .collect();
                let suffixes: Vec<_> = self
                    .params
                    .table
                    .suffixes()
                    .iter()
                    .map(|suffix| F::from_u32(suffix.suffix_mle::<XLEN>(LookupBits::new(0, 0))))
                    .collect();
                self.val = Some(self.params.table.combine(&prefixes, &suffixes));
                let gamma = self.params.gamma;
                let gamma_sqr = self.params.gamma_sqr;
                let raf_val = gamma
                    * self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                    + gamma_sqr * self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();

                self.raf_val = Some(raf_val);
                self.init_log_t_rounds();
            }
        } else {
            self.ra
                .as_mut()
                .unwrap()
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            self.eq_r_node_output.bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut FS,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            self.params.helper.get_output_operand(),
            SumcheckId::Raf,
            opening_point,
            self.ra.as_ref().unwrap().final_sumcheck_claim(),
        );
    }
}

pub struct RangecheckRafSumcheckVerifier<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper,
{
    params: RangecheckRafSumcheckParams<F, Helper>,
}

impl<F, Helper> RangecheckRafSumcheckVerifier<F, Helper>
where
    F: JoltField,
    Helper: ReadRafSumcheckHelper,
{
    pub fn new_from_verifier(
        node: &ComputationNode,
        verifier: &mut Verifier<F, impl Transcript>,
    ) -> Self {
        let params = RangecheckRafSumcheckParams::new(
            node.clone(),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Self::new(params, &mut verifier.accumulator, &mut verifier.transcript)
    }

    pub fn new(
        params: RangecheckRafSumcheckParams<F, Helper>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let [left_operand, right_operand] = params.helper.get_input_operands()[..] else {
            panic!("Expected exactly two input operands for RAF range check");
        };
        // Update accumulator
        opening_accumulator.append_virtual(
            transcript,
            left_operand,
            SumcheckId::Raf,
            params.r_node_output.clone(),
        );
        opening_accumulator.append_virtual(
            transcript,
            right_operand,
            SumcheckId::Raf,
            params.r_node_output.clone(),
        );

        Self { params }
    }
}

impl<F: JoltField, FS: Transcript, Helper: ReadRafSumcheckHelper> SumcheckInstanceVerifier<F, FS>
    for RangecheckRafSumcheckVerifier<F, Helper>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_node_output_prime) = opening_point.split_at(LOG_K);
        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            self.params.helper.get_output_operand(),
            SumcheckId::Raf,
        );
        let val_claim = self.params.table.evaluate_mle(&r_address_prime.r);
        let eq_eval = EqPolynomial::mle(&self.params.r_node_output.r, &r_node_output_prime.r);

        // RAF
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(&r_address_prime.r);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(&r_address_prime.r);
        // let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(&r_address_prime.r);
        let raf_claim = left_operand_eval + self.params.gamma * right_operand_eval;

        eq_eval * ra_claim * (val_claim + self.params.gamma * raf_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut FS,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            self.params.helper.get_output_operand(),
            SumcheckId::Raf,
            opening_point,
        );
    }
}

pub fn compute_lookup_indices_from_operands(operand_tensors: &[&Tensor<i32>]) -> Vec<LookupBits> {
    // Interleaved mode: requires exactly 2 operand tensors
    assert_eq!(
        operand_tensors.len(),
        2,
        "Interleaved operands mode requires exactly 2 input tensors, but got {}",
        operand_tensors.len()
    );

    let left_operand = operand_tensors[0];
    let right_operand = operand_tensors[1];

    // Validate that both tensors have the same length
    assert_eq!(
        left_operand.len(),
        right_operand.len(),
        "Interleaved operands must have the same length: left={}, right={}",
        left_operand.len(),
        right_operand.len()
    );

    // Interleave bits from both operands to form lookup indices
    left_operand
        .data()
        .par_iter()
        .zip(right_operand.data().par_iter())
        .map(|(&left_val, &right_val)| {
            // Cast to u64 for interleaving
            let left_bits = left_val as u32;
            let right_bits = right_val as u32;
            let interleaved = interleave_bits(left_bits, right_bits);
            LookupBits::new(interleaved, LOG_K)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::range_checking::{
        read_raf_checking::{
            RangecheckRafSumcheckParams, RangecheckRafSumcheckProver, RangecheckRafSumcheckVerifier,
        },
        sumcheck_instance::DivRangeCheckOperands,
    };
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{model, tensor::Tensor};
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            },
        },
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use serial_test::serial;

    #[serial]
    #[test]
    fn test_range_check() {
        let mut rng = StdRng::seed_from_u64(0x188);
        let log_T = 10;
        let T = 1 << log_T;
        let input = Tensor::random_small(&mut rng, &[T]);
        let model = model::test::div_model(T);
        let trace = model.trace(&[input.clone()]);
        let node = model[model.outputs()[0]].clone();

        let mut prover_transcript = Blake2bTranscript::new(&[]);
        let mut prover_accumulator = ProverOpeningAccumulator::<Fr>::new();

        let r_exec: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_T);
        let remainder: Vec<i32> = input.iter().map(|i| i % 128).collect();
        let in0_claim = MultilinearPolynomial::from(remainder.clone()).evaluate(&r_exec);
        prover_accumulator.append_virtual(
            &mut prover_transcript,
            VirtualPolynomial::DivRemainder(2),
            SumcheckId::Execution,
            OpeningPoint::new(r_exec.clone()),
            in0_claim,
        );

        let prover_params = RangecheckRafSumcheckParams::<Fr, DivRangeCheckOperands>::new(
            node.clone(),
            &prover_accumulator,
            &mut prover_transcript,
        );
        let mut prover = RangecheckRafSumcheckProver::initialize(
            prover_params,
            &trace,
            &mut prover_accumulator,
            &mut prover_transcript,
        );

        let (proof, r_sumcheck_prover) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bTranscript::new(&[]);
        let mut verifier_accumulator = VerifierOpeningAccumulator::<Fr>::new();

        let _r_exec: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);
        assert_eq!(r_exec, _r_exec);

        for (key, (_, value)) in prover_accumulator.openings.iter() {
            verifier_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *value));
        }

        verifier_accumulator.append_virtual(
            &mut verifier_transcript,
            VirtualPolynomial::DivRemainder(2),
            SumcheckId::Execution,
            OpeningPoint::new(r_exec.clone()),
        );

        let verifier_params = RangecheckRafSumcheckParams::<Fr, DivRangeCheckOperands>::new(
            node.clone(),
            &verifier_accumulator,
            &mut verifier_transcript,
        );

        let verifier = RangecheckRafSumcheckVerifier::<Fr, DivRangeCheckOperands>::new(
            verifier_params,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        );
        let r_sumcheck_verifier = Sumcheck::verify(
            &proof,
            &verifier,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(r_sumcheck_prover, r_sumcheck_verifier);
    }
}
