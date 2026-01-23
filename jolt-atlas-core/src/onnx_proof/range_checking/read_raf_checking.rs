use std::array;

use crate::onnx_proof::{
    lookup_tables::{
        prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
    op_lookups::{InterleavedBitsMarker, LOG_K},
};
use ark_std::Zero;
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::Tensor,
};
use common::{consts::XLEN, VirtualPolynomial};
use itertools::Itertools;
use joltworks::{
    field::{JoltField, MulTrunc},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide},
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

pub struct ReadRafSumcheckParams<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    pub gamma: F,
    pub gamma_sqr: F,
    /// log2(T): number of variables in the node output polynomial (last rounds bind input).
    pub log_T: usize,
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    /// Table for this node
    table: T,
}

impl<F, T> ReadRafSumcheckParams<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    pub fn new(
        computation_node: ComputationNode,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let (r_node_output, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.idx),
            SumcheckId::Execution,
        );
        let log_T = computation_node.num_output_elements().log_2();
        Self {
            gamma,
            gamma_sqr,
            log_T,
            r_node_output,
            computation_node,
            table: T::default(),
        }
    }
}

impl<F, T> SumcheckInstanceParams<F> for ReadRafSumcheckParams<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.idx),
            SumcheckId::Execution,
        );
        let (left_operand_claim, right_operand_claim) =
            if self.computation_node.is_interleaved_operands() {
                let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::Raf,
                );
                let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::NodeOutput(self.computation_node.inputs[1]),
                    SumcheckId::Raf,
                );
                (left_operand_claim, right_operand_claim)
            } else {
                let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::NodeOutput(self.computation_node.inputs[0]),
                    SumcheckId::Raf,
                );
                (F::zero(), right_operand_claim)
            };

        rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim
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

pub struct ReadRafSumcheckProver<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    params: ReadRafSumcheckParams<F, T>,
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
    /// Cycle indices with interleaved operands (used for left/right operand prefix-suffix Q).
    lookup_indices_uninterleave: Vec<usize>,
    /// Cycle indices with identity path (non-interleaved) used as the RAF flag source.
    lookup_indices_identity: Vec<usize>,
    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for right operand identity polynomial family.
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for left operand identity polynomial family.
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2>,
}

impl<F, T> ReadRafSumcheckProver<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    pub fn initialize(
        params: ReadRafSumcheckParams<F, T>,
        trace: &Trace,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let T = 1 << params.log_T;
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
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(trace, &params.computation_node);
        let is_interleaved_operands = params.computation_node.is_interleaved_operands();
        if is_interleaved_operands {
            let [left_operand_tensor, right_operand_tensor] = operands[..] else {
                panic!("Expected exactly two input tensors")
            };

            // Cache left/right operand claims.
            let left_operand_claim =
                MultilinearPolynomial::from(left_operand_tensor.into_container_data()) // TODO: make this work with from_i32
                    .evaluate(&params.r_node_output.r); // TODO: rm these clones
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
                SumcheckId::Raf,
                params.r_node_output.clone(),
                left_operand_claim,
            );
            let right_operand_claim =
                MultilinearPolynomial::from(right_operand_tensor.into_container_data())
                    .evaluate(&params.r_node_output.r);
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
                SumcheckId::Raf,
                params.r_node_output.clone(),
                right_operand_claim,
            );
        } else {
            let right_operand_tensor = operands[0];
            let right_operand_claim =
                MultilinearPolynomial::from(right_operand_tensor.into_container_data())
                    .evaluate(&params.r_node_output.r);
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
                SumcheckId::Raf,
                params.r_node_output.clone(),
                right_operand_claim,
            );
            let right_operand_claim = MultilinearPolynomial::from(right_operand_tensor.clone())
                .evaluate(&params.r_node_output.r);
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
                SumcheckId::Execution, // TODO: Add specialized sumcheck that relates raf and execution claims (signed vs unsigned claims)
                params.r_node_output.clone(),
                right_operand_claim,
            );
        };
        let lookup_indices =
            compute_lookup_indices_from_operands(&operands, is_interleaved_operands);
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);
        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), log_m, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), log_m, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), log_m, LOG_K);
        drop(_guard);
        drop(span);
        let eq_r_node_output =
            GruenSplitEqPolynomial::<F>::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        // TODO: Adjust [PrefixSuffixDecomposition::init_Q_dual] and [PrefixSuffixDecomposition::init_Q] to be compatible with jolt-atlas usage
        let (lookup_indices_uninterleave, lookup_indices_identity) = if is_interleaved_operands {
            ((0..T).collect(), vec![])
        } else {
            (vec![], (0..T).collect())
        };
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
            lookup_indices_identity,
            lookup_indices_uninterleave,
            right_operand_ps,
            left_operand_ps,
            identity_ps,

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
                    &self.lookup_indices_uninterleave,
                    &self.lookup_indices,
                )
            });
            s.spawn(|_| {
                self.identity_ps.init_Q(
                    &self.u_evals,
                    &self.lookup_indices_identity,
                    &self.lookup_indices,
                )
            });
        });

        self.init_suffix_polys(phase);

        self.identity_ps.init_P(&mut self.prefix_registry);
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
        let len = self.identity_ps.Q_len();
        let [left_0, left_2, right_0, right_2] = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let (i0, i2) = self.identity_ps.sumcheck_evals(b);
                let (r0, r2) = self.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = self.left_operand_ps.sumcheck_evals(b);
                [
                    *l0.as_unreduced_ref(),
                    *l2.as_unreduced_ref(),
                    *(i0 + r0).as_unreduced_ref(),
                    *(i2 + r2).as_unreduced_ref(),
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

impl<F: JoltField, FS: Transcript, T> SumcheckInstanceProver<F, FS> for ReadRafSumcheckProver<F, T>
where
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
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
                s.spawn(|_| self.identity_ps.bind(r_j));
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
                let raf_val = if self.params.computation_node.is_interleaved_operands() {
                    gamma * self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                        + gamma_sqr
                            * self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap()
                } else {
                    gamma_sqr * self.prefix_registry.checkpoints[Prefix::Identity].unwrap()
                };
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
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point,
            self.ra.as_ref().unwrap().final_sumcheck_claim(),
        );
    }
}

pub struct ReadRafSumcheckVerifier<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    params: ReadRafSumcheckParams<F, T>,
}

impl<F, T> ReadRafSumcheckVerifier<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    pub fn new(
        computation_node: ComputationNode,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ReadRafSumcheckParams::new(computation_node, opening_accumulator, transcript);
        // Update accumulator
        if params.computation_node.is_interleaved_operands() {
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
                SumcheckId::Raf,
                params.r_node_output.clone(),
            );
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
                SumcheckId::Raf,
                params.r_node_output.clone(),
            );
        } else {
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
                SumcheckId::Raf,
                params.r_node_output.clone(),
            );
            opening_accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
                SumcheckId::Execution,
                params.r_node_output.clone(),
            );
        }

        Self { params }
    }
}

impl<F: JoltField, FS: Transcript, T> SumcheckInstanceVerifier<F, FS>
    for ReadRafSumcheckVerifier<F, T>
where
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
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
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Execution,
        );
        let val_claim = self.params.table.evaluate_mle(&r_address_prime.r);
        let eq_eval = EqPolynomial::mle(&self.params.r_node_output.r, &r_node_output_prime.r);

        // RAF
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(&r_address_prime.r);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(&r_address_prime.r);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(&r_address_prime.r);
        let raf_flag_claim = F::from_bool(self.params.computation_node.is_interleaved_operands());
        let raf_claim = raf_flag_claim
            * (left_operand_eval + self.params.gamma * right_operand_eval)
            + (F::one() - raf_flag_claim) * self.params.gamma * identity_poly_eval;

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
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point,
        );
    }
}

pub fn compute_lookup_indices_from_operands(
    operand_tensors: &[&Tensor<i32>],
    is_interleaved_operands: bool,
) -> Vec<LookupBits> {
    if is_interleaved_operands {
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
    } else {
        // Single operand mode: requires exactly 1 input tensor
        assert_eq!(
            operand_tensors.len(),
            1,
            "Single operand mode requires exactly 1 input tensor, but got {}",
            operand_tensors.len()
        );

        let operand = operand_tensors[0];

        // Use tensor values directly as lookup indices
        operand
            .data()
            .par_iter()
            .map(|&value| {
                // Cast to u64 for consistent bit representation
                let index = value as u32 as u64;
                LookupBits::new(index, LOG_K)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Instant};

    use crate::onnx_proof::{
        lookup_tables::{
            and::AndTable, relu::ReluTable, JoltLookupTable, PrefixSuffixDecompositionTrait,
        },
        op_lookups::{
            self,
            ra_virtual::{
                InstructionRaSumcheckParams, InstructionRaSumcheckProver, RaSumcheckVerifier,
            },
            read_raf_checking::{
                ReadRafSumcheckParams, ReadRafSumcheckProver, ReadRafSumcheckVerifier,
            },
        },
        witness::{generate_node_output_ra, node_committed_polynomials},
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    };
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
            Model,
        },
        tensor::Tensor,
    };
    use common::{consts::XLEN, VirtualPolynomial};
    use joltworks::{
        config::OneHotParams,
        field::JoltField,
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme,
                dory::{DoryCommitmentScheme, DoryGlobals},
            },
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        subprotocols::{
            sumcheck::{BatchedSumcheck, Sumcheck},
            sumcheck_prover::SumcheckInstanceProver,
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use rayon::prelude::*;
    use serial_test::serial;

    /// Helper function to run a complete sumcheck proof and verification for a given table
    fn run_read_raf_sumcheck_test<T, PCS>(
        model: Model,
        trace: &Trace,
        log_T: usize,
        output_index: usize,
        computation_node: &atlas_onnx_tracer::node::ComputationNode,
    ) where
        T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN> + Default,
        PCS: CommitmentScheme<Field = Fr>,
    {
        DoryGlobals::reset();
        let one_hot_params = OneHotParams::new(log_T);
        let shared_pp = AtlasSharedPreprocessing::preprocess(model);
        let prover_pp = AtlasProverPreprocessing::<Fr, PCS>::new(shared_pp);
        let verifier_pp = AtlasVerifierPreprocessing::from(&prover_pp);

        DoryGlobals::initialize(1 << one_hot_params.log_k_chunk, 1 << log_T);
        let committed_polys = generate_node_output_ra::<Fr>(computation_node, trace);
        let (commitments, hints): (Vec<PCS::Commitment>, Vec<PCS::OpeningProofHint>) =
            committed_polys
                .par_iter()
                .map(|(_label, poly)| PCS::commit(poly, &prover_pp.generators))
                .unzip();

        let mut hint_map = HashMap::with_capacity(committed_polys.len());
        for (i, hint) in hints.into_iter().enumerate() {
            hint_map.insert(committed_polys[i].0, hint);
        }

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new(log_T);
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new(log_T);

        let r_node_output: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_T);
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);

        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(trace, computation_node);

        let rv_claim =
            MultilinearPolynomial::from(output.into_container_data()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            rv_claim,
        );

        let prover_params = ReadRafSumcheckParams::<Fr, T>::new(
            computation_node.clone(),
            &prover_opening_accumulator,
            prover_transcript,
        );
        let mut prover_sumcheck = ReadRafSumcheckProver::initialize(
            prover_params,
            trace,
            &mut prover_opening_accumulator,
            prover_transcript,
        );
        let time = Instant::now();
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover_opening_accumulator,
            prover_transcript,
        );
        println!("Proving time: {:?}", time.elapsed());

        // ra prover
        let one_hot_params = OneHotParams::new(log_T);
        let ra_params = InstructionRaSumcheckParams::new(
            computation_node.clone(),
            &OneHotParams::new(log_T),
            &prover_opening_accumulator,
        );
        let ra_prover_sumcheck = InstructionRaSumcheckProver::initialize(ra_params, trace);

        let lookups_hamming_weight_params = op_lookups::ra_hamming_weight_params(
            computation_node,
            &one_hot_params,
            &prover_opening_accumulator,
            prover_transcript,
        );
        let lookups_booleanity_params = op_lookups::ra_booleanity_params(
            computation_node,
            &one_hot_params,
            &prover_opening_accumulator,
            prover_transcript,
        );

        let (lookups_ra_booleanity, lookups_ra_hamming_weight) = op_lookups::gen_ra_one_hot_provers(
            lookups_hamming_weight_params,
            lookups_booleanity_params,
            trace,
            computation_node,
            &one_hot_params,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(ra_prover_sumcheck),
            Box::new(lookups_ra_booleanity),
            Box::new(lookups_ra_hamming_weight),
        ];
        let time = Instant::now();
        let (sumcheck_proof_6, _r_stage6) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover_opening_accumulator,
            prover_transcript,
        );
        println!("ra, bool, hw, time: {:?}", time.elapsed());

        // Prepare sumcheck
        let polynomial_map = HashMap::from_iter(committed_polys);
        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(prover_transcript);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc, prover_transcript);

        // Build RLC polynomial and combined hint
        let (joint_poly, hint) = prover_opening_accumulator.build_rlc_polynomial::<PCS>(
            polynomial_map,
            hint_map,
            &state,
        );

        // Dory opening proof
        let joint_opening_proof = PCS::prove(
            &prover_pp.generators,
            &joint_poly,
            &state.r_sumcheck,
            Some(hint),
            prover_transcript,
        );

        let stage7_sumcheck_claims = state.sumcheck_claims.clone();

        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );
        let verifier_sumcheck = ReadRafSumcheckVerifier::<Fr, T>::new(
            computation_node.clone(),
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );
        let _ = Sumcheck::verify(
            &proof,
            &verifier_sumcheck,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        let ra_verifier_sumcheck = RaSumcheckVerifier::new(
            computation_node.clone(),
            &OneHotParams::new(log_T),
            &verifier_opening_accumulator,
        );
        let (lookups_ra_booleanity, lookups_rs_hamming_weight) =
            op_lookups::new_ra_one_hot_verifiers(
                computation_node,
                &one_hot_params,
                &verifier_opening_accumulator,
                verifier_transcript,
            );

        let _ = BatchedSumcheck::verify(
            &sumcheck_proof_6,
            vec![
                &ra_verifier_sumcheck,
                &lookups_ra_booleanity,
                &lookups_rs_hamming_weight,
            ],
            &mut verifier_opening_accumulator,
            verifier_transcript,
        )
        .unwrap();

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator.prepare_for_sumcheck(&stage7_sumcheck_claims);

        // Verify sumcheck
        let r_sumcheck = verifier_opening_accumulator
            .verify_batch_opening_sumcheck(&accumulator_sumcheck_proof, verifier_transcript)
            .unwrap();

        // Finalize and store state in accumulator for Stage 8
        let verifier_state = verifier_opening_accumulator.finalize_batch_opening_sumcheck(
            r_sumcheck,
            &stage7_sumcheck_claims,
            verifier_transcript,
        );

        // Stage 8: Dory batch opening verification.
        // Build commitments map
        let node_committed_polys = node_committed_polynomials(computation_node);
        let mut commitments_map = HashMap::with_capacity(node_committed_polys.len());
        for (i, commitment) in commitments.into_iter().enumerate() {
            commitments_map.insert(node_committed_polys[i], commitment);
        }
        // Compute joint commitment
        let joint_commitment = verifier_opening_accumulator
            .compute_joint_commitment::<PCS>(&mut commitments_map, &verifier_state);

        // Verify joint opening
        verifier_opening_accumulator
            .verify_joint_opening::<_, PCS>(
                &verifier_pp.generators,
                &joint_opening_proof,
                &joint_commitment,
                &verifier_state,
                verifier_transcript,
            )
            .unwrap();

        prover_transcript.compare_to(verifier_transcript.clone());
    }

    #[serial]
    #[test]
    fn test_and() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x188);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = model::test::and2(&mut rng, T);
        let trace = model.trace(&[input]);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];

        run_read_raf_sumcheck_test::<AndTable<XLEN>, DoryCommitmentScheme>(
            model.clone(),
            &trace,
            log_T,
            output_index,
            computation_node,
        );
    }

    #[serial]
    #[test]
    fn test_relu() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x188);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = model::test::relu_model(T);
        let trace = model.trace(&[input]);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];

        run_read_raf_sumcheck_test::<ReluTable<XLEN>, DoryCommitmentScheme>(
            model.clone(),
            &trace,
            log_T,
            output_index,
            computation_node,
        );
    }

    #[serial]
    #[test]
    fn test_relu_small_T() {
        let log_T = 2;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x188);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = model::test::relu_model(T);
        let trace = model.trace(&[input]);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];

        run_read_raf_sumcheck_test::<ReluTable<XLEN>, DoryCommitmentScheme>(
            model.clone(),
            &trace,
            log_T,
            output_index,
            computation_node,
        );
    }
}
