// Instruction lookups: Read + RAF batched sumcheck
//
// Notation:
// - Field F. Let K = 2^{LOG_K}, T = 2^{log_T}.
// - Address index k ∈ {0..K-1}, cycle index j ∈ {0..T-1}.
// - eq(k; r_addr) := multilinear equality polynomial over LOG_K vars.
// - eq(j; r_reduction) := equality polynomials over LOG_T vars.
// - ra(k, j) is the selector arising from prefix/suffix condensation.
//   It is decomposed as the product of virtual sub selectors:
//   ra((k_0, k_1, ..., k_{n-1}), j) := ra_0(k_0, j) * ra_1(k_1, j) * ... * ra_{n-1}(k_{n-1}, j).
//   n is typically 1, 2, 4 or 8.
//   logically ra(k, j) = 1 when the j-th cycle's lookup key equals k, and 0 otherwise.// - Val_j(k) ∈ F is the lookup-table value selected by (j, k); concretely Val_j(k) = table_j(k)
//   if cycle j uses a table and 0 otherwise (materialized via prefix/suffix decomposition).
// - raf_flag(j) ∈ {0,1} is 1 iff the instruction at cycle j is NOT interleaved operands.
// - Let LeftPrefix_j, RightPrefix_j, IdentityPrefix_j ∈ F be the address-only (prefix) factors for
//   the left/right operand and identity polynomials at cycle j (from `PrefixSuffixDecomposition`).
//
// We introduce a batching challenge γ ∈ F. Define
//   RafVal_j(k) := (1 - raf_flag(j)) · (LeftPrefix_j + γ · RightPrefix_j)
//                  + raf_flag(j) · γ · IdentityPrefix_j.
// The overall γ-weights are arranged so that γ multiplies RafVal_j(k) in the final identity.
//
// Claims supplied by the accumulator (LHS), all claimed at `SumcheckId::InstructionClaimReduction`
// and `SumcheckId::SpartanProductVirtualization`:
// - rv         := ⟦LookupOutput⟧
// - left_op    := ⟦LeftLookupOperand⟧
// - right_op   := ⟦RightLookupOperand⟧
//   Combined as: rv + γ·left_op + γ^2·right_op
//
// Statement proved by this sumcheck (RHS), for random challenges
// r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T}:
//
//   rv(r_reduction) + γ·left_op(r_reduction) + γ^2·right_op(r_reduction)
//   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ].
//
// Prover structure:
// - First log(K) rounds bind address vars using prefix/suffix decomposition, accumulating:
//   Σ_k ra(k, j)·Val_j(k)  and  Σ_k ra(k, j)·RafVal_j(k)
//   for each j (via u_evals vectors and suffix polynomials).
// - Last log(T) rounds bind cycle vars producing a degree-3 univariate with the required previous-round claim.
// - The published univariate matches the RHS above; the verifier checks it against the LHS claims.

use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::Tensor,
};
use common::{consts::XLEN, VirtualPolynomial};
use joltworks::{
    config::{get_instruction_sumcheck_phases, OneHotParams},
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
        mles_product_sum::{eval_linear_prod_accumulate, finish_mles_product_sum_from_evals},
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
use num_traits::Zero;
use rayon::prelude::*;
use std::{array, iter::zip};
use strum::EnumCount;

use crate::onnx_proof::{
    lookup_tables::{
        prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
    op_lookups::{InterleavedBitsMarker, LOG_K},
};

pub struct ReadRafSumcheckParams<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    pub gamma: F,
    pub gamma_sqr: F,
    /// log2(T): number of input poly variables (last rounds bind input).
    pub log_T: usize,
    /// How many address variables each virtual ra polynomial has.
    pub ra_virtual_log_k_chunk: usize,
    pub r_tensor: OpeningPoint<BIG_ENDIAN, F>,
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
        computation_node: ComputationNode, // TODO: use Arc to avoid expensive clones
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let (r_tensor, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.idx),
            SumcheckId::Execution,
        );
        let log_T = computation_node.num_output_elements().log_2(); // TODO: Padding for non-power of two cases
        Self {
            gamma,
            gamma_sqr,
            log_T,
            ra_virtual_log_k_chunk: one_hot_params.lookups_ra_virtual_log_k_chunk,
            r_tensor,
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

        // TODO: Handle unary case
        let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.inputs[0]),
            SumcheckId::Execution,
        );
        let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.inputs[1]),
            SumcheckId::Execution,
        );

        rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim
    }

    fn degree(&self) -> usize {
        let n_virtual_ra_polys = LOG_K / self.ra_virtual_log_k_chunk;
        n_virtual_ra_polys + 2
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_tensor_prime) = challenges.split_at(LOG_K);
        let r_tensor_prime = r_tensor_prime.iter().copied().rev().collect::<Vec<_>>();

        OpeningPoint::new([r_address_prime.to_vec(), r_tensor_prime].concat())
    }
}

/// Sumcheck prover for [`ReadRafSumcheckVerifier`].
///
/// Binds address variables first using prefix/suffix decomposition to aggregate, per cycle j,
///   Σ_k ra(k, j)·Val_j(k) and Σ_k ra(k, j)·RafVal_j(k),
pub struct ReadRafSumcheckProver<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    /// Materialized `ra_i(k_i, j)` polynomials.
    /// Present only in the last log(T) rounds.
    ra_polys: Option<Vec<MultilinearPolynomial<F>>>,
    /// Running list of sumcheck challenges r_j (address then cycle) in binding order.
    r: Vec<F::Challenge>,
    /// Precomputed lookup keys k (bit-packed) per cycle j.
    lookup_indices: Vec<LookupBits>,
    /// Cycle indices with interleaved operands (used for left/right operand prefix-suffix Q).
    lookup_indices_uninterleave: Vec<usize>,
    /// Cycle indices with identity path (non-interleaved) used as the RAF flag source.
    lookup_indices_identity: Vec<usize>,
    /// Prefix checkpoints for each registered `Prefix` variant, updated every two rounds.
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    /// For each lookup table, dense polynomials holding suffix contributions in the current phase.
    suffix_polys: Vec<DensePolynomial<F>>,
    /// Expanding tables accumulating address-prefix products per phase.
    v: Vec<ExpandingTable<F>>,
    /// u_evals for read-checking and RAF: eq(r_reduction,j).
    u_evals: Vec<F>,
    /// Gruen-split equality polynomial over cycle vars.
    eq_r_tensor: GruenSplitEqPolynomial<F>,
    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for right operand identity polynomial family.
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for left operand identity polynomial family.
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2>,
    phases: usize,
    /// Materialized Val_j(k) over (address, cycle) after phase transitions.
    combined_val_polynomial: Option<F>,
    /// Materialized RafVal_j(k) (with γ-weights folded into prefixes) over (address, cycle).
    combined_raf_val_polynomial: Option<F>,
    params: ReadRafSumcheckParams<F, T>,
}

impl<F, T> ReadRafSumcheckProver<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
    /// Creates a prover-side instance for the Read+RAF batched sumcheck.
    ///
    /// Builds prover-side working state:
    /// - Precomputes per-cycle lookup index, interleaving flags, and table choices
    /// - Buckets cycles by table and by path (interleaved vs identity)
    /// - Allocates per-table suffix accumulators and u-evals for rv/raf parts
    /// - Instantiates the three RAF decompositions and Gruen EQs over cycles
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::initialize")]
    pub fn initialize(
        trace: &Trace,
        params: ReadRafSumcheckParams<F, T>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { output, operands } = Trace::layer_data(trace, &params.computation_node);
        let (r_tensor, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(params.computation_node.idx),
            SumcheckId::Execution,
        );
        // TODO: handle single operand cases
        let [left_operand_tensor, right_operand_tensor] = operands[..] else {
            panic!("Expected exactly two input tensors")
        };

        // Cache left/right operand claims.
        let left_operand_claim =
            MultilinearPolynomial::from(left_operand_tensor.clone()).evaluate(&r_tensor.r); // TODO: rm these clones
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
            SumcheckId::Execution,
            r_tensor.clone(),
            left_operand_claim,
        );
        let right_operand_claim =
            MultilinearPolynomial::from(right_operand_tensor.clone()).evaluate(&r_tensor.r);
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[1]),
            SumcheckId::Execution,
            r_tensor.clone(),
            right_operand_claim,
        );

        let is_interleaved_operands = params.computation_node.is_interleaved_operands();
        let T = output.len();
        let log_T = T.log_2();
        let phases = get_instruction_sumcheck_phases(log_T);
        println!("phases: {}", phases);
        let log_m = LOG_K / phases;
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

        let lookup_indices =
            compute_lookup_indices_from_operands(&operands, is_interleaved_operands);

        let suffix_polys: Vec<DensePolynomial<F>> = params
            .table
            .suffixes()
            .iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|_| DensePolynomial::default())
            .collect();

        // Build split-eq polynomials and u_evals.
        let span = tracing::span!(tracing::Level::INFO, "Compute u_evals");
        let _guard = span.enter();
        let eq_poly_r_tensor =
            GruenSplitEqPolynomial::<F>::new(&r_tensor.r, BindingOrder::LowToHigh);
        let u_evals = EqPolynomial::evals(&r_tensor.r);
        drop(_guard);
        drop(span);

        // TODO: Adjust [PrefixSuffixDecomposition::init_Q_dual] and [PrefixSuffixDecomposition::init_Q] to be compatible with jolt-atlas usage
        let (lookup_indices_uninterleave, lookup_indices_identity) = if is_interleaved_operands {
            ((0..T).collect(), vec![])
        } else {
            (vec![], (0..T).collect())
        };

        let prefix_checkpoints = vec![None.into(); Prefixes::COUNT];
        let mut res = Self {
            params,
            r: Vec::with_capacity(log_T + LOG_K),
            lookup_indices,

            // Prefix-suffix state (first log(K) rounds)
            prefix_checkpoints,
            suffix_polys,
            v: (0..phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            u_evals,
            lookup_indices_identity,
            lookup_indices_uninterleave,
            right_operand_ps,
            left_operand_ps,
            identity_ps,

            // State for last log(T) rounds
            ra_polys: None,
            combined_val_polynomial: None,
            combined_raf_val_polynomial: None,
            eq_r_tensor: eq_poly_r_tensor,
            prefix_registry: PrefixRegistry::new(),
            phases,
        };
        res.init_phase(0);
        res
    }

    /// To be called in the beginning of each phase, before any binding
    /// Phase initialization for address-binding:
    /// - Condenses prior-phase u-evals through the expanding-table v[phase-1]
    /// - Builds Q for RAF (Left/Right dual and Identity) from cycle buckets
    /// - Refreshes per-table read-checking suffix polynomials for this phase
    /// - Initializes/caches P via the shared `PrefixRegistry`
    /// - Resets the current expanding table accumulator for this phase
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_phase")]
    fn init_phase(&mut self, phase: usize) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
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
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        // TODO: parallelize with chunk_size
        let _chunk_size = (self.lookup_indices.len() / num_chunks).max(1);
        let new_suffix_polys: Vec<Vec<F>> = self
            .params
            .table
            .suffixes()
            .par_iter()
            .map(|s| {
                let mut Q = unsafe_allocate_zero_vec(m);
                self.lookup_indices.iter().enumerate().for_each(|(j, &k)| {
                    let (prefix_bits, suffix_bits) = k.split((self.phases - 1 - phase) * log_m);
                    let t = s.suffix_mle::<XLEN>(suffix_bits);
                    if t != 0 {
                        let u = self.u_evals[j];
                        Q[prefix_bits & m_mask] += u * F::from_u32(t);
                    }
                });
                Q
            })
            .collect();

        // Replace existing suffix polynomials
        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(poly, mut coeffs)| {
                *poly = DensePolynomial::new(std::mem::take(&mut coeffs))
            });
    }

    /// To be called before the last log(T) rounds
    /// Handoff between address and cycle rounds:
    /// - Materializes all virtual ra_i(k_i,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_sqr: F) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Drop stuff that's no longer needed
        drop_in_background_thread((
            std::mem::take(&mut self.u_evals),
            std::mem::take(&mut self.lookup_indices_uninterleave),
        ));

        let ra_polys: Vec<MultilinearPolynomial<F>> = {
            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomials");
            let _guard = span.enter();
            assert!(self.v.len().is_power_of_two());
            let n = LOG_K / self.params.ra_virtual_log_k_chunk;
            let chunk_size = self.v.len() / n;
            self.v
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_i, v_chunk)| {
                    let phase_offset = chunk_i * chunk_size;
                    let res = self
                        .lookup_indices
                        .par_iter()
                        .with_min_len(1024)
                        .map(|i| {
                            let mut acc = F::one();

                            for (phase, table) in zip(phase_offset.., v_chunk) {
                                let v: u64 = i.into();
                                let i_segment =
                                    ((v >> ((self.phases - 1 - phase) * log_m)) as usize) & m_mask;
                                acc *= table[i_segment];
                            }

                            acc
                        })
                        .collect::<Vec<F>>();
                    res.into()
                })
                .collect()
        };

        drop_in_background_thread(std::mem::take(&mut self.v));

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
        self.combined_val_polynomial = Some(self.params.table.combine(&prefixes, &suffixes));
        let combined_raf_val_polynomial = if self.params.computation_node.is_interleaved_operands()
        {
            gamma * self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap()
                + gamma_sqr * self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap()
        } else {
            gamma_sqr * self.prefix_registry.checkpoints[Prefix::Identity].unwrap()
        };
        self.combined_raf_val_polynomial = Some(combined_raf_val_polynomial);
        self.ra_polys = Some(ra_polys);
    }
}

impl<F, FS, T> SumcheckInstanceProver<F, FS> for ReadRafSumcheckProver<F, T>
where
    F: JoltField,
    FS: Transcript,
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
            let ra_polys = self.ra_polys.as_ref().unwrap();
            let val = self.combined_val_polynomial.as_ref().unwrap();
            let raf_val = self.combined_raf_val_polynomial.as_ref().unwrap();
            let n_evals = ra_polys.len() + 1;

            let mut sum_evals = self
                .eq_r_tensor
                .E_out_current()
                .par_iter()
                .enumerate()
                .map(|(j_out, e_out)| {
                    // Each pair is a linear polynomial.
                    let mut pairs = vec![(F::zero(), F::zero()); n_evals];
                    let mut evals_acc = vec![F::Unreduced::<9>::zero(); n_evals];

                    for (j_in, e_in) in self.eq_r_tensor.E_in_current().iter().enumerate() {
                        let j = self.eq_r_tensor.group_index(j_out, j_in);

                        let Some((val_pair, ra_pairs)) = pairs.split_first_mut() else {
                            unreachable!()
                        };

                        // v = val + raf_val
                        let v = *val + *raf_val;
                        // Load linear poly: eq * (val + raf_val).
                        *val_pair = (*e_in, *e_in);
                        // Load ra polys.
                        zip(ra_pairs, ra_polys).for_each(|(pair, ra_poly)| {
                            let eval_at_0 = ra_poly.get_bound_coeff(2 * j);
                            let eval_at_1 = ra_poly.get_bound_coeff(2 * j + 1);
                            *pair = (eval_at_0 * v, eval_at_1 * v);
                        });

                        // TODO: Use unreduced arithmetic in eval_linear_prod_assign.
                        eval_linear_prod_accumulate(&pairs, &mut evals_acc);
                    }

                    evals_acc
                        .into_iter()
                        .map(|v| F::from_montgomery_reduce(v) * e_out)
                        .collect::<Vec<F>>()
                })
                .reduce(
                    || vec![F::zero(); n_evals],
                    |a, b| zip(a, b).map(|(a, b)| a + b).collect(),
                );

            let current_scalar = self.eq_r_tensor.get_current_scalar();
            sum_evals.iter_mut().for_each(|v| *v *= current_scalar);
            finish_mles_product_sum_from_evals(&sum_evals, previous_claim, &self.eq_r_tensor)
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
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.suffix_polys
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
                });
                s.spawn(|_| self.identity_ps.bind(r_j));
                s.spawn(|_| self.right_operand_ps.bind(r_j));
                s.spawn(|_| self.left_operand_ps.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });
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
                if phase != self.phases - 1 {
                    // if not last phase, init next phase
                    self.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                self.init_log_t_rounds(self.params.gamma, self.params.gamma_sqr);
            }
        } else {
            // log(T) rounds

            self.eq_r_tensor.bind(r_j);
            self.ra_polys
                .as_mut()
                .unwrap()
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut FS,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Prover publishes new virtual openings derived by this sumcheck:
        // - Per-table LookupTableFlag(i) at r_cycle
        // - InstructionRa at r_sumcheck (ra MLE's final claim)
        // - InstructionRafFlag at r_cycle
        let (r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);
        let ra_polys = self.ra_polys.as_ref().unwrap();
        let mut r_address_chunks = r_address.r.chunks(LOG_K / ra_polys.len());
        for (i, ra_poly) in self.ra_polys.as_ref().unwrap().iter().enumerate() {
            let r_address = r_address_chunks.next().unwrap();
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutputRaD(self.params.computation_node.idx, i),
                SumcheckId::Execution,
                opening_point,
                ra_poly.final_sumcheck_claim(),
            );
        }
    }
}

impl<F, T> ReadRafSumcheckProver<F, T>
where
    F: JoltField,
    T: JoltLookupTable + PrefixSuffixDecompositionTrait<XLEN>,
{
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

    /// Read-checking part for address rounds.
    ///
    /// For each lookup table, evaluates Σ P(0)·Q^L, Σ P(2)·Q^L, Σ P(2)·Q^R via
    /// table-specific suffix families, then returns [g(0), g(2)] by the standard
    /// quadratic interpolation trick.
    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let len = self.suffix_polys[0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };

        let [eval_0, eval_2_left, eval_2_right] = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let lookup_bits = LookupBits::new(b as u64, log_len - 1);
                let prefixes_c0: Vec<_> = self
                    .params
                    .table
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            lookup_bits,
                            j,
                        )
                    })
                    .collect();
                let prefixes_c2: Vec<_> = self
                    .params
                    .table
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            lookup_bits,
                            j,
                        )
                    })
                    .collect();
                let suffixes_left: Vec<_> =
                    self.suffix_polys.iter().map(|suffix| suffix[b]).collect();
                let suffixes_right: Vec<_> = self
                    .suffix_polys
                    .iter()
                    .map(|suffix| suffix[b + len / 2])
                    .collect();
                [
                    self.params.table.combine(&prefixes_c0, &suffixes_left),
                    self.params.table.combine(&prefixes_c2, &suffixes_left),
                    self.params.table.combine(&prefixes_c2, &suffixes_right),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }
}

/// Instruction lookups: batched Read + RAF sumcheck.
///
/// Let K = 2^{LOG_K}, T = 2^{log_T}. For random r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T},
/// this sumcheck proves that the accumulator claims
///   rv + γ·left_op + γ^2·right_op
/// equal the double sum over (j, k):
///   Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ·RafVal_j(k)) ].
/// It is implemented as: first log(K) address-binding rounds (prefix/suffix condensation), then
/// last log(T) cycle-binding rounds driven by [`GruenSplitEqPolynomial`].
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
        one_hot_params: &OneHotParams,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (r_tensor, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.idx),
            SumcheckId::Execution,
        );
        let params = ReadRafSumcheckParams::new(
            computation_node.clone(),
            one_hot_params,
            opening_accumulator,
            transcript,
        );
        // Update accumulator
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(computation_node.inputs[0]),
            SumcheckId::Execution,
            r_tensor.clone(),
        );
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(computation_node.inputs[1]),
            SumcheckId::Execution,
            r_tensor,
        );
        Self { params }
    }
}

impl<F, FS, T> SumcheckInstanceVerifier<F, FS> for ReadRafSumcheckVerifier<F, T>
where
    F: JoltField,
    FS: Transcript,
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
        // Verifier's RHS reconstruction from virtual claims at r:
        //
        // Computes Val and RafVal contributions at r_address, forms EQ(r_cycle)
        // for InstructionClaimReduction sumcheck, multiplies by ra claim at r_sumcheck,
        // and returns the batched identity RHS to be matched against the LHS input claim.
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_tensor_prime) = opening_point.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(&r_address_prime.r);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(&r_address_prime.r);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(&r_address_prime.r);
        let val_claim = self
            .params
            .table
            .evaluate_mle::<F, F::Challenge>(&r_address_prime.r);

        let r_tensor = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let eq_eval_r_tensor = EqPolynomial::<F>::mle(&r_tensor, &r_tensor_prime.r);
        let n_virtual_ra_polys = LOG_K / self.params.ra_virtual_log_k_chunk;
        let ra_claim = (0..n_virtual_ra_polys)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::NodeOutputRaD(self.params.computation_node.idx, i),
                        SumcheckId::Execution,
                    )
                    .1
            })
            .product::<F>();
        let raf_flag_claim = F::from_bool(self.params.computation_node.is_interleaved_operands());
        let raf_claim = raf_flag_claim
            * (left_operand_eval + self.params.gamma * right_operand_eval)
            + (F::one() - raf_flag_claim) * self.params.gamma * identity_poly_eval;

        eq_eval_r_tensor * ra_claim * (val_claim + self.params.gamma * raf_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut FS,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Verifier requests the virtual openings that the prover must provide
        // for this sumcheck (same set as published by the prover-side cache).
        let (r_address, r_cycle) = r_sumcheck.split_at(LOG_K);
        for (i, r_address_chunk) in r_address
            .r
            .chunks(self.params.ra_virtual_log_k_chunk)
            .enumerate()
        {
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address_chunk, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::NodeOutputRaD(self.params.computation_node.idx, i),
                SumcheckId::Execution,
                opening_point,
            );
        }
    }
}

/// Converts input tensors into lookup indices for the Read+RAF sumcheck.
///
/// This function handles two different operand modes:
/// 1. **Interleaved operands**: When the operation requires two separate operands (e.g., AND, OR, XOR),
///    the lookup index is constructed by interleaving the bits of both operands. This requires exactly
///    two input tensors.
/// 2. **Single operand**: When the operation uses a single operand or identity mapping, the lookup
///    index is derived directly from the first (and only) tensor.
///
/// # Arguments
///
/// * `operand_tensors` - A slice of references to input tensors. Must contain either 1 or 2 tensors
///   depending on the operand mode.
/// * `is_interleaved_operands` - If `true`, expects 2 input tensors and interleaves their bits.
///   If `false`, expects 1 input tensor and uses it directly.
///
/// # Panics
///
/// Panics if the number of input tensors doesn't match the expected count for the operand mode:
/// - When `is_interleaved_operands` is `true`, requires exactly 2 tensors
/// - When `is_interleaved_operands` is `false`, requires exactly 1 tensor
///
/// # Returns
///
/// A vector of `LookupBits` representing the packed lookup indices for each element across
/// all input tensors. For interleaved mode, the indices are formed by bit-interleaving pairs
/// of values from both tensors. For single-operand mode, indices are derived from individual values.
///
/// # Performance
///
/// Uses parallel iterators (`rayon`) for efficient computation across large tensors.
fn compute_lookup_indices_from_operands(
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
                let index = value as u64;
                LookupBits::new(index, LOG_K)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
    };
    use common::{consts::XLEN, VirtualPolynomial};
    use joltworks::{
        config::OneHotParams,
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
        utils::interleave_bits,
    };
    use rand::{rngs::StdRng, SeedableRng};

    use crate::onnx_proof::{
        lookup_tables::{and::AndTable, JoltLookupTable},
        op_lookups::read_raf_checking::{
            ReadRafSumcheckParams, ReadRafSumcheckProver, ReadRafSumcheckVerifier,
        },
    };

    #[test]
    fn test_and() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x188);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = model::test::and2(&mut rng, T);
        let trace = model.trace(&[input]);

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

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData { operands, output } = Trace::layer_data(&trace, computation_node);

        {
            let left_operand = operands[0];
            let right_operand = operands[1];
            // Interleave bits from both operands to form lookup indices
            let expected_indices: Vec<u64> = left_operand
                .data()
                .iter()
                .zip(right_operand.data().iter())
                .map(|(&left_val, &right_val)| {
                    // Cast to u64 for interleaving
                    let left_bits = left_val as u32;
                    let right_bits = right_val as u32;
                    interleave_bits(left_bits, right_bits)
                })
                .collect();
            let table = AndTable::<XLEN>;
            let expected_rv: Vec<u64> = expected_indices
                .iter()
                .map(|i| table.materialize_entry(*i))
                .collect();
            let output_u32 = output.into_container_data();
            expected_rv.iter().enumerate().for_each(|(j, &e)| {
                assert_eq!(output_u32[j] as u64, e);
            });
        }

        let rv_claim =
            MultilinearPolynomial::from(output.into_container_data()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            rv_claim,
        );
        let one_hot_params = OneHotParams::new(log_T);
        let prover_params = ReadRafSumcheckParams::<Fr, AndTable<XLEN>>::new(
            computation_node.clone(),
            &one_hot_params,
            &prover_opening_accumulator,
            prover_transcript,
        );
        let mut prover_sumcheck = ReadRafSumcheckProver::initialize(
            &trace,
            prover_params,
            &mut prover_opening_accumulator,
            prover_transcript,
        );
        let (proof, r_sumcheck) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover_opening_accumulator,
            prover_transcript,
        );

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
        let verifier_sumcheck = ReadRafSumcheckVerifier::<Fr, AndTable<XLEN>>::new(
            computation_node.clone(),
            &one_hot_params,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );
        let res = Sumcheck::verify(
            &proof,
            &verifier_sumcheck,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );
        prover_transcript.compare_to(verifier_transcript.clone());
        let r_sumcheck_verif = res.unwrap();
        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }
}
