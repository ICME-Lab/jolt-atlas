//! The prover's side of the table reads ([`jolt_atlas_verifier::onnx_proof::lookup`] states
//! the relations): the small-table relation's prover ([`SmallTableProver`]) with its addresses
//! and setup claims from the trace, and a prefix-suffix read's operand and prover
//! ([`table_read_operand`], [`table_read_prover`]).

pub use jolt_atlas_verifier::onnx_proof::lookup::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{activation_clamped, softmax_last_axis, trig};
use crate::onnx_proof::relation::{OpRel, RelationChallenge, RelationParams, RelationSetup};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::compute_lookup_indices;
use crate::utils::n_bits_to_usize;
use atlas_algebra::poly::univariate::UnivariatePoly;
use atlas_algebra::{
    field::{FieldChallengeOps, JoltField},
    poly::{
        eq::EqPolynomial,
        identity::IdentityPolynomial,
        multilinear::{
            BindingOrder, EvaluateMle, MultilinearPolynomial, PolynomialBinding,
            PolynomialEvaluation,
        },
        signed_identity::SignedIdentityPoly,
    },
    utils::{lookup_bits::LookupBits, thread::unsafe_allocate_zero_vec},
};
use atlas_claims::SymbolicSumcheck;
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable, SiluClampTable, SoftmaxClampTable},
    relu::ReluTable,
    right_shift::RightShiftTable,
};
use atlas_sumcheck::engine::{prover::ProveRounds, SumcheckError};
use atlas_sumcheck::{
    opening_proof::{prover::ProverOpeningAccumulator, OpeningAccumulator},
    ps_shout::prover::ps_read_raf_prover,
    relation::prover::{DynRelationWitness, RelationWitness},
};
use atlas_trace::{Op, Tensor, Trace};
use atlas_transcript::Transcript;
use common::consts::XLEN;
use common::parallel::par_enabled;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};

/// Degree of the small-table sumcheck.
const DEGREE: usize = 2;

/// The addresses a node reads its small table at, over its cells: the downscaled period
/// remainder of a trig op, the clamped input of the others.
pub fn small_table_addresses(node: &ComputationNode, trace: &Trace, vars: usize) -> Tensor<i32> {
    match &node.operator {
        Op::Sin(_) | Op::Cos(_) => trig::downscaled(node, trace),
        _ => activation_clamped::clamped_input(node, trace, vars),
    }
}

/// The table index of an address: its two's-complement encoding over `def.vars` bits for a
/// signed table, the address itself otherwise.
fn table_index(def: &SmallTableDef, address: i32) -> usize {
    if def.signed {
        n_bits_to_usize(address, def.vars)
    } else {
        address as usize
    }
}

/// The claim of a [`small_table`]'s setup open: the addresses at the output point.
pub fn small_table_setup_claims<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    setup: &RelationSetup<F>,
) -> Vec<F> {
    let node = &relation.node;
    let def = relation
        .small_table()
        .expect("the relation was built for a small table");
    let addresses = small_table_addresses(node, trace, def.vars).padded_pow2();
    vec![addresses.evaluate_mle(&setup.incoming_point)]
}

/// The table index each padded cell of `node` reads, for the read's one-hot checks.
pub fn small_table_lookup_indices(
    node: &ComputationNode,
    def: &SmallTableDef,
    trace: &Trace,
) -> Vec<usize> {
    small_table_addresses(node, trace, def.vars)
        .padded_pow2()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&address| table_index(def, address))
        .collect()
}

/// The address identity of a small table: signed for two's-complement addresses.
enum AddressIdentity<F: JoltField> {
    Signed(SignedIdentityPoly<F>),
    Unsigned(IdentityPolynomial<F>),
}

impl<F: JoltField> AddressIdentity<F> {
    fn sumcheck_evals(&self, index: usize) -> Vec<F> {
        match self {
            Self::Signed(identity) => {
                identity.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh)
            }
            Self::Unsigned(identity) => {
                identity.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh)
            }
        }
    }

    fn bind(&mut self, r_j: F::Challenge) {
        match self {
            Self::Signed(identity) => identity.bind_parallel(r_j, BindingOrder::LowToHigh),
            Self::Unsigned(identity) => identity.bind_parallel(r_j, BindingOrder::LowToHigh),
        }
    }
}

/// The specialized prover of a [`small_table`]: the addresses' one-hot encoding at the output
/// point against the table and the address identity.
pub struct SmallTableProver<F: JoltField> {
    relation: OpRel<F>,
    setup: RelationSetup<F>,
    gamma: F,
    table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: AddressIdentity<F>,
}

impl<F: JoltField> SmallTableProver<F> {
    /// The prover for `params`, whose setup has appended the address opening where the table
    /// asks for it: the addresses read from `trace`.
    pub fn initialize(
        trace: &Trace,
        params: RelationParams<F, OpRel<F>>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let (relation, setup) = params.into_parts();
        crate::onnx_proof::relation::note_dispatch(&relation);
        let node = &relation.node;
        let def = *relation
            .small_table()
            .expect("the relation was built for a small table");
        let gamma = setup.challenge(&RelationChallenge::Gamma);
        let table_size = 1 << def.vars;
        let addresses = small_table_addresses(node, trace, def.vars);
        let index = |&address: &i32| table_index(&def, address);
        let indices: Vec<usize> = addresses
            .par_iter()
            .with_min_len(par_enabled())
            .map(index)
            .collect();
        assert!(indices.iter().all(|&index| index < table_size));

        let table = MultilinearPolynomial::from(
            def.materialize()
                .expect("the semantics define the table at the model scale"),
        );
        let input_onehot = MultilinearPolynomial::from(compute_ra_evals_from_usize_indices(
            &setup.incoming_point,
            &indices,
            table_size,
        ));
        assert_eq!(input_onehot.len(), table.len());
        let identity = if def.signed {
            AddressIdentity::Signed(SignedIdentityPoly::new(def.vars))
        } else {
            AddressIdentity::Unsigned(IdentityPolynomial::new(def.vars))
        };

        #[cfg(test)]
        {
            let input_claim =
                crate::onnx_proof::relation::relation_input_claim(&relation, &setup, accumulator)
                    .expect("the prover holds every claim it reads");
            let claim = (0..input_onehot.len())
                .map(|i| {
                    let address = match &identity {
                        AddressIdentity::Signed(_) => {
                            F::from_i32(crate::utils::usize_to_n_bits(i, def.vars))
                        }
                        AddressIdentity::Unsigned(_) => F::from_u32(i as u32),
                    };
                    input_onehot.get_bound_coeff(i) * (table.get_bound_coeff(i) + gamma * address)
                })
                .sum();
            crate::onnx_proof::testing::probe::probe_assert_eq!(input_claim, claim)
        }
        #[cfg(not(test))]
        let _ = accumulator;

        Self {
            relation,
            setup,
            gamma,
            table,
            input_onehot,
            identity,
        }
    }

    fn round_message(&mut self, previous_claim: F) -> UnivariatePoly<F> {
        let Self {
            input_onehot,
            table,
            identity,
            gamma,
            ..
        } = self;

        let evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra = input_onehot.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let t = table.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let id = identity.sumcheck_evals(i);
                [
                    ra[0] * (t[0] + id[0] * *gamma),
                    ra[1] * (t[1] + id[1] * *gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.input_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind(r_j);
    }
}

impl<F: JoltField> RelationWitness<F> for SmallTableProver<F> {
    type Relation = OpRel<F>;

    fn relation(&self) -> &OpRel<F> {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.input_onehot.final_claim()]
    }
}

impl<F: JoltField> ProveRounds<F> for SmallTableProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        Ok(self.round_message(previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// Compute one-hot read-address evaluations from non-negative table indices: entry `k` is the
/// sum of `eq(r, j)` over the cells `j` that read index `k`.
pub fn compute_ra_evals_from_usize_indices<F, U>(
    r: &[U],
    indices_usize: &[usize],
    table_size: usize,
) -> Vec<F>
where
    U: Copy + Send + Sync + Into<F>,
    F: JoltField + FieldChallengeOps<U>,
{
    let e = EqPolynomial::evals(r);
    let num_threads = rayon::current_num_threads();
    let chunk_size = indices_usize.len().div_ceil(num_threads);

    let partial_results: Vec<Vec<F>> = indices_usize
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local_ra = unsafe_allocate_zero_vec::<F>(table_size);
            let base_idx = chunk_idx * chunk_size;
            chunk.iter().enumerate().for_each(|(local_j, &k)| {
                let global_j = base_idx + local_j;
                local_ra[k] += e[global_j];
            });
            local_ra
        })
        .collect();

    let mut ra = unsafe_allocate_zero_vec::<F>(table_size);
    for partial in partial_results {
        ra.par_iter_mut()
            .zip(partial.par_iter())
            .with_min_len(par_enabled())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}

/// The operand values a prefix-suffix read's table is read at, over the node's padded cells:
/// the period remainder for a trig op's downscale, the address `u = c − x` for softmax's
/// clamp, the node's first operand for the others.
pub fn table_read_operand(read: &TableRead, trace: &Trace) -> Tensor<i64> {
    let node = &read.node;
    match read.table {
        PrefixSuffixTable::RightShift => TraceView::trig_period_advice(trace, node)
            .remainder
            .map(|v| v as i64),
        PrefixSuffixTable::SoftmaxClamp => {
            let witness = trace
                .softmax_witness(node.idx)
                .expect("softmax node has a softmax witness in the trace");
            softmax_last_axis::addresses(node, witness).map(|v| v as i64)
        }
        PrefixSuffixTable::Relu
        | PrefixSuffixTable::Clamp
        | PrefixSuffixTable::ActivationClamp
        | PrefixSuffixTable::SiluClamp => TraceView::operand_tensors(trace, node)[0]
            .padded_pow2()
            .map(|v| v as i64),
    }
}

/// The table address of each operand value of a prefix-suffix read: its `XLEN`-bit encoding.
fn table_read_addresses(read: &TableRead, operand: &Tensor<i64>) -> Vec<LookupBits> {
    match read.table {
        PrefixSuffixTable::RightShift => operand
            .iter()
            .map(|&v| LookupBits::new(v as u64, XLEN))
            .collect(),
        PrefixSuffixTable::Relu
        | PrefixSuffixTable::Clamp
        | PrefixSuffixTable::ActivationClamp
        | PrefixSuffixTable::SiluClamp
        | PrefixSuffixTable::SoftmaxClamp => compute_lookup_indices(&operand.map(|v| v as i32)),
    }
}

/// The prover of a prefix-suffix read, the mirror of [`table_read_verifier`]: the operand's
/// opening appended at the cycle point with its claim from the trace, then the read-raf prover
/// of the read's table set up. Returns the prover and the read's addresses, for its one-hot
/// checks.
pub fn table_read_prover<F: JoltField, T: Transcript>(
    read: &TableRead,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> (Box<dyn DynRelationWitness<F, T>>, Vec<usize>) {
    let operand = table_read_operand(read, trace);
    let r_cycle = read
        .cycle_point(&*accumulator)
        .expect("the prover holds every claim it reads");
    let claim = operand.evaluate_mle(&r_cycle.r);
    accumulator.append_virtual(transcript, read.operand, r_cycle, claim);
    let addresses = table_read_addresses(read, &operand);
    let indices: Vec<usize> = addresses.iter().map(|&x| x.into()).collect();
    let prover: Box<dyn DynRelationWitness<F, T>> = match read.table {
        PrefixSuffixTable::Relu => Box::new(ps_read_raf_prover::<F, T, ReluTable<XLEN>, XLEN>(
            read,
            addresses,
            accumulator,
            transcript,
        )),
        PrefixSuffixTable::Clamp => Box::new(ps_read_raf_prover::<F, T, ClampTable<XLEN>, XLEN>(
            read,
            addresses,
            accumulator,
            transcript,
        )),
        PrefixSuffixTable::ActivationClamp => Box::new(ps_read_raf_prover::<
            F,
            T,
            ActivationClampTable<XLEN>,
            XLEN,
        >(
            read, addresses, accumulator, transcript
        )),
        PrefixSuffixTable::SiluClamp => {
            Box::new(ps_read_raf_prover::<F, T, SiluClampTable<XLEN>, XLEN>(
                read,
                addresses,
                accumulator,
                transcript,
            ))
        }
        PrefixSuffixTable::RightShift => {
            Box::new(ps_read_raf_prover::<F, T, RightShiftTable<XLEN>, XLEN>(
                read,
                addresses,
                accumulator,
                transcript,
            ))
        }
        PrefixSuffixTable::SoftmaxClamp => {
            Box::new(ps_read_raf_prover::<F, T, SoftmaxClampTable<XLEN>, XLEN>(
                read,
                addresses,
                accumulator,
                transcript,
            ))
        }
    };
    (prover, indices)
}
