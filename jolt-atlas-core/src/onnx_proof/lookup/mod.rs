//! The prover's side of the table reads ([`jolt_atlas_verifier::onnx_proof::lookup`]).

pub use jolt_atlas_verifier::onnx_proof::lookup::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{activation_clamped, trig};
use crate::onnx_proof::relation::{OpRel, RelationChallenge, RelationParams, RelationSetup};
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
    utils::thread::unsafe_allocate_zero_vec,
};
use atlas_claims::SymbolicSumcheck;
use atlas_sumcheck::engine::{prover::ProveRounds, SumcheckError};
use atlas_sumcheck::{opening_proof::OpeningAccumulator, relation::prover::RelationWitness};
use atlas_trace::{Op, Tensor, Trace};
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
    let def = SmallTableDef::for_node(node).expect("the relation was built for a small table");
    let addresses = small_table_addresses(node, trace, def.vars).padded_pow2();
    vec![addresses.evaluate_mle(&setup.incoming_point)]
}

/// The table index each padded cell of `node` reads, for the read's one-hot checks.
pub fn small_table_lookup_indices(node: &ComputationNode, trace: &Trace) -> Vec<usize> {
    let def = SmallTableDef::for_node(node).expect("the node reads a small table");
    small_table_addresses(node, trace, def.vars)
        .padded_pow2()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&address| table_index(&def, address))
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
        let node = &relation.node;
        let def = SmallTableDef::for_node(node).expect("the relation was built for a small table");
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
