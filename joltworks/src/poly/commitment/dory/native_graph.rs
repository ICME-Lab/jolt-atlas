//! A registered tensor graph with one hiding opening and one BlindFold proof.
//! Nodes support exact Atlas arithmetic, axis reductions and table lookup.
//! Shared edges use the same committed tensor in both operator relations.
//! This is not an ONNX importer, a complete Qwen graph, or token provenance.

use super::{
    native_add::{AddRegistration, AddWitness},
    native_lookup::Lookup,
    native_mul::{MulRegistration, NativeMulWitness},
    native_reduce::{shape_bits, Reduction, ReductionRegistration, ReductionWitness},
    DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryScheme, DoryVerifierSetup,
};
use crate::{
    config::{OneHotConfig, OneHotParams},
    curve::{Bn254Curve, Bn254G1},
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, pedersen::PedersenGenerators},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{
            OpeningId, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    subprotocols::{
        blindfold::{
            assembly::NativeBlindFold,
            protocol::{BlindFoldProof, BlindFoldVerifierInput},
            witness::ExtraConstraintWitness,
            BlindFoldAccumulator,
        },
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, ZkSumcheckProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use common::CommittedPoly;
use std::collections::{BTreeMap, BTreeSet};

type SumcheckProof = ZkSumcheckProof<Fr, Bn254Curve, Blake2bTranscript>;
type Prover = Box<dyn SumcheckInstanceProver<Fr, Blake2bTranscript>>;
type Verifier = Box<dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>>;
fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}
fn tensor(i: usize) -> CommittedPoly {
    CommittedPoly::DivNodeQuotient(i)
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphMul {
    pub right: usize,
    pub shift: u8,
}
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphAdd {
    pub right: usize,
    pub subtract: bool,
}
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphReduce {
    pub axes: Vec<usize>,
    pub mean_scale: Option<u8>,
}
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphLookup {
    pub table: Vec<i32>,
    pub log_chunk: u8,
}

/// Exactly one operator must be present. Its output is tensor num_inputs+i.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphNode {
    pub input: usize,
    pub mul: Option<NativeGraphMul>,
    pub add: Option<NativeGraphAdd>,
    pub lookup: Option<NativeGraphLookup>,
    pub reduce: Option<NativeGraphReduce>,
}
impl NativeGraphNode {
    pub fn mul(left: usize, right: usize, shift: u8) -> Self {
        Self {
            input: left,
            mul: Some(NativeGraphMul { right, shift }),
            add: None,
            lookup: None,
            reduce: None,
        }
    }
    pub fn add(left: usize, right: usize) -> Self {
        Self {
            input: left,
            mul: None,
            lookup: None,
            reduce: None,
            add: Some(NativeGraphAdd {
                right,
                subtract: false,
            }),
        }
    }
    pub fn sub(left: usize, right: usize) -> Self {
        Self {
            input: left,
            mul: None,
            lookup: None,
            reduce: None,
            add: Some(NativeGraphAdd {
                right,
                subtract: true,
            }),
        }
    }
    pub fn sum(input: usize, axes: Vec<usize>) -> Self {
        Self {
            input,
            mul: None,
            add: None,
            lookup: None,
            reduce: Some(NativeGraphReduce {
                axes,
                mean_scale: None,
            }),
        }
    }
    pub fn mean_of_squares(input: usize, axes: Vec<usize>, scale: u8) -> Self {
        Self {
            input,
            mul: None,
            add: None,
            lookup: None,
            reduce: Some(NativeGraphReduce {
                axes,
                mean_scale: Some(scale),
            }),
        }
    }
    pub fn lookup(input: usize, table: Vec<i32>, log_chunk: u8) -> Self {
        Self {
            input,
            mul: None,
            add: None,
            lookup: Some(NativeGraphLookup { table, log_chunk }),
            reduce: None,
        }
    }
}

/// The caller registers this exact program, including wiring and output order.
/// Shapes fix row-major integer encoding. Reduced axes retain dimension one.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraph {
    pub context: Vec<u8>,
    pub input_shapes: Vec<Vec<usize>>,
    pub nodes: Vec<NativeGraphNode>,
    pub outputs: Vec<usize>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphStatement {
    pub graph: NativeGraph,
    pub commitments: BTreeMap<CommittedPoly, DoryCommitment>,
}

/// Private witness data and blinding hints never belong in the public bundle.
pub struct NativeGraphWitness {
    polynomials: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
    hints: BTreeMap<CommittedPoly, DoryHint>,
    arithmetic_ranges: BTreeMap<usize, Vec<Vec<u64>>>,
    lookup_indices: BTreeMap<usize, Vec<usize>>,
    outputs: Vec<Vec<i32>>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphProof {
    pub relations: SumcheckProof,
    pub indicators: Option<SumcheckProof>,
    pub openings: SumcheckProof,
    pub pcs: DoryProof,
    pub blindfold: BlindFoldProof<Fr, Bn254Curve>,
}

impl NativeGraph {
    pub fn num_inputs(&self) -> usize {
        self.input_shapes.len()
    }
    fn tensor_count(&self) -> usize {
        self.num_inputs() + self.nodes.len()
    }
    /// Derive every output shape from the registered inputs and operators.
    pub fn tensor_shapes(&self) -> Result<Vec<Vec<usize>>, ProofVerifyError> {
        if self.context.is_empty()
            || self.input_shapes.is_empty()
            || self.nodes.is_empty()
            || self
                .nodes
                .len()
                .checked_mul(8)
                .and_then(|n| n.checked_add(self.num_inputs()))
                .is_none()
        {
            return Err(invalid("Invalid registered native tensor graph"));
        }
        for shape in &self.input_shapes {
            shape_bits(shape)?;
        }
        let mut shapes = self.input_shapes.clone();
        let mut consumed = BTreeSet::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let output = self.num_inputs() + i;
            if node.input >= output {
                return Err(invalid("Graph input must precede its consumer"));
            }
            consumed.insert(node.input);
            let shape = &shapes[node.input];
            let result = match (&node.mul, &node.lookup, &node.add, &node.reduce) {
                (Some(m), None, None, None)
                    if m.right < output
                        && (1..=30).contains(&m.shift)
                        && shapes[m.right] == *shape =>
                {
                    consumed.insert(m.right);
                    shape.clone()
                }
                (None, Some(l), None, None)
                    if l.table.len() >= 2
                        && l.table.len().is_power_of_two()
                        && l.table.len().ilog2() <= 31
                        && matches!(l.log_chunk, 1 | 2 | 4 | 8) =>
                {
                    shape.clone()
                }
                (None, None, Some(a), None) if a.right < output && shapes[a.right] == *shape => {
                    consumed.insert(a.right);
                    shape.clone()
                }
                (None, None, None, Some(r)) => {
                    Reduction::new(shape, &r.axes, r.mean_scale)?.output_shape
                }
                _ => {
                    return Err(invalid(
                        "Exactly one supported operator with compatible shapes is required",
                    ))
                }
            };
            shapes.push(result);
        }
        if (0..self.num_inputs()).any(|i| !consumed.contains(&i))
            || self.outputs.is_empty()
            || self.outputs.iter().any(|i| *i >= self.tensor_count())
            || self.outputs.iter().copied().collect::<BTreeSet<_>>().len() != self.outputs.len()
        {
            return Err(invalid(
                "Unconstrained graph input or invalid registered outputs",
            ));
        }
        Ok(shapes)
    }
    pub fn max_log_rows(&self) -> Result<usize, ProofVerifyError> {
        Ok(self
            .tensor_shapes()?
            .iter()
            .map(|s| shape_bits(s).unwrap())
            .max()
            .unwrap())
    }
    fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
        if self
            .max_log_rows()?
            .checked_add(8)
            .is_none_or(|n| n > max_vars)
        {
            return Err(invalid("Native tensor exceeds the registered setup"));
        }
        Ok(())
    }
    fn multiplication(&self, i: usize) -> MulRegistration {
        let node = &self.nodes[i];
        let m = node.mul.as_ref().unwrap();
        let aux = self.tensor_count() + 4 * i;
        MulRegistration {
            tensors: [
                tensor(node.input),
                tensor(m.right),
                tensor(self.num_inputs() + i),
                tensor(aux),
                tensor(aux + 1),
                tensor(aux + 2),
            ],
            initial: SumcheckId::NodeExecution(4 * i),
            final_stage: SumcheckId::NodeExecution(4 * i + 1),
            namespace: 6 * i,
        }
    }
    fn addition(&self, i: usize) -> AddRegistration {
        let node = &self.nodes[i];
        let add = node.add.as_ref().unwrap();
        let aux = self.tensor_count() + 4 * i;
        AddRegistration {
            tensors: [
                tensor(node.input),
                tensor(add.right),
                tensor(self.num_inputs() + i),
                tensor(aux),
                tensor(aux + 1),
            ],
            initial: SumcheckId::NodeExecution(4 * i),
            final_stage: SumcheckId::NodeExecution(4 * i + 1),
            namespace: 6 * i,
        }
    }
    fn reduction(&self, i: usize, shape: &[usize]) -> ReductionRegistration {
        let node = &self.nodes[i];
        let r = node.reduce.as_ref().unwrap();
        let aux = self.tensor_count() + 4 * i;
        ReductionRegistration {
            tensors: [
                tensor(node.input),
                tensor(self.num_inputs() + i),
                tensor(aux),
                tensor(aux + 1),
                tensor(aux + 2),
                tensor(aux + 3),
            ],
            initial: SumcheckId::NodeExecution(4 * i),
            reduction_stage: SumcheckId::NodeExecution(4 * i + 1),
            clamp_stage: SumcheckId::NodeExecution(4 * i + 2),
            namespace: 6 * i,
            reduction: Reduction::new(shape, &r.axes, r.mean_scale).unwrap(),
        }
    }
    fn lookup(&self, i: usize) -> Lookup {
        let node = &self.nodes[i];
        let l = node.lookup.as_ref().unwrap();
        let log_k = l.table.len().ilog2() as usize;
        let source = SumcheckId::NodeExecution(4 * i);
        Lookup {
            params: OneHotParams::from_config_and_log_K(
                &OneHotConfig {
                    log_k_chunk: l.log_chunk,
                },
                log_k,
            ),
            log_k,
            input: OpeningId::new(tensor(node.input), source),
            output: OpeningId::new(tensor(self.num_inputs() + i), source),
            namespace: 6 * self.nodes.len() + i,
        }
    }
    fn required_keys(&self) -> BTreeSet<CommittedPoly> {
        let shapes = self.tensor_shapes().unwrap();
        let mut keys: BTreeSet<_> = (0..self.tensor_count()).map(tensor).collect();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(m) = &node.mul {
                let registration = self.multiplication(i);
                keys.extend(registration.tensors);
                keys.extend(registration.indicator_keys(m.shift));
            } else if node.add.is_some() {
                let registration = self.addition(i);
                keys.extend(registration.tensors);
                keys.extend(registration.indicator_keys());
            } else if node.reduce.is_some() {
                keys.extend(self.reduction(i, &shapes[node.input]).keys());
            } else {
                let lookup = self.lookup(i);
                keys.extend((0..lookup.params.instruction_d).map(|d| lookup.committed_poly(d)));
            }
        }
        keys
    }
    fn has_lookups(&self) -> bool {
        self.nodes.iter().any(|n| n.lookup.is_some())
    }
}

impl NativeGraphStatement {
    fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
        self.graph.validate(max_vars)?;
        if !self
            .graph
            .required_keys()
            .iter()
            .eq(self.commitments.keys())
        {
            return Err(invalid(
                "Every registered graph tensor and auxiliary commitment is required",
            ));
        }
        Ok(())
    }
    fn transcript(&self) -> Blake2bTranscript {
        let mut t = Blake2bTranscript::new(b"Atlas/private-tensor-graph/v3");
        t.append_serializable(self);
        t
    }
    pub fn input_commitment(&self, input: usize) -> Option<&DoryCommitment> {
        (input < self.graph.num_inputs())
            .then(|| self.commitments.get(&tensor(input)))
            .flatten()
    }
    pub fn output_commitment(&self, output: usize) -> Option<&DoryCommitment> {
        self.graph
            .outputs
            .get(output)
            .and_then(|id| self.commitments.get(&tensor(*id)))
    }
}

impl NativeGraphWitness {
    pub fn commit(
        graph: NativeGraph,
        inputs: Vec<Vec<i32>>,
        setup: &DoryProverSetup,
    ) -> Result<(NativeGraphStatement, Self), ProofVerifyError> {
        graph.validate(setup.verifier.max_log_n)?;
        let shapes = graph.tensor_shapes()?;
        if inputs.len() != graph.num_inputs()
            || inputs
                .iter()
                .zip(&graph.input_shapes)
                .any(|(v, s)| v.len() != 1 << shape_bits(s).unwrap())
        {
            return Err(invalid("Graph witness input shape mismatch"));
        }
        let mut values = inputs;
        let mut polynomials: BTreeMap<_, _> = values
            .iter()
            .enumerate()
            .map(|(i, v)| (tensor(i), MultilinearPolynomial::from(v.clone())))
            .collect();
        let mut arithmetic_ranges = BTreeMap::new();
        let mut lookup_indices = BTreeMap::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            if let Some(m) = &node.mul {
                let registration = graph.multiplication(i);
                let witness =
                    NativeMulWitness::uncommitted(&values[node.input], &values[m.right], m.shift)?;
                values.push(witness.output().to_vec());
                for (id, polynomial) in witness.polynomials {
                    let id = match id {
                        CommittedPoly::DivNodeQuotient(j) if j < 2 => continue,
                        CommittedPoly::DivNodeQuotient(j) => registration.tensors[j],
                        CommittedPoly::NodeOutputRaD(j, d) => {
                            CommittedPoly::NodeOutputRaD(registration.namespace + j, d)
                        }
                        _ => return Err(invalid("Unexpected multiplication witness polynomial")),
                    };
                    if polynomials.insert(id, polynomial).is_some() {
                        return Err(invalid("Graph witness namespace collision"));
                    }
                }
                arithmetic_ranges.insert(i, witness.range_values);
            } else if let Some(add) = &node.add {
                let registration = graph.addition(i);
                let witness =
                    AddWitness::new(&values[node.input], &values[add.right], add.subtract)?;
                values.push(witness.output);
                for (id, polynomial) in witness.polynomials {
                    let id = match id {
                        CommittedPoly::DivNodeQuotient(j) if j < 2 => continue,
                        CommittedPoly::DivNodeQuotient(j) => registration.tensors[j],
                        CommittedPoly::NodeOutputRaD(j, d) => {
                            CommittedPoly::NodeOutputRaD(registration.namespace + j, d)
                        }
                        _ => return Err(invalid("Unexpected addition witness polynomial")),
                    };
                    if polynomials.insert(id, polynomial).is_some() {
                        return Err(invalid("Graph witness namespace collision"));
                    }
                }
                arithmetic_ranges.insert(i, witness.range_values);
            } else if node.reduce.is_some() {
                let registration = graph.reduction(i, &shapes[node.input]);
                let witness = ReductionWitness::new(&values[node.input], &registration.reduction)?;
                values.push(witness.output);
                for (id, polynomial) in witness.polynomials {
                    let id = match id {
                        CommittedPoly::DivNodeQuotient(0) => continue,
                        CommittedPoly::DivNodeQuotient(j) => registration.tensors[j],
                        CommittedPoly::NodeOutputRaD(j, d) => {
                            CommittedPoly::NodeOutputRaD(registration.namespace + j, d)
                        }
                        _ => return Err(invalid("Unexpected reduction witness polynomial")),
                    };
                    if polynomials.insert(id, polynomial).is_some() {
                        return Err(invalid("Graph witness namespace collision"));
                    }
                }
                arithmetic_ranges.insert(i, witness.range_values);
            } else {
                let l = node.lookup.as_ref().unwrap();
                let lookup = graph.lookup(i);
                let indices = values[node.input]
                    .iter()
                    .map(|v| usize::try_from(*v))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| invalid("Negative table input"))?;
                if indices.iter().any(|v| *v >= l.table.len()) {
                    return Err(invalid("Graph table input out of range"));
                }
                let output = indices.iter().map(|j| l.table[*j]).collect::<Vec<_>>();
                polynomials.insert(
                    tensor(graph.num_inputs() + i),
                    MultilinearPolynomial::from(output.clone()),
                );
                values.push(output);
                for d in 0..lookup.params.instruction_d {
                    let digits = indices
                        .iter()
                        .map(|j| Some(u16::from(lookup.params.lookup_index_chunk(*j as u64, d))))
                        .collect();
                    polynomials.insert(
                        lookup.committed_poly(d),
                        MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                            digits,
                            lookup.params.k_chunk,
                        )),
                    );
                }
                lookup_indices.insert(i, indices);
            }
        }
        let outputs = graph.outputs.iter().map(|i| values[*i].clone()).collect();
        let mut statement = NativeGraphStatement {
            graph,
            commitments: BTreeMap::new(),
        };
        let mut hints = BTreeMap::new();
        for (id, p) in &polynomials {
            let (c, h) = DoryScheme::commit_zk(p, setup);
            statement.commitments.insert(*id, c);
            hints.insert(*id, h);
        }
        statement.validate(setup.verifier.max_log_n)?;
        Ok((
            statement,
            Self {
                polynomials,
                hints,
                arithmetic_ranges,
                lookup_indices,
                outputs,
            },
        ))
    }
    /// Prover output only, absent from the verifier's statement and proof.
    pub fn outputs(&self) -> &[Vec<i32>] {
        &self.outputs
    }
}

impl NativeGraphProof {
    fn check_generators(
        statement: &NativeGraphStatement,
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        super::native_opening::NativeOpeningProof::check_generators(setup, gens)?;
        if gens.message_generators.len() <= 3
            || statement.graph.nodes.iter().enumerate().any(|(i, n)| {
                n.lookup.is_some()
                    && statement.graph.lookup(i).params.instruction_d + 1
                        >= gens.message_generators.len()
            })
        {
            return Err(invalid(
                "Graph commitment width is too small for its sumcheck degrees",
            ));
        }
        Ok(())
    }
    pub fn prove(
        statement: &NativeGraphStatement,
        witness: NativeGraphWitness,
        setup: &DoryProverSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<Self, ProofVerifyError> {
        statement.validate(setup.verifier.max_log_n)?;
        Self::check_generators(statement, &DoryVerifierSetup(setup.verifier.clone()), gens)?;
        let NativeGraphWitness {
            polynomials,
            hints,
            arithmetic_ranges,
            lookup_indices,
            ..
        } = witness;
        let graph = &statement.graph;
        let shapes = graph.tensor_shapes()?;
        if !polynomials.keys().eq(statement.commitments.keys())
            || !hints.keys().eq(statement.commitments.keys())
            || arithmetic_ranges.len() + lookup_indices.len() != graph.nodes.len()
        {
            return Err(invalid(
                "Missing graph witness polynomials or operator data",
            ));
        }
        let mut t = statement.transcript();
        let mut a = ProverOpeningAccumulator::new();
        a.zk_mode = true;
        let mut provers: Vec<Prover> = vec![];
        for (i, node) in graph.nodes.iter().enumerate() {
            let log_rows = shape_bits(&shapes[node.input]).unwrap();
            if let Some(m) = &node.mul {
                let values = arithmetic_ranges
                    .get(&i)
                    .ok_or_else(|| invalid("Missing graph multiplication witness"))?;
                if values.len() != 6 || values.iter().any(|v| v.len() != 1 << log_rows) {
                    return Err(invalid("Invalid graph range witness"));
                }
                provers.extend(graph.multiplication(i).provers(
                    m.shift,
                    log_rows,
                    values,
                    &polynomials,
                    &mut a,
                    &mut t,
                ));
            } else if let Some(add) = &node.add {
                let values = arithmetic_ranges
                    .get(&i)
                    .ok_or_else(|| invalid("Missing graph addition witness"))?;
                if values.len() != 5 || values.iter().any(|v| v.len() != 1 << log_rows) {
                    return Err(invalid("Invalid addition range witness"));
                }
                provers.extend(graph.addition(i).provers(
                    add.subtract,
                    log_rows,
                    values,
                    &polynomials,
                    &mut a,
                    &mut t,
                ));
            } else if node.reduce.is_some() {
                let values = arithmetic_ranges
                    .get(&i)
                    .ok_or_else(|| invalid("Missing reduction witness"))?;
                provers.extend(graph.reduction(i, &shapes[node.input]).provers(
                    values,
                    &polynomials,
                    &mut a,
                    &mut t,
                )?);
            } else {
                let l = node.lookup.as_ref().unwrap();
                let lookup = graph.lookup(i);
                let indices = lookup_indices
                    .get(&i)
                    .ok_or_else(|| invalid("Missing graph lookup witness"))?;
                if indices.len() != 1 << log_rows || indices.iter().any(|j| *j >= l.table.len()) {
                    return Err(invalid("Invalid graph lookup witness"));
                }
                let r: Vec<Fr> = t.challenge_vector(log_rows);
                for id in [lookup.input, lookup.output] {
                    a.append_dense(
                        &mut t,
                        id,
                        r.clone(),
                        polynomials[&id.committed_poly().unwrap()].evaluate(&r),
                    );
                }
                provers.push(shout::read_raf_prover(
                    &lookup, indices, &l.table, &a, &mut t,
                ));
            }
        }
        a.take_pending_claims();
        a.take_pending_claim_ids();
        let mut bf = BlindFoldAccumulator::new();
        let mut rng = rand::thread_rng();
        let (relations, _, _) = BatchedSumcheck::prove_zk(
            provers
                .iter_mut()
                .map(|p| p.as_mut() as &mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>)
                .collect(),
            &mut a,
            &mut bf,
            &mut t,
            gens,
            &mut rng,
        );
        drop(provers);
        let indicators = if graph.has_lookups() {
            let mut provers: Vec<Prover> = vec![];
            for (i, node) in graph.nodes.iter().enumerate() {
                if node.lookup.is_some() {
                    provers.extend(shout::ra_onehot_provers_from_params(
                        graph.lookup(i).indicator_params(&a, &mut t),
                        &lookup_indices[&i],
                    ));
                }
            }
            Some(
                BatchedSumcheck::prove_zk(
                    provers
                        .iter_mut()
                        .map(|p| {
                            p.as_mut() as &mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>
                        })
                        .collect(),
                    &mut a,
                    &mut bf,
                    &mut t,
                    gens,
                    &mut rng,
                )
                .0,
            )
        } else {
            None
        };
        a.prepare_for_sumcheck(&polynomials, &mut t);
        let (openings, point) = a.prove_batch_opening_sumcheck_zk::<_, Bn254Curve, _>(
            &mut bf,
            &mut vec![],
            gens,
            &mut rng,
            &mut t,
        );
        let state = a.finalize_batch_opening_sumcheck(point, &mut t);
        // Every required tensor is opened. Silent unproved inputs are forbidden.
        if !state.polynomials.iter().eq(polynomials.keys()) {
            return Err(invalid("Graph polynomial lacks a required opening"));
        }
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let value = coefficients
            .iter()
            .zip(&state.sumcheck_claims)
            .map(|(x, y)| *x * *y)
            .sum();
        let (pcs, eval, blind) = DoryScheme::prove_rlc_zk(
            setup,
            &polynomials,
            &state.poly_coeffs,
            hints.into_values().collect(),
            &state.r_sumcheck,
            &mut t,
        )?;
        if eval != gens.commit(&[value], &blind) {
            return Err(invalid("Graph PCS evaluation mismatch"));
        }
        let data = bf.take_stage_data();
        let native = NativeBlindFold::new(
            NativeBlindFold::prover_relations(&data),
            &[relation],
            &coefficients,
            gens.message_generators.len(),
        )?;
        let values = a.openings.iter().map(|(id, (_, v))| (*id, *v)).collect();
        let blindfold = native.prove(
            &data,
            &values,
            vec![ExtraConstraintWitness {
                output_value: value,
                blinding: blind,
                challenge_values: coefficients,
                opening_values: state.sumcheck_claims,
            }],
            vec![eval],
            gens,
            &mut t,
        )?;
        Ok(Self {
            relations,
            indicators,
            openings,
            pcs,
            blindfold,
        })
    }

    pub fn verify(
        &self,
        statement: &NativeGraphStatement,
        setup: &DoryVerifierSetup,
        gens: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        statement.validate(setup.0.max_log_n)?;
        Self::check_generators(statement, setup, gens)?;
        let graph = &statement.graph;
        let shapes = graph.tensor_shapes()?;
        if self.indicators.is_some() != graph.has_lookups() {
            return Err(invalid(
                "Graph indicator proof presence does not match its operators",
            ));
        }
        let mut t = statement.transcript();
        let mut a = VerifierOpeningAccumulator::new_zk();
        let mut verifiers: Vec<Verifier> = vec![];
        for (i, node) in graph.nodes.iter().enumerate() {
            let log_rows = shape_bits(&shapes[node.input]).unwrap();
            if let Some(m) = &node.mul {
                verifiers.extend(
                    graph
                        .multiplication(i)
                        .verifiers(m.shift, log_rows, &mut a, &mut t),
                );
            } else if let Some(add) = &node.add {
                verifiers.extend(graph.addition(i).verifiers(
                    add.subtract,
                    log_rows,
                    &mut a,
                    &mut t,
                ));
            } else if node.reduce.is_some() {
                verifiers.extend(
                    graph
                        .reduction(i, &shapes[node.input])
                        .verifiers(&mut a, &mut t),
                );
            } else {
                let l = node.lookup.as_ref().unwrap();
                let lookup = graph.lookup(i);
                let r: Vec<Fr> = t.challenge_vector(log_rows);
                for id in [lookup.input, lookup.output] {
                    a.append_dense(&mut t, id, r.clone());
                }
                verifiers.push(shout::read_raf_verifier(
                    &lookup,
                    l.table.clone(),
                    &a,
                    &mut t,
                ));
            }
        }
        BatchedSumcheck::verify_zk_with_width(
            &self.relations,
            verifiers.iter().map(|p| p.as_ref()).collect(),
            &mut a,
            &mut t,
            gens.message_generators.len(),
        )?;
        if let Some(proof) = &self.indicators {
            let mut verifiers: Vec<Verifier> = vec![];
            for (i, node) in graph.nodes.iter().enumerate() {
                if node.lookup.is_some() {
                    verifiers.extend(shout::ra_onehot_verifiers_with_params(
                        graph.lookup(i).indicator_params(&a, &mut t),
                    ));
                }
            }
            BatchedSumcheck::verify_zk_with_width(
                proof,
                verifiers.iter().map(|p| p.as_ref()).collect(),
                &mut a,
                &mut t,
                gens.message_generators.len(),
            )?;
        }
        let point = a.verify_batch_opening_sumcheck_zk(
            &self.openings,
            &mut t,
            gens.message_generators.len(),
        )?;
        let state =
            a.finalize_batch_opening_sumcheck(point, &vec![Fr::zero(); a.num_sumchecks()], &mut t);
        if !state.polynomials.iter().eq(statement.commitments.keys()) {
            return Err(invalid("Graph commitment lacks a required opening"));
        }
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let commitments = state
            .polynomials
            .iter()
            .map(|p| statement.commitments[p])
            .collect::<Vec<_>>();
        let joint = DoryScheme::combine_commitments(&commitments, &state.poly_coeffs);
        let eval = self
            .pcs
            .0
            .y_com
            .map(|g| Bn254G1(g.0))
            .ok_or_else(|| invalid("Missing hidden graph evaluation"))?;
        DoryScheme::verify_zk(&self.pcs, setup, &mut t, &state.r_sumcheck, &eval, &joint)?;
        let native = NativeBlindFold::new(
            a.zk_stages,
            &[relation],
            &coefficients,
            gens.message_generators.len(),
        )?;
        let stages: Vec<_> = std::iter::once(&self.relations)
            .chain(self.indicators.iter())
            .chain(std::iter::once(&self.openings))
            .collect();
        native.verify(
            &self.blindfold,
            &BlindFoldVerifierInput {
                round_commitments: stages
                    .iter()
                    .flat_map(|s| s.round_commitments.iter().copied())
                    .collect(),
                output_claims_row_commitments: stages
                    .iter()
                    .flat_map(|s| s.output_claims_commitments.iter().copied())
                    .collect(),
                eval_commitments: vec![eval],
            },
            gens,
            &mut t,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> NativeGraph {
        NativeGraph {
            context: b"registered vector graph fixture".to_vec(),
            input_shapes: vec![vec![8]; 2],
            nodes: vec![
                NativeGraphNode::mul(0, 1, 14),
                NativeGraphNode::lookup(2, (0..8).rev().collect(), 2),
                NativeGraphNode::mul(3, 1, 14),
                NativeGraphNode::lookup(2, (0..8).map(|v| v * v).collect(), 2),
                NativeGraphNode::mul(4, 5, 1),
            ],
            outputs: vec![6, 3],
        }
    }
    fn inputs() -> Vec<Vec<i32>> {
        vec![(0..8).collect(), vec![1 << 14; 8]]
    }

    #[test]
    fn native_graph_shares_opening_and_blindfold_across_mixed_branches() {
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let (statement, witness) = NativeGraphWitness::commit(graph(), inputs(), &pp).unwrap();
        assert_eq!(witness.outputs()[0], vec![0, 3, 10, 18, 24, 25, 18, 0]);
        assert_eq!(witness.outputs()[1], vec![7, 6, 5, 4, 3, 2, 1, 0]);
        let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
        let mut bytes = vec![];
        proof.serialize_compressed(&mut bytes).unwrap();
        let proof = NativeGraphProof::deserialize_compressed(bytes.as_slice()).unwrap();
        proof.verify(&statement, &vp, &gens).unwrap();
        println!("five-node native graph proof bytes={}", bytes.len());
        let mut changed = statement.clone();
        changed.graph.nodes[2].input = 0;
        assert!(proof.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.graph.outputs.reverse();
        assert!(proof.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.graph.context.push(1);
        assert!(proof.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.graph.nodes.swap(0, 1);
        assert!(proof.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.graph.nodes.pop();
        assert!(proof.verify(&changed, &vp, &gens).is_err());
        let mut changed = statement.clone();
        changed.commitments.remove(&tensor(2));
        assert!(proof.verify(&changed, &vp, &gens).is_err());
        let mut changed = proof.clone();
        changed.indicators = None;
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = proof.clone();
        changed.pcs.0.y_com = None;
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = proof.clone();
        changed.relations.output_claims_commitments.clear();
        assert!(changed.verify(&statement, &vp, &gens).is_err());
        let mut changed = proof;
        changed.blindfold.folded_eval_outputs[0] += Fr::from(1u64);
        assert!(changed.verify(&statement, &vp, &gens).is_err());
    }

    #[test]
    fn native_graph_rejects_a_consistent_lookup_for_the_wrong_producer_values() {
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let mut graph = graph();
        graph.nodes.truncate(2);
        graph.outputs = vec![3];
        let (mut statement, mut witness) =
            NativeGraphWitness::commit(graph, inputs(), &pp).unwrap();
        let lookup = statement.graph.lookup(1);
        let wrong: Vec<usize> = (0..8).rev().collect();
        let table = &statement.graph.nodes[1].lookup.as_ref().unwrap().table;
        // The substituted consumer is a valid computation in isolation.
        let (consumer, cw) = super::super::native_lookup::NativeLookupWitness::commit(
            b"wrong but independently valid consumer".to_vec(),
            table.clone(),
            wrong.clone(),
            2,
            &pp,
        )
        .unwrap();
        super::super::native_lookup::NativeLookupProof::prove(&consumer, cw, &pp, &gens)
            .unwrap()
            .verify(&consumer, &vp, &gens)
            .unwrap();
        let output: Vec<i32> = wrong.iter().map(|i| table[*i]).collect();
        // Replace the entire consuming lookup's witness, including its output
        // and all indicators. The shared producer tensor remains unchanged.
        let mut replacements = vec![(tensor(3), MultilinearPolynomial::from(output))];
        for d in 0..lookup.params.instruction_d {
            let digits = wrong
                .iter()
                .map(|i| Some(u16::from(lookup.params.lookup_index_chunk(*i as u64, d))))
                .collect();
            replacements.push((
                lookup.committed_poly(d),
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    digits,
                    lookup.params.k_chunk,
                )),
            ));
        }
        for (id, p) in replacements {
            let (c, h) = DoryScheme::commit_zk(&p, &pp);
            witness.polynomials.insert(id, p);
            witness.hints.insert(id, h);
            statement.commitments.insert(id, c);
        }
        witness.lookup_indices.insert(1, wrong);
        match NativeGraphProof::prove(&statement, witness, &pp, &gens) {
            Err(_) => {}
            Ok(proof) => assert!(proof.verify(&statement, &vp, &gens).is_err()),
        }
    }

    #[test]
    fn native_graph_validates_registry_and_supports_each_operator_family() {
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        for kind in 0..3 {
            let graph = if kind == 2 {
                NativeGraph {
                    input_shapes: vec![vec![8]; 1],
                    nodes: vec![NativeGraphNode::mul(0, 0, 1)],
                    outputs: vec![1],
                    ..graph()
                }
            } else if kind == 0 {
                NativeGraph {
                    nodes: vec![NativeGraphNode::mul(0, 1, 14)],
                    outputs: vec![2],
                    ..graph()
                }
            } else {
                NativeGraph {
                    input_shapes: vec![vec![8]; 1],
                    nodes: vec![NativeGraphNode::lookup(
                        0,
                        vec![1, -1, 3, -3, 5, -5, 7, -7],
                        2,
                    )],
                    outputs: vec![1],
                    ..graph()
                }
            };
            let input = if kind == 0 {
                inputs()
            } else {
                vec![(0..8).collect()]
            };
            let (statement, witness) = NativeGraphWitness::commit(graph, input, &pp).unwrap();
            let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
            assert_eq!(proof.indicators.is_some(), kind == 1);
            proof.verify(&statement, &vp, &gens).unwrap();
        }
        let mut invalid_graph = graph();
        invalid_graph.nodes[0].input = 2;
        assert!(NativeGraphWitness::commit(invalid_graph, inputs(), &pp).is_err());
        let mut invalid_graph = graph();
        invalid_graph.nodes[0].lookup = Some(NativeGraphLookup {
            table: vec![0, 1],
            log_chunk: 1,
        });
        assert!(NativeGraphWitness::commit(invalid_graph, inputs(), &pp).is_err());
        let mut invalid_graph = graph();
        invalid_graph.nodes[0].mul = None;
        assert!(NativeGraphWitness::commit(invalid_graph, inputs(), &pp).is_err());
        let mut invalid_graph = graph();
        invalid_graph.outputs = vec![6, 6];
        assert!(NativeGraphWitness::commit(invalid_graph, inputs(), &pp).is_err());
        let invalid_graph = NativeGraph {
            input_shapes: vec![vec![8]; 3],
            nodes: vec![NativeGraphNode::mul(0, 1, 14)],
            outputs: vec![3],
            ..graph()
        };
        let mut data = inputs();
        data.push(vec![0; 8]);
        assert!(NativeGraphWitness::commit(invalid_graph, data, &pp).is_err());
    }

    #[test]
    fn native_graph_clamped_add_sub_match_atlas_and_bind_operator() {
        use atlas_onnx_tracer::{
            ops::{Add, Op, Sub},
            tensor::Tensor,
        };
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let left = vec![i32::MIN, i32::MAX, i32::MIN, i32::MAX, -1, 0, 123, -456];
        let right = vec![i32::MIN, i32::MAX, i32::MAX, i32::MIN, 1, -1, -456, 123];
        let a = Tensor::new(Some(&left), &[8]).unwrap();
        let b = Tensor::new(Some(&right), &[8]).unwrap();
        let expected = vec![
            Add.f(vec![&a, &b]).data().to_vec(),
            Sub.f(vec![&a, &b]).data().to_vec(),
        ];
        let graph = NativeGraph {
            input_shapes: vec![vec![8]; 2],
            nodes: vec![NativeGraphNode::add(0, 1), NativeGraphNode::sub(0, 1)],
            outputs: vec![2, 3],
            ..graph()
        };
        let (statement, witness) =
            NativeGraphWitness::commit(graph, vec![left, right], &pp).unwrap();
        assert_eq!(witness.outputs(), expected);
        let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
        let mut bytes = vec![];
        proof.serialize_compressed(&mut bytes).unwrap();
        let proof = NativeGraphProof::deserialize_compressed(bytes.as_slice()).unwrap();
        proof.verify(&statement, &vp, &gens).unwrap();
        let mut wrong = statement.clone();
        wrong.graph.nodes[0].add.as_mut().unwrap().subtract = true;
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong.graph.nodes[1].add.as_mut().unwrap().right = 0;
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong
            .commitments
            .remove(&statement.graph.addition(0).indicator_keys()[0]);
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement;
        wrong.graph.nodes[0].mul = Some(NativeGraphMul {
            right: 1,
            shift: 14,
        });
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
    }

    #[test]
    fn native_graph_add_sub_compose_with_mul_lookup_and_reused_operands() {
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let graph = NativeGraph {
            input_shapes: vec![vec![8]; 3],
            nodes: vec![
                NativeGraphNode::add(0, 1),
                NativeGraphNode::sub(0, 1),
                NativeGraphNode::mul(3, 2, 14),
                NativeGraphNode::sub(5, 3),
                NativeGraphNode::lookup(6, vec![3, 4], 1),
                NativeGraphNode::add(4, 7),
                NativeGraphNode::add(8, 8),
                NativeGraphNode::sub(9, 9),
            ],
            outputs: vec![3, 4, 5, 8, 10],
            ..graph()
        };
        let data = vec![vec![i32::MAX; 8], vec![1; 8], vec![1 << 14; 8]];
        let (statement, witness) = NativeGraphWitness::commit(graph, data, &pp).unwrap();
        assert_eq!(
            witness.outputs(),
            vec![
                vec![i32::MAX; 8],
                vec![i32::MAX - 1; 8],
                vec![i32::MAX; 8],
                vec![i32::MAX; 8],
                vec![0; 8]
            ]
        );
        let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
        proof.verify(&statement, &vp, &gens).unwrap();
        let mut wrong = statement;
        wrong.graph.nodes[5].input = 3;
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
    }

    #[test]
    fn native_graph_add_sub_reject_false_clamps_with_matching_range_commitments() {
        use super::super::native_mul::Range;
        use crate::field::JoltField;
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        // Replace the entire arithmetic witness and its indicator commitments.
        // Each tuple is (subtract, a, b, y, low, high). Some satisfy all field
        // arithmetic identities and fail only the required integer ranges.
        let cases = [
            (false, i32::MAX as i64, 1, i32::MIN as i64, 0, 1i64 << 32),
            (false, i32::MAX as i64, 1, 1i64 << 31, 0, 0),
            (false, 10, 20, 29, 0, 1),
            (false, 0, 0, i32::MIN as i64, 0, 1i64 << 31),
            (false, 0, 0, 1, 0, -1),
            (false, 1i64 << 31, 0, i32::MAX as i64, 0, 1),
            (true, 0, -1, -1, 0, 2),
            (true, i32::MIN as i64, 1, i32::MAX as i64, 1i64 << 32, 0),
        ];
        for (subtract, a, b, y, lo, hi) in cases {
            let graph = NativeGraph {
                input_shapes: vec![vec![8]; 2],
                nodes: vec![if subtract {
                    NativeGraphNode::sub(0, 1)
                } else {
                    NativeGraphNode::add(0, 1)
                }],
                outputs: vec![2],
                ..graph()
            };
            let (mut statement, mut witness) =
                NativeGraphWitness::commit(graph, vec![vec![0; 8]; 2], &pp).unwrap();
            let registration = statement.graph.addition(0);
            let raw = [a, b, y, lo, hi];
            let mut ranges = vec![];
            let mut replacements = vec![];
            for (j, value) in raw.iter().enumerate() {
                replacements.push((
                    registration.tensors[j],
                    MultilinearPolynomial::from(vec![Fr::from_i64(*value); 8]),
                ));
                let r = Range::new(j, 32, if j < 3 { 1 << 31 } else { 0 });
                let offset = value.wrapping_add(r.offset as i64) as u64;
                ranges.push(vec![offset; 8]);
                for d in 0..r.chunks() {
                    let digits = vec![Some(u16::from(r.digit(offset, d))); 8];
                    replacements.push((
                        CommittedPoly::NodeOutputRaD(registration.namespace + j, d),
                        MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                            digits,
                            1 << r.chunk,
                        )),
                    ));
                }
            }
            for (id, p) in replacements {
                let (c, h) = DoryScheme::commit_zk(&p, &pp);
                witness.polynomials.insert(id, p);
                witness.hints.insert(id, h);
                statement.commitments.insert(id, c);
            }
            witness.arithmetic_ranges.insert(0, ranges);
            match NativeGraphProof::prove(&statement, witness, &pp, &gens) {
                Err(_) => {}
                Ok(proof) => assert!(
                    proof.verify(&statement, &vp, &gens).is_err(),
                    "accepted false clamp {raw:?}"
                ),
            }
        }
    }

    #[test]
    fn native_graph_add_rejects_a_valid_consumer_for_different_hidden_values() {
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let standalone = NativeGraph {
            input_shapes: vec![vec![8]; 2],
            nodes: vec![NativeGraphNode::add(0, 1)],
            outputs: vec![2],
            ..graph()
        };
        let (s, w) =
            NativeGraphWitness::commit(standalone, vec![vec![9; 8], vec![1; 8]], &pp).unwrap();
        NativeGraphProof::prove(&s, w, &pp, &gens)
            .unwrap()
            .verify(&s, &vp, &gens)
            .unwrap();
        let graph = NativeGraph {
            input_shapes: vec![vec![8]; 2],
            nodes: vec![NativeGraphNode::add(0, 1), NativeGraphNode::add(2, 1)],
            outputs: vec![3],
            ..graph()
        };
        let (mut statement, mut witness) =
            NativeGraphWitness::commit(graph, vec![vec![0; 8], vec![1; 8]], &pp).unwrap();
        let replacement = AddWitness::new(&[9; 8], &[1; 8], false).unwrap();
        let registration = statement.graph.addition(1);
        for (id, p) in replacement.polynomials {
            let id = match id {
                CommittedPoly::DivNodeQuotient(j) if j < 2 => continue,
                CommittedPoly::DivNodeQuotient(j) => registration.tensors[j],
                CommittedPoly::NodeOutputRaD(j, d) => {
                    CommittedPoly::NodeOutputRaD(registration.namespace + j, d)
                }
                _ => panic!("unexpected witness key"),
            };
            let (c, h) = DoryScheme::commit_zk(&p, &pp);
            witness.polynomials.insert(id, p);
            witness.hints.insert(id, h);
            statement.commitments.insert(id, c);
        }
        witness
            .arithmetic_ranges
            .insert(1, replacement.range_values);
        match NativeGraphProof::prove(&statement, witness, &pp, &gens) {
            Err(_) => {}
            Ok(proof) => assert!(proof.verify(&statement, &vp, &gens).is_err()),
        }
    }

    fn reduction_graph(shape: Vec<usize>, node: NativeGraphNode) -> NativeGraph {
        NativeGraph {
            context: b"registered exact reduction".to_vec(),
            input_shapes: vec![shape],
            nodes: vec![node],
            outputs: vec![1],
        }
    }

    fn atlas_outputs(graph: &NativeGraph, inputs: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        use atlas_onnx_tracer::{
            ops::{Add, MeanOfSquares, Mul, Op, Sub, Sum},
            tensor::Tensor,
        };
        // Atlas represents a scalar with shape [1]; the native API also
        // accepts an empty shape. Use the Atlas scalar encoding for its kernel.
        let shapes = graph
            .tensor_shapes()
            .unwrap()
            .into_iter()
            .map(|s| if s.is_empty() { vec![1] } else { s })
            .collect::<Vec<_>>();
        let mut values = inputs;
        for node in &graph.nodes {
            let a = Tensor::new(Some(&values[node.input]), &shapes[node.input]).unwrap();
            let out = if let Some(r) = &node.reduce {
                if let Some(scale) = r.mean_scale {
                    let count = r
                        .axes
                        .iter()
                        .map(|axis| shapes[node.input][*axis])
                        .product();
                    MeanOfSquares {
                        axes: r.axes.clone(),
                        scale: i32::from(scale),
                        count,
                        padded_count: count,
                    }
                    .f(vec![&a])
                } else {
                    Sum {
                        axes: r.axes.clone(),
                    }
                    .f(vec![&a])
                }
            } else if let Some(m) = &node.mul {
                let b = Tensor::new(Some(&values[m.right]), &shapes[m.right]).unwrap();
                Mul {
                    scale: i32::from(m.shift),
                }
                .f(vec![&a, &b])
            } else if let Some(add) = &node.add {
                let b = Tensor::new(Some(&values[add.right]), &shapes[add.right]).unwrap();
                if add.subtract {
                    Sub.f(vec![&a, &b])
                } else {
                    Add.f(vec![&a, &b])
                }
            } else {
                let l = node.lookup.as_ref().unwrap();
                Tensor::new(
                    Some(
                        &values[node.input]
                            .iter()
                            .map(|x| l.table[*x as usize])
                            .collect::<Vec<_>>(),
                    ),
                    &shapes[node.input],
                )
                .unwrap()
            };
            assert_eq!(out.dims(), &shapes[values.len()]);
            values.push(out.data().to_vec());
        }
        graph.outputs.iter().map(|i| values[*i].clone()).collect()
    }

    #[test]
    fn native_graph_reductions_match_atlas_with_mixed_shapes_and_scalar_consumers() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let graph = NativeGraph {
            context: b"mixed tensor shapes".to_vec(),
            input_shapes: vec![vec![2, 4, 2], vec![2, 1, 2]],
            nodes: vec![
                NativeGraphNode::sum(0, vec![1]),
                NativeGraphNode::add(2, 1),
                NativeGraphNode::sum(0, vec![0, 2]),
                NativeGraphNode::mean_of_squares(0, vec![0, 2], 14),
                NativeGraphNode::mul(4, 5, 14),
                NativeGraphNode::sum(3, vec![0, 1, 2]),
                NativeGraphNode::sub(7, 7),
                NativeGraphNode::lookup(8, vec![3, 7], 1),
                NativeGraphNode::mul(9, 9, 1),
            ],
            outputs: vec![2, 3, 4, 5, 6, 7, 10],
        };
        let inputs = vec![
            vec![
                i32::MIN,
                1,
                0,
                -1,
                i32::MAX,
                20,
                -30,
                40,
                0,
                1,
                -i32::MAX,
                i32::MAX,
                1,
                2,
                -20,
                30,
            ],
            vec![1, -1, 2, -2],
        ];
        let expected = atlas_outputs(&graph, inputs.clone());
        let (statement, witness) = NativeGraphWitness::commit(graph, inputs, &pp).unwrap();
        assert_eq!(witness.outputs(), expected);
        let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
        let mut bytes = vec![];
        proof.serialize_compressed(&mut bytes).unwrap();
        NativeGraphProof::deserialize_compressed(bytes.as_slice())
            .unwrap()
            .verify(&statement, &vp, &gens)
            .unwrap();
        let mut wrong = statement.clone();
        wrong.graph.input_shapes[0] = vec![4, 2, 2];
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement.clone();
        wrong.graph.nodes[2].reduce.as_mut().unwrap().axes = vec![1];
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
        let mut wrong = statement;
        wrong.graph.nodes[3].reduce.as_mut().unwrap().mean_scale = Some(13);
        assert!(proof.verify(&wrong, &vp, &gens).is_err());
    }

    #[test]
    fn native_graph_reductions_clamp_once_and_reject_invalid_shapes_or_overflow() {
        let pp = DoryScheme::setup_prover(11);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        for (shape, op, values, expected) in [
            (
                vec![4],
                NativeGraphNode::sum(0, vec![0]),
                vec![i32::MAX, 1, -1, -i32::MAX],
                0,
            ),
            (
                vec![2],
                NativeGraphNode::mean_of_squares(0, vec![0], 14),
                vec![1 << 25, 0],
                i32::MAX,
            ),
            (
                vec![2],
                NativeGraphNode::mean_of_squares(0, vec![0], 0),
                vec![i32::MAX, i32::MAX],
                i32::MAX,
            ),
            (
                vec![],
                NativeGraphNode::mean_of_squares(0, vec![], 0),
                vec![-3],
                9,
            ),
        ] {
            let graph = reduction_graph(shape, op);
            let reference = atlas_outputs(&graph, vec![values.clone()]);
            let (s, w) = NativeGraphWitness::commit(graph, vec![values], &pp).unwrap();
            assert_eq!(w.outputs(), vec![vec![expected]]);
            assert_eq!(w.outputs(), reference);
            NativeGraphProof::prove(&s, w, &pp, &gens)
                .unwrap()
                .verify(&s, &vp, &gens)
                .unwrap();
        }
        let g = reduction_graph(vec![2], NativeGraphNode::mean_of_squares(0, vec![0], 0));
        assert!(NativeGraphWitness::commit(g, vec![vec![i32::MIN; 2]], &pp).is_err());
        for shape in [vec![0], vec![3], vec![1usize << 31], vec![1; 33]] {
            assert!(reduction_graph(shape, NativeGraphNode::sum(0, vec![]))
                .tensor_shapes()
                .is_err());
        }
        for axes in [vec![0, 0], vec![1, 0], vec![2]] {
            assert!(reduction_graph(vec![2, 2], NativeGraphNode::sum(0, axes))
                .tensor_shapes()
                .is_err());
        }
        assert!(
            reduction_graph(vec![2], NativeGraphNode::mean_of_squares(0, vec![0], 30))
                .tensor_shapes()
                .is_err()
        );
        let mut g = reduction_graph(vec![2, 2], NativeGraphNode::add(0, 1));
        g.input_shapes.push(vec![4]);
        g.outputs = vec![2];
        assert!(g.tensor_shapes().is_err());
    }

    #[test]
    fn native_graph_reductions_reject_valid_witness_for_wrong_axes() {
        let pp = DoryScheme::setup_prover(10);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let data = vec![1, 2, 3, 4];
        let g = reduction_graph(vec![2, 2], NativeGraphNode::sum(0, vec![0]));
        let (s, w) = NativeGraphWitness::commit(g, vec![data.clone()], &pp).unwrap();
        NativeGraphProof::prove(&s, w, &pp, &gens)
            .unwrap()
            .verify(&s, &vp, &gens)
            .unwrap();
        let g = reduction_graph(vec![2, 2], NativeGraphNode::sum(0, vec![1]));
        let (mut s, mut w) = NativeGraphWitness::commit(g, vec![data.clone()], &pp).unwrap();
        let registration = s.graph.reduction(0, &[2, 2]);
        let replacement =
            ReductionWitness::new(&data, &Reduction::new(&[2, 2], &[0], None).unwrap()).unwrap();
        for (id, p) in replacement.polynomials {
            let id = match id {
                CommittedPoly::DivNodeQuotient(0) => continue,
                CommittedPoly::DivNodeQuotient(j) => registration.tensors[j],
                CommittedPoly::NodeOutputRaD(j, d) => {
                    CommittedPoly::NodeOutputRaD(registration.namespace + j, d)
                }
                _ => panic!("unexpected reduction key"),
            };
            let (c, h) = DoryScheme::commit_zk(&p, &pp);
            w.polynomials.insert(id, p);
            w.hints.insert(id, h);
            s.commitments.insert(id, c);
        }
        w.arithmetic_ranges.insert(0, replacement.range_values);
        if let Ok(proof) = NativeGraphProof::prove(&s, w, &pp, &gens) {
            assert!(proof.verify(&s, &vp, &gens).is_err());
        }
    }

    #[test]
    fn native_graph_reductions_reject_false_integers_with_matching_commitments() {
        use super::super::native_mul::Range;
        let pp = DoryScheme::setup_prover(9);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        // Every column and every indicator is replaced consistently. The final
        // two cases satisfy the field identities and fail an integer range.
        let cases: Vec<(bool, Vec<i128>, [i128; 5])> = vec![
            (false, vec![0, 0], [-1, 0, 0, 1, 0]),
            (false, vec![0, 0], [1, 1, 0, 0, 0]),
            (true, vec![1, 1], [0, 2, 0, 0, 2]),
            (true, vec![1, 0], [1, 1, 0, 0, -1]),
            (
                true,
                vec![i128::from(i32::MIN); 2],
                [
                    i128::from(i32::MAX),
                    1i128 << 63,
                    0,
                    (1i128 << 62) - i128::from(i32::MAX),
                    0,
                ],
            ),
        ];
        for (square, input, raw) in cases {
            let op = if square {
                NativeGraphNode::mean_of_squares(0, vec![0], 0)
            } else {
                NativeGraphNode::sum(0, vec![0])
            };
            let g = reduction_graph(vec![2], op);
            let (mut s, mut w) = NativeGraphWitness::commit(g, vec![vec![0; 2]], &pp).unwrap();
            let registration = s.graph.reduction(0, &[2]);
            let columns = std::iter::once(input)
                .chain(
                    raw[..registration.reduction.columns() - 1]
                        .iter()
                        .map(|v| vec![*v]),
                )
                .collect::<Vec<_>>();
            let mut ranges = vec![];
            for (j, values) in columns.iter().enumerate() {
                let bits = if j < 2 {
                    32
                } else if j == 5 {
                    1
                } else {
                    64
                };
                let offset = if j < 2 {
                    1u64 << 31
                } else if j == 2 {
                    1u64 << 63
                } else {
                    0
                };
                let r = Range::new(j, bits, offset);
                let field = values
                    .iter()
                    .map(|v| {
                        if *v < 0 {
                            -Fr::from((-v) as u64)
                        } else {
                            Fr::from(*v as u64)
                        }
                    })
                    .collect::<Vec<_>>();
                let offsets = values
                    .iter()
                    .map(|v| (v + i128::from(offset)) as u64)
                    .collect::<Vec<_>>();
                let mut replacements =
                    vec![(registration.tensors[j], MultilinearPolynomial::from(field))];
                for d in 0..r.chunks() {
                    replacements.push((
                        CommittedPoly::NodeOutputRaD(registration.namespace + j, d),
                        MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                            offsets
                                .iter()
                                .map(|v| Some(u16::from(r.digit(*v, d))))
                                .collect(),
                            1 << r.chunk,
                        )),
                    ));
                }
                ranges.push(offsets);
                for (id, p) in replacements {
                    let (c, h) = DoryScheme::commit_zk(&p, &pp);
                    w.polynomials.insert(id, p);
                    w.hints.insert(id, h);
                    s.commitments.insert(id, c);
                }
            }
            w.arithmetic_ranges.insert(0, ranges);
            if let Ok(proof) = NativeGraphProof::prove(&s, w, &pp, &gens) {
                assert!(
                    proof.verify(&s, &vp, &gens).is_err(),
                    "accepted false reduction {raw:?}"
                );
            }
        }
    }
}
