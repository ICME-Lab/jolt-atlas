//! A registered vector graph with one hiding opening and one BlindFold proof.
//! Supported nodes are exact Atlas fused multiplication and public table lookup.
//! Shared edges use the same committed tensor in both operator relations.
//! This is not an ONNX importer, a complete Qwen graph, or token provenance.

use super::{
    native_lookup::Lookup,
    native_mul::{MulRegistration, NativeMulWitness},
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
pub struct NativeGraphLookup {
    pub table: Vec<i32>,
    pub log_chunk: u8,
}

/// Exactly one operator must be present. Its output is tensor num_inputs+i.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraphNode {
    pub input: usize,
    pub mul: Option<NativeGraphMul>,
    pub lookup: Option<NativeGraphLookup>,
}
impl NativeGraphNode {
    pub fn mul(left: usize, right: usize, shift: u8) -> Self {
        Self {
            input: left,
            mul: Some(NativeGraphMul { right, shift }),
            lookup: None,
        }
    }
    pub fn lookup(input: usize, table: Vec<i32>, log_chunk: u8) -> Self {
        Self {
            input,
            mul: None,
            lookup: Some(NativeGraphLookup { table, log_chunk }),
        }
    }
}

/// The caller registers this exact program, including wiring and output order.
/// All tensors use the same flattened, signed integer representation and size.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeGraph {
    pub context: Vec<u8>,
    pub log_rows: usize,
    pub num_inputs: usize,
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
    mul_ranges: BTreeMap<usize, Vec<Vec<u64>>>,
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
    fn tensor_count(&self) -> usize {
        self.num_inputs + self.nodes.len()
    }
    pub(super) fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
        if self.context.is_empty()
            || self.num_inputs == 0
            || self.nodes.is_empty()
            || self.log_rows == 0
            || self.log_rows >= usize::BITS as usize
            || self.log_rows.checked_add(8).is_none_or(|n| n > max_vars)
            || self
                .nodes
                .len()
                .checked_mul(8)
                .and_then(|n| n.checked_add(self.num_inputs))
                .is_none()
        {
            return Err(invalid("Invalid registered native graph dimensions"));
        }
        let mut consumed = BTreeSet::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let output = self.num_inputs + i;
            if node.input >= output {
                return Err(invalid("Graph input must precede its consumer"));
            }
            consumed.insert(node.input);
            match (&node.mul, &node.lookup) {
                (Some(m), None) if m.right < output && (1..=30).contains(&m.shift) => {
                    consumed.insert(m.right);
                }
                (None, Some(l))
                    if l.table.len() >= 2
                        && l.table.len().is_power_of_two()
                        && l.table.len().ilog2() <= 31
                        && matches!(l.log_chunk, 1 | 2 | 4 | 8) => {}
                _ => return Err(invalid("Exactly one supported graph operator is required")),
            }
        }
        // Every input must be constrained by an actual consuming operator.
        if (0..self.num_inputs).any(|i| !consumed.contains(&i))
            || self.outputs.is_empty()
            || self.outputs.iter().any(|i| *i >= self.tensor_count())
            || self.outputs.iter().copied().collect::<BTreeSet<_>>().len() != self.outputs.len()
        {
            return Err(invalid(
                "Unconstrained graph input or invalid registered outputs",
            ));
        }
        Ok(())
    }
    fn multiplication(&self, i: usize) -> MulRegistration {
        let node = &self.nodes[i];
        let m = node.mul.as_ref().unwrap();
        let aux = self.tensor_count() + 3 * i;
        MulRegistration {
            tensors: [
                tensor(node.input),
                tensor(m.right),
                tensor(self.num_inputs + i),
                tensor(aux),
                tensor(aux + 1),
                tensor(aux + 2),
            ],
            initial: SumcheckId::NodeExecution(2 * i),
            final_stage: SumcheckId::NodeExecution(2 * i + 1),
            namespace: 6 * i,
        }
    }
    fn lookup(&self, i: usize) -> Lookup {
        let node = &self.nodes[i];
        let l = node.lookup.as_ref().unwrap();
        let log_k = l.table.len().ilog2() as usize;
        let source = SumcheckId::NodeExecution(2 * i);
        Lookup {
            params: OneHotParams::from_config_and_log_K(
                &OneHotConfig {
                    log_k_chunk: l.log_chunk,
                },
                log_k,
            ),
            log_k,
            input: OpeningId::new(tensor(node.input), source),
            output: OpeningId::new(tensor(self.num_inputs + i), source),
            namespace: 6 * self.nodes.len() + i,
        }
    }
    fn required_keys(&self) -> BTreeSet<CommittedPoly> {
        let mut keys: BTreeSet<_> = (0..self.tensor_count()).map(tensor).collect();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(m) = &node.mul {
                let registration = self.multiplication(i);
                keys.extend(registration.tensors);
                keys.extend(registration.indicator_keys(m.shift));
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
    pub(super) fn validate(&self, max_vars: usize) -> Result<(), ProofVerifyError> {
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
        let mut t = Blake2bTranscript::new(b"Atlas/private-vector-graph/v1");
        t.append_serializable(self);
        t
    }
    pub fn input_commitment(&self, input: usize) -> Option<&DoryCommitment> {
        (input < self.graph.num_inputs)
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
        Self::commit_with_public_inputs(graph, inputs, setup, &BTreeMap::new())
    }
    pub(super) fn commit_with_public_inputs(
        graph: NativeGraph,
        inputs: Vec<Vec<i32>>,
        setup: &DoryProverSetup,
        public: &BTreeMap<CommittedPoly, (DoryCommitment, DoryHint)>,
    ) -> Result<(NativeGraphStatement, Self), ProofVerifyError> {
        graph.validate(setup.verifier.max_log_n)?;
        if public
            .keys()
            .any(|id| !matches!(id, CommittedPoly::DivNodeQuotient(i) if *i < graph.num_inputs))
        {
            return Err(invalid("Only registered public inputs may reuse commitments"));
        }
        if inputs.len() != graph.num_inputs || inputs.iter().any(|v| v.len() != 1 << graph.log_rows)
        {
            return Err(invalid("Graph witness input shape mismatch"));
        }
        let mut values = inputs;
        let mut polynomials: BTreeMap<_, _> = values
            .iter()
            .enumerate()
            .map(|(i, v)| (tensor(i), MultilinearPolynomial::from(v.clone())))
            .collect();
        let mut mul_ranges = BTreeMap::new();
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
                mul_ranges.insert(i, witness.range_values);
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
                    tensor(graph.num_inputs + i),
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
            let (c, h) = public
                .get(id)
                .cloned()
                .unwrap_or_else(|| DoryScheme::commit_zk(p, setup));
            statement.commitments.insert(*id, c);
            hints.insert(*id, h);
        }
        statement.validate(setup.verifier.max_log_n)?;
        Ok((
            statement,
            Self {
                polynomials,
                hints,
                mul_ranges,
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
            mul_ranges,
            lookup_indices,
            ..
        } = witness;
        let graph = &statement.graph;
        if !polynomials.keys().eq(statement.commitments.keys())
            || !hints.keys().eq(statement.commitments.keys())
            || mul_ranges.len() + lookup_indices.len() != graph.nodes.len()
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
            if let Some(m) = &node.mul {
                let values = mul_ranges
                    .get(&i)
                    .ok_or_else(|| invalid("Missing graph multiplication witness"))?;
                if values.len() != 6 || values.iter().any(|v| v.len() != 1 << graph.log_rows) {
                    return Err(invalid("Invalid graph range witness"));
                }
                provers.extend(graph.multiplication(i).provers(
                    m.shift,
                    graph.log_rows,
                    values,
                    &polynomials,
                    &mut a,
                    &mut t,
                ));
            } else {
                let l = node.lookup.as_ref().unwrap();
                let lookup = graph.lookup(i);
                let indices = lookup_indices
                    .get(&i)
                    .ok_or_else(|| invalid("Missing graph lookup witness"))?;
                if indices.len() != 1 << graph.log_rows
                    || indices.iter().any(|j| *j >= l.table.len())
                {
                    return Err(invalid("Invalid graph lookup witness"));
                }
                let r: Vec<Fr> = t.challenge_vector(graph.log_rows);
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
        if self.indicators.is_some() != graph.has_lookups() {
            return Err(invalid(
                "Graph indicator proof presence does not match its operators",
            ));
        }
        let mut t = statement.transcript();
        let mut a = VerifierOpeningAccumulator::new_zk();
        let mut verifiers: Vec<Verifier> = vec![];
        for (i, node) in graph.nodes.iter().enumerate() {
            if let Some(m) = &node.mul {
                verifiers.extend(graph.multiplication(i).verifiers(
                    m.shift,
                    graph.log_rows,
                    &mut a,
                    &mut t,
                ));
            } else {
                let l = node.lookup.as_ref().unwrap();
                let lookup = graph.lookup(i);
                let r: Vec<Fr> = t.challenge_vector(graph.log_rows);
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
            log_rows: 3,
            num_inputs: 2,
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
                    num_inputs: 1,
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
                    num_inputs: 1,
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
            num_inputs: 3,
            nodes: vec![NativeGraphNode::mul(0, 1, 14)],
            outputs: vec![3],
            ..graph()
        };
        let mut data = inputs();
        data.push(vec![0; 8]);
        assert!(NativeGraphWitness::commit(invalid_graph, data, &pp).is_err());
    }
}
