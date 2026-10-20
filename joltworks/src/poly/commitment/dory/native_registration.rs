//! Register public parameters once and reuse their exact Dory commitments.
//! The verifier must obtain this registration from a trusted model setup.
//! Private inputs, outputs and auxiliary tensors receive fresh hiding commitments.

use super::{
    native_graph::{NativeGraph, NativeGraphProof, NativeGraphStatement, NativeGraphWitness},
    DoryCommitment, DoryHint, DoryProverSetup, DoryScheme, DoryVerifierSetup,
};
use crate::{
    curve::Bn254Curve,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, pedersen::PedersenGenerators},
        multilinear_polynomial::MultilinearPolynomial,
    },
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::CommittedPoly;
use std::collections::BTreeMap;

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(message.into())
}
fn encoded(value: &impl CanonicalSerialize) -> Result<Vec<u8>, ProofVerifyError> {
    let mut bytes = vec![];
    value
        .serialize_compressed(&mut bytes)
        .map_err(|_| invalid("Invalid registered graph encoding"))?;
    Ok(bytes)
}

/// Trusted verifier data. Construct it from the intended graph and public
/// model parameters, then authenticate these bytes independently of the proof.
/// A prover-supplied registration does not identify an expected model.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeRegisteredGraph {
    pub graph: NativeGraph,
    pub public_inputs: BTreeMap<usize, DoryCommitment>,
    pub setup: DoryVerifierSetup,
}
impl NativeRegisteredGraph {
    /// Check the expected graph, every registered public input commitment and
    /// the complete native proof. Other input commitments remain hidden.
    pub fn verify(
        &self,
        proof: &NativeGraphProof,
        statement: &NativeGraphStatement,
        generators: &PedersenGenerators<Bn254Curve>,
    ) -> Result<(), ProofVerifyError> {
        self.graph.validate(self.setup.0.max_log_n)?;
        if encoded(&statement.graph)? != encoded(&self.graph)? {
            return Err(invalid("Proof graph differs from the registered model"));
        }
        for (&id, commitment) in &self.public_inputs {
            if statement.input_commitment(id) != Some(commitment) {
                return Err(invalid(
                    "Proof does not use the registered public parameter",
                ));
            }
        }
        proof.verify(statement, &self.setup, generators)
    }
}

/// Reusable public values, their commitments and prover opening hints.
/// This is preprocessing for a supported native graph, not an ONNX importer.
pub struct NativeGraphPreprocessing {
    registered: NativeRegisteredGraph,
    values: BTreeMap<usize, Vec<i32>>,
    commitments: BTreeMap<CommittedPoly, (DoryCommitment, DoryHint)>,
}
impl NativeGraphPreprocessing {
    pub fn new(
        graph: NativeGraph,
        public_inputs: BTreeMap<usize, Vec<i32>>,
        setup: &DoryProverSetup,
    ) -> Result<Self, ProofVerifyError> {
        graph.validate(setup.verifier.max_log_n)?;
        if graph
            .log_rows
            .checked_add(8)
            .is_none_or(|n| n > setup.verifier.max_log_n)
        {
            return Err(invalid("Registered graph exceeds the setup"));
        }
        let mut commitments = BTreeMap::new();
        let mut expected = BTreeMap::new();
        for (&id, values) in &public_inputs {
            if id >= graph.num_inputs {
                return Err(invalid("Registered public input is not an input tensor"));
            }
            if values.len() != 1 << graph.log_rows {
                return Err(invalid("Registered public parameter shape mismatch"));
            }
            // These parameters are public. Deterministic commitments make
            // their identity reusable without sharing any private input blind.
            let polynomial = MultilinearPolynomial::<Fr>::from(values.clone());
            let (commitment, hint) = DoryScheme::commit(&polynomial, setup);
            expected.insert(id, commitment);
            commitments.insert(CommittedPoly::DivNodeQuotient(id), (commitment, hint));
        }
        Ok(Self {
            registered: NativeRegisteredGraph {
                graph,
                public_inputs: expected,
                setup: DoryScheme::setup_verifier(setup),
            },
            values: public_inputs,
            commitments,
        })
    }
    pub fn registered(&self) -> &NativeRegisteredGraph {
        &self.registered
    }
    /// Supply exactly the inputs not fixed by public preprocessing. Cached
    /// parameter commitments and hints are reused; all other commitments are
    /// freshly blinded. Integer and operator checks remain mandatory.
    pub fn commit(
        &self,
        private_inputs: BTreeMap<usize, Vec<i32>>,
        setup: &DoryProverSetup,
    ) -> Result<(NativeGraphStatement, NativeGraphWitness), ProofVerifyError> {
        if encoded(&DoryScheme::setup_verifier(setup))? != encoded(&self.registered.setup)? {
            return Err(invalid(
                "Public parameter cache belongs to a different setup",
            ));
        }
        let expected =
            (0..self.registered.graph.num_inputs).filter(|i| !self.values.contains_key(i));
        if !expected.eq(private_inputs.keys().copied()) {
            return Err(invalid("Provide exactly the unregistered private inputs"));
        }
        let mut inputs = private_inputs;
        inputs.extend(self.values.iter().map(|(id, values)| (*id, values.clone())));
        NativeGraphWitness::commit_with_public_inputs(
            self.registered.graph.clone(),
            inputs.into_values().collect(),
            setup,
            &self.commitments,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::native_graph::NativeGraphNode;
    use super::*;
    fn graph() -> NativeGraph {
        NativeGraph {
            context: b"registered public parameters".to_vec(),
            log_rows: 2,
            num_inputs: 2,
            nodes: vec![NativeGraphNode::mul(0, 1, 1)],
            outputs: vec![2],
        }
    }
    #[test]
    fn native_registration_reuses_public_commitments_and_fresh_private_blinds() {
        let pp = DoryScheme::setup_prover(10);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let cache =
            NativeGraphPreprocessing::new(graph(), BTreeMap::from([(1, vec![2, 3, 4, 5])]), &pp)
                .unwrap();
        let bytes = encoded(cache.registered()).unwrap();
        let registered = NativeRegisteredGraph::deserialize_compressed(bytes.as_slice()).unwrap();
        let mut previous = None;
        for values in [vec![1, 2, 3, 4], vec![1, 2, 3, 4], vec![5, 6, 7, 8]] {
            let (st, wi) = cache
                .commit(BTreeMap::from([(0, values.clone())]), &pp)
                .unwrap();
            assert_eq!(st.input_commitment(1), registered.public_inputs.get(&1));
            if let Some(c) = previous {
                assert_ne!(st.input_commitment(0), Some(&c));
            }
            previous = st.input_commitment(0).copied();
            assert_eq!(
                wi.outputs(),
                vec![values
                    .iter()
                    .zip([2, 3, 4, 5])
                    .map(|(x, y)| (x * y) >> 1)
                    .collect::<Vec<_>>()]
            );
            let proof = NativeGraphProof::prove(&st, wi, &pp, &gens).unwrap();
            let proof =
                NativeGraphProof::deserialize_compressed(encoded(&proof).unwrap().as_slice())
                    .unwrap();
            registered.verify(&proof, &st, &gens).unwrap();
        }
    }
    #[test]
    fn native_registration_rejects_valid_proof_for_different_public_parameters() {
        let pp = DoryScheme::setup_prover(10);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        let expected =
            NativeGraphPreprocessing::new(graph(), BTreeMap::from([(1, vec![2, 3, 4, 5])]), &pp)
                .unwrap();
        let alternate =
            NativeGraphPreprocessing::new(graph(), BTreeMap::from([(1, vec![2, 3, 4, 6])]), &pp)
                .unwrap();
        let (st, wi) = alternate
            .commit(BTreeMap::from([(0, vec![1, 2, 3, 4])]), &pp)
            .unwrap();
        let proof = NativeGraphProof::prove(&st, wi, &pp, &gens).unwrap();
        proof
            .verify(&st, &DoryScheme::setup_verifier(&pp), &gens)
            .unwrap();
        alternate.registered().verify(&proof, &st, &gens).unwrap();
        assert!(expected.registered().verify(&proof, &st, &gens).is_err());
        let mut wrong = alternate.registered().clone();
        wrong.graph.context.push(0);
        assert!(wrong.verify(&proof, &st, &gens).is_err());
        let mut wrong = alternate.registered().clone();
        wrong.graph.nodes[0].mul.as_mut().unwrap().shift = 2;
        assert!(wrong.verify(&proof, &st, &gens).is_err());
        let mut wrong = alternate.registered().clone();
        wrong.public_inputs.insert(2, wrong.public_inputs[&1]);
        assert!(wrong.verify(&proof, &st, &gens).is_err());
    }
    #[test]
    fn native_registration_validates_input_partition_and_setup() {
        let pp = DoryScheme::setup_prover(10);
        assert!(
            NativeGraphPreprocessing::new(graph(), BTreeMap::from([(2, vec![0; 4])]), &pp).is_err()
        );
        assert!(
            NativeGraphPreprocessing::new(graph(), BTreeMap::from([(1, vec![0; 3])]), &pp).is_err()
        );
        assert!(NativeGraphPreprocessing::new(
            graph(),
            BTreeMap::new(),
            &DoryScheme::setup_prover(8)
        )
        .is_err());
        let cache =
            NativeGraphPreprocessing::new(graph(), BTreeMap::from([(1, vec![2; 4])]), &pp).unwrap();
        for private in [
            BTreeMap::new(),
            BTreeMap::from([(0, vec![1; 4]), (1, vec![2; 4])]),
            BTreeMap::from([(0, vec![1; 3])]),
            BTreeMap::from([(2, vec![1; 4])]),
        ] {
            assert!(cache.commit(private, &pp).is_err());
        }
        assert!(cache
            .commit(
                BTreeMap::from([(0, vec![1; 4])]),
                &DoryScheme::setup_prover(12)
            )
            .is_err());
    }
}
