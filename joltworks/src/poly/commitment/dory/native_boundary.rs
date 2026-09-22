//! Link an original graph tensor to a precommitted external witness through
//! a hiding evaluation. The external proof must open that prior commitment,
//! prove its conversion and check this same evaluation commitment. This module
//! alone does not prove a conversion or complete a private receipt.
use super::{
    native_graph::NativeGraphStatement, DoryCommitment, DoryHint, DoryProof, DoryProverSetup,
    DoryScheme, DoryVerifierSetup,
};
use crate::field::JoltField;
use crate::{
    curve::Bn254G1,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::CommittedPoly;
type Challenge = <Fr as JoltField>::Challenge;
use std::collections::BTreeMap;
fn invalid(s: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(s.into())
}

/// Public proof only. The tensor values and evaluation blind are never stored.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct NativeTensorBoundaryProof {
    pub opening: DoryProof,
    pub evaluation_commitment: Bn254G1,
}
/// Private input to the external conversion prover. Deliberately not serializable.
pub struct NativeBoundaryEvaluation {
    pub value: Fr,
    pub blind: Fr,
}
#[derive(CanonicalSerialize)]
struct Binding<'a> {
    context: &'a [u8],
    graph_context: &'a [u8],
    tensor: usize,
    shape: Vec<usize>,
    commitment: DoryCommitment,
    conversion_commitment: [u8; 32],
}
fn transcript(
    graph: &NativeGraphStatement,
    context: &[u8],
    tensor: usize,
    conversion_commitment: [u8; 32],
    max_vars: usize,
) -> Result<(Blake2bTranscript, Vec<Challenge>, DoryCommitment), ProofVerifyError> {
    if context.is_empty() {
        return Err(invalid("Boundary requires a registered context"));
    }
    graph.graph.validate(usize::BITS as usize)?;
    if tensor >= graph.graph.num_inputs + graph.graph.nodes.len() {
        return Err(invalid("Boundary tensor is outside the registered graph"));
    }
    let variables = graph.graph.log_rows;
    if variables > max_vars {
        return Err(invalid("Boundary tensor exceeds setup"));
    }
    let shape = vec![1 << variables];
    let commitment = *graph
        .commitments
        .get(&CommittedPoly::DivNodeQuotient(tensor))
        .ok_or_else(|| invalid("Missing original boundary tensor commitment"))?;
    let binding = Binding {
        context,
        graph_context: &graph.graph.context,
        tensor,
        shape,
        commitment,
        conversion_commitment,
    };
    let mut t = Blake2bTranscript::new(b"Atlas/native-boundary/v1");
    t.append_serializable(&binding);
    let point = t.challenge_vector_optimized::<Fr>(variables);
    Ok((t, point, commitment))
}
impl NativeTensorBoundaryProof {
    /// Both commitments must be fixed before this point. The conversion proof
    /// must recompute this derivation and open its entire source/destination
    /// witness, not merely claim the returned evaluation.
    pub fn evaluation_point(
        graph: &NativeGraphStatement,
        context: &[u8],
        tensor: usize,
        conversion_commitment: [u8; 32],
        setup: &DoryVerifierSetup,
    ) -> Result<Vec<Fr>, ProofVerifyError> {
        Ok(transcript(
            graph,
            context,
            tensor,
            conversion_commitment,
            setup.0.max_log_n,
        )?
        .1
        .into_iter()
        .map(Into::into)
        .collect())
    }
    pub(super) fn prove(
        graph: &NativeGraphStatement,
        context: &[u8],
        tensor: usize,
        conversion_commitment: [u8; 32],
        polynomial: &MultilinearPolynomial<Fr>,
        hint: DoryHint,
        setup: &DoryProverSetup,
    ) -> Result<(Self, NativeBoundaryEvaluation), ProofVerifyError> {
        let (mut t, point, _) = transcript(
            graph,
            context,
            tensor,
            conversion_commitment,
            setup.verifier.max_log_n,
        )?;
        if polynomial.get_num_vars() != point.len() {
            return Err(invalid("Boundary witness dimensions differ"));
        }
        let value = polynomial.evaluate(&point);
        let polynomials =
            BTreeMap::from([(CommittedPoly::DivNodeQuotient(tensor), polynomial.clone())]);
        let (opening, evaluation_commitment, blind) = DoryScheme::prove_rlc_zk(
            setup,
            &polynomials,
            &[Fr::from(1u64)],
            vec![hint],
            &point,
            &mut t,
        )?;
        Ok((
            Self {
                opening,
                evaluation_commitment,
            },
            NativeBoundaryEvaluation { value, blind },
        ))
    }
    /// The caller must first accept the registered graph proof and must also
    /// require the external conversion proof for the same witness commitment,
    /// point and evaluation commitment. Neither argument is optional.
    pub fn verify(
        &self,
        graph: &NativeGraphStatement,
        context: &[u8],
        tensor: usize,
        conversion_commitment: [u8; 32],
        setup: &DoryVerifierSetup,
    ) -> Result<(), ProofVerifyError> {
        let (mut t, point, commitment) = transcript(
            graph,
            context,
            tensor,
            conversion_commitment,
            setup.0.max_log_n,
        )?;
        DoryScheme::verify_zk(
            &self.opening,
            setup,
            &mut t,
            &point,
            &self.evaluation_commitment,
            &commitment,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::{
        commitment_scheme::CommitmentScheme,
        dory::native_graph::{NativeGraph, NativeGraphNode, NativeGraphProof, NativeGraphWitness},
    };
    #[test]
    fn boundary_opens_original_graph_tensors_and_hides_values() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        for input in [vec![0, 1], vec![0, 1, 1, 0]] {
            let graph = NativeGraph {
                context: b"registered model".to_vec(),
                log_rows: input.len().ilog2() as usize,
                num_inputs: 1,
                nodes: vec![NativeGraphNode::lookup(0, vec![11, -7], 1)],
                outputs: vec![1],
            };
            let (statement, witness) = NativeGraphWitness::commit(graph, vec![input], &pp).unwrap();
            let context = b"registered decoding and consumer invocation";
            let prior = [19u8; 32];
            for tensor in [0, 1] {
                let (proof, private) = witness
                    .prove_tensor_boundary(&statement, context, tensor, prior, &pp)
                    .unwrap();
                assert_eq!(
                    gens.commit(&[private.value], &private.blind),
                    proof.evaluation_commitment
                );
                let mut bytes = vec![];
                proof.serialize_compressed(&mut bytes).unwrap();
                let proof =
                    NativeTensorBoundaryProof::deserialize_compressed(bytes.as_slice()).unwrap();
                proof
                    .verify(&statement, context, tensor, prior, &vp)
                    .unwrap();
                assert!(proof
                    .verify(&statement, b"different invocation", tensor, prior, &vp)
                    .is_err());
                assert!(proof
                    .verify(&statement, context, tensor, [20u8; 32], &vp)
                    .is_err());
                assert!(proof
                    .verify(&statement, context, 1 - tensor, prior, &vp)
                    .is_err());
                let mut wrong = statement.clone();
                wrong.graph.context.push(0);
                assert!(proof.verify(&wrong, context, tensor, prior, &vp).is_err());
                let mut wrong = proof.clone();
                wrong.opening.0.y_com = None;
                assert!(wrong
                    .verify(&statement, context, tensor, prior, &vp)
                    .is_err());
            }
            NativeGraphProof::prove(&statement, witness, &pp, &gens)
                .unwrap()
                .verify(&statement, &vp, &gens)
                .unwrap();
        }
    }
    #[test]
    fn boundary_binds_prior_commitment_and_rejects_invalid_identifiers() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let graph = NativeGraph {
            context: b"registered source".to_vec(),
            log_rows: 2,
            num_inputs: 1,
            nodes: vec![NativeGraphNode::lookup(0, vec![4, 5], 1)],
            outputs: vec![1],
        };
        let (statement, witness) =
            NativeGraphWitness::commit(graph, vec![vec![0, 1, 0, 1]], &pp).unwrap();
        let a =
            NativeTensorBoundaryProof::evaluation_point(&statement, b"boundary", 1, [1; 32], &vp)
                .unwrap();
        let b =
            NativeTensorBoundaryProof::evaluation_point(&statement, b"boundary", 1, [2; 32], &vp)
                .unwrap();
        assert_ne!(a, b);
        assert!(witness
            .prove_tensor_boundary(&statement, b"", 1, [1; 32], &pp)
            .is_err());
        assert!(witness
            .prove_tensor_boundary(&statement, b"boundary", 2, [1; 32], &pp)
            .is_err());
    }
}
