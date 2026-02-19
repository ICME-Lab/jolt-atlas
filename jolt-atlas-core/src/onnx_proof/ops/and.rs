use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{node::ComputationNode, ops::And};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::impl_standard_sumcheck_proof_api;

impl_standard_sumcheck_proof_api!(And, MulParams, MulProver, MulVerifier);
