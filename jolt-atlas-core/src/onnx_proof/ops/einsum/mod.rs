use atlas_onnx_tracer::{
    model::{trace::Trace, Model},
    node::ComputationNode,
    ops::{Einsum, Operator},
};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
};

use crate::{
    onnx_proof::ops::einsum::{
        bmk_kbn_mbn::{BmkKbnMbnParams, BmkKbnMbnProver, BmkKbnMbnVerifier},
        k_nk_n::{KNkNParams, KNkNProver, KNkNVerifier},
        mbk_nbk_bmn::{MbkNbkBmnParams, MbkNbkBmnProver, MbkNbkBmnVerifier},
        mk_kn_mn::{MkKnMnParams, MkKnMnProver, MkKnMnVerifier},
    },
    utils::einsum::EINSUM_REGISTRY,
};

pub mod bmk_kbn_mbn;
pub mod k_nk_n;
pub mod mbk_nbk_bmn;
pub mod mk_kn_mn;

pub struct EinsumProver;

impl EinsumProver {
    pub fn sumcheck<F: JoltField, T: Transcript>(
        model: &Model,
        trace: &Trace,
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Box<dyn SumcheckInstanceProver<F, T>> {
        let config = match &computation_node.operator {
            Operator::Einsum(Einsum { equation }) => EINSUM_REGISTRY
                .iter()
                .find(|(pattern, _)| pattern == &equation.as_str())
                .map(|(_, config)| config)
                .unwrap_or_else(|| {
                    panic!("Einsum equation ({equation}) not supported by precompile system")
                }),
            _ => panic!("Unexpected operator"),
        };
        let einsum_dims = (config.dims_extractor)(&computation_node, model);
        match config.equation {
            "mk,kn->mn" => {
                let params = MkKnMnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(MkKnMnProver::initialize(trace, params))
            }
            "k,nk->n" => {
                let params = KNkNParams::new(computation_node, einsum_dims, accumulator);
                Box::new(KNkNProver::initialize(trace, params))
            }
            "bmk,kbn->bmn" => {
                let params = BmkKbnMbnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(BmkKbnMbnProver::initialize(trace, params))
            }
            "mbk,nbk->bmn" => {
                let params = MbkNbkBmnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(MbkNbkBmnProver::initialize(trace, params))
            }
            _ => panic!("unexpected equation"),
        }
    }
}

pub struct EinsumVerifier;

impl EinsumVerifier {
    pub fn sumcheck<F: JoltField, T: Transcript>(
        model: &Model,
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Box<dyn SumcheckInstanceVerifier<F, T>> {
        let config = match &computation_node.operator {
            Operator::Einsum(Einsum { equation }) => EINSUM_REGISTRY
                .iter()
                .find(|(pattern, _)| pattern == &equation.as_str())
                .map(|(_, config)| config)
                .unwrap_or_else(|| {
                    panic!("Einsum equation ({equation}) not supported by precompile system")
                }),
            _ => panic!("Unexpected operator"),
        };
        let einsum_dims = (config.dims_extractor)(&computation_node, model);
        match config.equation {
            "mk,kn->mn" => Box::new(MkKnMnVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            "k,nk->n" => Box::new(KNkNVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            "mbk,nbk->bmn" => Box::new(MbkNbkBmnVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            "bmk,kbn->bmn" => Box::new(BmkKbnMbnVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            _ => panic!("unexpected equation"),
        }
    }
}
