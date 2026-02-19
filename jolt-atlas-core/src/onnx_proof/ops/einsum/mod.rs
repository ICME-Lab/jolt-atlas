use crate::{
    onnx_proof::{
        ops::{
            einsum::{
                bmk_bkn_mbn::{BmkBknMbnParams, BmkBknMbnProver, BmkBknMbnVerifier},
                bmk_kbn_mbn::{BmkKbnMbnParams, BmkKbnMbnProver, BmkKbnMbnVerifier},
                k_nk_n::{KNkNParams, KNkNProver, KNkNVerifier},
                mbk_bnk_bmn::{MbkBnkBmnParams, MbkBnkBmnProver, MbkBnkBmnVerifier},
                mbk_nbk_bmn::{MbkNbkBmnParams, MbkNbkBmnProver, MbkNbkBmnVerifier},
                mk_kn_mn::{MkKnMnParams, MkKnMnProver, MkKnMnVerifier},
            },
            OperatorProofTrait, Prover, Verifier,
        },
        ProofId, ProofType,
    },
    utils::dims::EINSUM_REGISTRY,
};
use atlas_onnx_tracer::{
    model::{trace::Trace, Model},
    node::ComputationNode,
    ops::{Einsum, Operator},
};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
    subprotocols::{
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Einstein summation for batch matrix-matrix multiply: bmk,bkn->mbn
pub mod bmk_bkn_mbn;
/// Einstein summation for batch matrix-matrix multiply: bmk,kbn->mbn
pub mod bmk_kbn_mbn;
/// Einstein summation for vector-matrix multiply: k,nk->n
pub mod k_nk_n;
/// Einstein summation for batch matrix-matrix multiply: mbk,bnk->bmn
pub mod mbk_bnk_bmn;
/// Einstein summation for batch matrix-matrix multiply: mbk,nbk->bmn
pub mod mbk_nbk_bmn;
/// Einstein summation for matrix-matrix multiply: mk,kn->mn
pub mod mk_kn_mn;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Einsum {
    #[tracing::instrument(skip_all, name = "Einsum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut prover_sumcheck = EinsumProver::sumcheck(
            &prover.preprocessing.model,
            &prover.trace,
            node.clone(),
            &prover.accumulator,
        );
        let (proof, _) = Sumcheck::prove(
            &mut *prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip_all, name = "Einsum::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = EinsumVerifier::sumcheck(
            &verifier.preprocessing.model,
            node.clone(),
            &verifier.accumulator,
        );
        Sumcheck::verify(
            proof,
            &*verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}

/// Prover dispatcher for Einsum operations.
///
/// Routes to the appropriate Einsum variant based on the equation pattern.
pub struct EinsumProver;

impl EinsumProver {
    /// Create a sumcheck prover for the specified Einsum equation.
    ///
    /// Dispatches to one of the supported Einsum variants (mk,kn->mn, k,nk->n, etc.)
    /// based on the equation in the computation node.
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
                    panic!("Einsum equation ({equation}) not supported by Einsum proof system")
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
            "bmk,kbn->mbn" => {
                let params = BmkKbnMbnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(BmkKbnMbnProver::initialize(trace, params))
            }
            "bmk,bkn->mbn" => {
                let params = BmkBknMbnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(BmkBknMbnProver::initialize(trace, params))
            }
            "mbk,nbk->bmn" => {
                let params = MbkNbkBmnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(MbkNbkBmnProver::initialize(trace, params))
            }
            "mbk,bnk->bmn" => {
                let params = MbkBnkBmnParams::new(computation_node, einsum_dims, accumulator);
                Box::new(MbkBnkBmnProver::initialize(trace, params))
            }
            _ => panic!("unexpected equation: {}", config.equation),
        }
    }
}

/// Verifier dispatcher for Einsum operations.
///
/// Routes to the appropriate Einsum variant verifier based on the equation pattern.
pub struct EinsumVerifier;

impl EinsumVerifier {
    /// Create a sumcheck verifier for the specified Einsum equation.
    ///
    /// Dispatches to one of the supported Einsum variants (mk,kn->mn, k,nk->n, etc.)
    /// based on the equation in the computation node.
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
            "mbk,bnk->bmn" => Box::new(MbkBnkBmnVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            "bmk,kbn->mbn" => Box::new(BmkKbnMbnVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            "bmk,bkn->mbn" => Box::new(BmkBknMbnVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            _ => panic!("unexpected equation"),
        }
    }
}
