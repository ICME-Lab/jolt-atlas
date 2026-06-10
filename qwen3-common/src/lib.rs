pub mod claim;
pub mod commitment;
pub mod constants;
pub mod layer;
pub mod opening;
pub mod ops;
pub mod round;
pub mod shape;
pub mod trace;

pub use claim::{EvalClaim, append_eval_claim};
pub use commitment::{
    ChunkedCommitments, LayerBitCommitments, LayerCommitments, LayerHiddenCommitments,
};
pub use constants::{FIXED_SCALE, FRAC_BITS, SCALE};
pub use layer::{
    BitOpeningClaims, IopLayerProof, LayerLookupRanges, LayerOpeningClaims,
    LayerOpeningDomainLengths, LayerOpeningDomainMax, LayerRmsNormVerifierInput,
    LayerRopeVerifierInput, LayerShape, LayerSiluVerifierInput, LayerSoftmaxVerifierInput,
    LayerVerifierPublicInput, RaOpeningClaims, draw_hidden_out_claim, qwen3_layer_shape,
};
pub use opening::{
    ChunkedLayerPcsOpeningProof, LAYER_OPENING_CHUNK_SIZE, LAYER_OPENING_CHUNK_VARS,
    OpeningReductionOutput, OpeningReductionProof, VerifiedOpeningReduction,
};
pub use round::{
    RoundPolynomial, SumCheckRounds, VerifiedSumcheck, append_round_statement,
    sumcheck_initial_claim, verify_sumcheck_rounds,
};
pub use shape::MatrixShape;
pub use trace::{
    LayerRawWitness, LayerWeights, TraceError, TraceLayerRawInput, layer_raw_input_from_trace_dir,
    read_layer_weights,
};
