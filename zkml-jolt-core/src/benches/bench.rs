use crate::jolt::{JoltProverPreprocessing, JoltSNARK, execution_trace::jolt_execution_trace};
use ark_bn254::Fr;
use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme, utils::transcript::KeccakTranscript,
};
use onnx_tracer::{builder, graph::model::Model, model, tensor::Tensor};

type PCS = DoryCommitmentScheme<KeccakTranscript>;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    MultiClass,
    Sentiment,
    MLP,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::MultiClass => multiclass(),
        BenchType::Sentiment => sentiment(),
        BenchType::MLP => mlp(),
    }
}

fn prove_and_verify<F>(
    model_fn: F,
    input: Vec<i32>,
    input_shape: Vec<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: Fn() -> Model + 'static,
{
    let mut tasks = Vec::new();
    let task = move || {
        let model = model_fn();
        let program_bytecode = onnx_tracer::decode_model(model.clone());
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);
        let (raw_trace, program_output) =
            onnx_tracer::execution_trace(model, &Tensor::new(Some(&input), &input_shape).unwrap());
        let execution_trace = jolt_execution_trace(raw_trace.clone());
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace, &program_output);
        snark.verify((&pp).into(), program_output).unwrap();
    };
    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));
    tasks
}

fn multiclass() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_and_verify(
        builder::multiclass0,
        vec![8, 14, 30, 29, 0, 0, 0, 0],
        vec![1, 8],
    )
}

fn sentiment() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_and_verify(builder::sentiment0, vec![3, 4, 5, 0, 0], vec![1, 5])
}

// TODO: MLP model requires larger tensor size
// To run this benchmark, increase `onnx_tracer::constants::MAX_TENSOR_SIZE` to 1024
// This temporary workaround maintains performance for smaller models
// Issue tracked at: https://github.com/ICME-Lab/jolt-atlas/issues/21
fn mlp() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_and_verify(
        || model(&"../tests/perceptron_2.onnx".into()),
        vec![1, 2, 3, 4],
        vec![1, 4],
    )
}
