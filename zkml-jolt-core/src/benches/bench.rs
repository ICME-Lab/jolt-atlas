use crate::jolt::JoltSNARK;
use ark_bn254::Fr;
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{graph::model::Model, model, tensor::Tensor};

type PCS = DoryCommitmentScheme;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    MLP,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::MLP => mlp(),
    }
}

fn prove_and_verify<F, const N: usize>(
    model: F,
    input_data: [i32; N],
    shape: [usize; 2],
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: Fn() -> Model + 'static + Copy,
{
    let mut tasks = Vec::new();
    let task = move || {
        let input = Tensor::new(Some(&input_data), &shape).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model, 1 << 14);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    };
    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));
    tasks
}

fn mlp() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_and_verify(
        || model(&"../tests/perceptron_2.onnx".into()),
        [1, 2, 3, 4],
        [1, 4],
    )
}
