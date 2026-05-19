use std::{error::Error, time::Instant};

use qwen2_prover::{
    float::HybridOp,
    illm::{DiConfig, DiRebaseMethod},
    rebase::Rounding,
};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut max_targets = 3usize;
    let mut bits = 8u8;
    let mut op = None;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--op" => {
                let value = args.next().ok_or("--op requires a value")?;
                op = Some(parse_op(&value)?);
            }
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.parse()?;
            }
            "--max-targets" => {
                let value = args.next().ok_or("--max-targets requires a value")?;
                max_targets = value.parse()?;
            }
            "--full" => max_targets = usize::MAX,
            other => words.push(other.to_string()),
        }
    }
    let op = op.ok_or("--op is required")?;

    let text = if words.is_empty() {
        "hello world this is a test".to_string()
    } else {
        words.join(" ")
    };

    let cfg = DiConfig {
        bits,
        rounding: Rounding::Floor,
        rebase: DiRebaseMethod::Shift {
            multiplier_shift: 32,
        },
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let h = qwen2_prover::float::forward_with_hybrid_op_from_safetensors(&st, &ids, op, cfg)?;
    let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        max_targets,
    )?;

    println!("text: {text:?}");
    println!("op: {op:?}");
    println!("bits: {bits}");
    if max_targets == usize::MAX {
        println!("ppl(full): {ppl}");
    } else {
        println!("ppl(first {max_targets} targets): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn parse_op(value: &str) -> Result<HybridOp, Box<dyn Error>> {
    match value {
        "add" => Ok(HybridOp::Add),
        "add-bias" => Ok(HybridOp::AddBias),
        "add-residual" => Ok(HybridOp::AddResidual),
        "matmul" => Ok(HybridOp::Matmul),
        "matmul-qkv" => Ok(HybridOp::MatmulQkv),
        "matmul-o" => Ok(HybridOp::MatmulO),
        "matmul-attention" => Ok(HybridOp::MatmulAttention),
        "matmul-gate" => Ok(HybridOp::MatmulGate),
        "matmul-up" => Ok(HybridOp::MatmulUp),
        "matmul-gate-up" => Ok(HybridOp::MatmulGateUp),
        "matmul-gate-up-group1024" => Ok(HybridOp::MatmulGateUpGroup1024),
        "matmul-down" => Ok(HybridOp::MatmulDown),
        "matmul-down-group1024" => Ok(HybridOp::MatmulDownGroup1024),
        "matmul-mlp-group1024" => Ok(HybridOp::MatmulMlpGroup1024),
        "matmul-token-channel" => Ok(HybridOp::MatmulTokenChannel),
        "matmul-token-channel-paper" => Ok(HybridOp::MatmulTokenChannelPaper),
        "rope" => Ok(HybridOp::Rope),
        "silu" => Ok(HybridOp::Silu),
        "softmax-paper" => Ok(HybridOp::SoftmaxPaper),
        "softmax-lut" => Ok(HybridOp::SoftmaxLut),
        _ => Err(format!(
            "unknown --op {value:?}; expected add, add-bias, add-residual, matmul, matmul-qkv, matmul-o, matmul-attention, matmul-gate, matmul-up, matmul-gate-up, matmul-gate-up-group1024, matmul-down, matmul-down-group1024, matmul-mlp-group1024, matmul-token-channel, matmul-token-channel-paper, rope, silu, softmax-paper, softmax-lut"
        )
        .into()),
    }
}
