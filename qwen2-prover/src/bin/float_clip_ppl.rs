use std::{error::Error, time::Instant};

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut clip_gate_a = None;
    let mut clip_up_a = None;
    let mut clip_down_a = None;
    let mut clip_gate_y = None;
    let mut clip_up_y = None;
    let mut clip_down_y = None;
    let mut max_targets = usize::MAX;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--clip-gate-a" => {
                clip_gate_a = Some(
                    args.next()
                        .ok_or("--clip-gate-a requires a value")?
                        .parse()?,
                )
            }
            "--clip-up-a" => {
                clip_up_a = Some(args.next().ok_or("--clip-up-a requires a value")?.parse()?)
            }
            "--clip-n2-a" => {
                let value = args.next().ok_or("--clip-n2-a requires a value")?.parse()?;
                clip_gate_a = Some(value);
                clip_up_a = Some(value);
            }
            "--clip-down-a" => {
                clip_down_a = Some(
                    args.next()
                        .ok_or("--clip-down-a requires a value")?
                        .parse()?,
                )
            }
            "--clip-gate-y" => {
                clip_gate_y = Some(
                    args.next()
                        .ok_or("--clip-gate-y requires a value")?
                        .parse()?,
                )
            }
            "--clip-up-y" => {
                clip_up_y = Some(args.next().ok_or("--clip-up-y requires a value")?.parse()?)
            }
            "--clip-down-y" => {
                clip_down_y = Some(
                    args.next()
                        .ok_or("--clip-down-y requires a value")?
                        .parse()?,
                )
            }
            "--clip-mlp-y" => {
                let value = args
                    .next()
                    .ok_or("--clip-mlp-y requires a value")?
                    .parse()?;
                clip_gate_y = Some(value);
                clip_up_y = Some(value);
                clip_down_y = Some(value);
            }
            "--max-targets" => {
                max_targets = args
                    .next()
                    .ok_or("--max-targets requires a value")?
                    .parse()?
            }
            "--first3" => max_targets = 3,
            "--full" => max_targets = usize::MAX,
            other => words.push(other.to_string()),
        }
    }

    let text = if words.is_empty() {
        "hello world this is a test".to_string()
    } else {
        words.join(" ")
    };
    let cfg = ClipConfig {
        gate_a: clip_gate_a,
        up_a: clip_up_a,
        down_a: clip_down_a,
        gate_y: clip_gate_y,
        up_y: clip_up_y,
        down_y: clip_down_y,
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let h = forward_clip(&st, &ids, &r, cfg)?;
    let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        max_targets,
    )?;

    println!("text: {text:?}");
    println!("clip_gate_a: {clip_gate_a:?}");
    println!("clip_up_a: {clip_up_a:?}");
    println!("clip_down_a: {clip_down_a:?}");
    println!("clip_gate_y: {clip_gate_y:?}");
    println!("clip_up_y: {clip_up_y:?}");
    println!("clip_down_y: {clip_down_y:?}");
    if max_targets == usize::MAX {
        println!("ppl(full): {ppl}");
    } else {
        println!("ppl(first {max_targets}): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

#[derive(Clone, Copy)]
struct ClipConfig {
    gate_a: Option<f32>,
    up_a: Option<f32>,
    down_a: Option<f32>,
    gate_y: Option<f32>,
    up_y: Option<f32>,
    down_y: Option<f32>,
}

fn forward_clip(
    st: &safetensors::SafeTensors,
    ids: &[u32],
    r: &Rotary,
    cfg: ClipConfig,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut x = qwen2_prover::float::embed_from_safetensors(st, ids)?;
    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
        x = layer_clip(&x, &w, r, cfg);
    }
    let norm = qwen2_prover::float::final_norm_from_safetensors(st)?;
    Ok(qwen2_prover::float::rms_norm(
        &x,
        &norm,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
    ))
}

fn layer_clip(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: ClipConfig,
) -> Vec<f32> {
    let n1 = qwen2_prover::float::rms_norm(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut q = qwen2_prover::float::matmul(
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = qwen2_prover::float::matmul(
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = qwen2_prover::float::matmul(
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut v, &w.bv);
    let q = qwen2_prover::float::rope(&q, &r.rq);
    let k = qwen2_prover::float::rope(&k, &r.rk);
    let s = qwen2_prover::float::score_qk(&q, &k);
    let p = qwen2_prover::float::softmax(
        &s,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let c = qwen2_prover::float::attn_v(&p, &v);
    let a = qwen2_prover::float::matmul(
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    let h = qwen2_prover::float::add(x, &a);
    let n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);

    let mut gate_a = n2.clone();
    clip_in_place(&mut gate_a, cfg.gate_a);
    let mut gate = qwen2_prover::float::matmul(
        &gate_a,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    clip_in_place(&mut gate, cfg.gate_y);

    let mut up_a = n2;
    clip_in_place(&mut up_a, cfg.up_a);
    let mut up = qwen2_prover::float::matmul(
        &up_a,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    clip_in_place(&mut up, cfg.up_y);

    let mut mid = qwen2_prover::float::silu_mul(&gate, &up);
    clip_in_place(&mut mid, cfg.down_a);
    let mut down = qwen2_prover::float::matmul(
        &mid,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    clip_in_place(&mut down, cfg.down_y);
    qwen2_prover::float::add(&h, &down)
}

fn clip_in_place(xs: &mut [f32], threshold: Option<f32>) {
    let Some(threshold) = threshold else {
        return;
    };
    for x in xs {
        *x = x.clamp(-threshold, threshold);
    }
}
