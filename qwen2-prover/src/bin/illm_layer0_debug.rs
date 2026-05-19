use std::{error::Error, time::Instant};

use qwen2_prover::{
    illm::{self, DiConfig, DiRebaseMethod, QuantTensor},
    rebase::Rounding,
};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");
    let mut bits = 8u8;
    let mut text_args = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--bits" {
            bits = args
                .next()
                .ok_or("--bits requires a value")?
                .parse()
                .map_err(|err| format!("invalid --bits value: {err}"))?;
        } else {
            text_args.push(arg);
        }
    }
    let text = text_args.join(" ");
    let text = if text.is_empty() {
        "hello world this is a test".to_string()
    } else {
        text
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

    let fw = qwen2_prover::float::load_layer_from_safetensors(&st, 0)?;
    let fr = qwen2_prover::float::Rotary::new();
    let iw = illm::load_layer_from_safetensors(&st, 0, cfg)?;
    let ir = illm::IllmRotary::new();

    let fx = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;
    let ix = illm::embed_from_safetensors(&st, &ids, cfg)?;
    compare("embed", &fx, &ix);

    let fn1 = qwen2_prover::float::rms_norm(&fx, &fw.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let in1 = illm::di_rms_norm(&ix, &iw.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN, cfg);
    compare("rms1", &fn1, &in1);

    let mut fq = qwen2_prover::float::matmul(
        &fn1,
        &fw.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    qwen2_prover::float::add_rows(&mut fq, &fw.bq);
    let mut iq = illm::di_matmul(
        &in1,
        &iw.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        cfg,
    );
    iq = illm::di_add_rows(&iq, &iw.bq, qwen2_prover::SEQ, qwen2_prover::HIDDEN, cfg);
    compare("q_proj", &fq, &iq);

    let mut fk = qwen2_prover::float::matmul(
        &fn1,
        &fw.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut fk, &fw.bk);
    let mut ik = illm::di_matmul(
        &in1,
        &iw.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        cfg,
    );
    ik = illm::di_add_rows(
        &ik,
        &iw.bk,
        qwen2_prover::SEQ,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        cfg,
    );
    compare("k_proj", &fk, &ik);

    let mut fv = qwen2_prover::float::matmul(
        &fn1,
        &fw.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut fv, &fw.bv);
    let mut iv = illm::di_matmul(
        &in1,
        &iw.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        cfg,
    );
    iv = illm::di_add_rows(
        &iv,
        &iw.bv,
        qwen2_prover::SEQ,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        cfg,
    );
    compare("v_proj", &fv, &iv);

    let fq = qwen2_prover::float::rope(&fq, &fr.rq);
    let fk = qwen2_prover::float::rope(&fk, &fr.rk);
    let iq = illm::di_rope(&iq, &ir.rq, cfg);
    let ik = illm::di_rope(&ik, &ir.rk, cfg);
    compare("q_rope", &fq, &iq);
    compare("k_rope", &fk, &ik);

    let fs = finite_scores(qwen2_prover::float::score_qk(&fq, &fk));
    let is = illm::di_score_qk(&iq, &ik, cfg);
    compare("score_qk", &fs, &is);
    row_tops("score_qk", &fs, &is.dequantize(), qwen2_prover::SEQ);

    let fp = qwen2_prover::float::softmax(
        &fs,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let ip = illm::di_softmax(
        &is,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
        cfg,
    );
    compare("softmax", &fp, &ip);
    softmax_rows("softmax", &fp, &ip.dequantize(), qwen2_prover::SEQ);

    let fc = qwen2_prover::float::attn_v(&fp, &fv);
    let ic = illm::di_attn_v(&ip, &iv, cfg);
    compare("attn_v", &fc, &ic);

    let fa = qwen2_prover::float::matmul(
        &fc,
        &fw.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    let ia = illm::di_matmul(
        &ic,
        &iw.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        cfg,
    );
    compare("o_proj", &fa, &ia);

    let fh = qwen2_prover::float::add(&fx, &fa);
    let ih = illm::di_add(&ix, &ia, 32, cfg);
    compare("resid1", &fh, &ih);

    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn finite_scores(mut scores: Vec<f32>) -> Vec<f32> {
    for row in scores.chunks_mut(qwen2_prover::SEQ) {
        let min_finite = row
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::INFINITY, f32::min);
        let mask_value = min_finite - 16.0;
        for value in row {
            if !value.is_finite() {
                *value = mask_value;
            }
        }
    }
    scores
}

fn compare(name: &str, reference: &[f32], actual: &QuantTensor) {
    compare_f32(name, reference, &actual.dequantize());
}

fn compare_f32(name: &str, reference: &[f32], actual: &[f32]) {
    assert_eq!(reference.len(), actual.len());
    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut act_norm = 0.0f64;
    let mut mse = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut ref_min = f32::INFINITY;
    let mut ref_max = f32::NEG_INFINITY;
    let mut act_min = f32::INFINITY;
    let mut act_max = f32::NEG_INFINITY;
    for (&r, &a) in reference.iter().zip(actual) {
        let r64 = r as f64;
        let a64 = a as f64;
        dot += r64 * a64;
        ref_norm += r64 * r64;
        act_norm += a64 * a64;
        let diff = r64 - a64;
        mse += diff * diff;
        max_abs = max_abs.max(diff.abs());
        ref_min = ref_min.min(r);
        ref_max = ref_max.max(r);
        act_min = act_min.min(a);
        act_max = act_max.max(a);
    }
    mse /= reference.len() as f64;
    let cosine = dot / (ref_norm.sqrt() * act_norm.sqrt()).max(f64::MIN_POSITIVE);
    println!(
        "{name:<10} cos={cosine:>9.5} mse={mse:>12.5e} max_abs={max_abs:>10.5} ref=[{ref_min:>9.4},{ref_max:>9.4}] di=[{act_min:>9.4},{act_max:>9.4}]"
    );
}

fn softmax_rows(name: &str, reference: &[f32], actual: &[f32], cols: usize) {
    assert_eq!(reference.len(), actual.len());
    let rows = reference.len() / cols;
    let mut ref_sum_min = f32::INFINITY;
    let mut ref_sum_max = f32::NEG_INFINITY;
    let mut act_sum_min = f32::INFINITY;
    let mut act_sum_max = f32::NEG_INFINITY;
    let mut top_matches = 0usize;

    for row in 0..rows {
        let r = &reference[row * cols..(row + 1) * cols];
        let a = &actual[row * cols..(row + 1) * cols];
        let ref_sum = r.iter().sum::<f32>();
        let act_sum = a.iter().sum::<f32>();
        ref_sum_min = ref_sum_min.min(ref_sum);
        ref_sum_max = ref_sum_max.max(ref_sum);
        act_sum_min = act_sum_min.min(act_sum);
        act_sum_max = act_sum_max.max(act_sum);
        if argmax(r) == argmax(a) {
            top_matches += 1;
        }
    }

    println!(
        "{name:<10} rows ref_sum=[{ref_sum_min:>7.4},{ref_sum_max:>7.4}] di_sum=[{act_sum_min:>7.4},{act_sum_max:>7.4}] top_match={top_matches}/{rows}"
    );
}

fn row_tops(name: &str, reference: &[f32], actual: &[f32], cols: usize) {
    assert_eq!(reference.len(), actual.len());
    let rows = reference.len() / cols;
    let mut top_matches = 0usize;
    let mut ref_gap_sum = 0.0f64;
    let mut act_gap_sum = 0.0f64;

    for row in 0..rows {
        let r = &reference[row * cols..(row + 1) * cols];
        let a = &actual[row * cols..(row + 1) * cols];
        if argmax(r) == argmax(a) {
            top_matches += 1;
        }
        ref_gap_sum += top_gap(r) as f64;
        act_gap_sum += top_gap(a) as f64;
    }

    println!(
        "{name:<10} rows top_match={top_matches}/{rows} avg_gap ref={:>9.4} di={:>9.4}",
        ref_gap_sum / rows as f64,
        act_gap_sum / rows as f64
    );
}

fn top_gap(values: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &value in values {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    best - second
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap()
}
