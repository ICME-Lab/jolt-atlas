//! Benchmark: `SoftmaxLastAxis` proving/verify time at a given model scale.
//!
//! Isolates softmax from the rest of a model (unlike `gpt2_zk_bench.rs`, which times a full
//! GPT-2 forward pass) so its proving cost can be measured directly, at realistic GPT-2
//! attention-score magnitudes.
//!
//! Run with:
//! ```bash
//! cargo run --release --package jolt-atlas-core --example softmax_bench
//! ```

use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    tensor::Tensor,
    utils::quantize::mask_sentinel_magnitude,
};
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};

/// GPT-2 layer-0 causal attention scores at scale=12 (multiplier 4096), taken from
/// `jolt-atlas-core/src/onnx_proof/ops/softmax_last_axis/mod.rs`'s
/// `test_softmax_last_axis` fixture: 4 attention heads, 8x8 causal attention matrix each.
/// Masked (upper-triangular / future-token) entries use the scale=12 mask sentinel
/// `-(11 << 12)`; they are rescaled to the target benchmark scale below, along with the
/// real scores.
#[rustfmt::skip]
const GPT2_ATTN_SCORES_SCALE12: &[i32] = &[
    // Head 0 — moderate range
      -492, -45056, -45056, -45056, -45056, -45056, -45056, -45056,
      5440,  -6182, -45056, -45056, -45056, -45056, -45056, -45056,
      4040,  -4700,  -3995, -45056, -45056, -45056, -45056, -45056,
      3503,  -4425,    753,   -701, -45056, -45056, -45056, -45056,
     -2353,  -6284,  -4118,  -4413, -10752, -45056, -45056, -45056,
     -2247,  -7704,  -5686,  -6371, -12738, -12175, -45056, -45056,
      -652,    749,    826,  -1139,  -2820,  -5039,  -2898, -45056,
     -2622,  -2193,  -5682,  -6362,  -8036,  -4648,  -9872, -10283,
    // Head 1 — high variance
     13386, -45056, -45056, -45056, -45056, -45056, -45056, -45056,
     11265,  36900, -45056, -45056, -45056, -45056, -45056, -45056,
      9668,  19702,  39017, -45056, -45056, -45056, -45056, -45056,
     10015,  13052,  21247,  49207, -45056, -45056, -45056, -45056,
      7392,   6466,   1150,  -4119,  30542, -45056, -45056, -45056,
      4939,    493,    848,  -7209,   9139,  24853, -45056, -45056,
     16554,  12057,  15905,   8038,  13262,  12499,   5472, -45056,
     11299,   4069,   6294,   3081,   6412,   8400,  10154,   4726,
    // Head 2 — large positive range
     41677, -45056, -45056, -45056, -45056, -45056, -45056, -45056,
     30755,  46148, -45056, -45056, -45056, -45056, -45056, -45056,
     31081,  16155,  43315, -45056, -45056, -45056, -45056, -45056,
     29381,  15600,  12915,  47013, -45056, -45056, -45056, -45056,
     19723,  10507,   5700,   5104,  30550, -45056, -45056, -45056,
     16650,   9028,  11201,   6500,  11179,  20697, -45056, -45056,
     11184,   4791,    371,   1565,  -2793,  -5332,   2261, -45056,
     -2074,  -6790,  -7238,  -7885,  -5024,  -5917,  -2315,  -7353,
    // Head 3 — negative bias
     -7215, -45056, -45056, -45056, -45056, -45056, -45056, -45056,
     -7363, -18856, -45056, -45056, -45056, -45056, -45056, -45056,
    -12684, -10596, -20225, -45056, -45056, -45056, -45056, -45056,
    -11926, -11397, -10691, -16112, -45056, -45056, -45056, -45056,
    -16957, -19591, -19715, -17550, -16389, -45056, -45056, -45056,
    -20129, -23135, -23295, -21295, -17981, -17724, -45056, -45056,
     -2898,  -9452,  -7226,  -7040,  -6460, -13308,  -6362, -45056,
     -8241, -10310, -12414,  -5334,  -1045,  -8036, -10493, -12577,
];

const INPUT_SHAPE: [usize; 3] = [4, 8, 8];
const SOURCE_SCALE: i32 = 12;

/// Rescale the scale=12 fixture to `target_scale`, replacing masked cells with that scale's
/// own mask sentinel (`mask_sentinel_magnitude`), so the mask stays "just above cutoff" at
/// every benchmarked scale rather than reusing the scale=12 magnitude verbatim.
fn rescaled_scores(target_scale: i32) -> Vec<i32> {
    let mask_c = mask_sentinel_magnitude(target_scale) as i32;
    let mask_value = -(mask_c << target_scale);
    let source_mask = -(45056i32); // -(11 << 12), the scale=12 sentinel used in the fixture above.

    let factor = 2f64.powi(target_scale - SOURCE_SCALE);
    GPT2_ATTN_SCORES_SCALE12
        .iter()
        .map(|&v| {
            if v == source_mask {
                mask_value
            } else {
                (v as f64 * factor).round() as i32
            }
        })
        .collect()
}

fn softmax_model(scale: u32) -> Model {
    let mut b = ModelBuilder::with_scale(scale);
    let i = b.input(INPUT_SHAPE.to_vec());
    let res = b.softmax_last_axis(i);
    b.mark_output(res);
    b.build()
}

/// Same model, but using the `SoftmaxSatClampTable`-lookup-based proof variant (replaces the
/// `sat_diff` complementary-slackness sumcheck) instead of the original `SoftmaxLastAxis`.
/// Only supports the scale the crate was compiled for (`common::consts::MODEL_SCALE`) — see
/// `jolt_atlas_core::onnx_proof::ops::softmax_last_axis_satclamp`.
fn softmax_satclamp_model(scale: u32) -> Model {
    let mut b = ModelBuilder::with_scale(scale);
    let i = b.input(INPUT_SHAPE.to_vec());
    let res = b.softmax_last_axis_satclamp(i);
    b.mark_output(res);
    b.build()
}

/// Same model, but using the single flat exp lookup table variant (no digit split, no
/// `exp_hi*exp_lo` multiplication relation, no `r_exp` remainder) instead of the original
/// `SoftmaxLastAxis`. Only supports the scale the crate was compiled for
/// (`common::consts::MODEL_SCALE`) — see `jolt_atlas_core::onnx_proof::ops::softmax_last_axis_flatexp`.
fn softmax_flatexp_model(scale: u32) -> Model {
    let mut b = ModelBuilder::with_scale(scale);
    let i = b.input(INPUT_SHAPE.to_vec());
    let res = b.softmax_last_axis_flatexp(i);
    b.mark_output(res);
    b.build()
}

/// Prove + verify `model` on `input`, printing timing/size under `label`.
fn bench_model(label: &str, model: Model, input: Tensor<i32>) {
    println!("\n=== {label} ===");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let t = std::time::Instant::now();
    let (proof, io, _dbg) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
    let prove_time = t.elapsed();
    println!("prove:  {prove_time:.2?}");

    let proof_bytes = {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::new();
        proof
            .serialize_compressed(&mut buf)
            .expect("serialize proof");
        buf.len()
    };
    println!(
        "size:   {proof_bytes} bytes ({:.2} KiB)",
        proof_bytes as f64 / 1024.0
    );

    let t = std::time::Instant::now();
    proof.verify(&verifier_pp, &io, None).unwrap();
    let verify_time = t.elapsed();
    println!("verify: {verify_time:.2?}");
}

fn bench_softmax(scale: u32) {
    let data = rescaled_scores(scale as i32);
    let input = Tensor::new(Some(&data), &INPUT_SHAPE).unwrap();
    bench_model(
        &format!("SoftmaxLastAxis @ scale={scale}"),
        softmax_model(scale),
        input,
    );
}

/// Benchmark the sat-clamp variant, only at whatever scale `MODEL_SCALE` is currently compiled
/// to (it doesn't support other scales) — flip `MODEL_SCALE` in `common/src/consts.rs` and
/// recompile to get numbers at a different scale.
fn bench_softmax_satclamp() {
    let scale = common::consts::MODEL_SCALE as u32;
    let data = rescaled_scores(scale as i32);
    let input = Tensor::new(Some(&data), &INPUT_SHAPE).unwrap();
    bench_model(
        &format!("SoftmaxLastAxisSatClamp @ scale={scale} (MODEL_SCALE)"),
        softmax_satclamp_model(scale),
        input,
    );
}

/// Benchmark the flat-exp variant, only at whatever scale `MODEL_SCALE` is currently compiled
/// to (it doesn't support other scales) — flip `MODEL_SCALE` in `common/src/consts.rs` and
/// recompile to get numbers at a different scale.
fn bench_softmax_flatexp() {
    let scale = common::consts::MODEL_SCALE as u32;
    let data = rescaled_scores(scale as i32);
    let input = Tensor::new(Some(&data), &INPUT_SHAPE).unwrap();
    bench_model(
        &format!("SoftmaxLastAxisFlatExp @ scale={scale} (MODEL_SCALE)"),
        softmax_flatexp_model(scale),
        input,
    );
}

fn main() {
    for scale in [8, 12] {
        bench_softmax(scale);
    }
    bench_softmax_satclamp();
    bench_softmax_flatexp();
}
