use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use ark_bn254::Fr;
use joltworks::{field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript};
use qwen3_layer_prover::{
    Claim, Shape, TensorId,
    layer::{LayerShape, LayerTensorIds, LayerWeights, prove_layer, verify_layer},
    trace::build_layer_witness_from_trace_dir,
};
use serde_json::Value;

const FIXED_CACHE_MAGIC: &[u8; 16] = b"QWEN3AWYQ8CACHE1";
const FIXED_FRAC: u64 = 8;
const LAYERS: usize = 28;
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3072;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB: usize = 151_936;
const ROPE_THETA: f64 = 1_000_000.0;

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let seq = args.seq.unwrap_or(read_trace_seq_len(&args.trace)?);
    let shape = LayerShape {
        seq,
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        q_heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
    };
    let cache = args
        .q8_cache
        .unwrap_or_else(|| fixed_cache_path(&args.model));
    let t0 = Instant::now();
    let weights = read_layer_weights(&cache, args.layer, seq)?;
    eprintln!(
        "timing: read_layer_weights {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let traced = build_layer_witness_from_trace_dir(&args.trace, args.layer, &weights, &shape)?;
    eprintln!(
        "timing: build_layer_witness {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let hidden_shape = shape.hidden_shape();
    let point_len = hidden_shape.padded_power_of_two().point_len();
    let point_a = challenge_point(point_len, 3);
    let point_b = challenge_point(point_len, 37);
    let hidden_out_a = hidden_out_claim(&traced.hidden_out, &hidden_shape, point_a);
    let hidden_out_b = hidden_out_claim(&traced.hidden_out, &hidden_shape, point_b);
    let tensors = LayerTensorIds::default();

    let mut prover_transcript = Blake2bTranscript::default();
    let t0 = Instant::now();
    let proof = prove_layer::<Fr, _>(
        hidden_out_a.clone(),
        hidden_out_b.clone(),
        &traced.witness,
        &weights,
        &shape,
        &tensors,
        &mut prover_transcript,
    )?;
    eprintln!("timing: prove_layer {:.3}s", t0.elapsed().as_secs_f64());

    let mut verifier_transcript = Blake2bTranscript::default();
    let t0 = Instant::now();
    let claims = verify_layer::<Fr, _>(
        hidden_out_a,
        hidden_out_b,
        &proof.proof,
        &weights,
        &shape,
        &tensors,
        &mut verifier_transcript,
    )?;
    eprintln!("timing: verify_layer {:.3}s", t0.elapsed().as_secs_f64());
    if claims != proof.claims {
        return Err("verify_layer claims differ from prove_layer claims".into());
    }

    println!("prove_trace_layer: ok");
    println!("trace: {}", args.trace.display());
    println!("q8_cache: {}", cache.display());
    println!("layer: {}", args.layer);
    println!("seq: {}", seq);
    Ok(())
}

#[derive(Debug)]
struct Args {
    trace: PathBuf,
    model: PathBuf,
    q8_cache: Option<PathBuf>,
    layer: usize,
    seq: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut trace = None;
        let mut model = PathBuf::from("qwen3-awy/models/qwen3-0.6b/model.safetensors");
        let mut q8_cache = None;
        let mut layer = 0usize;
        let mut seq = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--trace" => {
                    trace = Some(PathBuf::from(args.next().ok_or("--trace requires a path")?))
                }
                "--model" => model = PathBuf::from(args.next().ok_or("--model requires a path")?),
                "--q8-cache" => {
                    q8_cache = Some(PathBuf::from(
                        args.next().ok_or("--q8-cache requires a path")?,
                    ));
                }
                "--layer" => layer = args.next().ok_or("--layer requires a value")?.parse()?,
                "--seq" => seq = Some(args.next().ok_or("--seq requires a value")?.parse()?),
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {arg:?}; run with --help").into()),
            }
        }
        if layer >= LAYERS {
            return Err(format!("--layer must be < {LAYERS}, got {layer}").into());
        }
        Ok(Self {
            trace: trace.ok_or("--trace is required")?,
            model,
            q8_cache,
            layer,
            seq,
        })
    }
}

fn print_help() {
    println!(
        "Usage:\n\
         cargo run --release -p qwen3-layer-prover --bin prove_trace_layer -- \\\n\
           --trace qwen3-awy/traces/fox_eos_full_awy \\\n\
           --model qwen3-awy/models/qwen3-0.6b/model.safetensors \\\n\
           --layer 0\n\n\
         Options:\n\
           --trace PATH     qwen3-awy --dump-full-awy output directory\n\
           --model PATH     safetensors path; used only to derive default .q8.bin cache path\n\
           --q8-cache PATH  explicit qwen3-awy fixed weight cache path\n\
           --layer N        layer index to prove\n\
           --seq N          override seq length instead of reading manifest metadata"
    );
}

fn fixed_cache_path(model: &Path) -> PathBuf {
    let mut path = model.to_path_buf();
    path.set_extension("q8.bin");
    path
}

fn read_trace_seq_len(trace: &Path) -> Result<usize, Box<dyn Error>> {
    let manifest = BufReader::new(File::open(trace.join("manifest.jsonl"))?);
    for line in manifest.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)?;
        if value.get("event").and_then(Value::as_str) == Some("metadata") {
            if let Some(seq) = value.get("tokens_total").and_then(Value::as_u64) {
                return Ok(seq as usize);
            }
        }
    }
    Err("trace metadata did not contain tokens_total".into())
}

fn read_layer_weights(
    path: &Path,
    target_layer: usize,
    seq: usize,
) -> Result<LayerWeights, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 16];
    r.read_exact(&mut magic)?;
    if &magic != FIXED_CACHE_MAGIC {
        return Err(format!("{} has invalid fixed cache magic", path.display()).into());
    }
    for expected in [
        FIXED_FRAC,
        LAYERS as u64,
        HIDDEN as u64,
        INTERMEDIATE as u64,
        HEADS as u64,
        KV_HEADS as u64,
        HEAD_DIM as u64,
        VOCAB as u64,
    ] {
        let got = read_u64(&mut r)?;
        if got != expected {
            return Err(format!(
                "{} fixed cache header mismatch: expected {expected}, got {got}",
                path.display()
            )
            .into());
        }
    }

    let _final_norm = read_i32_vec(&mut r)?;
    let _lm_head = read_i32_vec(&mut r)?;
    for layer in 0..LAYERS {
        let ln1 = read_i32_vec(&mut r)?;
        let ln2 = read_i32_vec(&mut r)?;
        let q_norm = read_i32_vec(&mut r)?;
        let k_norm = read_i32_vec(&mut r)?;
        let wq = read_i32_vec(&mut r)?;
        let wk = read_i32_vec(&mut r)?;
        let wv = read_i32_vec(&mut r)?;
        let wo = read_i32_vec(&mut r)?;
        let wg = read_i32_vec(&mut r)?;
        let wu = read_i32_vec(&mut r)?;
        let wd = read_i32_vec(&mut r)?;
        if layer == target_layer {
            let (rope_cos, rope_sin) = rope_tables(seq);
            return Ok(LayerWeights {
                rope_cos,
                rope_sin,
                rms_norm_atten: ln1,
                q_norm,
                k_norm,
                rms_norm_mlp: ln2,
                o_proj: wo,
                q_proj: wq,
                k_proj: wk,
                v_proj: wv,
                gate_proj: wg,
                up_proj: wu,
                down_proj: wd,
            });
        }
    }
    Err(format!("layer {target_layer} not found").into())
}

fn read_u64(r: &mut impl Read) -> Result<u64, Box<dyn Error>> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i32_vec(r: &mut impl Read) -> Result<Vec<i32>, Box<dyn Error>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * std::mem::size_of::<i32>()];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
        .collect())
}

fn rope_tables(seq: usize) -> (Vec<i32>, Vec<i32>) {
    let mut cos = vec![0; seq * (HEAD_DIM / 2)];
    let mut sin = vec![0; seq * (HEAD_DIM / 2)];
    for pos in 0..seq {
        for pair in 0..HEAD_DIM / 2 {
            let f = ROPE_THETA.powf(-((2 * pair) as f64) / HEAD_DIM as f64);
            let t = pos as f64 * f;
            // Match qwen3-awy runtime exactly: it stores the RoPE table as
            // f32 first and only quantizes that f32 value to Q8 inside
            // rope_fixed_i32_chunk.  Quantizing the f64 cos/sin directly can
            // differ by one Q8 unit at half-integer boundaries.
            cos[pos * (HEAD_DIM / 2) + pair] = quantize_q8_f32(t.cos() as f32);
            sin[pos * (HEAD_DIM / 2) + pair] = quantize_q8_f32(t.sin() as f32);
        }
    }
    (cos, sin)
}

fn quantize_q8_f32(x: f32) -> i32 {
    (x * 256.0).round() as i32
}

fn challenge_point(len: usize, seed: u64) -> Vec<Fr> {
    (0..len)
        .map(|idx| Fr::from(seed + (idx as u64) * 17))
        .collect()
}

fn hidden_out_claim(values: &[i32], shape: &Shape, point: Vec<Fr>) -> Claim<Fr> {
    Claim {
        tensor: TensorId::new("hidden_out"),
        logical_shape: shape.clone(),
        domain_shape: shape.padded_power_of_two(),
        value: eval_tensor(values, shape, &point),
        point,
    }
}

fn eval_tensor<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
    let eq_by_dim = split_point(shape, point)
        .into_iter()
        .map(EqPolynomial::<F>::evals)
        .collect::<Vec<_>>();
    let strides = row_major_strides(shape.dims());
    let mut out = F::zero();
    for (flat, &value) in values.iter().enumerate() {
        let mut weight = F::one();
        for (dim, (&stride, eq)) in strides.iter().zip(&eq_by_dim).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            weight *= eq[coord];
        }
        out += weight * F::from_i32(value);
    }
    out
}

fn split_point<'a, F>(shape: &Shape, point: &'a [F]) -> Vec<&'a [F]> {
    let mut offset = 0;
    shape
        .dims()
        .iter()
        .map(|dim| {
            let bits = dim.next_power_of_two().trailing_zeros() as usize;
            let out = &point[offset..offset + bits];
            offset += bits;
            out
        })
        .collect()
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for idx in (0..dims.len()).rev().skip(1) {
        strides[idx] = strides[idx + 1] * dims[idx + 1];
    }
    strides
}
