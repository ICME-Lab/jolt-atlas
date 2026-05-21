use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::Instant,
};

use ark_bn254::Fr;
use joltworks::{field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript};
use qwen3_layer_prover::{
    Claim, Shape, TensorId,
    decoder::{
        DecoderWeights, ParallelLayerWitness, prove_layers_parallel, verify_layers_parallel,
    },
    layer::{LayerShape, LayerTensorIds, LayerWeights},
    trace::build_layer_witness_from_trace_dir,
};
use rayon::prelude::*;
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

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let total_start = Instant::now();
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
    let layers_to_run = args.layers;
    let start = Instant::now();
    let weights = DecoderWeights {
        layers: read_layer_weights_prefix(&cache, seq, layers_to_run)?,
    };
    eprintln!(
        "timing: decoder.read_weights {:.3}s",
        start.elapsed().as_secs_f64()
    );

    let hidden_shape = shape.hidden_shape();
    let point_len = hidden_shape.padded_power_of_two().point_len();
    let start = Instant::now();
    let layer_witnesses = (0..layers_to_run)
        .into_par_iter()
        .map(|layer| {
            let traced = build_layer_witness_from_trace_dir(
                &args.trace,
                layer,
                &weights.layers[layer],
                &shape,
            )?;
            let hidden_out_a = hidden_out_claim(
                &traced.hidden_out,
                &hidden_shape,
                challenge_point(point_len, 3),
            );
            let hidden_out_b = hidden_out_claim(
                &traced.hidden_out,
                &hidden_shape,
                challenge_point(point_len, 37),
            );
            Ok(ParallelLayerWitness {
                hidden_out_a,
                hidden_out_b,
                witness: traced.witness,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error + Send + Sync>>>()?;
    eprintln!(
        "timing: decoder.build_witness {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let tensors = (0..layers_to_run)
        .map(|_| LayerTensorIds::default())
        .collect::<Vec<_>>();

    let start = Instant::now();
    let proof = prove_layers_parallel::<Fr, Blake2bTranscript>(
        &layer_witnesses,
        &weights,
        &shape,
        &tensors,
    )?;
    eprintln!(
        "timing: decoder.prove_layers_parallel {:.3}s",
        start.elapsed().as_secs_f64()
    );

    let start = Instant::now();
    let claims =
        verify_layers_parallel::<Fr, Blake2bTranscript>(&proof.proof, &weights, &shape, &tensors)?;
    eprintln!(
        "timing: decoder.verify_layers_parallel {:.3}s",
        start.elapsed().as_secs_f64()
    );
    if claims != proof.claims {
        return Err(
            "verify_layers_parallel claims differ from prove_layers_parallel claims".into(),
        );
    }

    println!("prove_trace_decoder: ok");
    println!("trace: {}", args.trace.display());
    println!("q8_cache: {}", cache.display());
    println!("layers: {layers_to_run}");
    println!("seq: {seq}");
    eprintln!(
        "timing: decoder.total {:.3}s",
        total_start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[derive(Debug)]
struct Args {
    trace: PathBuf,
    model: PathBuf,
    q8_cache: Option<PathBuf>,
    seq: Option<usize>,
    layers: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut trace = None;
        let mut model = PathBuf::from("qwen3-awy/models/qwen3-0.6b/model.safetensors");
        let mut q8_cache = None;
        let mut seq = None;
        let mut layers = LAYERS;
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
                "--seq" => seq = Some(args.next().ok_or("--seq requires a value")?.parse()?),
                "--layers" => {
                    layers = args.next().ok_or("--layers requires a value")?.parse()?;
                    if layers == 0 || layers > LAYERS {
                        return Err(
                            format!("--layers must be in 1..={LAYERS}, got {layers}").into()
                        );
                    }
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {arg:?}; run with --help").into()),
            }
        }
        Ok(Self {
            trace: trace.ok_or("--trace is required")?,
            model,
            q8_cache,
            seq,
            layers,
        })
    }
}

fn print_help() {
    println!(
        "Usage:\n\
         cargo run --release -p qwen3-layer-prover --bin prove_trace_decoder -- \\\n\
           --trace qwen3-awy/traces/fox_eos_full_awy \\\n\
           --model qwen3-awy/models/qwen3-0.6b/model.safetensors\n\n\
         Options:\n\
           --trace PATH     qwen3-awy --dump-full-awy output directory\n\
           --model PATH     safetensors path; used only to derive default .q8.bin cache path\n\
           --q8-cache PATH  explicit qwen3-awy fixed weight cache path\n\
           --seq N          override seq length instead of reading manifest metadata\n\
           --layers N       prove only the first N layers through the parallel path"
    );
}

fn fixed_cache_path(model: &Path) -> PathBuf {
    let mut path = model.to_path_buf();
    path.set_extension("q8.bin");
    path
}

fn read_trace_seq_len(trace: &Path) -> Result<usize, Box<dyn Error + Send + Sync>> {
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

fn read_layer_weights_prefix(
    path: &Path,
    seq: usize,
    layers_to_read: usize,
) -> Result<Vec<LayerWeights>, Box<dyn Error + Send + Sync>> {
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

    skip_i32_vec(&mut r)?;
    skip_i32_vec(&mut r)?;
    let (rope_cos, rope_sin) = rope_tables(seq);
    let mut layers = Vec::with_capacity(layers_to_read);
    for _ in 0..layers_to_read {
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
        layers.push(LayerWeights {
            rope_cos: rope_cos.clone(),
            rope_sin: rope_sin.clone(),
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
    Ok(layers)
}

fn read_u64(r: &mut impl Read) -> Result<u64, Box<dyn Error + Send + Sync>> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i32_vec(r: &mut impl Read) -> Result<Vec<i32>, Box<dyn Error + Send + Sync>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * std::mem::size_of::<i32>()];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
        .collect())
}

fn skip_i32_vec<R>(r: &mut R) -> Result<(), Box<dyn Error + Send + Sync>>
where
    R: Read + Seek,
{
    let len = read_u64(r)?;
    let bytes = len
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .ok_or("fixed cache vector byte length overflow")?;
    r.seek(SeekFrom::Current(bytes as i64))?;
    Ok(())
}

fn rope_tables(seq: usize) -> (Vec<i32>, Vec<i32>) {
    let mut cos = vec![0; seq * (HEAD_DIM / 2)];
    let mut sin = vec![0; seq * (HEAD_DIM / 2)];
    for pos in 0..seq {
        for pair in 0..HEAD_DIM / 2 {
            let f = ROPE_THETA.powf(-((2 * pair) as f64) / HEAD_DIM as f64);
            let t = pos as f64 * f;
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
