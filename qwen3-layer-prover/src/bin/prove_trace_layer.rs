use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use ark_bn254::{Bn254, Fr};
use joltworks::{
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Blake2bTranscript,
};
use qwen3_layer_prover::layer::{
    HiddenStateCommitments, LayerPolynomialMap, LayerShape, LayerTensorIds, LayerWeights,
    commit_layer_polynomials, prove_layer, verify_layer,
};
use qwen3_layer_prover::trace::build_layer_witness_from_trace_dir;
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

    let tensors = LayerTensorIds::default();
    type PCS = HyperKZG<Bn254>;
    let t0 = Instant::now();
    let traced = build_layer_witness_from_trace_dir(&args.trace, args.layer, &weights, &shape)?;
    eprintln!(
        "timing: build_layer_witness {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    if args.map_only {
        let t0 = Instant::now();
        let polynomials = LayerPolynomialMap::<Fr>::from_layer(
            &traced.hidden_out,
            &traced.witness,
            &weights,
            &shape,
            &tensors,
        );
        eprintln!(
            "timing: build_layer_polynomial_map {:.3}s",
            t0.elapsed().as_secs_f64()
        );
        println!("map_only: polynomial_count {}", polynomials.entries.len());
        for entry in &polynomials.entries {
            println!("poly {} len {}", entry.name, entry.poly.len());
        }
        println!("map_only: ok");
        return Ok(());
    }

    if args.commit_hidden_in_only {
        let pcs_num_vars = required_pcs_num_vars_for_slices([traced.witness.hidden_in.as_slice()]);
        let t0 = Instant::now();
        let pcs_setup = PCS::setup_prover(pcs_num_vars);
        eprintln!(
            "timing: setup_prover vars={} {:.3}s",
            pcs_num_vars,
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let (_hidden_in_commitment, _) = PCS::commit(
            &MultilinearPolynomial::from(pad_power_of_two(&traced.witness.hidden_in)),
            &pcs_setup,
        );
        eprintln!(
            "timing: commit_hidden_in {:.3}s",
            t0.elapsed().as_secs_f64()
        );
        println!("commit_hidden_in_only: ok");
        return Ok(());
    }
    if args.commit_hidden_only {
        let pcs_num_vars = required_pcs_num_vars_for_slices([
            traced.witness.hidden_in.as_slice(),
            traced.hidden_out.as_slice(),
        ]);
        let t0 = Instant::now();
        let pcs_setup = PCS::setup_prover(pcs_num_vars);
        eprintln!(
            "timing: setup_prover vars={} {:.3}s",
            pcs_num_vars,
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let (hidden_in_commitment, _) = PCS::commit(
            &MultilinearPolynomial::from(pad_power_of_two(&traced.witness.hidden_in)),
            &pcs_setup,
        );
        eprintln!(
            "timing: commit_hidden_in {:.3}s",
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let (hidden_out_commitment, _) = PCS::commit(
            &MultilinearPolynomial::from(pad_power_of_two(&traced.hidden_out)),
            &pcs_setup,
        );
        eprintln!(
            "timing: commit_hidden_out {:.3}s",
            t0.elapsed().as_secs_f64()
        );
        let _hidden_state_commitments = HiddenStateCommitments {
            hidden_in: hidden_in_commitment,
            hidden_out: hidden_out_commitment,
        };
        println!("commit_hidden_only: ok");
        return Ok(());
    }

    let polynomials_for_setup = LayerPolynomialMap::<Fr>::from_layer(
        &traced.hidden_out,
        &traced.witness,
        &weights,
        &shape,
        &tensors,
    );
    let pcs_num_vars = required_pcs_num_vars_for_polynomials(&polynomials_for_setup);
    let t0 = Instant::now();
    let pcs_setup = PCS::setup_prover(pcs_num_vars);
    eprintln!(
        "timing: setup_prover vars={} {:.3}s",
        pcs_num_vars,
        t0.elapsed().as_secs_f64()
    );
    let verifier_setup = PCS::setup_verifier(&pcs_setup);
    let (hidden_in_commitment, _) = PCS::commit(
        &MultilinearPolynomial::from(pad_power_of_two(&traced.witness.hidden_in)),
        &pcs_setup,
    );
    let (hidden_out_commitment, _) = PCS::commit(
        &MultilinearPolynomial::from(pad_power_of_two(&traced.hidden_out)),
        &pcs_setup,
    );
    let hidden_state_commitments = HiddenStateCommitments {
        hidden_in: hidden_in_commitment,
        hidden_out: hidden_out_commitment,
    };
    if args.commit_only {
        println!(
            "commit_only: polynomial_count {}",
            polynomials_for_setup.entries.len()
        );
        for entry in &polynomials_for_setup.entries {
            println!("poly {} len {}", entry.name, entry.poly.len());
        }

        let t0 = Instant::now();
        let commitments = commit_layer_polynomials::<Fr, PCS>(
            &polynomials_for_setup,
            hidden_state_commitments,
            &pcs_setup,
        );
        eprintln!(
            "timing: commit_layer_polynomials {:.3}s",
            t0.elapsed().as_secs_f64()
        );
        println!(
            "commit_only: commitment_count {}",
            commitments.entries.len()
        );
        println!("commit_only: ok");
        return Ok(());
    }

    let mut prover_transcript = Blake2bTranscript::default();
    let t0 = Instant::now();
    let proof = prove_layer::<Fr, _, PCS>(
        &args.trace,
        args.layer,
        hidden_state_commitments.clone(),
        &weights,
        &shape,
        &tensors,
        &pcs_setup,
        &mut prover_transcript,
    )?;
    eprintln!("timing: prove_layer {:.3}s", t0.elapsed().as_secs_f64());

    let mut verifier_transcript = Blake2bTranscript::default();
    let t0 = Instant::now();
    let claims = verify_layer::<Fr, _, PCS>(
        args.layer,
        &proof.proof,
        hidden_state_commitments,
        &verifier_setup,
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

fn required_pcs_num_vars_for_polynomials(polynomials: &LayerPolynomialMap<Fr>) -> usize {
    polynomials
        .entries
        .iter()
        .map(|entry| entry.poly.get_num_vars())
        .max()
        .unwrap_or(0)
}

fn required_pcs_num_vars_for_slices<'a>(slices: impl IntoIterator<Item = &'a [i32]>) -> usize {
    slices
        .into_iter()
        .map(|slice| ceil_log2_len(slice.len()))
        .max()
        .unwrap_or(0)
}

fn ceil_log2_len(len: usize) -> usize {
    debug_assert!(len > 0);
    len.next_power_of_two().trailing_zeros() as usize
}

#[derive(Debug)]
struct Args {
    trace: PathBuf,
    model: PathBuf,
    q8_cache: Option<PathBuf>,
    layer: usize,
    seq: Option<usize>,
    map_only: bool,
    commit_hidden_in_only: bool,
    commit_hidden_only: bool,
    commit_only: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut trace = None;
        let mut model = PathBuf::from("qwen3-awy/models/qwen3-0.6b/model.safetensors");
        let mut q8_cache = None;
        let mut layer = 0usize;
        let mut seq = None;
        let mut map_only = false;
        let mut commit_hidden_in_only = false;
        let mut commit_hidden_only = false;
        let mut commit_only = false;
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
                "--map-only" => map_only = true,
                "--commit-hidden-in-only" => commit_hidden_in_only = true,
                "--commit-hidden-only" => commit_hidden_only = true,
                "--commit-only" => commit_only = true,
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
            map_only,
            commit_hidden_in_only,
            commit_hidden_only,
            commit_only,
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
           --seq N          override seq length instead of reading manifest metadata\n\
           --map-only       stop after building the layer polynomial map\n\
           --commit-hidden-in-only stop after committing hidden_in only\n\
           --commit-hidden-only stop after committing hidden_in and hidden_out\n\
           --commit-only    stop after building and committing layer polynomials"
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

fn pad_power_of_two<T: Copy + Default>(values: &[T]) -> Vec<T> {
    let len = values.len().next_power_of_two();
    let mut out = vec![T::default(); len];
    out[..values.len()].copy_from_slice(values);
    out
}
