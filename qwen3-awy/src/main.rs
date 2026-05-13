use std::{
    error::Error,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use half::{bf16, f16};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use tokenizers::Tokenizer;

const DEFAULT_SEQ: usize = 128;
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3072;
const LAYERS: usize = 28;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const Q_DIM: usize = HEADS * HEAD_DIM;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
const KV_GROUP: usize = HEADS / KV_HEADS;
const VOCAB: usize = 151_936;
const ROPE_THETA: f64 = 1_000_000.0;
const DEFAULT_FIXED_FRAC: u8 = 8;
const EOS_IM_END: u32 = 151_645;
const EOS_END_OF_TEXT: u32 = 151_643;
const NO_THINK_SUFFIX: &str = "<think>\n\n</think>\n\n";
const ROUND_LUT_Q8: [i64; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];
const EXP_LUT_Q8: [i64; 9] = [0, 0, 1, 2, 5, 13, 35, 94, 256];
const SIGMOID_LUT_Q8: [i64; 16] = [
    0, 0, 1, 2, 5, 12, 31, 69, 128, 187, 225, 244, 251, 254, 255, 256,
];

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let started = Instant::now();
    let read_started = Instant::now();
    let bytes = std::fs::read(&args.model)?;
    let read_elapsed = read_started.elapsed();
    let deserialize_started = Instant::now();
    let st = SafeTensors::deserialize(&bytes)?;
    let deserialize_elapsed = deserialize_started.elapsed();
    // SmoothQuant/QuaRot hooks were intentionally removed from this runtime path.
    // Keep transforms offline or in separate experiment crates so the fixed path stays simple.
    if args.timing {
        println!("timing.read_safetensors: {:.3?}", read_elapsed);
        println!(
            "timing.deserialize_safetensors: {:.3?}",
            deserialize_elapsed
        );
    }
    let max_new_tokens = args.generate_tokens.unwrap_or(128);
    let generated = generate_with_kv_cache(&st, &args, max_new_tokens)?;
    println!("prompt: {:?}", args.text);
    println!("model: {}", args.model.display());
    println!("tokenizer: {}", args.tokenizer.display());
    println!("prompt_template: {}", prompt_template_name(args.thinking));
    println!("fixed: true");
    println!("fixed_frac: {}", DEFAULT_FIXED_FRAC);
    println!(
        "matmul_rebase_rounding: {}",
        args.cfg.matmul_rebase_rounding.name()
    );
    println!(
        "sigmoid_input_rounding: {}",
        args.cfg.sigmoid_input_rounding.name()
    );
    println!("seq_len: {}", args.seq_len);
    println!("sample: {}", args.sampling.enabled);
    println!("temperature: {:.3}", args.sampling.temperature);
    println!("top_p: {:.3}", args.sampling.top_p);
    println!(
        "top_k: {}",
        args.sampling
            .top_k
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "repetition_penalty: {:.3}",
        args.sampling.repetition_penalty
    );
    println!("prompt_tokens: {}", generated.prompt_tokens);
    println!("generated_tokens: {}", generated.generated_tokens);
    println!("total_tokens: {}", generated.total_tokens);
    println!("ended_with_eos: {}", generated.ended_with_eos);
    println!("text: {}", generated.text);
    println!("elapsed: {:.2?}", started.elapsed());
    Ok(())
}

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    text: String,
    seq_len: usize,
    generate_tokens: Option<usize>,
    stream: bool,
    thinking: bool,
    sampling: Sampling,
    timing: bool,
    cfg: ForwardConfig,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let mut model = root.join("qwen3-awy/models/qwen3-0.6b/model.safetensors");
        let mut tokenizer = root.join("qwen3-awy/models/qwen3-0.6b/tokenizer.json");
        let mut text = Vec::new();
        let mut seq_len = DEFAULT_SEQ;
        let mut generate_tokens = None;
        let mut stream = false;
        let mut thinking = false;
        let mut sampling = Sampling::default();
        let mut greedy = false;
        let mut timing = false;
        let mut cfg = ForwardConfig::default();

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = PathBuf::from(args.next().ok_or("--model requires a path")?),
                "--tokenizer" => {
                    tokenizer = PathBuf::from(args.next().ok_or("--tokenizer requires a path")?)
                }
                "--matmul-rebase-rounding" => {
                    cfg.matmul_rebase_rounding = RoundingMode::parse(
                        &args
                            .next()
                            .ok_or("--matmul-rebase-rounding requires round|floor|ceil")?,
                    )?;
                }
                "--sigmoid-input-rounding" => {
                    cfg.sigmoid_input_rounding = RoundingMode::parse(
                        &args
                            .next()
                            .ok_or("--sigmoid-input-rounding requires round|floor|ceil")?,
                    )?;
                }
                "--seq-len" => {
                    seq_len = args.next().ok_or("--seq-len requires a value")?.parse()?;
                    if seq_len < 2 {
                        return Err(err("--seq-len must be at least 2"));
                    }
                }
                "--generate" | "--generate-tokens" => {
                    generate_tokens = Some(
                        args.next()
                            .ok_or("--generate requires a token count")?
                            .parse()?,
                    );
                }
                "--stream" => stream = true,
                "--thinking" => thinking = true,
                "--no-think" => thinking = false,
                "--sample" => sampling.enabled = true,
                "--greedy" => greedy = true,
                "--seed" => {
                    sampling.enabled = true;
                    sampling.seed = args.next().ok_or("--seed requires a value")?.parse()?;
                }
                "--temperature" => {
                    sampling.enabled = true;
                    sampling.temperature = args
                        .next()
                        .ok_or("--temperature requires a value")?
                        .parse()?;
                    if sampling.temperature <= 0.0 {
                        return Err(err("--temperature must be positive"));
                    }
                }
                "--top-p" => {
                    sampling.enabled = true;
                    sampling.top_p = args.next().ok_or("--top-p requires a value")?.parse()?;
                    if !(0.0..=1.0).contains(&sampling.top_p) {
                        return Err(err("--top-p must be in 0..=1"));
                    }
                }
                "--top-k" => {
                    sampling.enabled = true;
                    sampling.top_k = Some(args.next().ok_or("--top-k requires a value")?.parse()?);
                    if sampling.top_k == Some(0) {
                        return Err(err("--top-k must be positive"));
                    }
                }
                "--repetition-penalty" => {
                    sampling.repetition_penalty = args
                        .next()
                        .ok_or("--repetition-penalty requires a value")?
                        .parse()?;
                    if sampling.repetition_penalty <= 0.0 {
                        return Err(err("--repetition-penalty must be positive"));
                    }
                }
                "--timing" => timing = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other if other.starts_with("--") => {
                    return Err(err(format!("unknown option {other:?}")));
                }
                other => text.push(other.to_string()),
            }
        }

        let text = if text.is_empty() {
            "hello world this is a test".to_string()
        } else {
            text.join(" ")
        };
        if greedy {
            sampling.enabled = false;
        }
        Ok(Self {
            model,
            tokenizer,
            text,
            seq_len,
            generate_tokens,
            stream,
            thinking,
            sampling,
            timing,
            cfg,
        })
    }
}

fn print_help() {
    println!(
        "qwen3-awy\n\
         \n\
         Run Qwen3 with fixed-point block internals.\n\
         \n\
         Options:\n\
           --model PATH          safetensors path\n\
           --tokenizer PATH      tokenizer.json path\n\
          --matmul-rebase-rounding round|floor|ceil MatMul accumulator rebase rounding; default round\n\
           --sigmoid-input-rounding round|floor|ceil sigmoid input integer rounding; default round\n\
           --seq-len N           context length; default 128\n\
           --generate N          generate up to N new tokens\n\
           --thinking            use Qwen3 thinking chat prompt instead of no-think\n\
           --no-think            use Qwen3 no-think prompt; default\n\
          --stream              print decoded text after each generated token\n\
           --sample              sample next token; default on for Qwen3\n\
           --greedy              disable sampling and use argmax\n\
           --seed N              deterministic sampling seed\n\
           --temperature T       sampling temperature; default 0.6\n\
           --top-p P             nucleus sampling cutoff; default 0.95\n\
           --top-k K             keep only top K candidates before top-p; default 20\n\
           --repetition-penalty R penalize previously generated tokens; default 1.0\n\
          --timing              print coarse timing breakdowns\n"
    );
}

#[derive(Clone, Copy, Debug)]
struct Sampling {
    enabled: bool,
    seed: u64,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            enabled: true,
            seed: 1,
            temperature: 0.6,
            top_p: 0.95,
            top_k: Some(20),
            repetition_penalty: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ForwardConfig {
    matmul_rebase_rounding: RoundingMode,
    sigmoid_input_rounding: RoundingMode,
}

impl Default for ForwardConfig {
    fn default() -> Self {
        Self {
            matmul_rebase_rounding: RoundingMode::Round,
            sigmoid_input_rounding: RoundingMode::Round,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RoundingMode {
    #[default]
    Round,
    Floor,
    Ceil,
}

impl RoundingMode {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "round" | "nearest" => Ok(Self::Round),
            "floor" => Ok(Self::Floor),
            "ceil" | "ceiling" => Ok(Self::Ceil),
            _ => Err(err(format!("unknown rounding mode {value:?}"))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Round => "round",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
        }
    }
}

fn encode_unpadded(path: &PathBuf, text: &str, thinking: bool) -> Result<Vec<u32>, Box<dyn Error>> {
    let tok = Tokenizer::from_file(path).map_err(|e| err(e.to_string()))?;
    let prompt = chat_prompt(text, thinking);
    let enc = tok
        .encode(prompt.as_str(), true)
        .map_err(|e| err(e.to_string()))?;
    Ok(enc.get_ids().to_vec())
}

fn chat_prompt(text: &str, thinking: bool) -> String {
    let suffix = if thinking { "" } else { NO_THINK_SUFFIX };
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{suffix}")
}

fn prompt_template_name(thinking: bool) -> &'static str {
    if thinking {
        "qwen3-thinking"
    } else {
        "qwen3-no-think"
    }
}

fn decode_tokens(path: &PathBuf, ids: &[u32]) -> Result<String, Box<dyn Error>> {
    let tok = Tokenizer::from_file(path).map_err(|e| err(e.to_string()))?;
    tok.decode(ids, true).map_err(|e| err(e.to_string()))
}

fn decode_generation_text(
    path: &PathBuf,
    ids: &[u32],
    prompt_tokens: usize,
) -> Result<String, Box<dyn Error>> {
    let start = prompt_tokens.min(ids.len());
    decode_tokens(path, &ids[start..])
}

fn is_eos(id: u32) -> bool {
    id == EOS_IM_END || id == EOS_END_OF_TEXT
}

fn tensor_f32(t: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
    match t.dtype() {
        Dtype::BF16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        Dtype::F16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        Dtype::F32 => Ok(t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        dt => Err(err(format!("unsupported tensor dtype {dt:?}"))),
    }
}

fn need_shape(t: &TensorView<'_>, shape: &[usize], name: &str) -> Result<(), Box<dyn Error>> {
    if t.shape() != shape {
        return Err(err(format!(
            "{name}: expected shape {:?}, got {:?}",
            shape,
            t.shape()
        )));
    }
    Ok(())
}

fn embed_fixed_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    let mut out = Vec::with_capacity(ids.len() * HIDDEN);
    for &id in ids {
        row_fixed_i32(&t, id as usize, HIDDEN, &mut out)?;
    }
    Ok(out)
}

fn embed_one_fixed_from_safetensors(
    st: &SafeTensors<'_>,
    id: u32,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    let mut out = Vec::with_capacity(HIDDEN);
    row_fixed_i32(&t, id as usize, HIDDEN, &mut out)?;
    Ok(out)
}

fn row_fixed_i32(
    t: &TensorView<'_>,
    row: usize,
    cols: usize,
    out: &mut Vec<i32>,
) -> Result<(), Box<dyn Error>> {
    if row >= VOCAB {
        return Err(err(format!("token id {row} exceeds vocab")));
    }
    let scale = (1u64 << DEFAULT_FIXED_FRAC) as f32;
    match t.dtype() {
        Dtype::BF16 => {
            let start = row * cols * 2;
            for b in t.data()[start..start + cols * 2].chunks_exact(2) {
                out.push(
                    (bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * scale).round()
                        as i32,
                );
            }
        }
        Dtype::F16 => {
            let start = row * cols * 2;
            for b in t.data()[start..start + cols * 2].chunks_exact(2) {
                out.push(
                    (f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * scale).round()
                        as i32,
                );
            }
        }
        Dtype::F32 => {
            let start = row * cols * 4;
            for b in t.data()[start..start + cols * 4].chunks_exact(4) {
                out.push((f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * scale).round() as i32);
            }
        }
        dt => return Err(err(format!("unsupported row dtype {dt:?}"))),
    }
    Ok(())
}

struct FixedProjectionWeights {
    wq: Vec<i32>,
    wk: Vec<i32>,
    wv: Vec<i32>,
    wo: Vec<i32>,
    wg: Vec<i32>,
    wu: Vec<i32>,
    wd: Vec<i32>,
}

struct FixedLayerWeights {
    ln1: Vec<i32>,
    ln2: Vec<i32>,
    q_norm: Vec<i32>,
    k_norm: Vec<i32>,
    q: FixedProjectionWeights,
}

fn quantized_linear_i32_transposed(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, &[out, input], name)?;
    let scale = (1u64 << DEFAULT_FIXED_FRAC) as f32;
    let mut ys = vec![0i32; out * input];
    match t.dtype() {
        Dtype::BF16 => {
            let data = t.data();
            ys.par_chunks_mut(out).enumerate().for_each(|(i, row)| {
                for o in 0..out {
                    let p = (o * input + i) * 2;
                    let x = bf16::from_bits(u16::from_le_bytes([data[p], data[p + 1]])).to_f32();
                    row[o] = (x * scale).round() as i32;
                }
            });
        }
        Dtype::F16 => {
            let data = t.data();
            ys.par_chunks_mut(out).enumerate().for_each(|(i, row)| {
                for o in 0..out {
                    let p = (o * input + i) * 2;
                    let x = f16::from_bits(u16::from_le_bytes([data[p], data[p + 1]])).to_f32();
                    row[o] = (x * scale).round() as i32;
                }
            });
        }
        Dtype::F32 => {
            let data = t.data();
            ys.par_chunks_mut(out).enumerate().for_each(|(i, row)| {
                for o in 0..out {
                    let p = (o * input + i) * 4;
                    let x = f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                    row[o] = (x * scale).round() as i32;
                }
            });
        }
        dt => return Err(err(format!("unsupported tensor dtype {dt:?}"))),
    }
    Ok(ys)
}

fn vec_fixed_i32(
    st: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, shape, name)?;
    let scale = (1u64 << DEFAULT_FIXED_FRAC) as f32;
    match t.dtype() {
        Dtype::BF16 => Ok(t
            .data()
            .par_chunks_exact(2)
            .map(|b| {
                (bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * scale).round() as i32
            })
            .collect()),
        Dtype::F16 => Ok(t
            .data()
            .par_chunks_exact(2)
            .map(|b| {
                (f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * scale).round() as i32
            })
            .collect()),
        Dtype::F32 => Ok(t
            .data()
            .par_chunks_exact(4)
            .map(|b| (f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * scale).round() as i32)
            .collect()),
        dt => Err(err(format!("unsupported tensor dtype {dt:?}"))),
    }
}

fn load_fixed_layer(
    st: &SafeTensors<'_>,
    layer: usize,
) -> Result<FixedLayerWeights, Box<dyn Error>> {
    let p = format!("model.layers.{layer}");
    Ok(FixedLayerWeights {
        ln1: vec_fixed_i32(st, &format!("{p}.input_layernorm.weight"), &[HIDDEN])?,
        ln2: vec_fixed_i32(
            st,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
        )?,
        q_norm: vec_fixed_i32(st, &format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM])?,
        k_norm: vec_fixed_i32(st, &format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM])?,
        q: FixedProjectionWeights {
            wq: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                Q_DIM,
                HIDDEN,
            )?,
            wk: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                KV_DIM,
                HIDDEN,
            )?,
            wv: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.v_proj.weight"),
                KV_DIM,
                HIDDEN,
            )?,
            wo: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                HIDDEN,
                Q_DIM,
            )?,
            wg: quantized_linear_i32_transposed(
                st,
                &format!("{p}.mlp.gate_proj.weight"),
                INTERMEDIATE,
                HIDDEN,
            )?,
            wu: quantized_linear_i32_transposed(
                st,
                &format!("{p}.mlp.up_proj.weight"),
                INTERMEDIATE,
                HIDDEN,
            )?,
            wd: quantized_linear_i32_transposed(
                st,
                &format!("{p}.mlp.down_proj.weight"),
                HIDDEN,
                INTERMEDIATE,
            )?,
        },
    })
}

fn load_fixed_layers(st: &SafeTensors<'_>) -> Result<Vec<FixedLayerWeights>, Box<dyn Error>> {
    (0..LAYERS)
        .map(|layer| load_fixed_layer(st, layer))
        .collect()
}

struct Rotary {
    rq: Vec<f32>,
    rk: Vec<f32>,
}

impl Rotary {
    fn new(seq: usize) -> Self {
        Self {
            rq: rot(seq, HEADS, HEAD_DIM),
            rk: rot(seq, KV_HEADS, HEAD_DIM),
        }
    }
}

fn rot(seq: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut xs = vec![0.0; seq * heads * head_dim];
    let half = head_dim / 2;
    for pos in 0..seq {
        for h in 0..heads {
            for p in 0..half {
                let f = ROPE_THETA.powf(-((2 * p) as f64) / head_dim as f64);
                let t = pos as f64 * f;
                let i = (pos * heads + h) * head_dim;
                xs[i + p] = t.cos() as f32;
                xs[i + half + p] = t.sin() as f32;
            }
        }
    }
    xs
}

fn generate_with_kv_cache(
    st: &SafeTensors<'_>,
    args: &Args,
    max_new_tokens: usize,
) -> Result<GenerationResult, Box<dyn Error>> {
    let mut real_ids = encode_unpadded(&args.tokenizer, &args.text, args.thinking)?;
    if real_ids.is_empty() {
        return Err(err("prompt produced no tokens"));
    }
    if real_ids.len() >= args.seq_len {
        return Err(err(format!(
            "prompt has {} tokens but seq_len is {}",
            real_ids.len(),
            args.seq_len
        )));
    }

    let prompt_tokens = real_ids.len();
    let load_layers_started = Instant::now();
    let fixed_layers = load_fixed_layers(st)?;
    let load_layers_elapsed = load_layers_started.elapsed();
    let load_lm_head_started = Instant::now();
    let lm_head = load_lm_head(st)?;
    let load_lm_head_elapsed = load_lm_head_started.elapsed();
    let rotary_started = Instant::now();
    let rotary = Rotary::new(args.seq_len);
    let rotary_elapsed = rotary_started.elapsed();
    let prefill_started = Instant::now();
    let (last_hidden, mut caches) =
        prefill_prompt_with_kv_cache(st, &fixed_layers, &rotary, &real_ids, &args.cfg)?;
    let prefill_elapsed = prefill_started.elapsed();
    let norm_started = Instant::now();
    let norm = vec_fixed_i32(st, "model.norm.weight", &[HIDDEN])?;
    let load_norm_elapsed = norm_started.elapsed();
    let first_lm_head_started = Instant::now();
    let mut logits = {
        let h_int = rms_norm_fixed_i32(&last_hidden, &norm, 1, HIDDEN);
        let h = fixed_i32_to_f32(&h_int);
        lm_head_scores_loaded(&lm_head, &h)?
    };
    let first_lm_head_elapsed = first_lm_head_started.elapsed();

    let mut ended_with_eos = false;
    let mut rng = SmallRng::new(args.sampling.seed);
    let mut streamed_text = decode_tokens(&args.tokenizer, &real_ids)?;
    let mut decode_model_elapsed = Duration::ZERO;
    let mut decode_lm_head_elapsed = Duration::ZERO;
    let mut choose_elapsed = Duration::ZERO;
    for _ in 0..max_new_tokens {
        let choose_started = Instant::now();
        apply_repetition_penalty(&mut logits, &real_ids, args.sampling.repetition_penalty);
        let next = choose_next_token(&logits, args.sampling, &mut rng) as u32;
        choose_elapsed += choose_started.elapsed();
        real_ids.push(next);
        stream_generated_delta(args, &real_ids, &mut streamed_text)?;
        if is_eos(next) {
            ended_with_eos = true;
            break;
        }
        if real_ids.len() >= args.seq_len {
            break;
        }
        let pos = real_ids.len() - 1;
        let decode_model_started = Instant::now();
        let hidden = decode_one_token_with_kv_cache(
            st,
            &fixed_layers,
            &rotary,
            next,
            pos,
            &mut caches,
            &args.cfg,
        )?;
        decode_model_elapsed += decode_model_started.elapsed();
        let decode_lm_head_started = Instant::now();
        let h_int = rms_norm_fixed_i32(&hidden, &norm, 1, HIDDEN);
        let h = fixed_i32_to_f32(&h_int);
        logits = lm_head_scores_loaded(&lm_head, &h)?;
        decode_lm_head_elapsed += decode_lm_head_started.elapsed();
    }

    if args.timing {
        println!(
            "timing.load_layers_bf16_to_i32_transpose: {:.3?}",
            load_layers_elapsed
        );
        println!("timing.load_lm_head_f32: {:.3?}", load_lm_head_elapsed);
        println!("timing.rotary_build: {:.3?}", rotary_elapsed);
        println!(
            "timing.prefill_prompt_with_kv_cache: {:.3?}",
            prefill_elapsed
        );
        println!("timing.load_final_norm: {:.3?}", load_norm_elapsed);
        println!("timing.first_lm_head: {:.3?}", first_lm_head_elapsed);
        println!("timing.decode_model_total: {:.3?}", decode_model_elapsed);
        println!(
            "timing.decode_lm_head_total: {:.3?}",
            decode_lm_head_elapsed
        );
        println!("timing.choose_token_total: {:.3?}", choose_elapsed);
    }
    let text = decode_generation_text(&args.tokenizer, &real_ids, prompt_tokens)?;
    Ok(GenerationResult {
        prompt_tokens,
        generated_tokens: real_ids.len() - prompt_tokens,
        total_tokens: real_ids.len(),
        ended_with_eos,
        text,
    })
}

fn stream_generated_delta(
    args: &Args,
    ids: &[u32],
    streamed_text: &mut String,
) -> Result<(), Box<dyn Error>> {
    if args.stream {
        let text = decode_tokens(&args.tokenizer, ids)?;
        let delta = text
            .strip_prefix(streamed_text.as_str())
            .unwrap_or(text.as_str());
        print!("{delta}");
        *streamed_text = text;
        std::io::stdout().flush()?;
    }
    Ok(())
}

fn apply_repetition_penalty(scores: &mut [f32], ids: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &id in ids {
        let idx = id as usize;
        if idx >= scores.len() {
            continue;
        }
        let score = &mut scores[idx];
        if *score > 0.0 {
            *score /= penalty;
        } else {
            *score *= penalty;
        }
    }
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(id, _)| id)
        .unwrap_or(EOS_IM_END as usize)
}

fn choose_next_token(scores: &[f32], sampling: Sampling, rng: &mut SmallRng) -> usize {
    if sampling.enabled {
        sample_token(scores, sampling, rng)
    } else {
        argmax(scores)
    }
}

fn sample_token(scores: &[f32], sampling: Sampling, rng: &mut SmallRng) -> usize {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(id, &score)| {
            (
                id,
                (((score - max_score) / sampling.temperature) as f64).exp(),
            )
        })
        .collect();
    let total = probs.iter().map(|(_, p)| *p).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return argmax(scores);
    }

    probs.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if let Some(top_k) = sampling.top_k {
        probs.truncate(top_k.min(probs.len()).max(1));
    }
    let total = probs.iter().map(|(_, p)| *p).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return argmax(scores);
    }
    let cutoff = sampling.top_p.clamp(0.0, 1.0) as f64;
    let mut kept_total = 0.0f64;
    let mut keep = 0usize;
    for &(_, p) in &probs {
        kept_total += p / total;
        keep += 1;
        if kept_total >= cutoff {
            break;
        }
    }
    probs.truncate(keep.max(1));

    let total = probs.iter().map(|(_, p)| *p).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return argmax(scores);
    }

    let mut draw = rng.next_f64() * total;
    for (id, prob) in probs {
        draw -= prob;
        if draw <= 0.0 {
            return id;
        }
    }
    scores.len().saturating_sub(1)
}

struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        const DEN: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DEN
    }
}

struct FixedLayerCache {
    k: Vec<i32>,
    v: Vec<i32>,
}

fn prefill_prompt_with_kv_cache(
    st: &SafeTensors<'_>,
    layers: &[FixedLayerWeights],
    r: &Rotary,
    ids: &[u32],
    cfg: &ForwardConfig,
) -> Result<(Vec<i32>, Vec<FixedLayerCache>), Box<dyn Error>> {
    assert_eq!(layers.len(), LAYERS);
    let seq = ids.len();
    let mut x = embed_fixed_from_safetensors(st, ids)?;
    let mut caches = Vec::with_capacity(LAYERS);
    for w in layers {
        let (next, cache) = prefill_layer_with_kv_cache(&x, w, r, seq, cfg);
        x = next;
        caches.push(cache);
    }
    Ok((x[(seq - 1) * HIDDEN..seq * HIDDEN].to_vec(), caches))
}

fn decode_one_token_with_kv_cache(
    st: &SafeTensors<'_>,
    layers: &[FixedLayerWeights],
    r: &Rotary,
    id: u32,
    pos: usize,
    caches: &mut [FixedLayerCache],
    cfg: &ForwardConfig,
) -> Result<Vec<i32>, Box<dyn Error>> {
    assert_eq!(layers.len(), caches.len());
    let mut x = embed_one_fixed_from_safetensors(st, id)?;
    for (w, cache) in layers.iter().zip(caches.iter_mut()) {
        x = decode_layer_with_kv_cache(&x, w, r, pos, cache, cfg);
    }
    Ok(x)
}

struct GenerationResult {
    prompt_tokens: usize,
    generated_tokens: usize,
    total_tokens: usize,
    ended_with_eos: bool,
    text: String,
}

fn prefill_layer_with_kv_cache(
    x: &[i32],
    w: &FixedLayerWeights,
    r: &Rotary,
    seq: usize,
    cfg: &ForwardConfig,
) -> (Vec<i32>, FixedLayerCache) {
    let n1 = rms_norm_fixed_i32(x, &w.ln1, seq, HIDDEN);
    let attn = run_attention_prefill_fixed(&n1, w, r, seq, cfg);
    let h = add_fixed_i32(x, &attn.hidden);
    (
        run_mlp_fixed(&h, w, seq, cfg),
        FixedLayerCache {
            k: attn.k,
            v: attn.v,
        },
    )
}

fn decode_layer_with_kv_cache(
    x: &[i32],
    w: &FixedLayerWeights,
    r: &Rotary,
    pos: usize,
    cache: &mut FixedLayerCache,
    cfg: &ForwardConfig,
) -> Vec<i32> {
    let n1 = rms_norm_fixed_i32(x, &w.ln1, 1, HIDDEN);
    let a = run_attention_decode_fixed(&n1, w, r, pos, cache, cfg);
    let h = add_fixed_i32(x, &a);
    run_mlp_fixed(&h, w, 1, cfg)
}

struct AttentionPrefillOutput {
    hidden: Vec<i32>,
    k: Vec<i32>,
    v: Vec<i32>,
}

fn run_attention_prefill_fixed(
    n1: &[i32],
    w: &FixedLayerWeights,
    r: &Rotary,
    seq: usize,
    cfg: &ForwardConfig,
) -> AttentionPrefillOutput {
    let q = matmul_fixed_i32(&n1, &w.q.wq, seq, HIDDEN, Q_DIM, cfg.matmul_rebase_rounding);
    let q = rms_norm_fixed_i32(&q, &w.q_norm, seq * HEADS, HEAD_DIM);
    let k = matmul_fixed_i32(
        &n1,
        &w.q.wk,
        seq,
        HIDDEN,
        KV_DIM,
        cfg.matmul_rebase_rounding,
    );
    let k = rms_norm_fixed_i32(&k, &w.k_norm, seq * KV_HEADS, HEAD_DIM);
    let v = matmul_fixed_i32(
        &n1,
        &w.q.wv,
        seq,
        HIDDEN,
        KV_DIM,
        cfg.matmul_rebase_rounding,
    );
    let q = rope_fixed_i32(&q, &r.rq[..q.len()]);
    let k = rope_fixed_i32(&k, &r.rk[..k.len()]);
    let s = score_qk_fixed_i32(&q, &k, seq);
    let p = softmax_fixed_i32(&s, HEADS * seq, seq);
    let c = attn_v_fixed_i32(&p, &v, seq);
    let a = matmul_fixed_i32(&c, &w.q.wo, seq, Q_DIM, HIDDEN, cfg.matmul_rebase_rounding);
    AttentionPrefillOutput { hidden: a, k, v }
}

fn run_attention_decode_fixed(
    n1: &[i32],
    w: &FixedLayerWeights,
    r: &Rotary,
    pos: usize,
    cache: &mut FixedLayerCache,
    cfg: &ForwardConfig,
) -> Vec<i32> {
    let q = matmul_fixed_i32(&n1, &w.q.wq, 1, HIDDEN, Q_DIM, cfg.matmul_rebase_rounding);
    let q = rms_norm_fixed_i32(&q, &w.q_norm, HEADS, HEAD_DIM);
    let k = matmul_fixed_i32(&n1, &w.q.wk, 1, HIDDEN, KV_DIM, cfg.matmul_rebase_rounding);
    let k = rms_norm_fixed_i32(&k, &w.k_norm, KV_HEADS, HEAD_DIM);
    let v = matmul_fixed_i32(&n1, &w.q.wv, 1, HIDDEN, KV_DIM, cfg.matmul_rebase_rounding);
    let q = rope_one_fixed_i32(&q, &r.rq, pos, HEADS);
    let k = rope_one_fixed_i32(&k, &r.rk, pos, KV_HEADS);
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    let context_len = cache.v.len() / KV_DIM;
    let c = attn_v_decode_fixed_i32(&q, &cache.k, &cache.v, context_len);
    matmul_fixed_i32(&c, &w.q.wo, 1, Q_DIM, HIDDEN, cfg.matmul_rebase_rounding)
}

fn run_mlp_fixed(x: &[i32], w: &FixedLayerWeights, rows: usize, cfg: &ForwardConfig) -> Vec<i32> {
    let n2 = rms_norm_fixed_i32(x, &w.ln2, rows, HIDDEN);
    let g = matmul_fixed_i32(
        &n2,
        &w.q.wg,
        rows,
        HIDDEN,
        INTERMEDIATE,
        cfg.matmul_rebase_rounding,
    );
    let u = matmul_fixed_i32(
        &n2,
        &w.q.wu,
        rows,
        HIDDEN,
        INTERMEDIATE,
        cfg.matmul_rebase_rounding,
    );
    let m = silu_mul_fixed_i32(&g, &u, cfg.sigmoid_input_rounding);
    let d = matmul_fixed_i32(
        &m,
        &w.q.wd,
        rows,
        INTERMEDIATE,
        HIDDEN,
        cfg.matmul_rebase_rounding,
    );
    add_fixed_i32(x, &d)
}

fn matmul_fixed_i32(
    a: &[i32],
    w: &[i32],
    m: usize,
    k: usize,
    n: usize,
    rounding: RoundingMode,
) -> Vec<i32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), k * n);
    let mut y = vec![0i32; m * n];
    if m == 1 {
        y.par_chunks_mut(256)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_col = chunk_idx * 256;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    matmul_fixed_i32_cols_neon(a, w, chunk, start_col, k, n, rounding);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    matmul_fixed_i32_cols_scalar(a, w, chunk, start_col, k, n, rounding);
                }
            });
        return y;
    }
    #[cfg(target_arch = "aarch64")]
    {
        y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            // SAFETY: aarch64 always has NEON. The helper uses unaligned loads
            // and falls back to scalar code for tail columns.
            unsafe {
                matmul_fixed_i32_row_neon(&a[r * k..(r + 1) * k], w, row, k, n, rounding);
            }
        });
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            matmul_fixed_i32_row_scalar(&a[r * k..(r + 1) * k], w, row, k, n, rounding);
        });
    }
    y
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_fixed_i32_cols_neon(
    a: &[i32],
    w: &[i32],
    out: &mut [i32],
    start_col: usize,
    k: usize,
    n: usize,
    rounding: RoundingMode,
) {
    use std::arch::aarch64::{
        int64x2_t, vaddq_s64, vdup_n_s32, vdupq_n_s64, vget_high_s32, vget_low_s32, vld1q_s32,
        vmull_s32, vshlq_s64, vst1q_s64,
    };

    let mut offset = 0usize;
    while offset + 4 <= out.len() {
        let col = start_col + offset;
        let mut tmp = [0i64; 4];
        unsafe {
            let mut acc_lo: int64x2_t = vdupq_n_s64(0);
            let mut acc_hi: int64x2_t = vdupq_n_s64(0);
            for i in 0..k {
                let ai = vdup_n_s32(a[i]);
                let wv = vld1q_s32(w.as_ptr().add(i * n + col));
                acc_lo = vaddq_s64(acc_lo, vmull_s32(ai, vget_low_s32(wv)));
                acc_hi = vaddq_s64(acc_hi, vmull_s32(ai, vget_high_s32(wv)));
            }
            // The floor case can rebase by SIMD arithmetic shift. Other modes
            // keep SIMD accumulation but use scalar post-processing for ties/signs.
            if rounding == RoundingMode::Floor {
                let shift = vdupq_n_s64(-(DEFAULT_FIXED_FRAC as i64));
                acc_lo = vshlq_s64(acc_lo, shift);
                acc_hi = vshlq_s64(acc_hi, shift);
            }
            vst1q_s64(tmp.as_mut_ptr(), acc_lo);
            vst1q_s64(tmp.as_mut_ptr().add(2), acc_hi);
        }
        for j in 0..4 {
            out[offset + j] = if rounding == RoundingMode::Floor {
                tmp[j] as i32
            } else {
                shift_signed_i64(tmp[j], DEFAULT_FIXED_FRAC, rounding) as i32
            };
        }
        offset += 4;
    }

    if offset < out.len() {
        matmul_fixed_i32_cols_scalar(a, w, &mut out[offset..], start_col + offset, k, n, rounding);
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_fixed_i32_cols_scalar(
    a: &[i32],
    w: &[i32],
    out: &mut [i32],
    start_col: usize,
    k: usize,
    n: usize,
    rounding: RoundingMode,
) {
    for (offset, slot) in out.iter_mut().enumerate() {
        let col = start_col + offset;
        let mut acc = 0i64;
        for i in 0..k {
            acc += a[i] as i64 * w[i * n + col] as i64;
        }
        *slot = shift_signed_i64(acc, DEFAULT_FIXED_FRAC, rounding) as i32;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn matmul_fixed_i32_row_scalar(
    a: &[i32],
    w: &[i32],
    row: &mut [i32],
    k: usize,
    n: usize,
    rounding: RoundingMode,
) {
    for c in 0..n {
        let mut acc = 0i64;
        for i in 0..k {
            acc += a[i] as i64 * w[i * n + c] as i64;
        }
        row[c] = shift_signed_i64(acc, DEFAULT_FIXED_FRAC, rounding) as i32;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_fixed_i32_row_neon(
    a: &[i32],
    w: &[i32],
    row: &mut [i32],
    k: usize,
    n: usize,
    rounding: RoundingMode,
) {
    use std::arch::aarch64::{
        int64x2_t, vaddq_s64, vdup_n_s32, vdupq_n_s64, vget_high_s32, vget_low_s32, vld1q_s32,
        vmull_s32, vshlq_s64, vst1q_s64,
    };

    let mut c = 0usize;
    while c + 4 <= n {
        let mut tmp = [0i64; 4];
        unsafe {
            let mut acc_lo: int64x2_t = vdupq_n_s64(0);
            let mut acc_hi: int64x2_t = vdupq_n_s64(0);
            for i in 0..k {
                let ai = vdup_n_s32(a[i]);
                let wv = vld1q_s32(w.as_ptr().add(i * n + c));
                acc_lo = vaddq_s64(acc_lo, vmull_s32(ai, vget_low_s32(wv)));
                acc_hi = vaddq_s64(acc_hi, vmull_s32(ai, vget_high_s32(wv)));
            }
            // The floor MatMul rebase is just an arithmetic right shift from
            // QX.(2 * frac) to QX.frac. Round/ceil need extra sign/tie
            // handling and stay on the scalar post-processing path.
            // We also store i64 lanes and cast to i32 below to preserve scalar
            // `as i32` semantics; saturating NEON narrowing would change behavior.
            if rounding == RoundingMode::Floor {
                let shift = vdupq_n_s64(-(DEFAULT_FIXED_FRAC as i64));
                acc_lo = vshlq_s64(acc_lo, shift);
                acc_hi = vshlq_s64(acc_hi, shift);
            }
            vst1q_s64(tmp.as_mut_ptr(), acc_lo);
            vst1q_s64(tmp.as_mut_ptr().add(2), acc_hi);
        }
        for j in 0..4 {
            row[c + j] = if rounding == RoundingMode::Floor {
                tmp[j] as i32
            } else {
                shift_signed_i64(tmp[j], DEFAULT_FIXED_FRAC, rounding) as i32
            };
        }
        c += 4;
    }

    if c < n {
        for col in c..n {
            let mut acc = 0i64;
            for i in 0..k {
                acc += a[i] as i64 * w[i * n + col] as i64;
            }
            row[col] = shift_signed_i64(acc, DEFAULT_FIXED_FRAC, rounding) as i32;
        }
    }
}

fn fixed_i32_to_f32(xs: &[i32]) -> Vec<f32> {
    let scale = (1u64 << DEFAULT_FIXED_FRAC) as f32;
    xs.par_iter().map(|&x| x as f32 / scale).collect()
}

fn add_fixed_i32(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    a.par_iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn rms_inv_from_square_sum(square_sum: i128, hidden_size: usize) -> i64 {
    debug_assert!(square_sum >= 0);
    debug_assert!(hidden_size > 0);
    let input_scale = (1u64 << DEFAULT_FIXED_FRAC) as f64;
    let output_scale = (1u64 << DEFAULT_FIXED_FRAC) as f64;
    let mean = square_sum as f64 / hidden_size as f64 / (input_scale * input_scale);
    let inv = 1.0 / (mean + 1e-6).sqrt();
    (inv * output_scale).round() as i64
}

fn rms_norm_fixed_i32(x: &[i32], w: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    let mut y = vec![0i32; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mut sq_acc = 0i128;
        for &xi in xs {
            sq_acc += xi as i128 * xi as i128;
        }
        let inv_int = rms_inv_from_square_sum(sq_acc, cols);
        for c in 0..cols {
            let norm_int = round_shift_signed_i64(xs[c] as i64 * inv_int, DEFAULT_FIXED_FRAC);
            row[c] = round_shift_signed_i64(norm_int * w[c] as i64, DEFAULT_FIXED_FRAC) as i32;
        }
    });
    y
}

fn rope_fixed_i32(x: &[i32], r: &[f32]) -> Vec<i32> {
    assert_eq!(x.len(), r.len());
    let mut y = vec![0i32; x.len()];
    let hd = if x.len() % HEAD_DIM == 0 {
        HEAD_DIM
    } else {
        x.len()
    };
    y.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let base = i * hd;
        rope_fixed_i32_chunk(&x[base..base + hd], &r[base..base + hd], out);
    });
    y
}

fn rope_one_fixed_i32(x: &[i32], r: &[f32], pos: usize, heads: usize) -> Vec<i32> {
    assert_eq!(x.len(), heads * HEAD_DIM);
    let mut y = vec![0i32; x.len()];
    for h in 0..heads {
        let base = h * HEAD_DIM;
        let rbase = (pos * heads + h) * HEAD_DIM;
        rope_fixed_i32_chunk(
            &x[base..base + HEAD_DIM],
            &r[rbase..rbase + HEAD_DIM],
            &mut y[base..base + HEAD_DIM],
        );
    }
    y
}

fn rope_fixed_i32_chunk(x: &[i32], r: &[f32], out: &mut [i32]) {
    let half = x.len() / 2;
    for d in 0..half {
        let a = x[d] as i64;
        let b = x[half + d] as i64;
        let c = quantize_fixed_i64_scalar(r[d]);
        let s = quantize_fixed_i64_scalar(r[half + d]);
        out[d] = round_shift_signed_i64(a * c - b * s, DEFAULT_FIXED_FRAC) as i32;
        out[half + d] = round_shift_signed_i64(b * c + a * s, DEFAULT_FIXED_FRAC) as i32;
    }
}

fn score_qk_fixed_i32(q: &[i32], k: &[i32], seq: usize) -> Vec<i32> {
    assert_eq!(q.len(), seq * HEADS * HEAD_DIM);
    assert_eq!(k.len(), seq * KV_DIM);
    let inv_sqrt_int =
        ((1.0 / (HEAD_DIM as f32).sqrt()) * (1u64 << DEFAULT_FIXED_FRAC) as f32).round() as i64;
    let mut y = vec![0i32; HEADS * seq * seq];
    y.par_chunks_mut(seq * seq).enumerate().for_each(|(h, ys)| {
        let kh = h / KV_GROUP;
        for i in 0..seq {
            for j in 0..seq {
                let o = i * seq + j;
                if j > i {
                    ys[o] = i32::MIN / 4;
                    continue;
                }
                let mut acc = 0i64;
                for d in 0..HEAD_DIM {
                    let qi = (i * HEADS + h) * HEAD_DIM + d;
                    let ki = (j * KV_HEADS + kh) * HEAD_DIM + d;
                    acc += q[qi] as i64 * k[ki] as i64;
                }
                let dot_int = round_shift_signed_i64(acc, DEFAULT_FIXED_FRAC);
                ys[o] = round_shift_signed_i64(dot_int * inv_sqrt_int, DEFAULT_FIXED_FRAC) as i32;
            }
        }
    });
    y
}

fn softmax_fixed_i32(x: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    assert_eq!(x.len(), rows * cols);
    let mut y = vec![0i32; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mx = xs.iter().copied().max().unwrap_or(0);
        let mut exps = vec![0i64; cols];
        let mut sum = 0i64;
        for c in 0..cols {
            if xs[c] <= i32::MIN / 8 {
                exps[c] = 0;
            } else {
                exps[c] = softmax_exp_coarse((xs[c] - mx) as i64);
            }
            sum += exps[c];
        }
        if sum == 0 {
            return;
        }
        for c in 0..cols {
            row[c] = div_round_i128((exps[c] as i128) << DEFAULT_FIXED_FRAC, sum as i128)
                .clamp(0, 1i128 << DEFAULT_FIXED_FRAC) as i32;
        }
    });
    y
}

fn div_round_i128(num: i128, den: i128) -> i128 {
    assert!(den > 0);
    let sign = num < 0;
    let abs_num = if sign { -num } else { num };
    let q = abs_num / den;
    let r = abs_num % den;
    let rounded_abs = if r * 2 >= den { q + 1 } else { q };
    if sign { -rounded_abs } else { rounded_abs }
}

fn softmax_exp_coarse(diff_int: i64) -> i64 {
    let clipped = diff_int.clamp(-(8i64 << DEFAULT_FIXED_FRAC), 0);
    let n = floor_shift_signed_i64(clipped, DEFAULT_FIXED_FRAC).clamp(-8, 0);
    let f_int = clipped - (n << DEFAULT_FIXED_FRAC);
    let exp_n = EXP_LUT_Q8[(n + 8) as usize];
    let corr = ((1i64 << DEFAULT_FIXED_FRAC) + f_int).max(0);
    round_shift_signed_i64(exp_n * corr, DEFAULT_FIXED_FRAC)
}

fn shift_signed_i64(x: i64, shift: u8, rounding: RoundingMode) -> i64 {
    match rounding {
        RoundingMode::Round => round_shift_signed_i64(x, shift),
        RoundingMode::Floor => floor_shift_signed_i64(x, shift),
        RoundingMode::Ceil => ceil_shift_signed_i64(x, shift),
    }
}

fn round_shift_signed_i64(x: i64, shift: u8) -> i64 {
    if shift == 0 {
        return x;
    }
    let q = floor_shift_signed_i64(x, shift);
    let denom = 1i64 << shift;
    let r = x - (q << shift);
    debug_assert!((0..denom).contains(&r));
    if shift == DEFAULT_FIXED_FRAC {
        q + ROUND_LUT_Q8[r as usize]
    } else if r * 2 >= denom {
        q + 1
    } else {
        q
    }
}

fn ceil_shift_signed_i64(x: i64, shift: u8) -> i64 {
    if shift == 0 {
        return x;
    }
    let denom = 1i128 << shift;
    let x = x as i128;
    let y = if x >= 0 {
        (x + denom - 1) / denom
    } else {
        -((-x) / denom)
    };
    y as i64
}

fn floor_shift_signed_i64(x: i64, shift: u8) -> i64 {
    if shift == 0 { x } else { x >> shift }
}

fn attn_v_fixed_i32(p: &[i32], v: &[i32], seq: usize) -> Vec<i32> {
    assert_eq!(p.len(), HEADS * seq * seq);
    assert_eq!(v.len(), seq * KV_DIM);
    let mut y = vec![0i32; seq * Q_DIM];
    y.par_chunks_mut(HEAD_DIM)
        .enumerate()
        .for_each(|(oh, out)| {
            let pos = oh / HEADS;
            let h = oh % HEADS;
            let kh = h / KV_GROUP;
            for (d, slot) in out.iter_mut().enumerate() {
                let mut acc = 0i64;
                for j in 0..seq {
                    let pi = (h * seq + pos) * seq + j;
                    let vi = (j * KV_HEADS + kh) * HEAD_DIM + d;
                    acc += p[pi] as i64 * v[vi] as i64;
                }
                *slot = round_shift_signed_i64(acc, DEFAULT_FIXED_FRAC) as i32;
            }
        });
    y
}

fn attn_v_decode_fixed_i32(q: &[i32], k: &[i32], v: &[i32], context_len: usize) -> Vec<i32> {
    assert_eq!(q.len(), Q_DIM);
    assert_eq!(k.len(), context_len * KV_DIM);
    assert_eq!(v.len(), context_len * KV_DIM);
    let mut y = vec![0i32; Q_DIM];
    let inv_sqrt_int =
        ((1.0 / (HEAD_DIM as f32).sqrt()) * (1u64 << DEFAULT_FIXED_FRAC) as f32).round() as i64;
    y.par_chunks_mut(HEAD_DIM).enumerate().for_each(|(h, out)| {
        let kh = h / KV_GROUP;
        let mut scores = vec![0i32; context_len];
        for (j, score) in scores.iter_mut().enumerate() {
            let mut acc = 0i64;
            for d in 0..HEAD_DIM {
                let qi = h * HEAD_DIM + d;
                let ki = (j * KV_HEADS + kh) * HEAD_DIM + d;
                acc += q[qi] as i64 * k[ki] as i64;
            }
            let dot_int = round_shift_signed_i64(acc, DEFAULT_FIXED_FRAC);
            *score = round_shift_signed_i64(dot_int * inv_sqrt_int, DEFAULT_FIXED_FRAC) as i32;
        }
        let mx = scores.iter().copied().max().unwrap_or(0);
        let mut exps = vec![0i64; context_len];
        let mut denom = 0i64;
        for j in 0..context_len {
            exps[j] = softmax_exp_coarse((scores[j] - mx) as i64);
            denom += exps[j];
        }
        for (d, slot) in out.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (j, &exp_int) in exps.iter().enumerate() {
                let p_int = div_round_i128((exp_int as i128) << DEFAULT_FIXED_FRAC, denom as i128)
                    .clamp(0, 1i128 << DEFAULT_FIXED_FRAC) as i64;
                let vi = (j * KV_HEADS + kh) * HEAD_DIM + d;
                acc += p_int * v[vi] as i64;
            }
            *slot = round_shift_signed_i64(acc, DEFAULT_FIXED_FRAC) as i32;
        }
    });
    y
}

fn silu_mul_fixed_i32(g: &[i32], u: &[i32], sigmoid_rounding: RoundingMode) -> Vec<i32> {
    assert_eq!(g.len(), u.len());
    g.par_iter()
        .zip(u)
        .map(|(&g, &u)| {
            let g_index = shift_signed_i64(g as i64, DEFAULT_FIXED_FRAC, sigmoid_rounding);
            let sig_int = sigmoid_from_integer_index(g_index);
            let silu_int = round_shift_signed_i64(g as i64 * sig_int, DEFAULT_FIXED_FRAC);
            round_shift_signed_i64(silu_int * u as i64, DEFAULT_FIXED_FRAC) as i32
        })
        .collect()
}

fn quantize_fixed_i64_scalar(x: f32) -> i64 {
    let scale = (1u64 << DEFAULT_FIXED_FRAC) as f32;
    (x * scale).round() as i64
}

fn sigmoid_from_integer_index(x: i64) -> i64 {
    let x = x.clamp(-8, 7);
    SIGMOID_LUT_Q8[(x + 8) as usize]
}

fn load_lm_head(st: &SafeTensors<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    tensor_f32(&t)
}

fn lm_head_scores_loaded(w: &[f32], x: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    if x.len() != HIDDEN {
        return Err(err(format!(
            "expected hidden length {HIDDEN}, got {}",
            x.len()
        )));
    }
    if w.len() != VOCAB * HIDDEN {
        return Err(err(format!(
            "expected lm_head length {}, got {}",
            VOCAB * HIDDEN,
            w.len()
        )));
    }

    Ok(w.par_chunks_exact(HIDDEN)
        .map(|row| x.iter().zip(row).map(|(&a, &b)| a * b).sum::<f32>())
        .collect())
}

fn err(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}
