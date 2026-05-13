use std::{
    collections::BTreeMap,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use half::{bf16, f16};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::json;
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
const LUT_FRAC: u8 = 8;
const EOS_IM_END: u32 = 151_645;
const EOS_END_OF_TEXT: u32 = 151_643;
const PAD: u32 = EOS_END_OF_TEXT;
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
    let ids = tokenize(&args.tokenizer, &args.text, args.seq_len, args.thinking)?;
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
    if let Some(path) = &args.calibration_jsonl {
        let out = args
            .activation_stats_out
            .clone()
            .ok_or("--calibration-jsonl requires --activation-stats-out")?;
        let stats = collect_activation_stats(&st, &args, path)?;
        stats.write_json(&out)?;
        println!("calibration_jsonl: {}", path.display());
        println!("activation_stats_out: {}", out.display());
        println!("samples: {}", stats.samples);
        println!("tokens: {}", stats.tokens);
        println!("elapsed: {:.2?}", started.elapsed());
        return Ok(());
    }
    if let Some(out) = &args.logit_compare_out {
        compare_logits(&st, &ids, &args, out)?;
        println!("logit_compare_out: {}", out.display());
        println!("elapsed: {:.2?}", started.elapsed());
        return Ok(());
    }
    if let Some(top_n) = args.outlier_tokens {
        let mut real_ids = encode_unpadded(&args.tokenizer, &args.text, args.thinking)?;
        real_ids.truncate(args.seq_len);
        report_down_outlier_tokens(&st, &args.tokenizer, &real_ids, top_n, false)?;
        println!("elapsed: {:.2?}", started.elapsed());
        return Ok(());
    }
    if let Some(top_n) = args.outlier_content_tokens {
        let mut real_ids = encode_unpadded(&args.tokenizer, &args.text, args.thinking)?;
        real_ids.truncate(args.seq_len);
        report_down_outlier_tokens(&st, &args.tokenizer, &real_ids, top_n, true)?;
        println!("elapsed: {:.2?}", started.elapsed());
        return Ok(());
    }
    if let Some((layer, pos, top_n)) = args.outlier_vector {
        let mut real_ids = encode_unpadded(&args.tokenizer, &args.text, args.thinking)?;
        real_ids.truncate(args.seq_len);
        report_down_outlier_vector(&st, &args.tokenizer, &real_ids, layer, pos, top_n)?;
        println!("elapsed: {:.2?}", started.elapsed());
        return Ok(());
    }
    if let Some(max_new_tokens) = args.generate_tokens {
        let generated = generate_greedy(&st, &args, max_new_tokens)?;
        println!("prompt: {:?}", args.text);
        println!("model: {}", args.model.display());
        println!("tokenizer: {}", args.tokenizer.display());
        println!("prompt_template: {}", prompt_template_name(args.thinking));
        println!("fixed: true");
        println!("fixed_frac: {}", args.cfg.fixed_frac);
        println!(
            "matmul_rebase_rounding: {}",
            args.cfg.matmul_rebase_rounding.name()
        );
        println!(
            "sigmoid_input_rounding: {}",
            args.cfg.sigmoid_input_rounding.name()
        );
        println!("seq_len: {}", args.seq_len);
        println!("kv_cache: {}", args.use_kv_cache);
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
        return Ok(());
    }
    let mut report = if args.report {
        Report::default()
    } else {
        Report::disabled()
    };
    let hidden = forward(&st, &ids, &args.cfg, &mut report)?;
    let ppl = perplexity(&st, &hidden, &ids, args.max_targets, &args.cfg, &mut report)?;

    println!("text: {:?}", args.text);
    println!("model: {}", args.model.display());
    println!("tokenizer: {}", args.tokenizer.display());
    println!("prompt_template: {}", prompt_template_name(args.thinking));
    println!("fixed: true");
    println!("fixed_frac: {}", args.cfg.fixed_frac);
    println!(
        "matmul_rebase_rounding: {}",
        args.cfg.matmul_rebase_rounding.name()
    );
    println!(
        "sigmoid_input_rounding: {}",
        args.cfg.sigmoid_input_rounding.name()
    );
    println!("seq_len: {}", args.seq_len);
    println!("input_tokens: {}", args.input_tokens);
    println!("context_tokens: {}", ids.len());
    if args.max_targets == usize::MAX {
        println!("targets: full");
    } else {
        println!("targets: first {}", args.max_targets);
    }
    println!("ppl_targets: {}", ppl.targets);
    println!("avg_nll: {:.6}", ppl.avg_nll);
    println!("ppl: {:.6}", ppl.ppl);
    println!("elapsed: {:.2?}", started.elapsed());
    if args.report {
        println!();
        report.print();
    }
    Ok(())
}

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    text: String,
    seq_len: usize,
    input_tokens: usize,
    generate_tokens: Option<usize>,
    use_kv_cache: bool,
    stream: bool,
    thinking: bool,
    sampling: Sampling,
    calibration_jsonl: Option<PathBuf>,
    activation_stats_out: Option<PathBuf>,
    calibration_limit: Option<usize>,
    logit_compare_out: Option<PathBuf>,
    logit_compare_top_k: usize,
    outlier_tokens: Option<usize>,
    outlier_content_tokens: Option<usize>,
    outlier_vector: Option<(usize, usize, usize)>,
    max_targets: usize,
    report: bool,
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
        let mut use_kv_cache = true;
        let mut stream = false;
        let mut thinking = false;
        let mut sampling = Sampling::default();
        let mut greedy = false;
        let mut calibration_jsonl = None;
        let mut activation_stats_out = None;
        let mut calibration_limit = None;
        let mut logit_compare_out = None;
        let mut logit_compare_top_k = 20usize;
        let mut outlier_tokens = None;
        let mut outlier_content_tokens = None;
        let mut outlier_vector = None;
        let mut max_targets = usize::MAX;
        let mut report = false;
        let mut timing = false;
        let mut cfg = ForwardConfig::default();

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = PathBuf::from(args.next().ok_or("--model requires a path")?),
                "--tokenizer" => {
                    tokenizer = PathBuf::from(args.next().ok_or("--tokenizer requires a path")?)
                }
                "--fixed-frac" => {
                    let frac = args
                        .next()
                        .ok_or("--fixed-frac requires a value")?
                        .parse()?;
                    if frac > 12 {
                        return Err(err("--fixed-frac must be in 0..=12"));
                    }
                    cfg.set_fixed_frac(frac);
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
                "--max-targets" => {
                    max_targets = args
                        .next()
                        .ok_or("--max-targets requires a value")?
                        .parse()?
                }
                "--generate" | "--generate-tokens" => {
                    generate_tokens = Some(
                        args.next()
                            .ok_or("--generate requires a token count")?
                            .parse()?,
                    );
                }
                "--no-kv-cache" => use_kv_cache = false,
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
                "--calibration-jsonl" => {
                    calibration_jsonl = Some(PathBuf::from(
                        args.next().ok_or("--calibration-jsonl requires a path")?,
                    ));
                }
                "--activation-stats-out" => {
                    activation_stats_out = Some(PathBuf::from(
                        args.next()
                            .ok_or("--activation-stats-out requires a path")?,
                    ));
                }
                "--calibration-limit" => {
                    calibration_limit = Some(
                        args.next()
                            .ok_or("--calibration-limit requires a value")?
                            .parse()?,
                    );
                }
                "--logit-compare-out" => {
                    logit_compare_out = Some(PathBuf::from(
                        args.next().ok_or("--logit-compare-out requires a path")?,
                    ));
                }
                "--logit-compare-top-k" => {
                    logit_compare_top_k = args
                        .next()
                        .ok_or("--logit-compare-top-k requires a value")?
                        .parse()?;
                    if logit_compare_top_k == 0 || logit_compare_top_k > VOCAB {
                        return Err(err(format!("--logit-compare-top-k must be in 1..={VOCAB}")));
                    }
                }
                "--outlier-tokens" => {
                    outlier_tokens = Some(
                        args.next()
                            .ok_or("--outlier-tokens requires a value")?
                            .parse()?,
                    );
                }
                "--outlier-content-tokens" => {
                    outlier_content_tokens = Some(
                        args.next()
                            .ok_or("--outlier-content-tokens requires a value")?
                            .parse()?,
                    );
                }
                "--outlier-vector" => {
                    let layer = args
                        .next()
                        .ok_or("--outlier-vector requires layer pos top_n")?
                        .parse()?;
                    let pos = args
                        .next()
                        .ok_or("--outlier-vector requires layer pos top_n")?
                        .parse()?;
                    let top_n = args
                        .next()
                        .ok_or("--outlier-vector requires layer pos top_n")?
                        .parse()?;
                    outlier_vector = Some((layer, pos, top_n));
                }
                "--full" => max_targets = usize::MAX,
                "--report" => report = true,
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
        let input_tokens = count_tokens(&tokenizer, &text, thinking)?;

        Ok(Self {
            model,
            tokenizer,
            text,
            seq_len,
            input_tokens,
            generate_tokens,
            use_kv_cache,
            stream,
            thinking,
            sampling,
            calibration_jsonl,
            activation_stats_out,
            calibration_limit,
            logit_compare_out,
            logit_compare_top_k,
            outlier_tokens,
            outlier_content_tokens,
            outlier_vector,
            max_targets,
            report,
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
           --fixed-frac N        fractional bits for QX.N fixed-point block runtime; default 8; range 0..=12\n\
           --matmul-rebase-rounding round|floor|ceil MatMul accumulator rebase rounding; default round\n\
           --sigmoid-input-rounding round|floor|ceil sigmoid input integer rounding; default round\n\
           --seq-len N           context length; default 128\n\
           --generate N          generate up to N new tokens\n\
           --thinking            use Qwen3 thinking chat prompt instead of no-think\n\
           --no-think            use Qwen3 no-think prompt; default\n\
           --no-kv-cache         use slow full-forward generation\n\
           --stream              print decoded text after each generated token\n\
           --sample              sample next token; default on for Qwen3\n\
           --greedy              disable sampling and use argmax\n\
           --seed N              deterministic sampling seed\n\
           --temperature T       sampling temperature; default 0.6\n\
           --top-p P             nucleus sampling cutoff; default 0.95\n\
           --top-k K             keep only top K candidates before top-p; default 20\n\
           --repetition-penalty R penalize previously generated tokens; default 1.0\n\
           --calibration-jsonl PATH   read generated texts from JSONL and collect stats\n\
           --activation-stats-out PATH write activation stats JSON\n\
           --calibration-limit N      limit calibration samples for quick tests\n\
           --logit-compare-out PATH   write per-position float vs quant top-k JSONL\n\
           --logit-compare-top-k N    top-k to save for logit compare; default 20\n\
           --outlier-tokens N    print top N down_proj output outliers by layer/token\n\
           --outlier-content-tokens N print top N outliers excluding template/control tokens\n\
           --outlier-vector L P N print top N channels for down_proj output at layer L, position P\n\
           --max-targets N       limit PPL targets\n\
           --full                use all non-EOS prefix targets\n\
           --report              print per-MatMul error and integer-width reports\n\
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
    a_bits: QuantBits,
    w_bits: QuantBits,
    y_bits: QuantBits,
    quant_mode: QuantMode,
    enabled: MatmulSet,
    fixed_frac: u8,
    matmul_rebase_rounding: RoundingMode,
    sigmoid_input_rounding: RoundingMode,
}

impl ForwardConfig {
    fn set_fixed_frac(&mut self, frac: u8) {
        self.fixed_frac = frac;
        self.a_bits = QuantBits(Some(frac));
        self.w_bits = QuantBits(Some(frac));
        self.y_bits = QuantBits(Some(frac));
    }
}

impl Default for ForwardConfig {
    fn default() -> Self {
        let mut cfg = Self {
            a_bits: QuantBits::default(),
            w_bits: QuantBits::default(),
            y_bits: QuantBits::default(),
            quant_mode: QuantMode::Fixed,
            enabled: MatmulSet::default(),
            fixed_frac: DEFAULT_FIXED_FRAC,
            matmul_rebase_rounding: RoundingMode::Round,
            sigmoid_input_rounding: RoundingMode::Round,
        };
        cfg.set_fixed_frac(DEFAULT_FIXED_FRAC);
        cfg
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

#[derive(Clone, Copy, Debug, Default)]
struct QuantBits(Option<u8>);

#[derive(Clone, Copy, Debug, Default)]
enum QuantMode {
    #[default]
    Affine,
    Fixed,
}

impl ForwardConfig {
    fn has_quant(self) -> bool {
        self.a_bits.0.is_some() || self.w_bits.0.is_some() || self.y_bits.0.is_some()
    }

    fn fixed_runtime(self) -> bool {
        self.has_quant() && matches!(self.quant_mode, QuantMode::Fixed)
    }
}

#[derive(Clone, Copy, Debug)]
struct MatmulSet {
    q: bool,
    k: bool,
    v: bool,
    o: bool,
    gate: bool,
    up: bool,
    down: bool,
    lm_head: bool,
}

impl Default for MatmulSet {
    fn default() -> Self {
        Self {
            q: true,
            k: true,
            v: true,
            o: true,
            gate: true,
            up: true,
            down: true,
            lm_head: false,
        }
    }
}

impl MatmulSet {
    fn enabled(self, site: MatmulSite) -> bool {
        match site {
            MatmulSite::Q => self.q,
            MatmulSite::K => self.k,
            MatmulSite::V => self.v,
            MatmulSite::O => self.o,
            MatmulSite::Gate => self.gate,
            MatmulSite::Up => self.up,
            MatmulSite::Down => self.down,
            MatmulSite::LmHead => self.lm_head,
        }
    }

    fn is_model_default(self) -> bool {
        self.q && self.k && self.v && self.o && self.gate && self.up && self.down && !self.lm_head
    }
}

#[derive(Clone, Copy, Debug)]
enum MatmulSite {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    LmHead,
}

impl MatmulSite {
    fn name(self) -> &'static str {
        match self {
            MatmulSite::Q => "q_proj",
            MatmulSite::K => "k_proj",
            MatmulSite::V => "v_proj",
            MatmulSite::O => "o_proj",
            MatmulSite::Gate => "gate_proj",
            MatmulSite::Up => "up_proj",
            MatmulSite::Down => "down_proj",
            MatmulSite::LmHead => "lm_head",
        }
    }
}

fn count_tokens(path: &PathBuf, text: &str, thinking: bool) -> Result<usize, Box<dyn Error>> {
    let tok = Tokenizer::from_file(path).map_err(|e| err(e.to_string()))?;
    let prompt = chat_prompt(text, thinking);
    let enc = tok
        .encode(prompt.as_str(), true)
        .map_err(|e| err(e.to_string()))?;
    Ok(enc.get_ids().len())
}

fn tokenize(
    path: &PathBuf,
    text: &str,
    seq_len: usize,
    thinking: bool,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let tok = Tokenizer::from_file(path).map_err(|e| err(e.to_string()))?;
    let prompt = chat_prompt(text, thinking);
    let enc = tok
        .encode(prompt.as_str(), true)
        .map_err(|e| err(e.to_string()))?;
    let mut ids = enc.get_ids().to_vec();
    ids.truncate(seq_len);
    ids.resize(seq_len, PAD);
    Ok(ids)
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

fn no_think_chat_prompt(text: &str) -> String {
    chat_prompt(text, false)
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

fn pad_to_seq(mut ids: Vec<u32>, seq_len: usize) -> Vec<u32> {
    ids.truncate(seq_len);
    ids.resize(seq_len, PAD);
    ids
}

fn is_eos(id: u32) -> bool {
    id == EOS_IM_END || id == EOS_END_OF_TEXT
}

fn is_eos_usize(id: usize) -> bool {
    id == EOS_IM_END as usize || id == EOS_END_OF_TEXT as usize
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

fn vec_f32(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, shape, name)?;
    tensor_f32(&t)
}

fn transpose(xs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(xs.len(), rows * cols);
    let mut ys = vec![0.0; xs.len()];
    for r in 0..rows {
        for c in 0..cols {
            ys[c * rows + r] = xs[r * cols + c];
        }
    }
    ys
}

fn linear_f32(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, &[out, input], name)?;
    Ok(transpose(&tensor_f32(&t)?, out, input))
}

fn row_f32(
    t: &TensorView<'_>,
    row: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> Result<(), Box<dyn Error>> {
    match t.dtype() {
        Dtype::BF16 => {
            let start = row * cols * 2;
            for b in t.data()[start..start + cols * 2].chunks_exact(2) {
                out.push(bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            }
        }
        Dtype::F16 => {
            let start = row * cols * 2;
            for b in t.data()[start..start + cols * 2].chunks_exact(2) {
                out.push(f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            }
        }
        Dtype::F32 => {
            let start = row * cols * 4;
            for b in t.data()[start..start + cols * 4].chunks_exact(4) {
                out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
        }
        dt => return Err(err(format!("unsupported row dtype {dt:?}"))),
    }
    Ok(())
}

fn embed_from_safetensors(st: &SafeTensors<'_>, ids: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    let mut out = Vec::with_capacity(ids.len() * HIDDEN);
    for &id in ids {
        let id = id as usize;
        if id >= VOCAB {
            return Err(err(format!("token id {id} exceeds vocab")));
        }
        row_f32(&t, id, HIDDEN, &mut out)?;
    }
    Ok(out)
}

fn embed_one_from_safetensors(st: &SafeTensors<'_>, id: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    if id as usize >= VOCAB {
        return Err(err(format!("token id {id} exceeds vocab")));
    }
    let mut out = Vec::with_capacity(HIDDEN);
    row_f32(&t, id as usize, HIDDEN, &mut out)?;
    Ok(out)
}

fn embed_fixed_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
    frac_bits: u8,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    let mut out = Vec::with_capacity(ids.len() * HIDDEN);
    for &id in ids {
        row_fixed_i32(&t, id as usize, HIDDEN, frac_bits, &mut out)?;
    }
    Ok(out)
}

fn embed_one_fixed_from_safetensors(
    st: &SafeTensors<'_>,
    id: u32,
    frac_bits: u8,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    let mut out = Vec::with_capacity(HIDDEN);
    row_fixed_i32(&t, id as usize, HIDDEN, frac_bits, &mut out)?;
    Ok(out)
}

fn row_fixed_i32(
    t: &TensorView<'_>,
    row: usize,
    cols: usize,
    frac_bits: u8,
    out: &mut Vec<i32>,
) -> Result<(), Box<dyn Error>> {
    if row >= VOCAB {
        return Err(err(format!("token id {row} exceeds vocab")));
    }
    let scale = (1u64 << frac_bits) as f32;
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

struct LayerWeights {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    wq: Vec<f32>,
    bq: Vec<f32>,
    q_norm: Vec<f32>,
    wk: Vec<f32>,
    bk: Vec<f32>,
    k_norm: Vec<f32>,
    wv: Vec<f32>,
    bv: Vec<f32>,
    wo: Vec<f32>,
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
}

struct QuantizedLayerWeights {
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
    q: QuantizedLayerWeights,
}

fn quantize_layers_fixed(layers: &[LayerWeights], frac_bits: u8) -> Vec<QuantizedLayerWeights> {
    layers
        .par_iter()
        .map(|w| QuantizedLayerWeights {
            wq: quantize_fixed_i32(&w.wq, frac_bits),
            wk: quantize_fixed_i32(&w.wk, frac_bits),
            wv: quantize_fixed_i32(&w.wv, frac_bits),
            wo: quantize_fixed_i32(&w.wo, frac_bits),
            wg: quantize_fixed_i32(&w.wg, frac_bits),
            wu: quantize_fixed_i32(&w.wu, frac_bits),
            wd: quantize_fixed_i32(&w.wd, frac_bits),
        })
        .collect()
}

fn quantized_linear_i32_transposed(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
    frac_bits: u8,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, &[out, input], name)?;
    let scale = (1u64 << frac_bits) as f32;
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
    frac_bits: u8,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, shape, name)?;
    let scale = (1u64 << frac_bits) as f32;
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
    frac_bits: u8,
) -> Result<FixedLayerWeights, Box<dyn Error>> {
    let p = format!("model.layers.{layer}");
    Ok(FixedLayerWeights {
        ln1: vec_fixed_i32(
            st,
            &format!("{p}.input_layernorm.weight"),
            &[HIDDEN],
            frac_bits,
        )?,
        ln2: vec_fixed_i32(
            st,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
            frac_bits,
        )?,
        q_norm: vec_fixed_i32(
            st,
            &format!("{p}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
            frac_bits,
        )?,
        k_norm: vec_fixed_i32(
            st,
            &format!("{p}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
            frac_bits,
        )?,
        q: QuantizedLayerWeights {
            wq: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.q_proj.weight"),
                Q_DIM,
                HIDDEN,
                frac_bits,
            )?,
            wk: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.k_proj.weight"),
                KV_DIM,
                HIDDEN,
                frac_bits,
            )?,
            wv: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.v_proj.weight"),
                KV_DIM,
                HIDDEN,
                frac_bits,
            )?,
            wo: quantized_linear_i32_transposed(
                st,
                &format!("{p}.self_attn.o_proj.weight"),
                HIDDEN,
                Q_DIM,
                frac_bits,
            )?,
            wg: quantized_linear_i32_transposed(
                st,
                &format!("{p}.mlp.gate_proj.weight"),
                INTERMEDIATE,
                HIDDEN,
                frac_bits,
            )?,
            wu: quantized_linear_i32_transposed(
                st,
                &format!("{p}.mlp.up_proj.weight"),
                INTERMEDIATE,
                HIDDEN,
                frac_bits,
            )?,
            wd: quantized_linear_i32_transposed(
                st,
                &format!("{p}.mlp.down_proj.weight"),
                HIDDEN,
                INTERMEDIATE,
                frac_bits,
            )?,
        },
    })
}

fn load_fixed_layers(
    st: &SafeTensors<'_>,
    frac_bits: u8,
) -> Result<Vec<FixedLayerWeights>, Box<dyn Error>> {
    (0..LAYERS)
        .map(|layer| load_fixed_layer(st, layer, frac_bits))
        .collect()
}

fn load_layer(st: &SafeTensors<'_>, layer: usize) -> Result<LayerWeights, Box<dyn Error>> {
    let p = format!("model.layers.{layer}");
    Ok(LayerWeights {
        ln1: vec_f32(st, &format!("{p}.input_layernorm.weight"), &[HIDDEN])?,
        ln2: vec_f32(
            st,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
        )?,
        wq: linear_f32(st, &format!("{p}.self_attn.q_proj.weight"), Q_DIM, HIDDEN)?,
        bq: vec![0.0; Q_DIM],
        q_norm: vec_f32(st, &format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM])?,
        wk: linear_f32(st, &format!("{p}.self_attn.k_proj.weight"), KV_DIM, HIDDEN)?,
        bk: vec![0.0; KV_DIM],
        k_norm: vec_f32(st, &format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM])?,
        wv: linear_f32(st, &format!("{p}.self_attn.v_proj.weight"), KV_DIM, HIDDEN)?,
        bv: vec![0.0; KV_DIM],
        wo: linear_f32(st, &format!("{p}.self_attn.o_proj.weight"), HIDDEN, Q_DIM)?,
        wg: linear_f32(
            st,
            &format!("{p}.mlp.gate_proj.weight"),
            INTERMEDIATE,
            HIDDEN,
        )?,
        wu: linear_f32(st, &format!("{p}.mlp.up_proj.weight"), INTERMEDIATE, HIDDEN)?,
        wd: linear_f32(
            st,
            &format!("{p}.mlp.down_proj.weight"),
            HIDDEN,
            INTERMEDIATE,
        )?,
    })
}

fn load_layers(st: &SafeTensors<'_>) -> Result<Vec<LayerWeights>, Box<dyn Error>> {
    (0..LAYERS).map(|layer| load_layer(st, layer)).collect()
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

fn forward(
    st: &SafeTensors<'_>,
    ids: &[u32],
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let seq = ids.len();
    let r = Rotary::new(seq);
    let mut x = embed_from_safetensors(st, ids)?;
    x = maybe_fixed_q8(x, cfg);
    for layer in 0..LAYERS {
        let w = load_layer(st, layer)?;
        x = layer_forward(&x, &w, &r, seq, cfg, report);
    }
    let w = vec_f32(st, "model.norm.weight", &[HIDDEN])?;
    Ok(rms_norm_cfg(&x, &w, seq, HIDDEN, cfg))
}

fn collect_activation_stats(
    st: &SafeTensors<'_>,
    args: &Args,
    path: &PathBuf,
) -> Result<ActivationStats, Box<dyn Error>> {
    let layers = load_layers(st)?;
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| err(e.to_string()))?;
    let mut stats = ActivationStats::new();
    let mut report = Report::disabled();
    let cfg = ForwardConfig {
        a_bits: QuantBits(None),
        w_bits: QuantBits(None),
        y_bits: QuantBits(None),
        quant_mode: QuantMode::Affine,
        enabled: args.cfg.enabled,
        ..ForwardConfig::default()
    };

    for (line_no, line) in reader.lines().enumerate() {
        if let Some(limit) = args.calibration_limit
            && stats.samples >= limit
        {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)?;
        let text = value
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| err(format!("line {} missing string field 'text'", line_no + 1)))?;
        let prompt = no_think_chat_prompt(text);
        let enc = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| err(e.to_string()))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(args.seq_len);
        if ids.is_empty() {
            continue;
        }
        forward_with_activation_stats(st, &layers, &ids, &cfg, &mut report, &mut stats)?;
        stats.samples += 1;
        stats.tokens += ids.len();
        println!(
            "calibrated {:03}: tokens={} text_len={}",
            stats.samples,
            ids.len(),
            text.len()
        );
    }
    Ok(stats)
}

fn compare_logits(
    st: &SafeTensors<'_>,
    ids: &[u32],
    args: &Args,
    out: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| err(e.to_string()))?;
    let mut float_report = Report::disabled();
    let mut quant_report = Report::disabled();
    let float_cfg = ForwardConfig {
        a_bits: QuantBits(None),
        w_bits: QuantBits(None),
        y_bits: QuantBits(None),
        quant_mode: QuantMode::Affine,
        enabled: args.cfg.enabled,
        ..ForwardConfig::default()
    };
    let float_hidden = forward(st, ids, &float_cfg, &mut float_report)?;
    let quant_hidden = forward(st, ids, &args.cfg, &mut quant_report)?;
    let mut file = File::create(out)?;
    let mut count = 0usize;

    for pos in 0..ids.len() - 1 {
        let target = ids[pos + 1] as usize;
        if is_eos_usize(target) {
            break;
        }
        let xf = &float_hidden[pos * HIDDEN..(pos + 1) * HIDDEN];
        let xq = &quant_hidden[pos * HIDDEN..(pos + 1) * HIDDEN];
        let float_logits = lm_head_scores(st, xf, &float_cfg, &mut float_report)?;
        let quant_logits = lm_head_scores(st, xq, &args.cfg, &mut quant_report)?;
        let float_probs = softmax_vector(&float_logits);
        let quant_probs = softmax_vector(&quant_logits);
        let float_top = top_k_indices(&float_logits, args.logit_compare_top_k);
        let quant_top = top_k_indices(&quant_logits, args.logit_compare_top_k);
        let float_rank = ranks_from_logits(&float_logits);
        let quant_rank = ranks_from_logits(&quant_logits);
        let float_top_set: std::collections::BTreeSet<_> = float_top.iter().copied().collect();
        let quant_top_set: std::collections::BTreeSet<_> = quant_top.iter().copied().collect();
        let overlap = float_top_set.intersection(&quant_top_set).count();
        let record = json!({
            "pos": pos,
            "context_tail": context_tail(&tokenizer, &ids[..=pos], 80)?,
            "target": token_record(target, &tokenizer, &float_logits, &float_probs, &quant_logits, &quant_probs, &float_rank, &quant_rank)?,
            "float_topk": top_records(&float_top, &tokenizer, &float_logits, &float_probs, &quant_logits, &quant_probs, &float_rank, &quant_rank)?,
            "quant_topk": top_records(&quant_top, &tokenizer, &quant_logits, &quant_probs, &float_logits, &float_probs, &quant_rank, &float_rank)?,
            "metrics": {
                "top1_match": float_top.first() == quant_top.first(),
                "topk_overlap": overlap,
                "topk": args.logit_compare_top_k,
                "float_top1_quant_rank": quant_rank[float_top[0]],
                "quant_top1_float_rank": float_rank[quant_top[0]],
                "float_entropy": entropy(&float_probs),
                "quant_entropy": entropy(&quant_probs),
                "float_top1_logit_margin": float_logits[float_top[0]] - float_logits[float_top[1]],
                "quant_top1_logit_margin": quant_logits[quant_top[0]] - quant_logits[quant_top[1]],
                "float_top1_prob_margin": float_probs[float_top[0]] - float_probs[float_top[1]],
                "quant_top1_prob_margin": quant_probs[quant_top[0]] - quant_probs[quant_top[1]],
                "target_logprob_delta_quant_minus_float": quant_probs[target].ln() - float_probs[target].ln(),
                "js_divergence": js_divergence(&float_probs, &quant_probs),
                "logit_cosine": cosine(&float_logits, &quant_logits),
            }
        });
        writeln!(file, "{}", serde_json::to_string(&record)?)?;
        count += 1;
        println!(
            "compared pos {:03}: top1_match={} topk_overlap={}",
            pos,
            float_top.first() == quant_top.first(),
            overlap
        );
    }
    println!("logit_compare_positions: {count}");
    Ok(())
}

fn forward_with_activation_stats(
    st: &SafeTensors<'_>,
    layers: &[LayerWeights],
    ids: &[u32],
    cfg: &ForwardConfig,
    report: &mut Report,
    stats: &mut ActivationStats,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let seq = ids.len();
    let r = Rotary::new(seq);
    let mut x = embed_from_safetensors(st, ids)?;
    for (layer, w) in layers.iter().enumerate() {
        x = layer_forward_with_stats(&x, w, &r, seq, cfg, report, stats, layer);
    }
    let w = vec_f32(st, "model.norm.weight", &[HIDDEN])?;
    Ok(rms_norm(&x, &w, seq, HIDDEN))
}

#[derive(Clone)]
struct DownOutlierRecord {
    absmax: f32,
    min: f32,
    max: f32,
    layer: usize,
    pos: usize,
    token_id: u32,
}

fn report_down_outlier_tokens(
    st: &SafeTensors<'_>,
    tokenizer_path: &PathBuf,
    ids: &[u32],
    top_n: usize,
    content_only: bool,
) -> Result<(), Box<dyn Error>> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| err(e.to_string()))?;
    let mut records = collect_down_outlier_tokens(st, ids)?;
    if content_only {
        records.retain(|rec| is_content_token(rec.token_id, &tokenizer));
    }
    records.sort_by(|a, b| b.absmax.total_cmp(&a.absmax));
    println!(
        "{:<5} {:<5} {:<7} {:>12} {:>12} {:>12} token",
        "rank", "layer", "pos", "absmax", "min", "max"
    );
    for (rank, rec) in records.into_iter().take(top_n).enumerate() {
        let token = tokenizer
            .decode(&[rec.token_id], true)
            .unwrap_or_else(|_| format!("<{}>", rec.token_id))
            .replace('\n', "\\n");
        println!(
            "{:<5} {:<5} {:<7} {:>12.3} {:>12.3} {:>12.3} {:?} ({})",
            rank + 1,
            rec.layer,
            rec.pos,
            rec.absmax,
            rec.min,
            rec.max,
            token,
            rec.token_id
        );
    }
    Ok(())
}

fn is_content_token(token_id: u32, tokenizer: &Tokenizer) -> bool {
    if token_id >= 151_643 {
        return false;
    }
    let Ok(token) = tokenizer.decode(&[token_id], true) else {
        return false;
    };
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return false;
    }
    !matches!(trimmed, "user" | "assistant" | "system")
}

fn collect_down_outlier_tokens(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<DownOutlierRecord>, Box<dyn Error>> {
    let seq = ids.len();
    let r = Rotary::new(seq);
    let layers = load_layers(st)?;
    let mut x = embed_from_safetensors(st, ids)?;
    let mut records = Vec::with_capacity(LAYERS * seq);
    for (layer, w) in layers.iter().enumerate() {
        let n1 = rms_norm(&x, &w.ln1, seq, HIDDEN);
        let mut q = matmul(&n1, &w.wq, seq, HIDDEN, Q_DIM);
        add_rows(&mut q, &w.bq);
        let q = rms_norm_heads(&q, &w.q_norm, seq, HEADS);
        let mut k = matmul(&n1, &w.wk, seq, HIDDEN, KV_DIM);
        add_rows(&mut k, &w.bk);
        let k = rms_norm_heads(&k, &w.k_norm, seq, KV_HEADS);
        let mut v = matmul(&n1, &w.wv, seq, HIDDEN, KV_DIM);
        add_rows(&mut v, &w.bv);
        let q = rope(&q, &r.rq[..q.len()]);
        let k = rope(&k, &r.rk[..k.len()]);
        let s = score_qk(&q, &k, seq);
        let p = softmax(&s, HEADS * seq, seq);
        let c = attn_v(&p, &v, seq);
        let a = matmul(&c, &w.wo, seq, Q_DIM, HIDDEN);
        let h = add(&x, &a);
        let n2 = rms_norm(&h, &w.ln2, seq, HIDDEN);
        let g = matmul(&n2, &w.wg, seq, HIDDEN, INTERMEDIATE);
        let u = matmul(&n2, &w.wu, seq, HIDDEN, INTERMEDIATE);
        let m = silu_mul(&g, &u);
        let d = matmul(&m, &w.wd, seq, INTERMEDIATE, HIDDEN);
        for (pos, row) in d.chunks_exact(HIDDEN).enumerate() {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut absmax = 0.0f32;
            for &value in row {
                min = min.min(value);
                max = max.max(value);
                absmax = absmax.max(value.abs());
            }
            records.push(DownOutlierRecord {
                absmax,
                min,
                max,
                layer,
                pos,
                token_id: ids[pos],
            });
        }
        x = add(&h, &d);
    }
    Ok(records)
}

fn report_down_outlier_vector(
    st: &SafeTensors<'_>,
    tokenizer_path: &PathBuf,
    ids: &[u32],
    target_layer: usize,
    target_pos: usize,
    top_n: usize,
) -> Result<(), Box<dyn Error>> {
    if target_layer >= LAYERS {
        return Err(err(format!("layer must be in 0..{LAYERS}")));
    }
    if target_pos >= ids.len() {
        return Err(err(format!(
            "pos must be in 0..{} for this input",
            ids.len()
        )));
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| err(e.to_string()))?;
    let row = down_output_row(st, ids, target_layer, target_pos)?;
    let token = tokenizer
        .decode(&[ids[target_pos]], true)
        .unwrap_or_else(|_| format!("<{}>", ids[target_pos]))
        .replace('\n', "\\n");
    let min = row.iter().copied().fold(f32::INFINITY, f32::min);
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = row.iter().copied().sum::<f32>() / row.len() as f32;
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32).sqrt();
    let absmax = row.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let above_100 = row.iter().filter(|v| v.abs() >= 100.0).count();
    let above_1000 = row.iter().filter(|v| v.abs() >= 1000.0).count();
    println!(
        "layer={target_layer} pos={target_pos} token={:?} ({})",
        token, ids[target_pos]
    );
    println!(
        "min={min:.3} max={max:.3} mean={mean:.3} rms={rms:.3} absmax={absmax:.3} |x|>=100: {above_100} |x|>=1000: {above_1000}"
    );
    let mut indexed: Vec<_> = row.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    println!("{:<5} {:<8} {:>12}", "rank", "channel", "value");
    for (rank, (channel, value)) in indexed.into_iter().take(top_n).enumerate() {
        println!("{:<5} {:<8} {:>12.3}", rank + 1, channel, value);
    }
    Ok(())
}

fn down_output_row(
    st: &SafeTensors<'_>,
    ids: &[u32],
    target_layer: usize,
    target_pos: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let seq = ids.len();
    let r = Rotary::new(seq);
    let layers = load_layers(st)?;
    let mut x = embed_from_safetensors(st, ids)?;
    for (layer, w) in layers.iter().enumerate() {
        let n1 = rms_norm(&x, &w.ln1, seq, HIDDEN);
        let mut q = matmul(&n1, &w.wq, seq, HIDDEN, Q_DIM);
        add_rows(&mut q, &w.bq);
        let q = rms_norm_heads(&q, &w.q_norm, seq, HEADS);
        let mut k = matmul(&n1, &w.wk, seq, HIDDEN, KV_DIM);
        add_rows(&mut k, &w.bk);
        let k = rms_norm_heads(&k, &w.k_norm, seq, KV_HEADS);
        let mut v = matmul(&n1, &w.wv, seq, HIDDEN, KV_DIM);
        add_rows(&mut v, &w.bv);
        let q = rope(&q, &r.rq[..q.len()]);
        let k = rope(&k, &r.rk[..k.len()]);
        let s = score_qk(&q, &k, seq);
        let p = softmax(&s, HEADS * seq, seq);
        let c = attn_v(&p, &v, seq);
        let a = matmul(&c, &w.wo, seq, Q_DIM, HIDDEN);
        let h = add(&x, &a);
        let n2 = rms_norm(&h, &w.ln2, seq, HIDDEN);
        let g = matmul(&n2, &w.wg, seq, HIDDEN, INTERMEDIATE);
        let u = matmul(&n2, &w.wu, seq, HIDDEN, INTERMEDIATE);
        let m = silu_mul(&g, &u);
        let d = matmul(&m, &w.wd, seq, INTERMEDIATE, HIDDEN);
        if layer == target_layer {
            return Ok(d[target_pos * HIDDEN..(target_pos + 1) * HIDDEN].to_vec());
        }
        x = add(&h, &d);
    }
    unreachable!()
}

fn generate_greedy(
    st: &SafeTensors<'_>,
    args: &Args,
    max_new_tokens: usize,
) -> Result<GenerationResult, Box<dyn Error>> {
    if args.use_kv_cache {
        return generate_greedy_cached(st, args, max_new_tokens);
    }
    generate_greedy_uncached(st, args, max_new_tokens)
}

fn generate_greedy_uncached(
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
    let mut ended_with_eos = false;
    let mut rng = SmallRng::new(args.sampling.seed);
    let mut streamed_text = decode_tokens(&args.tokenizer, &real_ids)?;
    for _ in 0..max_new_tokens {
        let last_pos = real_ids.len() - 1;
        let ids = pad_to_seq(real_ids.clone(), args.seq_len);
        let mut report = if args.report {
            Report::default()
        } else {
            Report::disabled()
        };
        let hidden = forward(st, &ids, &args.cfg, &mut report)?;
        let x = &hidden[last_pos * HIDDEN..(last_pos + 1) * HIDDEN];
        let mut scores = lm_head_scores(st, x, &args.cfg, &mut report)?;
        apply_repetition_penalty(&mut scores, &real_ids, args.sampling.repetition_penalty);
        let next = choose_next_token(&scores, args.sampling, &mut rng) as u32;
        real_ids.push(next);
        maybe_stream_generation(args, &real_ids, &mut streamed_text)?;
        if is_eos(next) {
            ended_with_eos = true;
            break;
        }
        if real_ids.len() >= args.seq_len {
            break;
        }
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

fn generate_greedy_cached(
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
    let mut report = if args.report {
        Report::default()
    } else {
        Report::disabled()
    };
    let use_direct_fixed_weights = args.cfg.fixed_runtime()
        && !args.report
        && args.cfg.enabled.is_model_default()
        && args.cfg.a_bits.0.is_some()
        && args.cfg.w_bits.0.is_some()
        && args.cfg.y_bits.0.is_some();
    if use_direct_fixed_weights {
        let load_layers_started = Instant::now();
        let fixed_layers = load_fixed_layers(st, args.cfg.w_bits.0.unwrap())?;
        let load_layers_elapsed = load_layers_started.elapsed();
        let quantize_weights_elapsed = Duration::ZERO;
        let load_lm_head_started = Instant::now();
        let lm_head = load_lm_head(st)?;
        let load_lm_head_elapsed = load_lm_head_started.elapsed();
        let rotary_started = Instant::now();
        let rotary = Rotary::new(args.seq_len);
        let rotary_elapsed = rotary_started.elapsed();
        let prefill_started = Instant::now();
        let (last_hidden, mut caches) =
            prefill_fixed_cached(st, &fixed_layers, &rotary, &real_ids, &args.cfg)?;
        let prefill_elapsed = prefill_started.elapsed();
        let norm_started = Instant::now();
        let norm = vec_fixed_i32(st, "model.norm.weight", &[HIDDEN], args.cfg.fixed_frac)?;
        let load_norm_elapsed = norm_started.elapsed();
        let first_lm_head_started = Instant::now();
        let mut logits = {
            let h_int = rms_norm_fixed_i32(&last_hidden, &norm, 1, HIDDEN, args.cfg.fixed_frac);
            let h = fixed_i32_to_f32(&h_int, args.cfg.fixed_frac);
            lm_head_scores_loaded(&lm_head, &h, &args.cfg, &mut report)?
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
            maybe_stream_generation(args, &real_ids, &mut streamed_text)?;
            if is_eos(next) {
                ended_with_eos = true;
                break;
            }
            if real_ids.len() >= args.seq_len {
                break;
            }
            let pos = real_ids.len() - 1;
            let decode_model_started = Instant::now();
            let hidden = decode_one_fixed_cached(
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
            let h_int = rms_norm_fixed_i32(&hidden, &norm, 1, HIDDEN, args.cfg.fixed_frac);
            let h = fixed_i32_to_f32(&h_int, args.cfg.fixed_frac);
            logits = lm_head_scores_loaded(&lm_head, &h, &args.cfg, &mut report)?;
            decode_lm_head_elapsed += decode_lm_head_started.elapsed();
        }

        if args.timing {
            println!(
                "timing.load_layers_bf16_to_i32_transpose: {:.3?}",
                load_layers_elapsed
            );
            println!(
                "timing.quantize_layer_weights_i32: {:.3?}",
                quantize_weights_elapsed
            );
            println!("timing.load_lm_head_f32: {:.3?}", load_lm_head_elapsed);
            println!("timing.rotary_build: {:.3?}", rotary_elapsed);
            println!("timing.prefill_cached: {:.3?}", prefill_elapsed);
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
        return Ok(GenerationResult {
            prompt_tokens,
            generated_tokens: real_ids.len() - prompt_tokens,
            total_tokens: real_ids.len(),
            ended_with_eos,
            text,
        });
    }
    let load_layers_started = Instant::now();
    let layers = load_layers(st)?;
    let load_layers_elapsed = load_layers_started.elapsed();
    let quantize_weights_started = Instant::now();
    let quantized_layers = if matches!(args.cfg.quant_mode, QuantMode::Fixed)
        && args.cfg.a_bits.0.is_some()
        && args.cfg.w_bits.0.is_some()
    {
        Some(quantize_layers_fixed(&layers, args.cfg.w_bits.0.unwrap()))
    } else {
        None
    };
    let quantize_weights_elapsed = quantize_weights_started.elapsed();
    let load_lm_head_started = Instant::now();
    let lm_head = load_lm_head(st)?;
    let load_lm_head_elapsed = load_lm_head_started.elapsed();
    let rotary_started = Instant::now();
    let rotary = Rotary::new(args.seq_len);
    let rotary_elapsed = rotary_started.elapsed();
    let prefill_started = Instant::now();
    let (last_hidden, mut caches) = prefill_cached(
        st,
        &layers,
        quantized_layers.as_deref(),
        &rotary,
        &real_ids,
        &args.cfg,
        &mut report,
    )?;
    let prefill_elapsed = prefill_started.elapsed();
    let norm_started = Instant::now();
    let norm = vec_f32(st, "model.norm.weight", &[HIDDEN])?;
    let load_norm_elapsed = norm_started.elapsed();
    let first_lm_head_started = Instant::now();
    let mut logits = {
        let h = rms_norm_cfg(&last_hidden, &norm, 1, HIDDEN, &args.cfg);
        lm_head_scores_loaded(&lm_head, &h, &args.cfg, &mut report)?
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
        maybe_stream_generation(args, &real_ids, &mut streamed_text)?;
        if is_eos(next) {
            ended_with_eos = true;
            break;
        }
        if real_ids.len() >= args.seq_len {
            break;
        }
        let pos = real_ids.len() - 1;
        let decode_model_started = Instant::now();
        let hidden = decode_one_cached(
            st,
            &layers,
            quantized_layers.as_deref(),
            &rotary,
            next,
            pos,
            &mut caches,
            &args.cfg,
            &mut report,
        )?;
        decode_model_elapsed += decode_model_started.elapsed();
        let decode_lm_head_started = Instant::now();
        let h = rms_norm_cfg(&hidden, &norm, 1, HIDDEN, &args.cfg);
        logits = lm_head_scores_loaded(&lm_head, &h, &args.cfg, &mut report)?;
        decode_lm_head_elapsed += decode_lm_head_started.elapsed();
    }

    if args.report {
        println!();
        report.print();
        println!();
    }
    if args.timing {
        println!(
            "timing.load_layers_f32_transpose: {:.3?}",
            load_layers_elapsed
        );
        println!(
            "timing.quantize_layer_weights_i32: {:.3?}",
            quantize_weights_elapsed
        );
        println!("timing.load_lm_head_f32: {:.3?}", load_lm_head_elapsed);
        println!("timing.rotary_build: {:.3?}", rotary_elapsed);
        println!("timing.prefill_cached: {:.3?}", prefill_elapsed);
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

fn maybe_stream_generation(
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

struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct FixedLayerCache {
    k: Vec<i32>,
    v: Vec<i32>,
}

fn prefill_cached(
    st: &SafeTensors<'_>,
    layers: &[LayerWeights],
    quantized_layers: Option<&[QuantizedLayerWeights]>,
    r: &Rotary,
    ids: &[u32],
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Result<(Vec<f32>, Vec<LayerCache>), Box<dyn Error>> {
    assert_eq!(layers.len(), LAYERS);
    let seq = ids.len();
    let mut x = maybe_fixed_q8(embed_from_safetensors(st, ids)?, cfg);
    let mut caches = Vec::with_capacity(LAYERS);
    for (layer, w) in layers.iter().enumerate() {
        let (next, cache) = layer_prefill_cached(
            &x,
            w,
            quantized_layers.map(|ql| &ql[layer]),
            &r,
            seq,
            cfg,
            report,
        );
        x = next;
        caches.push(cache);
    }
    Ok((x[(seq - 1) * HIDDEN..seq * HIDDEN].to_vec(), caches))
}

fn prefill_fixed_cached(
    st: &SafeTensors<'_>,
    layers: &[FixedLayerWeights],
    r: &Rotary,
    ids: &[u32],
    cfg: &ForwardConfig,
) -> Result<(Vec<i32>, Vec<FixedLayerCache>), Box<dyn Error>> {
    assert_eq!(layers.len(), LAYERS);
    let seq = ids.len();
    let mut x = embed_fixed_from_safetensors(st, ids, cfg.fixed_frac)?;
    let mut caches = Vec::with_capacity(LAYERS);
    for w in layers {
        let (next, cache) = layer_prefill_fixed_cached(&x, w, r, seq, cfg);
        x = next;
        caches.push(cache);
    }
    Ok((x[(seq - 1) * HIDDEN..seq * HIDDEN].to_vec(), caches))
}

fn decode_one_cached(
    st: &SafeTensors<'_>,
    layers: &[LayerWeights],
    quantized_layers: Option<&[QuantizedLayerWeights]>,
    r: &Rotary,
    id: u32,
    pos: usize,
    caches: &mut [LayerCache],
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Result<Vec<f32>, Box<dyn Error>> {
    assert_eq!(layers.len(), caches.len());
    let mut x = maybe_fixed_q8(embed_one_from_safetensors(st, id)?, cfg);
    for (layer, (w, cache)) in layers.iter().zip(caches.iter_mut()).enumerate() {
        x = layer_decode_cached(
            &x,
            w,
            quantized_layers.map(|ql| &ql[layer]),
            &r,
            pos,
            cache,
            cfg,
            report,
        );
    }
    Ok(x)
}

fn decode_one_fixed_cached(
    st: &SafeTensors<'_>,
    layers: &[FixedLayerWeights],
    r: &Rotary,
    id: u32,
    pos: usize,
    caches: &mut [FixedLayerCache],
    cfg: &ForwardConfig,
) -> Result<Vec<i32>, Box<dyn Error>> {
    assert_eq!(layers.len(), caches.len());
    let mut x = embed_one_fixed_from_safetensors(st, id, cfg.fixed_frac)?;
    for (w, cache) in layers.iter().zip(caches.iter_mut()) {
        x = layer_decode_fixed_cached(&x, w, r, pos, cache, cfg);
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

fn layer_forward(
    x: &[f32],
    w: &LayerWeights,
    r: &Rotary,
    seq: usize,
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Vec<f32> {
    let n1 = rms_norm_cfg(x, &w.ln1, seq, HIDDEN, cfg);
    let mut q = matmul_awq(&n1, &w.wq, seq, HIDDEN, Q_DIM, MatmulSite::Q, cfg, report);
    add_rows(&mut q, &w.bq);
    fixed_q8_in_place_if(&mut q, cfg);
    let q = rms_norm_heads_cfg(&q, &w.q_norm, seq, HEADS, cfg);
    let mut k = matmul_awq(&n1, &w.wk, seq, HIDDEN, KV_DIM, MatmulSite::K, cfg, report);
    add_rows(&mut k, &w.bk);
    fixed_q8_in_place_if(&mut k, cfg);
    let k = rms_norm_heads_cfg(&k, &w.k_norm, seq, KV_HEADS, cfg);
    let mut v = matmul_awq(&n1, &w.wv, seq, HIDDEN, KV_DIM, MatmulSite::V, cfg, report);
    add_rows(&mut v, &w.bv);
    fixed_q8_in_place_if(&mut v, cfg);
    let q = rope_cfg(&q, &r.rq[..q.len()], cfg);
    let k = rope_cfg(&k, &r.rk[..k.len()], cfg);
    let s = score_qk_cfg(&q, &k, seq, cfg);
    let p = softmax_cfg(&s, HEADS * seq, seq, cfg);
    let c = attn_v_cfg(&p, &v, seq, cfg);
    let a = matmul_awq(&c, &w.wo, seq, Q_DIM, HIDDEN, MatmulSite::O, cfg, report);
    let h = add_cfg(x, &a, cfg);
    let n2 = rms_norm_cfg(&h, &w.ln2, seq, HIDDEN, cfg);
    let g = matmul_awq(
        &n2,
        &w.wg,
        seq,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Gate,
        cfg,
        report,
    );
    let u = matmul_awq(
        &n2,
        &w.wu,
        seq,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Up,
        cfg,
        report,
    );
    let m = silu_mul_cfg(&g, &u, cfg);
    let d = matmul_awq(
        &m,
        &w.wd,
        seq,
        INTERMEDIATE,
        HIDDEN,
        MatmulSite::Down,
        cfg,
        report,
    );
    add_cfg(&h, &d, cfg)
}

fn layer_forward_with_stats(
    x: &[f32],
    w: &LayerWeights,
    r: &Rotary,
    seq: usize,
    cfg: &ForwardConfig,
    report: &mut Report,
    stats: &mut ActivationStats,
    layer: usize,
) -> Vec<f32> {
    let n1 = rms_norm(x, &w.ln1, seq, HIDDEN);
    stats.update_hidden(layer, StatSite::AttnIn, &n1, seq);
    let mut q = matmul_awq(&n1, &w.wq, seq, HIDDEN, Q_DIM, MatmulSite::Q, cfg, report);
    add_rows(&mut q, &w.bq);
    let q = rms_norm_heads(&q, &w.q_norm, seq, HEADS);
    let mut k = matmul_awq(&n1, &w.wk, seq, HIDDEN, KV_DIM, MatmulSite::K, cfg, report);
    add_rows(&mut k, &w.bk);
    let k = rms_norm_heads(&k, &w.k_norm, seq, KV_HEADS);
    let mut v = matmul_awq(&n1, &w.wv, seq, HIDDEN, KV_DIM, MatmulSite::V, cfg, report);
    add_rows(&mut v, &w.bv);
    let q = rope(&q, &r.rq[..q.len()]);
    let k = rope(&k, &r.rk[..k.len()]);
    let s = score_qk_cfg(&q, &k, seq, cfg);
    let p = softmax_cfg(&s, HEADS * seq, seq, cfg);
    let c = attn_v(&p, &v, seq);
    let a = matmul_awq(&c, &w.wo, seq, Q_DIM, HIDDEN, MatmulSite::O, cfg, report);
    let h = add(x, &a);
    let n2 = rms_norm(&h, &w.ln2, seq, HIDDEN);
    stats.update_hidden(layer, StatSite::MlpIn, &n2, seq);
    let g = matmul_awq(
        &n2,
        &w.wg,
        seq,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Gate,
        cfg,
        report,
    );
    let u = matmul_awq(
        &n2,
        &w.wu,
        seq,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Up,
        cfg,
        report,
    );
    let m = silu_mul_cfg(&g, &u, cfg);
    stats.update_intermediate(layer, StatSite::DownIn, &m, seq);
    let d = matmul_awq(
        &m,
        &w.wd,
        seq,
        INTERMEDIATE,
        HIDDEN,
        MatmulSite::Down,
        cfg,
        report,
    );
    add(&h, &d)
}

fn layer_prefill_cached(
    x: &[f32],
    w: &LayerWeights,
    qw: Option<&QuantizedLayerWeights>,
    r: &Rotary,
    seq: usize,
    cfg: &ForwardConfig,
    report: &mut Report,
) -> (Vec<f32>, LayerCache) {
    let n1 = rms_norm_cfg(x, &w.ln1, seq, HIDDEN, cfg);
    let mut q = matmul_cached_awq(
        &n1,
        &w.wq,
        qw.map(|q| q.wq.as_slice()),
        seq,
        HIDDEN,
        Q_DIM,
        MatmulSite::Q,
        cfg,
        report,
    );
    add_rows(&mut q, &w.bq);
    fixed_q8_in_place_if(&mut q, cfg);
    let q = rms_norm_heads_cfg(&q, &w.q_norm, seq, HEADS, cfg);
    let mut k = matmul_cached_awq(
        &n1,
        &w.wk,
        qw.map(|q| q.wk.as_slice()),
        seq,
        HIDDEN,
        KV_DIM,
        MatmulSite::K,
        cfg,
        report,
    );
    add_rows(&mut k, &w.bk);
    fixed_q8_in_place_if(&mut k, cfg);
    let k = rms_norm_heads_cfg(&k, &w.k_norm, seq, KV_HEADS, cfg);
    let mut v = matmul_cached_awq(
        &n1,
        &w.wv,
        qw.map(|q| q.wv.as_slice()),
        seq,
        HIDDEN,
        KV_DIM,
        MatmulSite::V,
        cfg,
        report,
    );
    add_rows(&mut v, &w.bv);
    fixed_q8_in_place_if(&mut v, cfg);
    let q = rope_cfg(&q, &r.rq[..q.len()], cfg);
    let k = rope_cfg(&k, &r.rk[..k.len()], cfg);
    let s = score_qk_cfg(&q, &k, seq, cfg);
    let p = softmax_cfg(&s, HEADS * seq, seq, cfg);
    let c = attn_v_cfg(&p, &v, seq, cfg);
    let a = matmul_cached_awq(
        &c,
        &w.wo,
        qw.map(|q| q.wo.as_slice()),
        seq,
        Q_DIM,
        HIDDEN,
        MatmulSite::O,
        cfg,
        report,
    );
    let h = add_cfg(x, &a, cfg);
    let n2 = rms_norm_cfg(&h, &w.ln2, seq, HIDDEN, cfg);
    let g = matmul_cached_awq(
        &n2,
        &w.wg,
        qw.map(|q| q.wg.as_slice()),
        seq,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Gate,
        cfg,
        report,
    );
    let u = matmul_cached_awq(
        &n2,
        &w.wu,
        qw.map(|q| q.wu.as_slice()),
        seq,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Up,
        cfg,
        report,
    );
    let m = silu_mul_cfg(&g, &u, cfg);
    let d = matmul_cached_awq(
        &m,
        &w.wd,
        qw.map(|q| q.wd.as_slice()),
        seq,
        INTERMEDIATE,
        HIDDEN,
        MatmulSite::Down,
        cfg,
        report,
    );
    (add_cfg(&h, &d, cfg), LayerCache { k, v })
}

fn layer_prefill_fixed_cached(
    x: &[i32],
    w: &FixedLayerWeights,
    r: &Rotary,
    seq: usize,
    cfg: &ForwardConfig,
) -> (Vec<i32>, FixedLayerCache) {
    let frac = cfg.fixed_frac;
    let n1 = rms_norm_fixed_i32(x, &w.ln1, seq, HIDDEN, frac);
    let q = matmul_fixed_i32(
        &n1,
        &w.q.wq,
        seq,
        HIDDEN,
        Q_DIM,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let q = rms_norm_fixed_i32(&q, &w.q_norm, seq * HEADS, HEAD_DIM, frac);
    let k = matmul_fixed_i32(
        &n1,
        &w.q.wk,
        seq,
        HIDDEN,
        KV_DIM,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let k = rms_norm_fixed_i32(&k, &w.k_norm, seq * KV_HEADS, HEAD_DIM, frac);
    let v = matmul_fixed_i32(
        &n1,
        &w.q.wv,
        seq,
        HIDDEN,
        KV_DIM,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let q = rope_fixed_i32(&q, &r.rq[..q.len()], frac);
    let k = rope_fixed_i32(&k, &r.rk[..k.len()], frac);
    let s = score_qk_fixed_i32(&q, &k, seq, frac);
    let p = softmax_fixed_i32(&s, HEADS * seq, seq, frac);
    let c = attn_v_fixed_i32(&p, &v, seq, frac);
    let a = matmul_fixed_i32(
        &c,
        &w.q.wo,
        seq,
        Q_DIM,
        HIDDEN,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let h = add_fixed_i32(x, &a);
    let n2 = rms_norm_fixed_i32(&h, &w.ln2, seq, HIDDEN, frac);
    let g = matmul_fixed_i32(
        &n2,
        &w.q.wg,
        seq,
        HIDDEN,
        INTERMEDIATE,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let u = matmul_fixed_i32(
        &n2,
        &w.q.wu,
        seq,
        HIDDEN,
        INTERMEDIATE,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let m = silu_mul_fixed_i32(&g, &u, frac, cfg.sigmoid_input_rounding);
    let d = matmul_fixed_i32(
        &m,
        &w.q.wd,
        seq,
        INTERMEDIATE,
        HIDDEN,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    (add_fixed_i32(&h, &d), FixedLayerCache { k, v })
}

fn layer_decode_cached(
    x: &[f32],
    w: &LayerWeights,
    qw: Option<&QuantizedLayerWeights>,
    r: &Rotary,
    pos: usize,
    cache: &mut LayerCache,
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Vec<f32> {
    let n1 = rms_norm_cfg(x, &w.ln1, 1, HIDDEN, cfg);
    let mut q = matmul_cached_awq(
        &n1,
        &w.wq,
        qw.map(|q| q.wq.as_slice()),
        1,
        HIDDEN,
        Q_DIM,
        MatmulSite::Q,
        cfg,
        report,
    );
    add_rows(&mut q, &w.bq);
    fixed_q8_in_place_if(&mut q, cfg);
    let q = rms_norm_heads_cfg(&q, &w.q_norm, 1, HEADS, cfg);
    let mut k = matmul_cached_awq(
        &n1,
        &w.wk,
        qw.map(|q| q.wk.as_slice()),
        1,
        HIDDEN,
        KV_DIM,
        MatmulSite::K,
        cfg,
        report,
    );
    add_rows(&mut k, &w.bk);
    fixed_q8_in_place_if(&mut k, cfg);
    let k = rms_norm_heads_cfg(&k, &w.k_norm, 1, KV_HEADS, cfg);
    let mut v = matmul_cached_awq(
        &n1,
        &w.wv,
        qw.map(|q| q.wv.as_slice()),
        1,
        HIDDEN,
        KV_DIM,
        MatmulSite::V,
        cfg,
        report,
    );
    add_rows(&mut v, &w.bv);
    fixed_q8_in_place_if(&mut v, cfg);
    let q = rope_one_cfg(&q, &r.rq, pos, HEADS, cfg);
    let k = rope_one_cfg(&k, &r.rk, pos, KV_HEADS, cfg);
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    let context_len = cache.v.len() / KV_DIM;
    let c = attn_v_decode_cfg(&q, &cache.k, &cache.v, context_len, cfg);
    let a = matmul_cached_awq(
        &c,
        &w.wo,
        qw.map(|q| q.wo.as_slice()),
        1,
        Q_DIM,
        HIDDEN,
        MatmulSite::O,
        cfg,
        report,
    );
    let h = add_cfg(x, &a, cfg);
    let n2 = rms_norm_cfg(&h, &w.ln2, 1, HIDDEN, cfg);
    let g = matmul_cached_awq(
        &n2,
        &w.wg,
        qw.map(|q| q.wg.as_slice()),
        1,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Gate,
        cfg,
        report,
    );
    let u = matmul_cached_awq(
        &n2,
        &w.wu,
        qw.map(|q| q.wu.as_slice()),
        1,
        HIDDEN,
        INTERMEDIATE,
        MatmulSite::Up,
        cfg,
        report,
    );
    let m = silu_mul_cfg(&g, &u, cfg);
    let d = matmul_cached_awq(
        &m,
        &w.wd,
        qw.map(|q| q.wd.as_slice()),
        1,
        INTERMEDIATE,
        HIDDEN,
        MatmulSite::Down,
        cfg,
        report,
    );
    add_cfg(&h, &d, cfg)
}

fn layer_decode_fixed_cached(
    x: &[i32],
    w: &FixedLayerWeights,
    r: &Rotary,
    pos: usize,
    cache: &mut FixedLayerCache,
    cfg: &ForwardConfig,
) -> Vec<i32> {
    let frac = cfg.fixed_frac;
    let n1 = rms_norm_fixed_i32(x, &w.ln1, 1, HIDDEN, frac);
    let q = matmul_fixed_i32(
        &n1,
        &w.q.wq,
        1,
        HIDDEN,
        Q_DIM,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let q = rms_norm_fixed_i32(&q, &w.q_norm, HEADS, HEAD_DIM, frac);
    let k = matmul_fixed_i32(
        &n1,
        &w.q.wk,
        1,
        HIDDEN,
        KV_DIM,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let k = rms_norm_fixed_i32(&k, &w.k_norm, KV_HEADS, HEAD_DIM, frac);
    let v = matmul_fixed_i32(
        &n1,
        &w.q.wv,
        1,
        HIDDEN,
        KV_DIM,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let q = rope_one_fixed_i32(&q, &r.rq, pos, HEADS, frac);
    let k = rope_one_fixed_i32(&k, &r.rk, pos, KV_HEADS, frac);
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    let context_len = cache.v.len() / KV_DIM;
    let c = attn_v_decode_fixed_i32(&q, &cache.k, &cache.v, context_len, frac);
    let a = matmul_fixed_i32(
        &c,
        &w.q.wo,
        1,
        Q_DIM,
        HIDDEN,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let h = add_fixed_i32(x, &a);
    let n2 = rms_norm_fixed_i32(&h, &w.ln2, 1, HIDDEN, frac);
    let g = matmul_fixed_i32(
        &n2,
        &w.q.wg,
        1,
        HIDDEN,
        INTERMEDIATE,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let u = matmul_fixed_i32(
        &n2,
        &w.q.wu,
        1,
        HIDDEN,
        INTERMEDIATE,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    let m = silu_mul_fixed_i32(&g, &u, frac, cfg.sigmoid_input_rounding);
    let d = matmul_fixed_i32(
        &m,
        &w.q.wd,
        1,
        INTERMEDIATE,
        HIDDEN,
        frac,
        frac,
        frac,
        cfg.matmul_rebase_rounding,
    );
    add_fixed_i32(&h, &d)
}

fn matmul_awq(
    a: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    site: MatmulSite,
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Vec<f32> {
    if !cfg.enabled.enabled(site) || !cfg.has_quant() {
        return matmul(a, w, m, k, n);
    }
    if matches!(cfg.quant_mode, QuantMode::Fixed) {
        let y = matmul_awq_fixed_int(a, w, m, k, n, site, cfg);
        if report.enabled {
            let reference = matmul(a, w, m, k, n);
            report.push(site.name(), &reference, &y);
            report.push_int_width(site.name(), a, w, &y);
        }
        return y;
    }
    let aq = quantize_rows_optional(a, m, k, cfg.a_bits, cfg.quant_mode);
    let wq = quantize_cols_optional(w, k, n, cfg.w_bits, cfg.quant_mode);
    let y = matmul(&aq, &wq, m, k, n);
    let yq = quantize_rows_optional(&y, m, n, cfg.y_bits, cfg.quant_mode);
    if report.enabled {
        let reference = matmul(a, w, m, k, n);
        report.push(site.name(), &reference, &yq);
    }
    yq
}

fn matmul_awq_fixed_int(
    a: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    site: MatmulSite,
    cfg: &ForwardConfig,
) -> Vec<f32> {
    match (fixed_a_frac_for_site(site, cfg), cfg.w_bits.0) {
        (Some(a_frac), Some(w_frac)) => matmul_fixed_int(
            a,
            w,
            m,
            k,
            n,
            a_frac,
            w_frac,
            cfg.y_bits.0,
            cfg.matmul_rebase_rounding,
        ),
        _ => {
            let aq = quantize_rows_optional(a, m, k, cfg.a_bits, cfg.quant_mode);
            let wq = quantize_cols_optional(w, k, n, cfg.w_bits, cfg.quant_mode);
            let y = matmul(&aq, &wq, m, k, n);
            quantize_rows_optional(&y, m, n, cfg.y_bits, cfg.quant_mode)
        }
    }
}

fn fixed_a_frac_for_site(site: MatmulSite, cfg: &ForwardConfig) -> Option<u8> {
    if cfg.fixed_runtime()
        && matches!(
            site,
            MatmulSite::Q | MatmulSite::K | MatmulSite::V | MatmulSite::Gate | MatmulSite::Up
        )
    {
        Some(cfg.fixed_frac)
    } else {
        cfg.a_bits.0
    }
}

fn matmul_fixed_int(
    a: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    a_frac: u8,
    w_frac: u8,
    y_frac: Option<u8>,
    rounding: RoundingMode,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), k * n);
    let a_scale = (1i64 << a_frac) as f64;
    let w_scale = (1i64 << w_frac) as f64;
    let acc_frac = a_frac + w_frac;
    let mut y = vec![0.0; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for c in 0..n {
            let mut acc = 0i64;
            for i in 0..k {
                let ai = (a[r * k + i] as f64 * a_scale).round() as i64;
                let wi = (w[i * n + c] as f64 * w_scale).round() as i64;
                acc += ai * wi;
            }
            row[c] = match y_frac {
                Some(y_frac) => {
                    let y_int = rebase_fixed_acc(acc, acc_frac, y_frac, rounding);
                    (y_int as f64 / (1u64 << y_frac) as f64) as f32
                }
                None => (acc as f64 / (1u64 << acc_frac) as f64) as f32,
            };
        }
    });
    y
}

fn rebase_fixed_acc(acc: i64, from_frac: u8, to_frac: u8, rounding: RoundingMode) -> i64 {
    if from_frac == to_frac {
        return acc;
    }
    if from_frac > to_frac {
        shift_signed_i64(acc, from_frac - to_frac, rounding)
    } else {
        acc << (to_frac - from_frac)
    }
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
    if shift == LUT_FRAC {
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

fn matmul_cached_awq(
    a: &[f32],
    w: &[f32],
    w_int: Option<&[i32]>,
    m: usize,
    k: usize,
    n: usize,
    site: MatmulSite,
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Vec<f32> {
    let Some(w_int) = w_int else {
        return matmul_awq(a, w, m, k, n, site, cfg, report);
    };
    if !cfg.enabled.enabled(site) || !cfg.has_quant() || !matches!(cfg.quant_mode, QuantMode::Fixed)
    {
        return matmul(a, w, m, k, n);
    }
    let Some(a_frac) = fixed_a_frac_for_site(site, cfg) else {
        return matmul_awq(a, w, m, k, n, site, cfg, report);
    };
    let Some(w_frac) = cfg.w_bits.0 else {
        return matmul_awq(a, w, m, k, n, site, cfg, report);
    };
    assert_eq!(w_int.len(), k * n);
    let a_int = quantize_fixed_i32(a, a_frac);
    let y = matmul_fixed_prequant(
        &a_int,
        w_int,
        m,
        k,
        n,
        a_frac + w_frac,
        cfg.y_bits.0,
        cfg.matmul_rebase_rounding,
    );
    if report.enabled {
        let reference = matmul(a, w, m, k, n);
        report.push(site.name(), &reference, &y);
        report.push_int_width(site.name(), a, w, &y);
    }
    y
}

fn quantize_fixed_i32(xs: &[f32], frac_bits: u8) -> Vec<i32> {
    let scale = (1u64 << frac_bits) as f64;
    xs.par_iter()
        .map(|&x| (x as f64 * scale).round() as i32)
        .collect()
}

fn matmul_fixed_prequant(
    a: &[i32],
    w: &[i32],
    m: usize,
    k: usize,
    n: usize,
    acc_frac: u8,
    y_frac: Option<u8>,
    rounding: RoundingMode,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), k * n);
    let mut y = vec![0.0; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for c in 0..n {
            let mut acc = 0i64;
            for i in 0..k {
                acc += a[r * k + i] as i64 * w[i * n + c] as i64;
            }
            row[c] = match y_frac {
                Some(y_frac) => {
                    let y_int = rebase_fixed_acc(acc, acc_frac, y_frac, rounding);
                    (y_int as f64 / (1u64 << y_frac) as f64) as f32
                }
                None => (acc as f64 / (1u64 << acc_frac) as f64) as f32,
            };
        }
    });
    y
}

fn matmul_fixed_i32(
    a: &[i32],
    w: &[i32],
    m: usize,
    k: usize,
    n: usize,
    a_frac: u8,
    w_frac: u8,
    y_frac: u8,
    rounding: RoundingMode,
) -> Vec<i32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), k * n);
    let acc_frac = a_frac + w_frac;
    let mut y = vec![0i32; m * n];
    if m == 1 {
        y.par_chunks_mut(256)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_col = chunk_idx * 256;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    matmul_fixed_i32_cols_neon(
                        a, w, chunk, start_col, k, n, acc_frac, y_frac, rounding,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    matmul_fixed_i32_cols_scalar(
                        a, w, chunk, start_col, k, n, acc_frac, y_frac, rounding,
                    );
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
                matmul_fixed_i32_row_neon(
                    &a[r * k..(r + 1) * k],
                    w,
                    row,
                    k,
                    n,
                    acc_frac,
                    y_frac,
                    rounding,
                );
            }
        });
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            matmul_fixed_i32_row_scalar(
                &a[r * k..(r + 1) * k],
                w,
                row,
                k,
                n,
                acc_frac,
                y_frac,
                rounding,
            );
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
    acc_frac: u8,
    y_frac: u8,
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
            if rounding == RoundingMode::Floor && acc_frac >= y_frac {
                let shift = vdupq_n_s64(-((acc_frac - y_frac) as i64));
                acc_lo = vshlq_s64(acc_lo, shift);
                acc_hi = vshlq_s64(acc_hi, shift);
            }
            vst1q_s64(tmp.as_mut_ptr(), acc_lo);
            vst1q_s64(tmp.as_mut_ptr().add(2), acc_hi);
        }
        for j in 0..4 {
            out[offset + j] = if rounding == RoundingMode::Floor && acc_frac >= y_frac {
                tmp[j] as i32
            } else {
                rebase_fixed_acc(tmp[j], acc_frac, y_frac, rounding) as i32
            };
        }
        offset += 4;
    }

    if offset < out.len() {
        matmul_fixed_i32_cols_scalar(
            a,
            w,
            &mut out[offset..],
            start_col + offset,
            k,
            n,
            acc_frac,
            y_frac,
            rounding,
        );
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
    acc_frac: u8,
    y_frac: u8,
    rounding: RoundingMode,
) {
    for (offset, slot) in out.iter_mut().enumerate() {
        let col = start_col + offset;
        let mut acc = 0i64;
        for i in 0..k {
            acc += a[i] as i64 * w[i * n + col] as i64;
        }
        *slot = rebase_fixed_acc(acc, acc_frac, y_frac, rounding) as i32;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn matmul_fixed_i32_row_scalar(
    a: &[i32],
    w: &[i32],
    row: &mut [i32],
    k: usize,
    n: usize,
    acc_frac: u8,
    y_frac: u8,
    rounding: RoundingMode,
) {
    for c in 0..n {
        let mut acc = 0i64;
        for i in 0..k {
            acc += a[i] as i64 * w[i * n + c] as i64;
        }
        row[c] = rebase_fixed_acc(acc, acc_frac, y_frac, rounding) as i32;
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
    acc_frac: u8,
    y_frac: u8,
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
            // QX.(a_frac + w_frac) to QX.y_frac. Round/ceil need extra
            // sign/tie handling and stay on the scalar post-processing path.
            // We also store i64 lanes and cast to i32 below to preserve scalar
            // `as i32` semantics; saturating NEON narrowing would change behavior.
            if rounding == RoundingMode::Floor && acc_frac >= y_frac {
                let shift = vdupq_n_s64(-((acc_frac - y_frac) as i64));
                acc_lo = vshlq_s64(acc_lo, shift);
                acc_hi = vshlq_s64(acc_hi, shift);
            }
            vst1q_s64(tmp.as_mut_ptr(), acc_lo);
            vst1q_s64(tmp.as_mut_ptr().add(2), acc_hi);
        }
        for j in 0..4 {
            row[c + j] = if rounding == RoundingMode::Floor && acc_frac >= y_frac {
                tmp[j] as i32
            } else {
                rebase_fixed_acc(tmp[j], acc_frac, y_frac, rounding) as i32
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
            row[col] = rebase_fixed_acc(acc, acc_frac, y_frac, rounding) as i32;
        }
    }
}

fn fixed_i32_to_f32(xs: &[i32], frac: u8) -> Vec<f32> {
    let scale = (1u64 << frac) as f32;
    xs.par_iter().map(|&x| x as f32 / scale).collect()
}

fn add_fixed_i32(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    a.par_iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut y = vec![0.0; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for c in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[r * k + i] * b[i * n + c];
            }
            row[c] = sum;
        }
    });
    y
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.par_iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn add_cfg(a: &[f32], b: &[f32], cfg: &ForwardConfig) -> Vec<f32> {
    maybe_fixed_q8(add(a, b), cfg)
}

fn add_rows(x: &mut [f32], b: &[f32]) {
    assert_eq!(x.len() % b.len(), 0);
    x.par_chunks_mut(b.len()).for_each(|row| {
        for (x, &b) in row.iter_mut().zip(b) {
            *x += b;
        }
    });
}

fn maybe_fixed_q8(xs: Vec<f32>, cfg: &ForwardConfig) -> Vec<f32> {
    if cfg.fixed_runtime() {
        fixed_q8_vec(&xs, cfg.fixed_frac)
    } else {
        xs
    }
}

fn fixed_q8_in_place_if(xs: &mut [f32], cfg: &ForwardConfig) {
    if cfg.fixed_runtime() {
        fixed_q8_in_place(xs, cfg.fixed_frac);
    }
}

fn fixed_q8_vec(xs: &[f32], frac: u8) -> Vec<f32> {
    xs.par_iter()
        .map(|&x| quantize_fixed_scalar(x, frac))
        .collect()
}

fn fixed_q8_in_place(xs: &mut [f32], frac: u8) {
    xs.par_iter_mut()
        .for_each(|x| *x = quantize_fixed_scalar(*x, frac));
}

fn rms_norm(x: &[f32], w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let ss = xs.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ss + 1e-6).sqrt();
        for c in 0..cols {
            row[c] = xs[c] * inv * w[c];
        }
    });
    y
}

fn rms_norm_cfg(x: &[f32], w: &[f32], rows: usize, cols: usize, cfg: &ForwardConfig) -> Vec<f32> {
    if cfg.fixed_runtime() {
        rms_norm_fixed_with_float_rsqrt(x, w, rows, cols, cfg.fixed_frac)
    } else {
        rms_norm(x, w, rows, cols)
    }
}

fn rms_norm_heads(x: &[f32], w: &[f32], rows: usize, heads: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * heads * HEAD_DIM);
    assert_eq!(w.len(), HEAD_DIM);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(HEAD_DIM).enumerate().for_each(|(i, row)| {
        let xs = &x[i * HEAD_DIM..(i + 1) * HEAD_DIM];
        let ss = xs.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32;
        let inv = 1.0 / (ss + 1e-6).sqrt();
        for d in 0..HEAD_DIM {
            row[d] = xs[d] * inv * w[d];
        }
    });
    y
}

fn rms_norm_heads_cfg(
    x: &[f32],
    w: &[f32],
    rows: usize,
    heads: usize,
    cfg: &ForwardConfig,
) -> Vec<f32> {
    if cfg.fixed_runtime() {
        rms_norm_fixed_with_float_rsqrt(x, w, rows * heads, HEAD_DIM, cfg.fixed_frac)
    } else {
        rms_norm_heads(x, w, rows, heads)
    }
}

fn rms_norm_fixed_with_float_rsqrt(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    frac: u8,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mut sq_acc = 0i128;
        let mut x_ints = Vec::with_capacity(cols);
        for &value in xs {
            let xi = quantize_fixed_i64_scalar(value, frac);
            sq_acc += xi as i128 * xi as i128;
            x_ints.push(xi);
        }
        let inv_int = rms_inv_from_square_sum(sq_acc, cols, frac, frac);
        for c in 0..cols {
            let w_int = quantize_fixed_i64_scalar(w[c], frac);
            let norm_int = round_shift_signed_i64(x_ints[c] * inv_int, frac);
            let out_int = round_shift_signed_i64(norm_int * w_int, frac);
            row[c] = out_int as f32 / (1u64 << frac) as f32;
        }
    });
    y
}

fn rms_inv_from_square_sum(
    square_sum: i128,
    hidden_size: usize,
    input_frac: u8,
    output_frac: u8,
) -> i64 {
    debug_assert!(square_sum >= 0);
    debug_assert!(hidden_size > 0);
    let input_scale = (1u64 << input_frac) as f64;
    let output_scale = (1u64 << output_frac) as f64;
    let mean = square_sum as f64 / hidden_size as f64 / (input_scale * input_scale);
    let inv = 1.0 / (mean + 1e-6).sqrt();
    (inv * output_scale).round() as i64
}

fn rms_norm_fixed_i32(x: &[i32], w: &[i32], rows: usize, cols: usize, frac: u8) -> Vec<i32> {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    let mut y = vec![0i32; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mut sq_acc = 0i128;
        for &xi in xs {
            sq_acc += xi as i128 * xi as i128;
        }
        let inv_int = rms_inv_from_square_sum(sq_acc, cols, frac, frac);
        for c in 0..cols {
            let norm_int = round_shift_signed_i64(xs[c] as i64 * inv_int, frac);
            row[c] = round_shift_signed_i64(norm_int * w[c] as i64, frac) as i32;
        }
    });
    y
}

fn rope(x: &[f32], r: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), r.len());
    let mut y = vec![0.0; x.len()];
    let hd = if x.len() % HEAD_DIM == 0 {
        HEAD_DIM
    } else {
        x.len()
    };
    y.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let base = i * hd;
        let half = hd / 2;
        for d in 0..half {
            let a = x[base + d];
            let b = x[base + half + d];
            let c = r[base + d];
            let s = r[base + half + d];
            out[d] = a * c - b * s;
            out[half + d] = b * c + a * s;
        }
    });
    y
}

fn rope_cfg(x: &[f32], r: &[f32], cfg: &ForwardConfig) -> Vec<f32> {
    if !cfg.fixed_runtime() {
        return rope(x, r);
    }
    assert_eq!(x.len(), r.len());
    let mut y = vec![0.0; x.len()];
    let hd = if x.len() % HEAD_DIM == 0 {
        HEAD_DIM
    } else {
        x.len()
    };
    y.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let base = i * hd;
        rope_fixed_chunk(
            &x[base..base + hd],
            &r[base..base + hd],
            out,
            cfg.fixed_frac,
        );
    });
    y
}

fn rope_one(x: &[f32], r: &[f32], pos: usize, heads: usize) -> Vec<f32> {
    assert_eq!(x.len(), heads * HEAD_DIM);
    let mut y = vec![0.0; x.len()];
    for h in 0..heads {
        let base = h * HEAD_DIM;
        let rbase = (pos * heads + h) * HEAD_DIM;
        let half = HEAD_DIM / 2;
        for d in 0..half {
            let a = x[base + d];
            let b = x[base + half + d];
            let c = r[rbase + d];
            let s = r[rbase + half + d];
            y[base + d] = a * c - b * s;
            y[base + half + d] = b * c + a * s;
        }
    }
    y
}

fn rope_one_cfg(x: &[f32], r: &[f32], pos: usize, heads: usize, cfg: &ForwardConfig) -> Vec<f32> {
    if !cfg.fixed_runtime() {
        return rope_one(x, r, pos, heads);
    }
    assert_eq!(x.len(), heads * HEAD_DIM);
    let mut y = vec![0.0; x.len()];
    for h in 0..heads {
        let base = h * HEAD_DIM;
        let rbase = (pos * heads + h) * HEAD_DIM;
        rope_fixed_chunk(
            &x[base..base + HEAD_DIM],
            &r[rbase..rbase + HEAD_DIM],
            &mut y[base..base + HEAD_DIM],
            cfg.fixed_frac,
        );
    }
    y
}

fn rope_fixed_chunk(x: &[f32], r: &[f32], out: &mut [f32], frac: u8) {
    let half = x.len() / 2;
    for d in 0..half {
        let a = quantize_fixed_i64_scalar(x[d], frac);
        let b = quantize_fixed_i64_scalar(x[half + d], frac);
        let c = quantize_fixed_i64_scalar(r[d], frac);
        let s = quantize_fixed_i64_scalar(r[half + d], frac);
        let y0 = round_shift_signed_i64(a * c - b * s, frac);
        let y1 = round_shift_signed_i64(b * c + a * s, frac);
        out[d] = y0 as f32 / (1u64 << frac) as f32;
        out[half + d] = y1 as f32 / (1u64 << frac) as f32;
    }
}

fn rope_fixed_i32(x: &[i32], r: &[f32], frac: u8) -> Vec<i32> {
    assert_eq!(x.len(), r.len());
    let mut y = vec![0i32; x.len()];
    let hd = if x.len() % HEAD_DIM == 0 {
        HEAD_DIM
    } else {
        x.len()
    };
    y.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let base = i * hd;
        rope_fixed_i32_chunk(&x[base..base + hd], &r[base..base + hd], out, frac);
    });
    y
}

fn rope_one_fixed_i32(x: &[i32], r: &[f32], pos: usize, heads: usize, frac: u8) -> Vec<i32> {
    assert_eq!(x.len(), heads * HEAD_DIM);
    let mut y = vec![0i32; x.len()];
    for h in 0..heads {
        let base = h * HEAD_DIM;
        let rbase = (pos * heads + h) * HEAD_DIM;
        rope_fixed_i32_chunk(
            &x[base..base + HEAD_DIM],
            &r[rbase..rbase + HEAD_DIM],
            &mut y[base..base + HEAD_DIM],
            frac,
        );
    }
    y
}

fn rope_fixed_i32_chunk(x: &[i32], r: &[f32], out: &mut [i32], frac: u8) {
    let half = x.len() / 2;
    for d in 0..half {
        let a = x[d] as i64;
        let b = x[half + d] as i64;
        let c = quantize_fixed_i64_scalar(r[d], frac);
        let s = quantize_fixed_i64_scalar(r[half + d], frac);
        out[d] = round_shift_signed_i64(a * c - b * s, frac) as i32;
        out[half + d] = round_shift_signed_i64(b * c + a * s, frac) as i32;
    }
}

fn score_qk(q: &[f32], k: &[f32], seq: usize) -> Vec<f32> {
    assert_eq!(q.len(), seq * HEADS * HEAD_DIM);
    assert_eq!(k.len(), seq * KV_DIM);
    let mut y = vec![0.0; HEADS * seq * seq];
    y.par_chunks_mut(seq * seq).enumerate().for_each(|(h, ys)| {
        let kh = h / KV_GROUP;
        for i in 0..seq {
            for j in 0..seq {
                let o = i * seq + j;
                if j > i {
                    ys[o] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM {
                    let qi = (i * HEADS + h) * HEAD_DIM + d;
                    let ki = (j * KV_HEADS + kh) * HEAD_DIM + d;
                    s += q[qi] * k[ki];
                }
                ys[o] = s / (HEAD_DIM as f32).sqrt();
            }
        }
    });
    y
}

fn score_qk_cfg(q: &[f32], k: &[f32], seq: usize, cfg: &ForwardConfig) -> Vec<f32> {
    if !cfg.fixed_runtime() {
        return score_qk(q, k, seq);
    }
    assert_eq!(q.len(), seq * HEADS * HEAD_DIM);
    assert_eq!(k.len(), seq * KV_DIM);
    let inv_sqrt_int =
        ((1.0 / (HEAD_DIM as f32).sqrt()) * (1u64 << cfg.fixed_frac) as f32).round() as i64;
    let mut y = vec![0.0; HEADS * seq * seq];
    y.par_chunks_mut(seq * seq).enumerate().for_each(|(h, ys)| {
        let kh = h / KV_GROUP;
        for i in 0..seq {
            for j in 0..seq {
                let o = i * seq + j;
                if j > i {
                    ys[o] = f32::NEG_INFINITY;
                    continue;
                }
                let mut acc = 0i64;
                for d in 0..HEAD_DIM {
                    let qi = (i * HEADS + h) * HEAD_DIM + d;
                    let ki = (j * KV_HEADS + kh) * HEAD_DIM + d;
                    let q_int = quantize_fixed_i64_scalar(q[qi], cfg.fixed_frac);
                    let k_int = quantize_fixed_i64_scalar(k[ki], cfg.fixed_frac);
                    acc += q_int * k_int;
                }
                let dot_int = round_shift_signed_i64(acc, cfg.fixed_frac);
                let score_int = round_shift_signed_i64(dot_int * inv_sqrt_int, cfg.fixed_frac);
                ys[o] = score_int as f32 / (1u64 << cfg.fixed_frac) as f32;
            }
        }
    });
    y
}

fn score_qk_fixed_i32(q: &[i32], k: &[i32], seq: usize, frac: u8) -> Vec<i32> {
    assert_eq!(q.len(), seq * HEADS * HEAD_DIM);
    assert_eq!(k.len(), seq * KV_DIM);
    let inv_sqrt_int = ((1.0 / (HEAD_DIM as f32).sqrt()) * (1u64 << frac) as f32).round() as i64;
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
                let dot_int = round_shift_signed_i64(acc, frac);
                ys[o] = round_shift_signed_i64(dot_int * inv_sqrt_int, frac) as i32;
            }
        }
    });
    y
}

fn softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for c in 0..cols {
            row[c] = (xs[c] - mx).exp();
            sum += row[c];
        }
        for value in row {
            *value /= sum;
        }
    });
    y
}

fn softmax_cfg(x: &[f32], rows: usize, cols: usize, cfg: &ForwardConfig) -> Vec<f32> {
    if !cfg.fixed_runtime() {
        return softmax(x, rows, cols);
    }
    assert_eq!(x.len(), rows * cols);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let xs_int: Vec<i64> = xs
            .iter()
            .map(|&value| {
                if value.is_finite() {
                    quantize_fixed_i64_scalar(value, cfg.fixed_frac)
                } else {
                    i64::MIN / 4
                }
            })
            .collect();
        let mx = xs_int.iter().copied().max().unwrap_or(0);
        let mut exps = vec![0i64; cols];
        let mut sum = 0i64;
        for c in 0..cols {
            if xs_int[c] <= i64::MIN / 8 {
                exps[c] = 0;
            } else {
                let diff = xs_int[c] - mx;
                exps[c] = softmax_exp_coarse(diff, cfg.fixed_frac);
            }
            sum += exps[c];
        }
        if sum == 0 {
            return;
        }
        for c in 0..cols {
            let p_int = div_round_i128((exps[c] as i128) << cfg.fixed_frac, sum as i128)
                .clamp(0, 1i128 << cfg.fixed_frac) as i64;
            row[c] = p_int as f32 / (1u64 << cfg.fixed_frac) as f32;
        }
    });
    y
}

fn softmax_fixed_i32(x: &[i32], rows: usize, cols: usize, frac: u8) -> Vec<i32> {
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
                exps[c] = softmax_exp_coarse((xs[c] - mx) as i64, frac);
            }
            sum += exps[c];
        }
        if sum == 0 {
            return;
        }
        for c in 0..cols {
            row[c] = div_round_i128((exps[c] as i128) << frac, sum as i128).clamp(0, 1i128 << frac)
                as i32;
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

fn softmax_exp_coarse(diff_int: i64, frac: u8) -> i64 {
    let clipped = diff_int.clamp(-(8i64 << frac), 0);
    let n = floor_shift_signed_i64(clipped, frac).clamp(-8, 0);
    let f_int = clipped - (n << frac);
    let exp_n = if frac == LUT_FRAC {
        EXP_LUT_Q8[(n + 8) as usize]
    } else {
        ((n as f32).exp() * (1u64 << frac) as f32).round() as i64
    };
    let corr = ((1i64 << frac) + f_int).max(0);
    round_shift_signed_i64(exp_n * corr, frac)
}

fn floor_shift_signed_i64(x: i64, shift: u8) -> i64 {
    if shift == 0 { x } else { x >> shift }
}

fn attn_v(p: &[f32], v: &[f32], seq: usize) -> Vec<f32> {
    assert_eq!(p.len(), HEADS * seq * seq);
    assert_eq!(v.len(), seq * KV_DIM);
    let mut y = vec![0.0; seq * Q_DIM];
    y.par_chunks_mut(HEAD_DIM)
        .enumerate()
        .for_each(|(oh, out)| {
            let pos = oh / HEADS;
            let h = oh % HEADS;
            let kh = h / KV_GROUP;
            for (d, slot) in out.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for j in 0..seq {
                    let pi = (h * seq + pos) * seq + j;
                    let vi = (j * KV_HEADS + kh) * HEAD_DIM + d;
                    s += p[pi] * v[vi];
                }
                *slot = s;
            }
        });
    y
}

fn attn_v_cfg(p: &[f32], v: &[f32], seq: usize, cfg: &ForwardConfig) -> Vec<f32> {
    if !cfg.fixed_runtime() {
        return attn_v(p, v, seq);
    }
    assert_eq!(p.len(), HEADS * seq * seq);
    assert_eq!(v.len(), seq * KV_DIM);
    let mut y = vec![0.0; seq * Q_DIM];
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
                    let p_int = quantize_fixed_i64_scalar(p[pi], cfg.fixed_frac);
                    let v_int = quantize_fixed_i64_scalar(v[vi], cfg.fixed_frac);
                    acc += p_int * v_int;
                }
                let y_int = round_shift_signed_i64(acc, cfg.fixed_frac);
                *slot = y_int as f32 / (1u64 << cfg.fixed_frac) as f32;
            }
        });
    y
}

fn attn_v_fixed_i32(p: &[i32], v: &[i32], seq: usize, frac: u8) -> Vec<i32> {
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
                *slot = round_shift_signed_i64(acc, frac) as i32;
            }
        });
    y
}

fn attn_v_decode(q: &[f32], k: &[f32], v: &[f32], context_len: usize) -> Vec<f32> {
    assert_eq!(q.len(), Q_DIM);
    assert_eq!(k.len(), context_len * KV_DIM);
    assert_eq!(v.len(), context_len * KV_DIM);
    let mut y = vec![0.0; Q_DIM];
    y.par_chunks_mut(HEAD_DIM).enumerate().for_each(|(h, out)| {
        let kh = h / KV_GROUP;
        let mut scores = vec![0.0f32; context_len];
        for j in 0..context_len {
            let mut s = 0.0f32;
            for d in 0..HEAD_DIM {
                let qi = h * HEAD_DIM + d;
                let ki = (j * KV_HEADS + kh) * HEAD_DIM + d;
                s += q[qi] * k[ki];
            }
            scores[j] = s / (HEAD_DIM as f32).sqrt();
        }
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for score in &mut scores {
            *score = (*score - mx).exp();
            denom += *score;
        }
        for (d, slot) in out.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for (j, &p) in scores.iter().enumerate() {
                let vi = (j * KV_HEADS + kh) * HEAD_DIM + d;
                s += (p / denom) * v[vi];
            }
            *slot = s;
        }
    });
    y
}

fn attn_v_decode_cfg(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    context_len: usize,
    cfg: &ForwardConfig,
) -> Vec<f32> {
    if !cfg.fixed_runtime() {
        return attn_v_decode(q, k, v, context_len);
    }
    assert_eq!(q.len(), Q_DIM);
    assert_eq!(k.len(), context_len * KV_DIM);
    assert_eq!(v.len(), context_len * KV_DIM);
    let mut y = vec![0.0; Q_DIM];
    let inv_sqrt_int =
        ((1.0 / (HEAD_DIM as f32).sqrt()) * (1u64 << cfg.fixed_frac) as f32).round() as i64;
    y.par_chunks_mut(HEAD_DIM).enumerate().for_each(|(h, out)| {
        let kh = h / KV_GROUP;
        let mut scores = vec![0.0f32; context_len];
        for j in 0..context_len {
            let mut acc = 0i64;
            for d in 0..HEAD_DIM {
                let qi = h * HEAD_DIM + d;
                let ki = (j * KV_HEADS + kh) * HEAD_DIM + d;
                let q_int = quantize_fixed_i64_scalar(q[qi], cfg.fixed_frac);
                let k_int = quantize_fixed_i64_scalar(k[ki], cfg.fixed_frac);
                acc += q_int * k_int;
            }
            let dot_int = round_shift_signed_i64(acc, cfg.fixed_frac);
            let score_int = round_shift_signed_i64(dot_int * inv_sqrt_int, cfg.fixed_frac);
            scores[j] = score_int as f32 / (1u64 << cfg.fixed_frac) as f32;
        }
        let scores_int: Vec<i64> = scores
            .iter()
            .map(|&score| quantize_fixed_i64_scalar(score, cfg.fixed_frac))
            .collect();
        let mx = scores_int.iter().copied().max().unwrap_or(0);
        let mut exps = vec![0i64; context_len];
        let mut denom = 0i64;
        for j in 0..context_len {
            let diff = scores_int[j] - mx;
            exps[j] = softmax_exp_coarse(diff, cfg.fixed_frac);
            denom += exps[j];
        }
        for (d, slot) in out.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (j, &exp_int) in exps.iter().enumerate() {
                let p_int = div_round_i128((exp_int as i128) << cfg.fixed_frac, denom as i128)
                    .clamp(0, 1i128 << cfg.fixed_frac) as i64;
                let vi = (j * KV_HEADS + kh) * HEAD_DIM + d;
                let v_int = quantize_fixed_i64_scalar(v[vi], cfg.fixed_frac);
                acc += p_int * v_int;
            }
            let y_int = round_shift_signed_i64(acc, cfg.fixed_frac);
            *slot = y_int as f32 / (1u64 << cfg.fixed_frac) as f32;
        }
    });
    y
}

fn attn_v_decode_fixed_i32(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    context_len: usize,
    frac: u8,
) -> Vec<i32> {
    assert_eq!(q.len(), Q_DIM);
    assert_eq!(k.len(), context_len * KV_DIM);
    assert_eq!(v.len(), context_len * KV_DIM);
    let mut y = vec![0i32; Q_DIM];
    let inv_sqrt_int = ((1.0 / (HEAD_DIM as f32).sqrt()) * (1u64 << frac) as f32).round() as i64;
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
            let dot_int = round_shift_signed_i64(acc, frac);
            *score = round_shift_signed_i64(dot_int * inv_sqrt_int, frac) as i32;
        }
        let mx = scores.iter().copied().max().unwrap_or(0);
        let mut exps = vec![0i64; context_len];
        let mut denom = 0i64;
        for j in 0..context_len {
            exps[j] = softmax_exp_coarse((scores[j] - mx) as i64, frac);
            denom += exps[j];
        }
        for (d, slot) in out.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (j, &exp_int) in exps.iter().enumerate() {
                let p_int = div_round_i128((exp_int as i128) << frac, denom as i128)
                    .clamp(0, 1i128 << frac) as i64;
                let vi = (j * KV_HEADS + kh) * HEAD_DIM + d;
                acc += p_int * v[vi] as i64;
            }
            *slot = round_shift_signed_i64(acc, frac) as i32;
        }
    });
    y
}

fn silu_mul(g: &[f32], u: &[f32]) -> Vec<f32> {
    assert_eq!(g.len(), u.len());
    g.par_iter()
        .zip(u)
        .map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b)
        .collect()
}

fn silu_mul_cfg(g: &[f32], u: &[f32], cfg: &ForwardConfig) -> Vec<f32> {
    if cfg.fixed_runtime() {
        return silu_mul_fixed(g, u, cfg.fixed_frac, cfg.sigmoid_input_rounding);
    }
    silu_mul(g, u)
}

fn silu_mul_fixed(g: &[f32], u: &[f32], frac: u8, sigmoid_rounding: RoundingMode) -> Vec<f32> {
    assert_eq!(g.len(), u.len());
    g.par_iter()
        .zip(u)
        .map(|(&g, &u)| {
            let g_int = quantize_fixed_i64_scalar(g, frac);
            let u_int = quantize_fixed_i64_scalar(u, frac);
            let g_index = shift_signed_i64(g_int, frac, sigmoid_rounding);
            let sig_int = sigmoid_from_integer_index(g_index, frac);
            let silu_int = round_shift_signed_i64(g_int * sig_int, frac);
            let out_int = round_shift_signed_i64(silu_int * u_int, frac);
            out_int as f32 / (1u64 << frac) as f32
        })
        .collect()
}

fn silu_mul_fixed_i32(g: &[i32], u: &[i32], frac: u8, sigmoid_rounding: RoundingMode) -> Vec<i32> {
    assert_eq!(g.len(), u.len());
    g.par_iter()
        .zip(u)
        .map(|(&g, &u)| {
            let g_index = shift_signed_i64(g as i64, frac, sigmoid_rounding);
            let sig_int = sigmoid_from_integer_index(g_index, frac);
            let silu_int = round_shift_signed_i64(g as i64 * sig_int, frac);
            round_shift_signed_i64(silu_int * u as i64, frac) as i32
        })
        .collect()
}

fn quantize_fixed_i64_scalar(x: f32, frac: u8) -> i64 {
    let scale = (1u64 << frac) as f32;
    (x * scale).round() as i64
}

fn sigmoid_from_integer_index(x: i64, frac: u8) -> i64 {
    let x = x.clamp(-8, 7);
    if frac == LUT_FRAC {
        SIGMOID_LUT_Q8[(x + 8) as usize]
    } else {
        let x = x as f32;
        (1.0 / (1.0 + (-x).exp()) * (1u64 << frac) as f32).round() as i64
    }
}

fn quantize_fixed_scalar(x: f32, frac: u8) -> f32 {
    let scale = (1u64 << frac) as f32;
    (x * scale).round() / scale
}

fn quantize_rows_optional(
    xs: &[f32],
    rows: usize,
    cols: usize,
    bits: QuantBits,
    mode: QuantMode,
) -> Vec<f32> {
    let Some(bits) = bits.0 else {
        return xs.to_vec();
    };
    let mut out = vec![0.0; xs.len()];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let row_xs = &xs[row * cols..(row + 1) * cols];
            quantize_slice(row_xs, out_row, bits, mode);
        });
    assert_eq!(out.len(), rows * cols);
    out
}

fn quantize_cols_optional(
    xs: &[f32],
    rows: usize,
    cols: usize,
    bits: QuantBits,
    mode: QuantMode,
) -> Vec<f32> {
    let Some(bits) = bits.0 else {
        return xs.to_vec();
    };
    let mut out = vec![0.0; xs.len()];
    for col in 0..cols {
        let mut tmp = vec![0.0; rows];
        let mut q = vec![0.0; rows];
        for row in 0..rows {
            tmp[row] = xs[row * cols + col];
        }
        quantize_slice(&tmp, &mut q, bits, mode);
        for row in 0..rows {
            out[row * cols + col] = q[row];
        }
    }
    out
}

fn quantize_slice(xs: &[f32], out: &mut [f32], bits: u8, mode: QuantMode) {
    debug_assert_eq!(xs.len(), out.len());
    if matches!(mode, QuantMode::Fixed) {
        quantize_slice_fixed(xs, out, bits);
        return;
    }
    let min = xs.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let qmax = ((1u64 << bits) - 1) as f64;
    if min == max {
        out.fill(min as f32);
        return;
    }
    let scale = (max - min) / qmax;
    let zp = (-min / scale).round();
    for (dst, &x) in out.iter_mut().zip(xs) {
        let q = ((x as f64 / scale) + zp).floor().clamp(0.0, qmax);
        *dst = ((q - zp) * scale) as f32;
    }
}

fn quantize_slice_fixed(xs: &[f32], out: &mut [f32], frac_bits: u8) {
    debug_assert_eq!(xs.len(), out.len());
    let scale = (1u64 << frac_bits) as f64;
    for (dst, &x) in out.iter_mut().zip(xs) {
        *dst = ((x as f64 * scale).round() / scale) as f32;
    }
}

fn lm_head_scores(
    st: &SafeTensors<'_>,
    x: &[f32],
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if x.len() != HIDDEN {
        return Err(err(format!(
            "expected hidden length {HIDDEN}, got {}",
            x.len()
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    let enabled = cfg.enabled.enabled(MatmulSite::LmHead) && cfg.has_quant();
    let aq = if enabled {
        quantize_rows_optional(x, 1, HIDDEN, cfg.a_bits, cfg.quant_mode)
    } else {
        x.to_vec()
    };
    let mut row = Vec::with_capacity(HIDDEN);
    let mut scores = Vec::with_capacity(VOCAB);
    let mut reference = if enabled {
        Vec::with_capacity(VOCAB)
    } else {
        Vec::new()
    };
    for id in 0..VOCAB {
        row.clear();
        row_f32(&t, id, HIDDEN, &mut row)?;
        if enabled {
            reference.push(x.iter().zip(&row).map(|(&a, &b)| a * b).sum::<f32>());
            let wq = quantize_rows_optional(&row, 1, HIDDEN, cfg.w_bits, cfg.quant_mode);
            scores.push(aq.iter().zip(&wq).map(|(&a, &b)| a * b).sum::<f32>());
        } else {
            scores.push(x.iter().zip(&row).map(|(&a, &b)| a * b).sum::<f32>());
        }
    }
    if enabled {
        scores = quantize_rows_optional(&scores, 1, VOCAB, cfg.y_bits, cfg.quant_mode);
        report.push(MatmulSite::LmHead.name(), &reference, &scores);
    }
    Ok(scores)
}

fn load_lm_head(st: &SafeTensors<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(&t, &[VOCAB, HIDDEN], "model.embed_tokens.weight")?;
    tensor_f32(&t)
}

fn lm_head_scores_loaded(
    w: &[f32],
    x: &[f32],
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Result<Vec<f32>, Box<dyn Error>> {
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

    let enabled = cfg.enabled.enabled(MatmulSite::LmHead) && cfg.has_quant();
    let aq = if enabled {
        quantize_rows_optional(x, 1, HIDDEN, cfg.a_bits, cfg.quant_mode)
    } else {
        x.to_vec()
    };
    let mut scores: Vec<f32> = w
        .par_chunks_exact(HIDDEN)
        .map(|row| {
            if enabled {
                let wq = quantize_rows_optional(row, 1, HIDDEN, cfg.w_bits, cfg.quant_mode);
                aq.iter().zip(&wq).map(|(&a, &b)| a * b).sum::<f32>()
            } else {
                x.iter().zip(row).map(|(&a, &b)| a * b).sum::<f32>()
            }
        })
        .collect();

    if enabled {
        scores = quantize_rows_optional(&scores, 1, VOCAB, cfg.y_bits, cfg.quant_mode);
        if report.enabled {
            let reference: Vec<f32> = w
                .par_chunks_exact(HIDDEN)
                .map(|row| x.iter().zip(row).map(|(&a, &b)| a * b).sum::<f32>())
                .collect();
            report.push(MatmulSite::LmHead.name(), &reference, &scores);
        }
    }
    Ok(scores)
}

fn softmax_vector(logits: &[f32]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&x| ((x - max_logit) as f64).exp())
        .collect();
    let sum = probs.iter().sum::<f64>();
    if sum > 0.0 && sum.is_finite() {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(ids.len());
    ids.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    ids.truncate(k);
    ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    ids
}

fn ranks_from_logits(logits: &[f32]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.par_sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let mut ranks = vec![0usize; logits.len()];
    for (rank, id) in ids.into_iter().enumerate() {
        ranks[id] = rank + 1;
    }
    ranks
}

fn decode_token(tokenizer: &Tokenizer, id: usize) -> Result<String, Box<dyn Error>> {
    tokenizer
        .decode(&[id as u32], false)
        .map_err(|e| err(e.to_string()))
}

fn context_tail(
    tokenizer: &Tokenizer,
    ids: &[u32],
    chars: usize,
) -> Result<String, Box<dyn Error>> {
    let text = tokenizer
        .decode(ids, true)
        .map_err(|e| err(e.to_string()))?;
    let mut tail = text.chars().rev().take(chars).collect::<Vec<_>>();
    tail.reverse();
    Ok(tail.into_iter().collect())
}

fn token_record(
    id: usize,
    tokenizer: &Tokenizer,
    primary_logits: &[f32],
    primary_probs: &[f64],
    other_logits: &[f32],
    other_probs: &[f64],
    primary_ranks: &[usize],
    other_ranks: &[usize],
) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(json!({
        "id": id,
        "token": decode_token(tokenizer, id)?,
        "prob": primary_probs[id],
        "logit": primary_logits[id],
        "rank": primary_ranks[id],
        "other_prob": other_probs[id],
        "other_logit": other_logits[id],
        "other_rank": other_ranks[id],
    }))
}

fn top_records(
    ids: &[usize],
    tokenizer: &Tokenizer,
    primary_logits: &[f32],
    primary_probs: &[f64],
    other_logits: &[f32],
    other_probs: &[f64],
    primary_ranks: &[usize],
    other_ranks: &[usize],
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    ids.iter()
        .map(|&id| {
            token_record(
                id,
                tokenizer,
                primary_logits,
                primary_probs,
                other_logits,
                other_probs,
                primary_ranks,
                other_ranks,
            )
        })
        .collect()
}

fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut js = 0.0;
    for (&a, &b) in p.iter().zip(q) {
        let m = 0.5 * (a + b);
        if a > 0.0 && m > 0.0 {
            js += 0.5 * a * (a / m).ln();
        }
        if b > 0.0 && m > 0.0 {
            js += 0.5 * b * (b / m).ln();
        }
    }
    js
}

fn entropy(p: &[f64]) -> f64 {
    p.iter()
        .copied()
        .filter(|&x| x > 0.0)
        .map(|x| -x * x.ln())
        .sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / ((aa.sqrt() * bb.sqrt()) + 1e-12)
}

fn nll_from_scores(scores: &[f32], target: usize) -> f64 {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = scores
        .iter()
        .map(|&s| ((s - max_score) as f64).exp())
        .sum::<f64>();
    max_score as f64 + sum_exp.ln() - scores[target] as f64
}

fn perplexity(
    st: &SafeTensors<'_>,
    hidden: &[f32],
    ids: &[u32],
    max_targets: usize,
    cfg: &ForwardConfig,
    report: &mut Report,
) -> Result<PplResult, Box<dyn Error>> {
    let seq = ids.len();
    let expected = seq * HIDDEN;
    if hidden.len() != expected {
        return Err(err(format!(
            "expected hidden length {expected}, got {}",
            hidden.len()
        )));
    }
    let mut nll = 0.0;
    let mut n = 0usize;
    for pos in 0..seq - 1 {
        if n >= max_targets {
            break;
        }
        if is_eos(ids[pos + 1]) {
            break;
        }
        let x = &hidden[pos * HIDDEN..(pos + 1) * HIDDEN];
        let scores = lm_head_scores(st, x, cfg, report)?;
        nll += nll_from_scores(&scores, ids[pos + 1] as usize);
        n += 1;
    }
    if n == 0 {
        return Err(err("no target tokens for perplexity"));
    }
    let avg_nll = nll / n as f64;
    Ok(PplResult {
        targets: n,
        avg_nll,
        ppl: avg_nll.exp(),
    })
}

struct PplResult {
    targets: usize,
    avg_nll: f64,
    ppl: f64,
}

#[derive(Clone, Copy)]
enum StatSite {
    AttnIn,
    MlpIn,
    DownIn,
}

struct ActivationStats {
    samples: usize,
    tokens: usize,
    attn_in: Vec<Vec<f32>>,
    mlp_in: Vec<Vec<f32>>,
    down_in: Vec<Vec<f32>>,
}

impl ActivationStats {
    fn new() -> Self {
        Self {
            samples: 0,
            tokens: 0,
            attn_in: vec![vec![0.0; HIDDEN]; LAYERS],
            mlp_in: vec![vec![0.0; HIDDEN]; LAYERS],
            down_in: vec![vec![0.0; INTERMEDIATE]; LAYERS],
        }
    }

    fn update_hidden(&mut self, layer: usize, site: StatSite, xs: &[f32], rows: usize) {
        assert_eq!(xs.len(), rows * HIDDEN);
        let dst = match site {
            StatSite::AttnIn => &mut self.attn_in[layer],
            StatSite::MlpIn => &mut self.mlp_in[layer],
            StatSite::DownIn => unreachable!("down_in is not hidden-sized"),
        };
        update_channel_max(dst, xs, rows, HIDDEN);
    }

    fn update_intermediate(&mut self, layer: usize, site: StatSite, xs: &[f32], rows: usize) {
        assert_eq!(xs.len(), rows * INTERMEDIATE);
        let dst = match site {
            StatSite::DownIn => &mut self.down_in[layer],
            StatSite::AttnIn | StatSite::MlpIn => unreachable!("site is not intermediate-sized"),
        };
        update_channel_max(dst, xs, rows, INTERMEDIATE);
    }

    fn write_json(&self, path: &PathBuf) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let layers: Vec<_> = (0..LAYERS)
            .map(|layer| {
                json!({
                    "layer": layer,
                    "attn_in": {
                        "channels": HIDDEN,
                        "max_abs": self.attn_in[layer],
                        "global_max_abs": self.attn_in[layer].iter().copied().fold(0.0f32, f32::max),
                    },
                    "mlp_in": {
                        "channels": HIDDEN,
                        "max_abs": self.mlp_in[layer],
                        "global_max_abs": self.mlp_in[layer].iter().copied().fold(0.0f32, f32::max),
                    },
                    "down_in": {
                        "channels": INTERMEDIATE,
                        "max_abs": self.down_in[layer],
                        "global_max_abs": self.down_in[layer].iter().copied().fold(0.0f32, f32::max),
                    },
                })
            })
            .collect();
        let value = json!({
            "samples": self.samples,
            "tokens": self.tokens,
            "layers": layers,
        });
        std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
        Ok(())
    }
}

fn update_channel_max(dst: &mut [f32], xs: &[f32], rows: usize, cols: usize) {
    assert_eq!(dst.len(), cols);
    for row in xs.chunks_exact(cols).take(rows) {
        for (d, &x) in dst.iter_mut().zip(row) {
            *d = (*d).max(x.abs());
        }
    }
}

struct Report {
    enabled: bool,
    sites: BTreeMap<&'static str, Vec<ErrorStats>>,
    int_width_sites: BTreeMap<&'static str, Vec<IntWidthStats>>,
}

impl Report {
    fn disabled() -> Self {
        Self {
            enabled: false,
            sites: BTreeMap::new(),
            int_width_sites: BTreeMap::new(),
        }
    }

    fn push(&mut self, site: &'static str, reference: &[f32], actual: &[f32]) {
        if !self.enabled {
            return;
        }
        self.sites
            .entry(site)
            .or_default()
            .push(ErrorStats::new(reference, actual));
    }

    fn push_int_width(&mut self, site: &'static str, a: &[f32], w: &[f32], y: &[f32]) {
        if !self.enabled {
            return;
        }
        self.int_width_sites
            .entry(site)
            .or_default()
            .push(IntWidthStats::new(a, w, y));
    }

    fn print(&self) {
        if self.sites.is_empty() {
            println!("MatMul quantization report: no quantized MatMuls");
        } else {
            println!(
                "{:<12} {:>6} {:>12} {:>12} {:>12} {:>23} {:>23}",
                "site", "count", "cosine", "mse", "max_err", "float_range", "quant_range"
            );
            for (&site, rows) in &self.sites {
                let n = rows.len().max(1) as f64;
                let cosine = rows.iter().map(|s| s.cosine).sum::<f64>() / n;
                let mse = rows.iter().map(|s| s.mse).sum::<f64>() / n;
                let max_err = rows.iter().map(|s| s.max_err).fold(0.0, f64::max);
                let ref_min = rows.iter().map(|s| s.ref_min).fold(f32::INFINITY, f32::min);
                let ref_max = rows
                    .iter()
                    .map(|s| s.ref_max)
                    .fold(f32::NEG_INFINITY, f32::max);
                let act_min = rows.iter().map(|s| s.act_min).fold(f32::INFINITY, f32::min);
                let act_max = rows
                    .iter()
                    .map(|s| s.act_max)
                    .fold(f32::NEG_INFINITY, f32::max);
                println!(
                    "{site:<12} {:>6} {:>12.8} {:>12.5e} {:>12.5} [{ref_min:>9.3},{ref_max:>9.3}] [{act_min:>9.3},{act_max:>9.3}]",
                    rows.len(),
                    cosine,
                    mse,
                    max_err
                );
            }
        }
        if !self.int_width_sites.is_empty() {
            println!();
            println!(
                "{:<12} {:>6} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}",
                "site", "count", "A_QX.0b", "W_QX.0b", "Y_QX.0b", "A_max", "W_max", "Y_max"
            );
            for (&site, rows) in &self.int_width_sites {
                let a = rows.iter().map(|s| s.a_qx0_signed_bits).max().unwrap_or(0);
                let w = rows.iter().map(|s| s.w_qx0_signed_bits).max().unwrap_or(0);
                let y = rows.iter().map(|s| s.y_qx0_signed_bits).max().unwrap_or(0);
                let a_max = rows.iter().map(|s| s.a_abs_max).fold(0.0, f32::max);
                let w_max = rows.iter().map(|s| s.w_abs_max).fold(0.0, f32::max);
                let y_max = rows.iter().map(|s| s.y_abs_max).fold(0.0, f32::max);
                println!(
                    "{site:<12} {:>6} {a:>10} {w:>10} {y:>10} {a_max:>12.3} {w_max:>12.3} {y_max:>12.3}",
                    rows.len()
                );
            }
        }
    }
}

impl Default for Report {
    fn default() -> Self {
        Self {
            enabled: true,
            sites: BTreeMap::new(),
            int_width_sites: BTreeMap::new(),
        }
    }
}

struct IntWidthStats {
    a_qx0_signed_bits: u8,
    w_qx0_signed_bits: u8,
    y_qx0_signed_bits: u8,
    a_abs_max: f32,
    w_abs_max: f32,
    y_abs_max: f32,
}

impl IntWidthStats {
    fn new(a: &[f32], w: &[f32], y: &[f32]) -> Self {
        Self {
            a_qx0_signed_bits: qx0_signed_bits(a),
            w_qx0_signed_bits: qx0_signed_bits(w),
            y_qx0_signed_bits: qx0_signed_bits(y),
            a_abs_max: abs_max(a),
            w_abs_max: abs_max(w),
            y_abs_max: abs_max(y),
        }
    }
}

fn abs_max(xs: &[f32]) -> f32 {
    xs.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

fn qx0_signed_bits(xs: &[f32]) -> u8 {
    let min = xs.iter().map(|&x| x.round() as i64).min().unwrap_or(0);
    let max = xs.iter().map(|&x| x.round() as i64).max().unwrap_or(0);
    for bits in 1..=63 {
        let lo = -(1i64 << (bits - 1));
        let hi = (1i64 << (bits - 1)) - 1;
        if lo <= min && max <= hi {
            return bits as u8;
        }
    }
    64
}

struct ErrorStats {
    cosine: f64,
    mse: f64,
    max_err: f64,
    ref_min: f32,
    ref_max: f32,
    act_min: f32,
    act_max: f32,
}

impl ErrorStats {
    fn new(reference: &[f32], actual: &[f32]) -> Self {
        assert_eq!(reference.len(), actual.len());
        let mut dot = 0.0f64;
        let mut rn = 0.0f64;
        let mut an = 0.0f64;
        let mut sq = 0.0f64;
        let mut max_err = 0.0f64;
        let mut ref_min = f32::INFINITY;
        let mut ref_max = f32::NEG_INFINITY;
        let mut act_min = f32::INFINITY;
        let mut act_max = f32::NEG_INFINITY;
        for (&r, &a) in reference.iter().zip(actual) {
            let r64 = r as f64;
            let a64 = a as f64;
            let diff = r64 - a64;
            dot += r64 * a64;
            rn += r64 * r64;
            an += a64 * a64;
            sq += diff * diff;
            max_err = max_err.max(diff.abs());
            ref_min = ref_min.min(r);
            ref_max = ref_max.max(r);
            act_min = act_min.min(a);
            act_max = act_max.max(a);
        }
        Self {
            cosine: dot / (rn.sqrt() * an.sqrt()).max(f64::MIN_POSITIVE),
            mse: sq / reference.len().max(1) as f64,
            max_err,
            ref_min,
            ref_max,
            act_min,
            act_max,
        }
    }
}

fn err(message: impl Into<String>) -> Box<dyn Error> {
    message.into().into()
}
