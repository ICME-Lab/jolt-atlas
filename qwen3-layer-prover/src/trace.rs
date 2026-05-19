use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

use serde_json::Value;
use thiserror::Error;

use crate::{
    layer::{LayerShape, LayerWeights, LayerWitness},
    ops::round::ROUND_FRAC_BITS,
};

const FIXED_SCALE: i64 = 1 << ROUND_FRAC_BITS;

#[derive(Debug, Error)]
pub enum TraceWitnessError {
    #[error("io error while reading trace: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid trace manifest json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("trace tensor label {0:?} was not found")]
    MissingTensor(String),

    #[error("trace tensor {label:?} has dtype {actual:?}, expected {expected:?}")]
    DtypeMismatch {
        label: String,
        expected: String,
        actual: String,
    },

    #[error("trace tensor {label:?} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        label: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("trace tensor {label:?} has {actual} values, expected {expected}")]
    LenMismatch {
        label: String,
        expected: usize,
        actual: usize,
    },

    #[error("checkpoint {label:?} differs at index {index}: expected {expected}, got {actual}")]
    CheckpointMismatch {
        label: String,
        index: usize,
        expected: i32,
        actual: i32,
    },

    #[error("invalid layer shape: {0}")]
    InvalidShape(String),
}

pub type TraceWitnessResult<T> = std::result::Result<T, TraceWitnessError>;

#[derive(Debug, Clone)]
pub struct TraceLayerWitness {
    pub witness: LayerWitness,
    pub hidden_out: Vec<i32>,
}

#[derive(Debug, Clone)]
struct TensorRef {
    dtype: String,
    shape: Vec<usize>,
    file: PathBuf,
}

#[derive(Debug, Clone)]
struct TraceIndex {
    root: PathBuf,
    tensors: HashMap<String, TensorRef>,
}

/// Build the `LayerWitness` expected by `prove_layer` from a compact
/// `qwen3-awy --dump-full-awy` trace.
///
/// The trace intentionally stores only i32 checkpoints. This builder
/// regenerates derived witness data deterministically: accumulators, round
/// frac bits, RMS square sums, SiLU lookup selectors, and softmax advice.
///
/// The QK-score witness matches the qwen3-awy runtime exactly:
/// `dot = round((Q @ K^T) / 2^8)` followed by
/// `score = round(dot * round(2^8 / sqrt(head_dim)) / 2^8)`.
/// If the trace was produced by a runtime with a different scaling or rounding
/// order, this builder reports a checkpoint mismatch instead of constructing a
/// witness that cannot satisfy `prove_layer`.
pub fn build_layer_witness_from_trace_dir(
    trace_dir: impl AsRef<Path>,
    layer: usize,
    weights: &LayerWeights,
    shape: &LayerShape,
) -> TraceWitnessResult<TraceLayerWitness> {
    validate_shape(shape)?;
    let trace = TraceIndex::load(trace_dir)?;
    let hidden_in = trace.i32(&format!("layer{layer}.ln1.A"), &[shape.seq, shape.hidden])?;
    let hidden_out = trace.i32(
        &format!("layer{layer}.residual_mlp.Y"),
        &[shape.seq, shape.hidden],
    )?;

    let rms_norm_atten = rms_norm(&hidden_in, &weights.rms_norm_atten, shape.seq, shape.hidden);
    check(
        &format!("layer{layer}.ln1.Y"),
        &trace.i32(&format!("layer{layer}.ln1.Y"), &[shape.seq, shape.hidden])?,
        &rms_norm_atten.output,
    )?;

    let q_proj = matmul(
        &rms_norm_atten.output,
        &weights.q_proj,
        shape.seq,
        shape.hidden,
        shape.attention_width(),
    );
    let k_proj = matmul(
        &rms_norm_atten.output,
        &weights.k_proj,
        shape.seq,
        shape.hidden,
        shape.kv_heads * shape.head_dim,
    );
    let v_proj = matmul(
        &rms_norm_atten.output,
        &weights.v_proj,
        shape.seq,
        shape.hidden,
        shape.kv_heads * shape.head_dim,
    );
    check(
        &format!("layer{layer}.q_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.q_proj.Y"),
            &[shape.seq, shape.attention_width()],
        )?,
        &q_proj.output,
    )?;
    check(
        &format!("layer{layer}.k_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.k_proj.Y"),
            &[shape.seq, shape.kv_heads * shape.head_dim],
        )?,
        &k_proj.output,
    )?;
    check(
        &format!("layer{layer}.v_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.v_proj.Y"),
            &[shape.seq, shape.kv_heads * shape.head_dim],
        )?,
        &v_proj.output,
    )?;

    let q_norm = rms_norm(
        &q_proj.output,
        &weights.q_norm,
        shape.seq * shape.q_heads,
        shape.head_dim,
    );
    let k_norm = rms_norm(
        &k_proj.output,
        &weights.k_norm,
        shape.seq * shape.kv_heads,
        shape.head_dim,
    );
    check(
        &format!("layer{layer}.q_norm.Y"),
        &trace.i32(
            &format!("layer{layer}.q_norm.Y"),
            &[shape.seq, shape.q_heads, shape.head_dim],
        )?,
        &q_norm.output,
    )?;
    check(
        &format!("layer{layer}.k_norm.Y"),
        &trace.i32(
            &format!("layer{layer}.k_norm.Y"),
            &[shape.seq, shape.kv_heads, shape.head_dim],
        )?,
        &k_norm.output,
    )?;

    let q_rope = rope(
        &q_norm.output,
        &weights.rope_cos,
        &weights.rope_sin,
        shape.seq,
        shape.q_heads,
        shape.head_dim,
    );
    let k_rope = rope(
        &k_norm.output,
        &weights.rope_cos,
        &weights.rope_sin,
        shape.seq,
        shape.kv_heads,
        shape.head_dim,
    );
    check(
        &format!("layer{layer}.q_rope.Y"),
        &trace.i32(
            &format!("layer{layer}.q_rope.Y"),
            &[shape.seq, shape.q_heads, shape.head_dim],
        )?,
        &q_rope.output,
    )?;
    check(
        &format!("layer{layer}.k_rope.Y"),
        &trace.i32(
            &format!("layer{layer}.k_rope.Y"),
            &[shape.seq, shape.kv_heads, shape.head_dim],
        )?,
        &k_rope.output,
    )?;

    let qk_score = qk_score(&q_rope.output, &k_rope.output, shape);
    check(
        &format!("layer{layer}.attention_scores.Y"),
        &trace.i32(
            &format!("layer{layer}.attention_scores.Y"),
            &[shape.q_heads, shape.seq, shape.seq],
        )?,
        &qk_score.output,
    )?;

    let softmax = causal_softmax(&qk_score.output, shape);
    check(
        &format!("layer{layer}.softmax.Y"),
        &trace.i32(
            &format!("layer{layer}.softmax.Y"),
            &[shape.q_heads, shape.seq, shape.seq],
        )?,
        &softmax.output,
    )?;

    let context = pv_matmul(&softmax.output, &v_proj.output, shape);
    check(
        &format!("layer{layer}.attention_value.Y"),
        &trace.i32(
            &format!("layer{layer}.attention_value.Y"),
            &[shape.seq, shape.attention_width()],
        )?,
        &context.output,
    )?;

    let o_proj = matmul(
        &context.output,
        &weights.o_proj,
        shape.seq,
        shape.attention_width(),
        shape.hidden,
    );
    check(
        &format!("layer{layer}.o_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.o_proj.Y"),
            &[shape.seq, shape.hidden],
        )?,
        &o_proj.output,
    )?;

    let residual_add_attn = hidden_in
        .iter()
        .zip(&o_proj.output)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    check(
        &format!("layer{layer}.residual_attn.Y"),
        &trace.i32(
            &format!("layer{layer}.residual_attn.Y"),
            &[shape.seq, shape.hidden],
        )?,
        &residual_add_attn,
    )?;

    let rms_norm_mlp = rms_norm(
        &residual_add_attn,
        &weights.rms_norm_mlp,
        shape.seq,
        shape.hidden,
    );
    check(
        &format!("layer{layer}.ln2.Y"),
        &trace.i32(&format!("layer{layer}.ln2.Y"), &[shape.seq, shape.hidden])?,
        &rms_norm_mlp.output,
    )?;

    let gate_proj = matmul(
        &rms_norm_mlp.output,
        &weights.gate_proj,
        shape.seq,
        shape.hidden,
        shape.intermediate,
    );
    let up_proj = matmul(
        &rms_norm_mlp.output,
        &weights.up_proj,
        shape.seq,
        shape.hidden,
        shape.intermediate,
    );
    check(
        &format!("layer{layer}.gate_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.gate_proj.Y"),
            &[shape.seq, shape.intermediate],
        )?,
        &gate_proj.output,
    )?;
    check(
        &format!("layer{layer}.up_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.up_proj.Y"),
            &[shape.seq, shape.intermediate],
        )?,
        &up_proj.output,
    )?;

    let silu = silu(&gate_proj.output);
    let silu_up = hadamard(&silu.output, &up_proj.output);
    check(
        &format!("layer{layer}.silu_gate_times_up.Y"),
        &trace.i32(
            &format!("layer{layer}.silu_gate_times_up.Y"),
            &[shape.seq, shape.intermediate],
        )?,
        &silu_up.output,
    )?;

    let down_proj = matmul(
        &silu_up.output,
        &weights.down_proj,
        shape.seq,
        shape.intermediate,
        shape.hidden,
    );
    check(
        &format!("layer{layer}.down_proj.Y"),
        &trace.i32(
            &format!("layer{layer}.down_proj.Y"),
            &[shape.seq, shape.hidden],
        )?,
        &down_proj.output,
    )?;

    let hidden_out_expected = residual_add_attn
        .iter()
        .zip(&down_proj.output)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    check(
        &format!("layer{layer}.residual_mlp.Y"),
        &hidden_out,
        &hidden_out_expected,
    )?;

    Ok(TraceLayerWitness {
        hidden_out,
        witness: LayerWitness {
            hidden_in: hidden_in.clone(),
            rms_norm_atten_sum_x2: rms_norm_atten.sum_x2,
            rms_norm_atten_norm_acc: rms_norm_atten.norm_acc,
            rms_norm_atten_norm: rms_norm_atten.norm,
            rms_norm_atten_norm_frac_bits: rms_norm_atten.norm_frac_bits,
            rms_norm_atten_acc: rms_norm_atten.acc,
            rms_norm_atten_a: rms_norm_atten.output.clone(),
            rms_norm_atten_b: rms_norm_atten.output.clone(),
            rms_norm_atten_c: rms_norm_atten.output,
            rms_norm_atten_frac_bits: rms_norm_atten.frac_bits,
            context_acc: context.acc,
            context: context.output.clone(),
            context_frac_bits: context.frac_bits,
            o_proj: o_proj.output.clone(),
            o_proj_acc: o_proj.acc,
            o_proj_frac_bits: o_proj.frac_bits,
            softmax: softmax.output,
            softmax_acc: softmax.acc,
            softmax_floor: softmax.floor,
            softmax_floor_frac_bits: softmax.floor_frac_bits,
            softmax_frac_bits: softmax.frac_bits,
            softmax_max_index: softmax.max_index,
            softmax_min_diff: softmax.min_diff,
            softmax_max_diff: softmax.max_diff,
            softmax_input_frac_bits: softmax.input_frac_bits,
            softmax_ra: softmax.ra,
            softmax_exp_acc: softmax.exp_acc,
            softmax_exp_frac_bits: softmax.exp_frac_bits,
            qk_score: qk_score.output,
            qk_score_acc: qk_score.acc,
            qk_score_dot: qk_score.dot,
            qk_score_dot_frac_bits: qk_score.dot_frac_bits,
            qk_score_scale_acc: qk_score.scale_acc,
            qk_score_frac_bits: qk_score.frac_bits,
            q_rope_acc: q_rope.acc,
            q_rope: q_rope.output,
            q_rope_frac_bits: q_rope.frac_bits,
            k_rope_acc: k_rope.acc,
            k_rope: k_rope.output,
            k_rope_frac_bits: k_rope.frac_bits,
            q_proj: q_proj.output.clone(),
            k_proj: k_proj.output.clone(),
            q_norm_sum_x2: q_norm.sum_x2,
            k_norm_sum_x2: k_norm.sum_x2,
            q_norm_norm_acc: q_norm.norm_acc,
            k_norm_norm_acc: k_norm.norm_acc,
            q_norm_norm: q_norm.norm,
            k_norm_norm: k_norm.norm,
            q_norm_norm_frac_bits: q_norm.norm_frac_bits,
            k_norm_norm_frac_bits: k_norm.norm_frac_bits,
            q_norm_acc: q_norm.acc,
            k_norm_acc: k_norm.acc,
            q_norm: q_norm.output,
            k_norm: k_norm.output,
            q_norm_frac_bits: q_norm.frac_bits,
            k_norm_frac_bits: k_norm.frac_bits,
            q_proj_acc: q_proj.acc,
            q_proj_frac_bits: q_proj.frac_bits,
            k_proj_acc: k_proj.acc,
            k_proj_frac_bits: k_proj.frac_bits,
            v_proj_acc: v_proj.acc,
            v_proj_frac_bits: v_proj.frac_bits,
            softmax_max: softmax.max,
            softmax_exp: softmax.exp,
            softmax_sum: softmax.sum,
            v_proj: v_proj.output,
            residual_add_attn_a: residual_add_attn.clone(),
            residual_add_attn_b: residual_add_attn,
            rms_norm_mlp_sum_x2: rms_norm_mlp.sum_x2,
            rms_norm_mlp_norm_acc: rms_norm_mlp.norm_acc,
            rms_norm_mlp_norm: rms_norm_mlp.norm,
            rms_norm_mlp_norm_frac_bits: rms_norm_mlp.norm_frac_bits,
            rms_norm_mlp_acc: rms_norm_mlp.acc,
            rms_norm_mlp_a: rms_norm_mlp.output.clone(),
            rms_norm_mlp_b: rms_norm_mlp.output,
            rms_norm_mlp_frac_bits: rms_norm_mlp.frac_bits,
            gate_proj_acc: gate_proj.acc,
            gate_proj: gate_proj.output.clone(),
            gate_proj_frac_bits: gate_proj.frac_bits,
            silu_acc: silu.acc,
            silu: silu.output,
            silu_min_n: silu.min_n,
            silu_max_n: silu.max_n,
            silu_frac_bits: silu.frac_bits,
            silu_ra: silu.ra,
            silu_out_frac_bits: silu.out_frac_bits,
            silu_up_acc: silu_up.acc,
            silu_up: silu_up.output,
            silu_up_frac_bits: silu_up.frac_bits,
            up_proj_acc: up_proj.acc,
            up_proj: up_proj.output,
            up_proj_frac_bits: up_proj.frac_bits,
            down_proj_acc: down_proj.acc,
            down_proj: down_proj.output,
            down_proj_frac_bits: down_proj.frac_bits,
        },
    })
}

impl TraceIndex {
    fn load(root: impl AsRef<Path>) -> TraceWitnessResult<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest = BufReader::new(File::open(root.join("manifest.jsonl"))?);
        let mut tensors = HashMap::new();
        for line in manifest.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            collect_tensors(&root, &serde_json::from_str::<Value>(&line)?, &mut tensors);
        }
        Ok(Self { root, tensors })
    }

    fn i32(&self, label: &str, expected_shape: &[usize]) -> TraceWitnessResult<Vec<i32>> {
        let tensor = self
            .tensors
            .get(label)
            .ok_or_else(|| TraceWitnessError::MissingTensor(label.to_string()))?;
        if tensor.dtype != "i32_le" {
            return Err(TraceWitnessError::DtypeMismatch {
                label: label.to_string(),
                expected: "i32_le".to_string(),
                actual: tensor.dtype.clone(),
            });
        }
        if tensor.shape != expected_shape {
            return Err(TraceWitnessError::ShapeMismatch {
                label: label.to_string(),
                expected: expected_shape.to_vec(),
                actual: tensor.shape.clone(),
            });
        }
        let mut bytes = Vec::new();
        File::open(self.root.join(&tensor.file))?.read_to_end(&mut bytes)?;
        let expected_len = expected_shape.iter().product::<usize>();
        if bytes.len() != expected_len * std::mem::size_of::<i32>() {
            return Err(TraceWitnessError::LenMismatch {
                label: label.to_string(),
                expected: expected_len,
                actual: bytes.len() / std::mem::size_of::<i32>(),
            });
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
            .collect())
    }
}

fn collect_tensors(root: &Path, value: &Value, tensors: &mut HashMap<String, TensorRef>) {
    match value {
        Value::Object(map) => {
            if let (Some(Value::String(label)), Some(Value::String(dtype)), Some(file)) =
                (map.get("label"), map.get("dtype"), map.get("file"))
            {
                if let (Some(shape), Some(file)) =
                    (shape_from_json(map.get("shape")), file.as_str())
                {
                    let path = PathBuf::from(file);
                    let file = path
                        .strip_prefix(root)
                        .map_or(path.clone(), Path::to_path_buf);
                    tensors.insert(
                        label.clone(),
                        TensorRef {
                            dtype: dtype.clone(),
                            shape,
                            file,
                        },
                    );
                }
            }
            for child in map.values() {
                collect_tensors(root, child, tensors);
            }
        }
        Value::Array(xs) => {
            for child in xs {
                collect_tensors(root, child, tensors);
            }
        }
        _ => {}
    }
}

fn shape_from_json(value: Option<&Value>) -> Option<Vec<usize>> {
    value?
        .as_array()?
        .iter()
        .map(|dim| dim.as_u64().map(|dim| dim as usize))
        .collect()
}

#[derive(Debug, Clone)]
struct RoundFixture {
    acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
struct RmsFixture {
    sum_x2: Vec<i64>,
    norm_acc: Vec<i64>,
    norm: Vec<i32>,
    norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
struct QkFixture {
    acc: Vec<i64>,
    dot: Vec<i32>,
    dot_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    scale_acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
struct SiluFixture {
    acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    ra: Vec<u8>,
    out_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    min_n: i64,
    max_n: i64,
}

#[derive(Debug, Clone)]
struct SoftmaxFixture {
    output: Vec<i32>,
    acc: Vec<i64>,
    floor: Vec<i32>,
    floor_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    sum: Vec<i32>,
    max_index: Vec<usize>,
    max: Vec<i32>,
    min_diff: i64,
    max_diff: i64,
    input_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    ra: Vec<u8>,
    exp_acc: Vec<i64>,
    exp: Vec<i32>,
    exp_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

fn validate_shape(shape: &LayerShape) -> TraceWitnessResult<()> {
    if shape.seq == 0
        || shape.hidden == 0
        || shape.intermediate == 0
        || shape.q_heads == 0
        || shape.kv_heads == 0
        || shape.head_dim == 0
    {
        return Err(TraceWitnessError::InvalidShape(format!("{shape:?}")));
    }
    if shape.q_heads % shape.kv_heads != 0 {
        return Err(TraceWitnessError::InvalidShape(format!(
            "q_heads must be divisible by kv_heads: {shape:?}"
        )));
    }
    Ok(())
}

fn check(label: &str, actual: &[i32], expected: &[i32]) -> TraceWitnessResult<()> {
    if actual.len() != expected.len() {
        return Err(TraceWitnessError::LenMismatch {
            label: label.to_string(),
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual != expected {
            return Err(TraceWitnessError::CheckpointMismatch {
                label: label.to_string(),
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn matmul(a: &[i32], w: &[i32], m: usize, k: usize, n: usize) -> RoundFixture {
    let mut acc = vec![0_i64; m * n];
    for row in 0..m {
        for col in 0..n {
            acc[row * n + col] = (0..k)
                .map(|kk| i64::from(a[row * k + kk]) * i64::from(w[kk * n + col]))
                .sum();
        }
    }
    round_fixture(acc)
}

fn rms_norm(input: &[i32], weight: &[i32], rows: usize, cols: usize) -> RmsFixture {
    let mut sum_x2 = vec![0_i64; rows];
    let mut norm_acc = vec![0_i64; rows * cols];
    for row in 0..rows {
        sum_x2[row] = input[row * cols..(row + 1) * cols]
            .iter()
            .map(|&value| i64::from(value) * i64::from(value))
            .sum();
        let inv = rms_inv_from_square_sum(sum_x2[row], cols);
        for col in 0..cols {
            norm_acc[row * cols + col] = i64::from(input[row * cols + col]) * inv;
        }
    }
    let norm = round_fixture(norm_acc);
    let acc = norm
        .output
        .iter()
        .enumerate()
        .map(|(idx, &value)| i64::from(value) * i64::from(weight[idx % cols]))
        .collect::<Vec<_>>();
    let rounded = round_fixture(acc);
    RmsFixture {
        sum_x2,
        norm_acc: norm.acc,
        norm: norm.output,
        norm_frac_bits: norm.frac_bits,
        acc: rounded.acc,
        output: rounded.output,
        frac_bits: rounded.frac_bits,
    }
}

fn rope(
    input: &[i32],
    cos: &[i32],
    sin: &[i32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> RoundFixture {
    let mut acc = vec![0_i64; seq * heads * head_dim];
    for s in 0..seq {
        for h in 0..heads {
            for pair in 0..head_dim / 2 {
                let base = (s * heads + h) * head_dim;
                let lo = base + pair;
                let hi = base + head_dim / 2 + pair;
                let coeff = s * (head_dim / 2) + pair;
                let x0 = i64::from(input[lo]);
                let x1 = i64::from(input[hi]);
                let c = i64::from(cos[coeff]);
                let si = i64::from(sin[coeff]);
                acc[lo] = x0 * c - x1 * si;
                acc[hi] = x0 * si + x1 * c;
            }
        }
    }
    round_fixture(acc)
}

fn qk_score(q: &[i32], k: &[i32], shape: &LayerShape) -> QkFixture {
    let mut acc = vec![0_i64; shape.q_heads * shape.seq * shape.seq];
    let group = shape.q_heads / shape.kv_heads;
    for h in 0..shape.q_heads {
        let kh = h / group;
        for i in 0..shape.seq {
            for j in 0..shape.seq {
                let out = (h * shape.seq + i) * shape.seq + j;
                acc[out] = (0..shape.head_dim)
                    .map(|d| {
                        let qi = (i * shape.q_heads + h) * shape.head_dim + d;
                        let ki = (j * shape.kv_heads + kh) * shape.head_dim + d;
                        i64::from(q[qi]) * i64::from(k[ki])
                    })
                    .sum();
            }
        }
    }
    let dot = round_fixture(acc);
    let inv_sqrt = ((1.0 / (shape.head_dim as f64).sqrt()) * 256.0).round() as i64;
    let scale_acc = dot
        .output
        .iter()
        .map(|&value| i64::from(value) * inv_sqrt)
        .collect::<Vec<_>>();
    let score = round_fixture(scale_acc);
    QkFixture {
        acc: dot.acc,
        dot: dot.output,
        dot_frac_bits: dot.frac_bits,
        scale_acc: score.acc,
        output: score.output,
        frac_bits: score.frac_bits,
    }
}

fn pv_matmul(p: &[i32], v: &[i32], shape: &LayerShape) -> RoundFixture {
    let mut acc = vec![0_i64; shape.seq * shape.q_heads * shape.head_dim];
    let group = shape.q_heads / shape.kv_heads;
    for i in 0..shape.seq {
        for h in 0..shape.q_heads {
            let kh = h / group;
            for d in 0..shape.head_dim {
                let out = (i * shape.q_heads + h) * shape.head_dim + d;
                acc[out] = (0..shape.seq)
                    .map(|j| {
                        let pi = (h * shape.seq + i) * shape.seq + j;
                        let vi = (j * shape.kv_heads + kh) * shape.head_dim + d;
                        i64::from(p[pi]) * i64::from(v[vi])
                    })
                    .sum();
            }
        }
    }
    round_fixture(acc)
}

fn hadamard(lhs: &[i32], rhs: &[i32]) -> RoundFixture {
    round_fixture(
        lhs.iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| i64::from(lhs) * i64::from(rhs))
            .collect(),
    )
}

fn silu(gate: &[i32]) -> SiluFixture {
    let rounded = gate
        .iter()
        .map(|&value| round_q8(i64::from(value)))
        .collect::<Vec<_>>();
    let min_n = i64::from(*rounded.iter().min().unwrap_or(&0));
    let max_n = i64::from(*rounded.iter().max().unwrap_or(&0));
    let entries = (max_n - min_n + 1) as usize;
    let mut ra = vec![0; gate.len() * entries];
    let mut acc = vec![0_i64; gate.len()];
    for (idx, &gate_value) in gate.iter().enumerate() {
        let n = i64::from(rounded[idx]);
        ra[idx * entries + (n - min_n) as usize] = 1;
        acc[idx] = silu_base(n) + (i64::from(gate_value) - n * FIXED_SCALE) * silu_slope(n);
    }
    let rounded = round_fixture(acc);
    SiluFixture {
        acc: rounded.acc,
        output: rounded.output,
        frac_bits: frac_bits_i32(gate),
        ra,
        out_frac_bits: rounded.frac_bits,
        min_n,
        max_n,
    }
}

fn causal_softmax(scores: &[i32], shape: &LayerShape) -> SoftmaxFixture {
    let rows = shape.q_heads * shape.seq;
    let cols = shape.seq;
    let mut output = vec![0; rows * cols];
    let mut acc = vec![0; rows * cols];
    let mut floor = vec![0; rows * cols];
    let mut sum = vec![0; rows];
    let mut max_index = vec![0; rows];
    let mut max = vec![0; rows];
    let mut exp = vec![256; rows * cols];
    let mut exp_acc = vec![256_i64 * 256_i64; rows * cols];
    let mut diffs = vec![0_i64; rows * cols];
    let mut min_diff = 0_i64;
    let max_diff = 0_i64;
    for row in 0..rows {
        let query_pos = row % shape.seq;
        let row_scores = &scores[row * cols..(row + 1) * cols];
        let (idx, value) = row_scores[..=query_pos]
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .unwrap();
        max_index[row] = idx;
        max[row] = *value;
        for col in 0..cols {
            let out = row * cols + col;
            if col <= query_pos {
                let diff = i64::from(scores[out]) - i64::from(max[row]);
                let n = floor_q8(diff);
                diffs[out] = i64::from(n);
                min_diff = min_diff.min(i64::from(n));
                exp_acc[out] = softmax_exp_acc_q8(diff);
                exp[out] = softmax_exp_coarse_q8(diff);
                sum[row] += exp[out];
            }
        }
        let inv = inv_sum_q16(sum[row]);
        for col in 0..cols {
            let idx = row * cols + col;
            if col <= query_pos {
                acc[idx] = i64::from(exp[idx]) * inv;
                floor[idx] = floor_q8(acc[idx]);
                output[idx] = round_q8(i64::from(floor[idx]));
            }
        }
    }
    let entries = (max_diff - min_diff + 1) as usize;
    let mut ra = vec![0; scores.len() * entries];
    for (idx, &diff) in diffs.iter().enumerate() {
        ra[idx * entries + (diff - min_diff) as usize] = 1;
    }
    let input_frac_bits = frac_bits_i64(
        &scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| {
                let row = idx / cols;
                let col = idx % cols;
                if col <= row % shape.seq {
                    i64::from(score) - i64::from(max[row])
                } else {
                    0
                }
            })
            .collect::<Vec<_>>(),
    );
    let floor_frac_bits = frac_bits_i64(&acc);
    let frac_bits = frac_bits_i32(&floor);
    let exp_frac_bits = frac_bits_i64(&exp_acc);
    SoftmaxFixture {
        output,
        acc,
        floor,
        floor_frac_bits,
        sum,
        max_index,
        max,
        min_diff,
        max_diff,
        input_frac_bits,
        ra,
        exp_acc,
        exp,
        exp_frac_bits,
        frac_bits,
    }
}

fn round_fixture(acc: Vec<i64>) -> RoundFixture {
    RoundFixture {
        output: acc.iter().map(|&value| round_q8(value)).collect(),
        frac_bits: frac_bits_i64(&acc),
        acc,
    }
}

fn rms_inv_from_square_sum(square_sum: i64, hidden_size: usize) -> i64 {
    let mean = square_sum as f64 / hidden_size as f64 / (256.0 * 256.0);
    ((1.0 / (mean + 1e-6).sqrt()) * 256.0).round() as i64
}

fn silu_base(n: i64) -> i64 {
    let n_q8 = n * FIXED_SCALE;
    n_q8 * sigmoid_q8(n)
}

fn silu_slope(n: i64) -> i64 {
    let n_q8 = n * FIXED_SCALE;
    let sig = sigmoid_q8(n);
    let sig_slope = i64::from(round_q8(sig * (FIXED_SCALE - sig)));
    sig + i64::from(round_q8(n_q8 * sig_slope))
}

fn sigmoid_q8(n: i64) -> i64 {
    ((1.0 / (1.0 + (-(n as f64)).exp())) * 256.0).round() as i64
}

fn softmax_exp_coarse_q8(delta_q8: i64) -> i32 {
    round_q8(softmax_exp_acc_q8(delta_q8))
}

fn softmax_exp_acc_q8(delta_q8: i64) -> i64 {
    let n = delta_q8.div_euclid(FIXED_SCALE);
    let f = delta_q8 - n * FIXED_SCALE;
    let exp_n = exp_lut_q8(n);
    let corr = (FIXED_SCALE + f).max(0);
    exp_n * corr
}

fn exp_lut_q8(n: i64) -> i64 {
    let n = n.clamp(-16, 0);
    (f64::exp(n as f64) * FIXED_SCALE as f64).round() as i64
}

fn inv_sum_q16(sum: i32) -> i64 {
    ((1_i64 << 24) as f64 / f64::from(sum)).round() as i64
}

fn frac_bits_i64(values: &[i64]) -> [Vec<u8>; ROUND_FRAC_BITS] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((value.rem_euclid(FIXED_SCALE) >> bit) & 1) as u8)
            .collect()
    })
}

fn frac_bits_i32(values: &[i32]) -> [Vec<u8>; ROUND_FRAC_BITS] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((i64::from(*value).rem_euclid(FIXED_SCALE) >> bit) & 1) as u8)
            .collect()
    })
}

fn round_q8(value: i64) -> i32 {
    ((value + ((value.rem_euclid(FIXED_SCALE) >> 7) * FIXED_SCALE) - value.rem_euclid(FIXED_SCALE))
        / FIXED_SCALE) as i32
}

fn floor_q8(value: i64) -> i32 {
    value.div_euclid(FIXED_SCALE) as i32
}
