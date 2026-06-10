use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

use itertools::Itertools;
use qwen3_common::{FIXED_SCALE, FRAC_BITS};
use serde_json::Value;
use thiserror::Error;

use qwen3_prover::{
    layer::{LayerProverInput, LayerShape},
    layer_input::{LayerRawWitness, LayerWeights, layer_prover_input},
};

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

#[derive(Debug, Error)]
pub enum TraceError {
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

    #[error("invalid q8 cache: {0}")]
    InvalidQ8Cache(String),

    #[error("layer input conversion failed")]
    LayerInput,
}

pub struct TraceLayerInput {
    pub shape: LayerShape,
    pub hidden_out: Vec<i32>,
    pub raw_witness: LayerRawWitness,
    pub input: LayerProverInput,
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

#[derive(Debug, Clone)]
struct RoundFixture {
    output: Vec<i32>,
    frac_bits: [Vec<u8>; FRAC_BITS],
}

#[derive(Debug, Clone)]
struct RmsFixture {
    sum_x2: Vec<i64>,
    norm: Vec<i32>,
    norm_frac_bits: [Vec<u8>; FRAC_BITS],
    output: Vec<i32>,
    frac_bits: [Vec<u8>; FRAC_BITS],
}

#[derive(Debug, Clone)]
struct QkFixture {
    dot: Vec<i32>,
    dot_frac_bits: [Vec<u8>; FRAC_BITS],
    output: Vec<i32>,
    frac_bits: [Vec<u8>; FRAC_BITS],
}

#[derive(Debug, Clone)]
struct SiluFixture {
    output: Vec<i32>,
    frac_bits: [Vec<u8>; FRAC_BITS],
    ra: Vec<u8>,
    out_frac_bits: [Vec<u8>; FRAC_BITS],
    min_n: i64,
    max_n: i64,
}

#[derive(Debug, Clone)]
struct SoftmaxFixture {
    output: Vec<i32>,
    acc: Vec<i64>,
    floor: Vec<i32>,
    floor_frac_bits: [Vec<u8>; FRAC_BITS],
    sum: Vec<i32>,
    max: Vec<i32>,
    max_index: Vec<usize>,
    min_diff: i64,
    max_diff: i64,
    ra: Vec<u8>,
    exp_acc: Vec<i64>,
    exp: Vec<i32>,
    exp_frac_bits: [Vec<u8>; FRAC_BITS],
    frac_bits: [Vec<u8>; FRAC_BITS],
}

pub fn layer_input_from_trace_dir(
    trace_dir: impl AsRef<Path>,
    q8_cache: impl AsRef<Path>,
    layer: usize,
) -> Result<TraceLayerInput, TraceError> {
    let trace_dir = trace_dir.as_ref();
    let seq = read_trace_seq_len(trace_dir)?;
    let shape = LayerShape {
        seq,
        q_heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
    };
    let weights = read_layer_weights(q8_cache, layer, seq)?;
    let hidden_out_and_witness =
        build_raw_witness_from_trace_dir(trace_dir, layer, &weights, shape)?;
    let raw_witness = hidden_out_and_witness;
    let hidden_out = raw_witness.hidden_out.clone();
    let input =
        layer_prover_input(shape, weights, raw_witness.clone()).ok_or(TraceError::LayerInput)?;
    Ok(TraceLayerInput {
        shape,
        hidden_out,
        raw_witness,
        input,
    })
}

fn build_raw_witness_from_trace_dir(
    trace_dir: impl AsRef<Path>,
    layer: usize,
    weights: &LayerWeights,
    shape: LayerShape,
) -> Result<LayerRawWitness, TraceError> {
    validate_shape(shape)?;
    let trace = TraceIndex::load(trace_dir)?;
    let hidden_in = trace.i32(&format!("layer{layer}.ln1.A"), &[shape.seq, shape.hidden])?;
    let rms_norm_atten = rms_norm_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.ln1"),
        &hidden_in,
        &weights.rms_norm_atten,
        shape.seq,
        shape.hidden,
        &[shape.seq, shape.hidden],
    )?;
    let q_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.q_proj"),
        &[shape.seq, attention_width(shape)],
        || {
            matmul(
                &rms_norm_atten.output,
                &weights.q_proj,
                shape.seq,
                shape.hidden,
                attention_width(shape),
            )
        },
    )?;
    let k_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.k_proj"),
        &[shape.seq, shape.kv_heads * shape.head_dim],
        || {
            matmul(
                &rms_norm_atten.output,
                &weights.k_proj,
                shape.seq,
                shape.hidden,
                shape.kv_heads * shape.head_dim,
            )
        },
    )?;
    let v_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.v_proj"),
        &[shape.seq, shape.kv_heads * shape.head_dim],
        || {
            matmul(
                &rms_norm_atten.output,
                &weights.v_proj,
                shape.seq,
                shape.hidden,
                shape.kv_heads * shape.head_dim,
            )
        },
    )?;
    let q_norm = rms_norm_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.q_norm"),
        &q_proj.output,
        &weights.q_norm,
        shape.seq * shape.q_heads,
        shape.head_dim,
        &[shape.seq, shape.q_heads, shape.head_dim],
    )?;
    let k_norm = rms_norm_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.k_norm"),
        &k_proj.output,
        &weights.k_norm,
        shape.seq * shape.kv_heads,
        shape.head_dim,
        &[shape.seq, shape.kv_heads, shape.head_dim],
    )?;
    let q_rope = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.q_rope"),
        &[shape.seq, shape.q_heads, shape.head_dim],
        || {
            rope(
                &q_norm.output,
                &weights.rope_cos,
                &weights.rope_sin,
                shape.seq,
                shape.q_heads,
                shape.head_dim,
            )
        },
    )?;
    let k_rope = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.k_rope"),
        &[shape.seq, shape.kv_heads, shape.head_dim],
        || {
            rope(
                &k_norm.output,
                &weights.rope_cos,
                &weights.rope_sin,
                shape.seq,
                shape.kv_heads,
                shape.head_dim,
            )
        },
    )?;
    let qk_score = qk_score_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.attention_scores"),
        &q_rope.output,
        &k_rope.output,
        shape,
    )?;
    let softmax = causal_softmax_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.softmax"),
        &qk_score.output,
        shape,
    )?;
    let context = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.attention_value"),
        &[shape.seq, attention_width(shape)],
        || pv_matmul(&softmax.output, &v_proj.output, shape),
    )?;
    let o_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.o_proj"),
        &[shape.seq, shape.hidden],
        || {
            matmul(
                &context.output,
                &weights.o_proj,
                shape.seq,
                attention_width(shape),
                shape.hidden,
            )
        },
    )?;
    let residual_add_attn = hidden_in
        .iter()
        .zip_eq(&o_proj.output)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let rms_norm_mlp = rms_norm_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.ln2"),
        &residual_add_attn,
        &weights.rms_norm_mlp,
        shape.seq,
        shape.hidden,
        &[shape.seq, shape.hidden],
    )?;
    let gate_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.gate_proj"),
        &[shape.seq, shape.intermediate],
        || {
            matmul(
                &rms_norm_mlp.output,
                &weights.gate_proj,
                shape.seq,
                shape.hidden,
                shape.intermediate,
            )
        },
    )?;
    let up_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.up_proj"),
        &[shape.seq, shape.intermediate],
        || {
            matmul(
                &rms_norm_mlp.output,
                &weights.up_proj,
                shape.seq,
                shape.hidden,
                shape.intermediate,
            )
        },
    )?;
    let silu = silu(&gate_proj.output);
    let silu_up = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.silu_gate_times_up"),
        &[shape.seq, shape.intermediate],
        || mul(&silu.output, &up_proj.output),
    )?;
    let down_proj = round_from_trace_or_compute(
        &trace,
        &format!("layer{layer}.down_proj"),
        &[shape.seq, shape.hidden],
        || {
            matmul(
                &silu_up.output,
                &weights.down_proj,
                shape.seq,
                shape.intermediate,
                shape.hidden,
            )
        },
    )?;
    let hidden_out = residual_add_attn
        .iter()
        .zip_eq(&down_proj.output)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>();

    Ok(LayerRawWitness {
        hidden_in,
        hidden_out,
        rms_norm_atten_sum_x2: rms_norm_atten.sum_x2,
        rms_norm_atten_norm: rms_norm_atten.norm,
        rms_norm_atten_norm_frac_bits: rms_norm_atten.norm_frac_bits,
        rms_norm_atten_a: rms_norm_atten.output.clone(),
        rms_norm_atten_b: rms_norm_atten.output.clone(),
        rms_norm_atten_c: rms_norm_atten.output,
        rms_norm_atten_output_frac_bits: rms_norm_atten.frac_bits,
        context: context.output,
        pv_matmul_output_frac_bits: context.frac_bits,
        o_proj: o_proj.output,
        o_proj_output_frac_bits: o_proj.frac_bits,
        softmax: softmax.output,
        softmax_acc: softmax.acc,
        softmax_floor: softmax.floor,
        softmax_floor_frac_bits: softmax.floor_frac_bits,
        softmax_output_frac_bits: softmax.frac_bits,
        softmax_min_diff: softmax.min_diff,
        softmax_max_diff: softmax.max_diff,
        softmax_lookup_ra: softmax.ra,
        softmax_exp_acc: softmax.exp_acc,
        softmax_exp_frac_bits: softmax.exp_frac_bits,
        qk_score: qk_score.output,
        qk_score_dot: qk_score.dot,
        qk_score_dot_output_frac_bits: qk_score.dot_frac_bits,
        qk_score_output_frac_bits: qk_score.frac_bits,
        q_rope: q_rope.output,
        q_rope_output_frac_bits: q_rope.frac_bits,
        k_rope: k_rope.output,
        k_rope_output_frac_bits: k_rope.frac_bits,
        q_proj: q_proj.output,
        k_proj: k_proj.output,
        q_norm_sum_x2: q_norm.sum_x2,
        k_norm_sum_x2: k_norm.sum_x2,
        q_norm_norm: q_norm.norm,
        k_norm_norm: k_norm.norm,
        q_norm_norm_frac_bits: q_norm.norm_frac_bits,
        k_norm_norm_frac_bits: k_norm.norm_frac_bits,
        q_norm: q_norm.output,
        k_norm: k_norm.output,
        q_norm_output_frac_bits: q_norm.frac_bits,
        k_norm_output_frac_bits: k_norm.frac_bits,
        q_proj_output_frac_bits: q_proj.frac_bits,
        k_proj_output_frac_bits: k_proj.frac_bits,
        v_proj_output_frac_bits: v_proj.frac_bits,
        softmax_max: softmax.max,
        softmax_max_index: softmax.max_index,
        softmax_exp: softmax.exp,
        softmax_sum: softmax.sum,
        v_proj: v_proj.output,
        residual_add_attn_a: residual_add_attn.clone(),
        residual_add_attn_b: residual_add_attn,
        rms_norm_mlp_sum_x2: rms_norm_mlp.sum_x2,
        rms_norm_mlp_norm: rms_norm_mlp.norm,
        rms_norm_mlp_norm_frac_bits: rms_norm_mlp.norm_frac_bits,
        rms_norm_mlp_a: rms_norm_mlp.output.clone(),
        rms_norm_mlp_b: rms_norm_mlp.output,
        rms_norm_mlp_output_frac_bits: rms_norm_mlp.frac_bits,
        gate_proj: gate_proj.output,
        gate_proj_output_frac_bits: gate_proj.frac_bits,
        silu: silu.output,
        silu_min_n: silu.min_n,
        silu_max_n: silu.max_n,
        silu_input_frac_bits: silu.frac_bits,
        silu_lookup_ra: silu.ra,
        silu_output_frac_bits: silu.out_frac_bits,
        silu_up: silu_up.output,
        silu_up_output_frac_bits: silu_up.frac_bits,
        up_proj: up_proj.output,
        up_proj_output_frac_bits: up_proj.frac_bits,
        down_proj: down_proj.output,
        down_proj_output_frac_bits: down_proj.frac_bits,
    })
}

fn read_trace_seq_len(trace: &Path) -> Result<usize, TraceError> {
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
    Err(TraceError::InvalidShape(
        "trace metadata did not contain tokens_total".to_string(),
    ))
}

pub fn read_layer_weights(
    path: impl AsRef<Path>,
    target_layer: usize,
    seq: usize,
) -> Result<LayerWeights, TraceError> {
    let path = path.as_ref();
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic)?;
    if &magic != FIXED_CACHE_MAGIC {
        return Err(TraceError::InvalidQ8Cache(format!(
            "{} has invalid fixed cache magic",
            path.display()
        )));
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
        let got = read_u64(&mut reader)?;
        if got != expected {
            return Err(TraceError::InvalidQ8Cache(format!(
                "{} fixed cache header mismatch: expected {expected}, got {got}",
                path.display()
            )));
        }
    }
    let _final_norm = read_i32_vec(&mut reader)?;
    let _lm_head = read_i32_vec(&mut reader)?;
    for layer in 0..LAYERS {
        let ln1 = read_i32_vec(&mut reader)?;
        let ln2 = read_i32_vec(&mut reader)?;
        let q_norm = read_i32_vec(&mut reader)?;
        let k_norm = read_i32_vec(&mut reader)?;
        let wq = read_i32_vec(&mut reader)?;
        let wk = read_i32_vec(&mut reader)?;
        let wv = read_i32_vec(&mut reader)?;
        let wo = read_i32_vec(&mut reader)?;
        let wg = read_i32_vec(&mut reader)?;
        let wu = read_i32_vec(&mut reader)?;
        let wd = read_i32_vec(&mut reader)?;
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
    Err(TraceError::InvalidQ8Cache(format!(
        "layer {target_layer} not found"
    )))
}

impl TraceIndex {
    fn load(root: impl AsRef<Path>) -> Result<Self, TraceError> {
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

    fn has(&self, label: &str) -> bool {
        self.tensors.contains_key(label)
    }

    fn i32(&self, label: &str, expected_shape: &[usize]) -> Result<Vec<i32>, TraceError> {
        let tensor = self
            .tensors
            .get(label)
            .ok_or_else(|| TraceError::MissingTensor(label.to_string()))?;
        self.read_tensor_i32(label, tensor, expected_shape)
    }

    fn i64(&self, label: &str, expected_shape: &[usize]) -> Result<Vec<i64>, TraceError> {
        let tensor = self
            .tensors
            .get(label)
            .ok_or_else(|| TraceError::MissingTensor(label.to_string()))?;
        self.read_tensor_i64(label, tensor, expected_shape)
    }

    fn read_tensor_i32(
        &self,
        label: &str,
        tensor: &TensorRef,
        expected_shape: &[usize],
    ) -> Result<Vec<i32>, TraceError> {
        validate_tensor_ref(label, tensor, "i32_le", expected_shape)?;
        let mut bytes = Vec::new();
        File::open(self.root.join(&tensor.file))?.read_to_end(&mut bytes)?;
        let expected_len = expected_shape.iter().product::<usize>();
        if bytes.len() != expected_len * std::mem::size_of::<i32>() {
            return Err(TraceError::LenMismatch {
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

    fn read_tensor_i64(
        &self,
        label: &str,
        tensor: &TensorRef,
        expected_shape: &[usize],
    ) -> Result<Vec<i64>, TraceError> {
        validate_tensor_ref(label, tensor, "i64_le", expected_shape)?;
        let mut bytes = Vec::new();
        File::open(self.root.join(&tensor.file))?.read_to_end(&mut bytes)?;
        let expected_len = expected_shape.iter().product::<usize>();
        if bytes.len() != expected_len * std::mem::size_of::<i64>() {
            return Err(TraceError::LenMismatch {
                label: label.to_string(),
                expected: expected_len,
                actual: bytes.len() / std::mem::size_of::<i64>(),
            });
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunk length is 8")))
            .collect())
    }
}

fn validate_tensor_ref(
    label: &str,
    tensor: &TensorRef,
    expected_dtype: &str,
    expected_shape: &[usize],
) -> Result<(), TraceError> {
    if tensor.dtype != expected_dtype {
        return Err(TraceError::DtypeMismatch {
            label: label.to_string(),
            expected: expected_dtype.to_string(),
            actual: tensor.dtype.clone(),
        });
    }
    if tensor.shape != expected_shape {
        return Err(TraceError::ShapeMismatch {
            label: label.to_string(),
            expected: expected_shape.to_vec(),
            actual: tensor.shape.clone(),
        });
    }
    Ok(())
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

fn round_from_trace_or_compute(
    trace: &TraceIndex,
    label: &str,
    shape: &[usize],
    compute: impl FnOnce() -> RoundFixture,
) -> Result<RoundFixture, TraceError> {
    let acc_label = format!("{label}.Acc");
    if trace.has(&acc_label) {
        return Ok(round_fixture(trace.i64(&acc_label, shape)?));
    }
    let computed = compute();
    check(
        &format!("{label}.Y"),
        &trace.i32(&format!("{label}.Y"), shape)?,
        &computed.output,
    )?;
    Ok(computed)
}

fn rms_norm_from_trace_or_compute(
    trace: &TraceIndex,
    label: &str,
    input: &[i32],
    weight: &[i32],
    rows: usize,
    cols: usize,
    trace_shape: &[usize],
) -> Result<RmsFixture, TraceError> {
    let acc_label = format!("{label}.Acc");
    if !trace.has(&acc_label) {
        let computed = rms_norm(input, weight, rows, cols);
        check(
            &format!("{label}.Y"),
            &trace.i32(&format!("{label}.Y"), trace_shape)?,
            &computed.output,
        )?;
        return Ok(computed);
    }
    let mut sum_x2 = vec![0_i64; rows];
    for row in 0..rows {
        sum_x2[row] = input[row * cols..(row + 1) * cols]
            .iter()
            .map(|&value| i64::from(value) * i64::from(value))
            .sum();
    }
    let norm_acc_label = format!("{label}.NormAcc");
    let norm_acc = if trace.has(&norm_acc_label) {
        trace.i64(&norm_acc_label, trace_shape)?
    } else {
        let mut norm_acc = vec![0_i64; rows * cols];
        for row in 0..rows {
            let inv = rms_inv_from_square_sum(sum_x2[row], cols);
            for col in 0..cols {
                norm_acc[row * cols + col] = i64::from(input[row * cols + col]) * inv;
            }
        }
        norm_acc
    };
    let norm = round_fixture(norm_acc);
    let rounded = round_fixture(trace.i64(&acc_label, trace_shape)?);
    Ok(RmsFixture {
        sum_x2,
        norm: norm.output,
        norm_frac_bits: norm.frac_bits,
        output: rounded.output,
        frac_bits: rounded.frac_bits,
    })
}

fn qk_score_from_trace_or_compute(
    trace: &TraceIndex,
    label: &str,
    q: &[i32],
    k: &[i32],
    shape: LayerShape,
) -> Result<QkFixture, TraceError> {
    let acc_label = format!("{label}.Acc");
    if !trace.has(&acc_label) {
        let computed = qk_score(q, k, shape);
        check(
            &format!("{label}.Y"),
            &trace.i32(
                &format!("{label}.Y"),
                &[shape.q_heads, shape.seq, shape.seq],
            )?,
            &computed.output,
        )?;
        return Ok(computed);
    }
    let dot_acc = trace.i64(
        &format!("{label}.DotAcc"),
        &[shape.q_heads, shape.seq, shape.seq],
    )?;
    let dot = round_fixture(dot_acc);
    let score = round_fixture(trace.i64(&acc_label, &[shape.q_heads, shape.seq, shape.seq])?);
    Ok(QkFixture {
        dot: dot.output,
        dot_frac_bits: dot.frac_bits,
        output: score.output,
        frac_bits: score.frac_bits,
    })
}

fn causal_softmax_from_trace_or_compute(
    trace: &TraceIndex,
    label: &str,
    scores: &[i32],
    shape: LayerShape,
) -> Result<SoftmaxFixture, TraceError> {
    let mut computed = causal_softmax(scores, shape);
    let acc_label = format!("{label}.Acc");
    if trace.has(&acc_label) {
        computed.acc = trace.i64(&acc_label, &[shape.q_heads, shape.seq, shape.seq])?;
        computed.floor = computed.acc.iter().map(|&value| floor_q8(value)).collect();
        computed.output = computed
            .floor
            .iter()
            .map(|&value| round_q8(i64::from(value)))
            .collect();
        computed.floor_frac_bits = frac_bits_i64(&computed.acc);
        computed.frac_bits = frac_bits_i32(&computed.floor);
    } else {
        check(
            &format!("{label}.Y"),
            &trace.i32(
                &format!("{label}.Y"),
                &[shape.q_heads, shape.seq, shape.seq],
            )?,
            &computed.output,
        )?;
    }
    Ok(computed)
}

fn validate_shape(shape: LayerShape) -> Result<(), TraceError> {
    if shape.seq == 0
        || shape.hidden == 0
        || shape.intermediate == 0
        || shape.q_heads == 0
        || shape.kv_heads == 0
        || shape.head_dim == 0
    {
        return Err(TraceError::InvalidShape(format!("{shape:?}")));
    }
    if shape.q_heads % shape.kv_heads != 0 {
        return Err(TraceError::InvalidShape(format!(
            "q_heads must be divisible by kv_heads: {shape:?}"
        )));
    }
    Ok(())
}

fn check(label: &str, actual: &[i32], expected: &[i32]) -> Result<(), TraceError> {
    if actual.len() != expected.len() {
        return Err(TraceError::LenMismatch {
            label: label.to_string(),
            expected: expected.len(),
            actual: actual.len(),
        });
    }
    for (index, (&actual, &expected)) in actual.iter().zip_eq(expected).enumerate() {
        if actual != expected {
            return Err(TraceError::CheckpointMismatch {
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
        norm: norm.output,
        norm_frac_bits: norm.frac_bits,
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

fn qk_score(q: &[i32], k: &[i32], shape: LayerShape) -> QkFixture {
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
        dot: dot.output,
        dot_frac_bits: dot.frac_bits,
        output: score.output,
        frac_bits: score.frac_bits,
    }
}

fn pv_matmul(p: &[i32], v: &[i32], shape: LayerShape) -> RoundFixture {
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

fn mul(lhs: &[i32], rhs: &[i32]) -> RoundFixture {
    round_fixture(
        lhs.iter()
            .zip_eq(rhs)
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
        output: rounded.output,
        frac_bits: frac_bits_i32(gate),
        ra,
        out_frac_bits: rounded.frac_bits,
        min_n,
        max_n,
    }
}

fn causal_softmax(scores: &[i32], shape: LayerShape) -> SoftmaxFixture {
    let rows = shape.q_heads * shape.seq;
    let cols = shape.seq;
    let mut output = vec![0; rows * cols];
    let mut acc = vec![0; rows * cols];
    let mut floor = vec![0; rows * cols];
    let mut sum = vec![0; rows];
    let mut max = vec![0; rows];
    let mut max_index = vec![0; rows];
    let mut exp = vec![256; rows * cols];
    let mut exp_acc = vec![256_i64 * 256_i64; rows * cols];
    let mut diffs = vec![0_i64; rows * cols];
    let mut min_diff = 0_i64;
    let max_diff = 0_i64;
    for row in 0..rows {
        let query_pos = row % shape.seq;
        let row_scores = &scores[row * cols..(row + 1) * cols];
        let (selected_col, value) = row_scores[..=query_pos]
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .unwrap();
        max[row] = *value;
        max_index[row] = selected_col;
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
    SoftmaxFixture {
        output,
        acc: acc.clone(),
        floor: floor.clone(),
        floor_frac_bits: frac_bits_i64(&acc),
        sum,
        max,
        max_index,
        min_diff,
        max_diff,
        ra,
        exp_acc: exp_acc.clone(),
        exp,
        exp_frac_bits: frac_bits_i64(&exp_acc),
        frac_bits: frac_bits_i32(&floor),
    }
}

fn round_fixture(acc: Vec<i64>) -> RoundFixture {
    RoundFixture {
        output: acc.iter().map(|&value| round_q8(value)).collect(),
        frac_bits: frac_bits_i64(&acc),
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

fn frac_bits_i64(values: &[i64]) -> [Vec<u8>; FRAC_BITS] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((value.rem_euclid(FIXED_SCALE) >> bit) & 1) as u8)
            .collect()
    })
}

fn frac_bits_i32(values: &[i32]) -> [Vec<u8>; FRAC_BITS] {
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

fn read_u64(reader: &mut impl Read) -> Result<u64, TraceError> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i32_vec(reader: &mut impl Read) -> Result<Vec<i32>, TraceError> {
    let len = read_u64(reader)? as usize;
    let mut bytes = vec![0u8; len * std::mem::size_of::<i32>()];
    reader.read_exact(&mut bytes)?;
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
            cos[pos * (HEAD_DIM / 2) + pair] = quantize_q8_f32(t.cos() as f32);
            sin[pos * (HEAD_DIM / 2) + pair] = quantize_q8_f32(t.sin() as f32);
        }
    }
    (cos, sin)
}

fn quantize_q8_f32(x: f32) -> i32 {
    (x * 256.0).round() as i32
}

fn attention_width(shape: LayerShape) -> usize {
    shape.q_heads * shape.head_dim
}
