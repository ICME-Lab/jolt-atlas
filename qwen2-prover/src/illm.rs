use std::error::Error;

use safetensors::{SafeTensors, tensor::TensorView};

use crate::rebase::{
    QuantParams, Rounding, di_rebase, di_rebase_shift_multiplier, di_rebase_shift_slice,
    make_shift_multiplier_floor, make_shift_multiplier_from_ratio, q_for_bits,
};

#[derive(Clone, Debug, PartialEq)]
pub struct QuantTensor {
    pub values: Vec<i128>,
    pub params: QuantParams,
}

impl QuantTensor {
    pub fn new(values: Vec<i128>, params: QuantParams) -> Self {
        let q = q_for_bits(params.bits);
        debug_assert!(
            values.iter().all(|&v| 0 <= v && v <= q),
            "quantized tensor contains values outside [0, {q}]"
        );
        Self { values, params }
    }

    pub fn centered(&self) -> impl Iterator<Item = i128> + '_ {
        self.values
            .iter()
            .map(|&value| value - self.params.zero_point)
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&value| ((value - self.params.zero_point) as f64 * self.params.scale) as f32)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiRebaseMethod {
    Exact,
    Shift { multiplier_shift: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiConfig {
    pub bits: u8,
    pub rounding: Rounding,
    pub rebase: DiRebaseMethod,
}

impl DiConfig {
    pub fn w8a8() -> Self {
        Self {
            bits: 8,
            rounding: Rounding::Floor,
            rebase: DiRebaseMethod::Shift {
                multiplier_shift: 32,
            },
        }
    }
}

const SOFTMAX_BITS: u8 = 8;
const SOFTMAX_EXP_BITS: u8 = 16;
const SOFTMAX_INPUT_BITS: u8 = 8;
const SOFTMAX_DEFAULT_CLIP: i128 = 15;
const SOFTMAX_SCALE_SHIFT: u32 = 8;
const DI_EXP_LOG2E_MULTIPLIER: i128 = 369;
const DI_EXP_LOG2E_SHIFT: u32 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiSoftmaxConfig {
    pub input_bits: u8,
    pub output_bits: u8,
    pub exp_bits: u8,
    pub clip: i128,
    pub clip_input: bool,
    pub scale_shift: u32,
    pub normalize_shift: u32,
}

impl DiSoftmaxConfig {
    pub fn paper_default() -> Self {
        Self {
            input_bits: SOFTMAX_INPUT_BITS,
            output_bits: SOFTMAX_BITS,
            exp_bits: SOFTMAX_EXP_BITS,
            clip: SOFTMAX_DEFAULT_CLIP,
            clip_input: true,
            scale_shift: SOFTMAX_SCALE_SHIFT,
            normalize_shift: 32,
        }
    }

    pub fn without_input_clip(mut self) -> Self {
        self.clip_input = false;
        self
    }
}

pub fn di_matmul(
    lhs: &QuantTensor,
    rhs: &QuantTensor,
    m: usize,
    k: usize,
    n: usize,
    cfg: DiConfig,
) -> QuantTensor {
    assert_eq!(lhs.values.len(), m * k);
    assert_eq!(rhs.values.len(), k * n);

    let lhs_zp = lhs.params.zero_point;
    let rhs_zp = rhs.params.zero_point;
    let mut raw = vec![0i128; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i128;
            for inner in 0..k {
                let a = lhs.values[row * k + inner] - lhs_zp;
                let b = rhs.values[inner * n + col] - rhs_zp;
                acc += a * b;
            }
            raw[row * n + col] = acc;
        }
    }

    rebase_observed(&raw, lhs.params.scale * rhs.params.scale, cfg)
}

pub fn di_add(
    lhs: &QuantTensor,
    rhs: &QuantTensor,
    multiplier_shift: u32,
    cfg: DiConfig,
) -> QuantTensor {
    assert_eq!(lhs.values.len(), rhs.values.len());
    let acc_scale = lhs.params.scale.min(rhs.params.scale);
    let lhs_multiplier = make_shift_multiplier_from_ratio(
        lhs.params.scale / acc_scale,
        multiplier_shift,
        cfg.rounding,
    );
    let rhs_multiplier = make_shift_multiplier_from_ratio(
        rhs.params.scale / acc_scale,
        multiplier_shift,
        cfg.rounding,
    );

    let raw = lhs
        .centered()
        .zip(rhs.centered())
        .map(|(a, b)| lhs_multiplier.apply(a) + rhs_multiplier.apply(b))
        .collect::<Vec<_>>();

    rebase_observed(&raw, acc_scale, cfg)
}

pub fn di_reciprocal(input: &QuantTensor, cfg: DiConfig) -> QuantTensor {
    let raw_real = input
        .values
        .iter()
        .map(|&value| {
            let x = (value - input.params.zero_point) as f64 * input.params.scale;
            assert!(x > 0.0, "DI-Reciprocal input must be positive");
            1.0 / x
        })
        .collect::<Vec<_>>();
    quantize_real_observed(&raw_real, cfg)
}

pub fn di_silu(input: &QuantTensor, cfg: DiConfig) -> QuantTensor {
    di_lut(input, cfg, |x| x / (1.0 + (-x).exp()))
}

pub fn di_mul(lhs: &QuantTensor, rhs: &QuantTensor, cfg: DiConfig) -> QuantTensor {
    assert_eq!(lhs.values.len(), rhs.values.len());
    let raw = lhs
        .centered()
        .zip(rhs.centered())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();
    rebase_observed(&raw, lhs.params.scale * rhs.params.scale, cfg)
}

pub fn di_softmax(input: &QuantTensor, rows: usize, cols: usize, cfg: DiConfig) -> QuantTensor {
    let normalize_shift = match cfg.rebase {
        DiRebaseMethod::Exact => 64,
        DiRebaseMethod::Shift { multiplier_shift } => multiplier_shift,
    };
    di_clipped_softmax_paper(
        input,
        rows,
        cols,
        DiSoftmaxConfig {
            normalize_shift,
            ..DiSoftmaxConfig::paper_default()
        },
    )
}

pub fn di_clipped_softmax_paper(
    input: &QuantTensor,
    rows: usize,
    cols: usize,
    cfg: DiSoftmaxConfig,
) -> QuantTensor {
    assert_eq!(input.values.len(), rows * cols);
    assert!(cfg.clip > 0);

    let input_q = q_for_bits(cfg.input_bits);
    let output_q = q_for_bits(cfg.output_bits);
    let clipped_scale = cfg.clip as f64 / input_q as f64;
    let to_clipped = make_shift_multiplier_from_ratio(
        input.params.scale / clipped_scale,
        cfg.scale_shift,
        Rounding::Floor,
    );

    let mut values = Vec::with_capacity(input.values.len());
    for row in input.values.chunks_exact(cols) {
        let max = *row.iter().max().unwrap();
        let exp = row
            .iter()
            .map(|&value| {
                let delta = value - max;
                let clipped = to_clipped.apply(delta).max(-input_q);
                debug_assert!(
                    -input_q <= clipped && clipped <= 0,
                    "clipped softmax input out of range: {clipped}, expected [-{input_q}, 0]"
                );
                di_exp_paper(clipped, cfg.clip, cfg.scale_shift, cfg.exp_bits)
            })
            .collect::<Vec<_>>();
        let sum = exp.iter().sum::<i128>();
        assert!(sum > 0, "softmax exp row sum must be positive");
        let multiplier = make_shift_multiplier_floor(output_q, sum, cfg.normalize_shift);
        values.extend(exp.into_iter().map(|value| {
            let y = multiplier.apply(value);
            debug_assert!(
                0 <= y && y <= output_q,
                "softmax output out of range: {y}, expected [0, {output_q}]"
            );
            y
        }));
    }
    QuantTensor::new(
        values,
        QuantParams {
            scale: 1.0 / output_q as f64,
            zero_point: 0,
            bits: cfg.output_bits,
        },
    )
}

pub fn di_clipped_softmax_f32(
    input: &[f32],
    rows: usize,
    cols: usize,
    cfg: DiSoftmaxConfig,
) -> QuantTensor {
    assert_eq!(input.len(), rows * cols);
    assert!(cfg.clip > 0);

    let quantized = quantize_softmax_input_per_row(input, rows, cols, cfg);
    di_clipped_softmax_per_row(&quantized, cfg)
}

pub fn di_lut_softmax_f32(
    input: &[f32],
    rows: usize,
    cols: usize,
    cfg: DiSoftmaxConfig,
) -> QuantTensor {
    let quantized = quantize_softmax_input_per_row(input, rows, cols, cfg);
    di_lut_softmax_per_row(&quantized, cfg)
}

#[derive(Clone, Debug, PartialEq)]
pub struct PerRowSoftmaxInput {
    pub values: Vec<i128>,
    pub scales: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

pub fn quantize_softmax_input_per_row(
    input: &[f32],
    rows: usize,
    cols: usize,
    cfg: DiSoftmaxConfig,
) -> PerRowSoftmaxInput {
    assert_eq!(input.len(), rows * cols);
    assert!(cfg.input_bits > 0);
    let q = q_for_bits(cfg.input_bits);
    let mut values = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(rows);
    for row in input.chunks_exact(cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut min_delta = 0.0f64;
        for &value in row {
            let delta = if value.is_finite() {
                (value - max) as f64
            } else {
                -(cfg.clip as f64)
            };
            let delta = if cfg.clip_input {
                delta.max(-(cfg.clip as f64))
            } else {
                delta
            };
            min_delta = min_delta.min(delta);
        }
        let scale = (-min_delta).max(f64::MIN_POSITIVE) / q as f64;
        scales.push(scale);
        for &value in row {
            let delta = if value.is_finite() {
                (value - max) as f64
            } else {
                -(cfg.clip as f64)
            };
            let delta = if cfg.clip_input {
                delta.max(-(cfg.clip as f64))
            } else {
                delta
            };
            let quantized = (delta / scale).floor() as i128;
            let quantized = quantized.max(-q);
            debug_assert!(
                -q <= quantized && quantized <= 0,
                "per-row softmax input out of range: {quantized}, expected [-{q}, 0]"
            );
            values.push(quantized);
        }
    }
    PerRowSoftmaxInput {
        values,
        scales,
        rows,
        cols,
    }
}

pub fn di_clipped_softmax_per_row(input: &PerRowSoftmaxInput, cfg: DiSoftmaxConfig) -> QuantTensor {
    assert_eq!(input.values.len(), input.rows * input.cols);
    assert_eq!(input.scales.len(), input.rows);
    assert!(cfg.clip > 0);

    let input_q = q_for_bits(cfg.input_bits);
    let output_q = q_for_bits(cfg.output_bits);
    let clipped_scale = cfg.clip as f64 / input_q as f64;
    let mut values = Vec::with_capacity(input.values.len());
    for (row_idx, row) in input.values.chunks_exact(input.cols).enumerate() {
        let to_clipped = make_shift_multiplier_from_ratio(
            input.scales[row_idx] / clipped_scale,
            cfg.scale_shift,
            Rounding::Floor,
        );
        let exp = row
            .iter()
            .map(|&value| {
                let mut clipped = to_clipped.apply(value);
                if cfg.clip_input {
                    clipped = clipped.max(-input_q);
                }
                debug_assert!(
                    -input_q <= clipped && clipped <= 0,
                    "clipped softmax input out of range: {clipped}, expected [-{input_q}, 0]"
                );
                di_exp_paper(clipped, cfg.clip, cfg.scale_shift, cfg.exp_bits)
            })
            .collect::<Vec<_>>();
        let sum = exp.iter().sum::<i128>();
        assert!(sum > 0, "softmax exp row sum must be positive");
        let multiplier = make_shift_multiplier_floor(output_q, sum, cfg.normalize_shift);
        values.extend(exp.into_iter().map(|value| {
            let y = multiplier.apply(value);
            debug_assert!(
                0 <= y && y <= output_q,
                "softmax output out of range: {y}, expected [0, {output_q}]"
            );
            y
        }));
    }

    QuantTensor::new(
        values,
        QuantParams {
            scale: 1.0 / output_q as f64,
            zero_point: 0,
            bits: cfg.output_bits,
        },
    )
}

pub fn di_lut_softmax_per_row(input: &PerRowSoftmaxInput, cfg: DiSoftmaxConfig) -> QuantTensor {
    assert_eq!(input.values.len(), input.rows * input.cols);
    assert_eq!(input.scales.len(), input.rows);
    assert!(cfg.clip > 0);

    let input_q = q_for_bits(cfg.input_bits);
    let output_q = q_for_bits(cfg.output_bits);
    let exp_scale = 1i128 << cfg.exp_bits;
    let clipped_scale = cfg.clip as f64 / input_q as f64;
    let mut values = Vec::with_capacity(input.values.len());
    for (row_idx, row) in input.values.chunks_exact(input.cols).enumerate() {
        let to_clipped = make_shift_multiplier_from_ratio(
            input.scales[row_idx] / clipped_scale,
            cfg.scale_shift,
            Rounding::Floor,
        );
        let exp = row
            .iter()
            .map(|&value| {
                let mut clipped = to_clipped.apply(value);
                if cfg.clip_input {
                    clipped = clipped.max(-input_q);
                }
                debug_assert!(
                    clipped <= 0,
                    "softmax LUT input must be max-subtracted: {clipped}"
                );
                let x = clipped as f64 * clipped_scale;
                ((x.exp() * exp_scale as f64).floor() as i128).max(0)
            })
            .collect::<Vec<_>>();
        let sum = exp.iter().sum::<i128>();
        assert!(sum > 0, "softmax exp row sum must be positive");
        let multiplier = make_shift_multiplier_floor(output_q, sum, cfg.normalize_shift);
        values.extend(exp.into_iter().map(|value| {
            let y = multiplier.apply(value);
            debug_assert!(
                0 <= y && y <= output_q,
                "softmax output out of range: {y}, expected [0, {output_q}]"
            );
            y
        }));
    }

    QuantTensor::new(
        values,
        QuantParams {
            scale: 1.0 / output_q as f64,
            zero_point: 0,
            bits: cfg.output_bits,
        },
    )
}

fn di_exp_paper(x: i128, m_x: i128, k_x: u32, output_bits: u8) -> i128 {
    assert!(x <= 0, "DI-Exp input must be max-subtracted");
    assert!(m_x > 0);
    let n = DI_EXP_LOG2E_SHIFT + k_x;
    assert!(n < 127);

    let scaled = -(x * m_x * DI_EXP_LOG2E_MULTIPLIER);
    let p = scaled >> n;
    let r = scaled - (p << n);
    let y = (1i128 << n) - (r >> 1);
    let shift = n as i128 + p - output_bits as i128;
    let y = shift_i128(y, shift);
    let q = 1i128 << output_bits;
    debug_assert!(
        0 <= y && y <= q,
        "DI-Exp output out of range: {y}, expected [0, {q}]"
    );
    y
}

fn shift_i128(value: i128, shift: i128) -> i128 {
    if shift >= 0 {
        value >> shift as u32
    } else {
        value << (-shift) as u32
    }
}

pub fn di_add_rows(
    input: &QuantTensor,
    bias: &[f32],
    rows: usize,
    cols: usize,
    cfg: DiConfig,
) -> QuantTensor {
    assert_eq!(input.values.len(), rows * cols);
    assert_eq!(bias.len(), cols);
    let repeated_bias = (0..rows)
        .flat_map(|_| bias.iter().copied())
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    let bias = quantize_real_observed(&repeated_bias, cfg);
    di_add(input, &bias, 32, cfg)
}

pub fn di_lut(input: &QuantTensor, cfg: DiConfig, f: impl Fn(f64) -> f64) -> QuantTensor {
    let q = q_for_bits(input.params.bits);
    let lut = (0..=q)
        .map(|value| {
            let x = (value - input.params.zero_point) as f64 * input.params.scale;
            f(x)
        })
        .collect::<Vec<_>>();

    let raw_real = input
        .values
        .iter()
        .map(|&value| {
            assert!(0 <= value && value <= q);
            lut[value as usize]
        })
        .collect::<Vec<_>>();

    quantize_real_observed(&raw_real, cfg)
}

pub fn quantize_f32_observed(raw: &[f32], cfg: DiConfig) -> QuantTensor {
    let raw = raw.iter().map(|&value| value as f64).collect::<Vec<_>>();
    quantize_real_observed(&raw, cfg)
}

pub fn rebase_observed(raw: &[i128], raw_scale: f64, cfg: DiConfig) -> QuantTensor {
    let (p_min, p_max) = observed_i128_range(raw);
    if p_min == p_max {
        let q = q_for_bits(cfg.bits);
        return QuantTensor::new(
            vec![0; raw.len()],
            QuantParams {
                scale: raw_scale / q as f64,
                zero_point: -p_min * q,
                bits: cfg.bits,
            },
        );
    }
    assert!(p_max > p_min);
    let (values, params) = match cfg.rebase {
        DiRebaseMethod::Exact => di_rebase(raw, raw_scale, p_min, p_max, cfg.bits, cfg.rounding),
        DiRebaseMethod::Shift { multiplier_shift } => {
            let multiplier =
                di_rebase_shift_multiplier(p_min, p_max, cfg.bits, multiplier_shift, cfg.rounding);
            let values = di_rebase_shift_slice(raw, p_min, p_max, cfg.bits, multiplier);
            let params = crate::rebase::di_rebase_output_params(raw_scale, p_min, p_max, cfg.bits);
            (values, params)
        }
    };
    QuantTensor::new(values, params)
}

pub fn quantize_real_observed(raw: &[f64], cfg: DiConfig) -> QuantTensor {
    let (min, max) = observed_f64_range(raw);
    if min == max {
        let q = q_for_bits(cfg.bits);
        return QuantTensor::new(
            vec![0; raw.len()],
            QuantParams {
                scale: 1.0 / q as f64,
                zero_point: (-min * q as f64).round() as i128,
                bits: cfg.bits,
            },
        );
    }
    assert!(max > min);
    let q = q_for_bits(cfg.bits);
    let scale = (max - min) / q as f64;
    let zero_point = (-min / scale).round() as i128;
    let values = raw
        .iter()
        .map(|&value| {
            let y = match cfg.rounding {
                Rounding::Floor => ((value - min) / scale).floor(),
                Rounding::Nearest => ((value - min) / scale).round(),
            } as i128;
            debug_assert!(
                0 <= y && y <= q,
                "DI-LUT output out of range: {y}, expected [0, {q}]"
            );
            y
        })
        .collect::<Vec<_>>();
    QuantTensor::new(
        values,
        QuantParams {
            scale,
            zero_point,
            bits: cfg.bits,
        },
    )
}

pub fn observed_i128_range(values: &[i128]) -> (i128, i128) {
    assert!(!values.is_empty());
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    (min, max)
}

pub fn observed_f64_range(values: &[f64]) -> (f64, f64) {
    assert!(!values.is_empty());
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        assert!(value.is_finite());
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

fn need_shape(t: &TensorView<'_>, shape: &[usize], name: &str) -> Result<(), Box<dyn Error>> {
    if t.shape() != shape {
        return Err(crate::weights::err(format!(
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
    crate::float::tensor_f32(&t)
}

fn linear_qtensor(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
    cfg: DiConfig,
) -> Result<QuantTensor, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, &[out, input], name)?;
    let w = crate::float::transpose(&crate::float::tensor_f32(&t)?, out, input);
    Ok(quantize_f32_observed(&w, cfg))
}

pub struct IllmLayerWeights {
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub wq: QuantTensor,
    pub bq: Vec<f32>,
    pub wk: QuantTensor,
    pub bk: Vec<f32>,
    pub wv: QuantTensor,
    pub bv: Vec<f32>,
    pub wo: QuantTensor,
    pub wg: QuantTensor,
    pub wu: QuantTensor,
    pub wd: QuantTensor,
}

pub fn load_layer_from_safetensors(
    st: &SafeTensors<'_>,
    layer: usize,
    cfg: DiConfig,
) -> Result<IllmLayerWeights, Box<dyn Error>> {
    let p = format!("model.layers.{layer}");
    Ok(IllmLayerWeights {
        ln1: vec_f32(st, &format!("{p}.input_layernorm.weight"), &[crate::HIDDEN])?,
        ln2: vec_f32(
            st,
            &format!("{p}.post_attention_layernorm.weight"),
            &[crate::HIDDEN],
        )?,
        wq: linear_qtensor(
            st,
            &format!("{p}.self_attn.q_proj.weight"),
            crate::HIDDEN,
            crate::HIDDEN,
            cfg,
        )?,
        bq: vec_f32(st, &format!("{p}.self_attn.q_proj.bias"), &[crate::HIDDEN])?,
        wk: linear_qtensor(
            st,
            &format!("{p}.self_attn.k_proj.weight"),
            crate::KV_HEADS * crate::HEAD_DIM,
            crate::HIDDEN,
            cfg,
        )?,
        bk: vec_f32(
            st,
            &format!("{p}.self_attn.k_proj.bias"),
            &[crate::KV_HEADS * crate::HEAD_DIM],
        )?,
        wv: linear_qtensor(
            st,
            &format!("{p}.self_attn.v_proj.weight"),
            crate::KV_HEADS * crate::HEAD_DIM,
            crate::HIDDEN,
            cfg,
        )?,
        bv: vec_f32(
            st,
            &format!("{p}.self_attn.v_proj.bias"),
            &[crate::KV_HEADS * crate::HEAD_DIM],
        )?,
        wo: linear_qtensor(
            st,
            &format!("{p}.self_attn.o_proj.weight"),
            crate::HIDDEN,
            crate::HIDDEN,
            cfg,
        )?,
        wg: linear_qtensor(
            st,
            &format!("{p}.mlp.gate_proj.weight"),
            crate::INTERMEDIATE,
            crate::HIDDEN,
            cfg,
        )?,
        wu: linear_qtensor(
            st,
            &format!("{p}.mlp.up_proj.weight"),
            crate::INTERMEDIATE,
            crate::HIDDEN,
            cfg,
        )?,
        wd: linear_qtensor(
            st,
            &format!("{p}.mlp.down_proj.weight"),
            crate::HIDDEN,
            crate::INTERMEDIATE,
            cfg,
        )?,
    })
}

pub fn embed_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
    cfg: DiConfig,
) -> Result<QuantTensor, Box<dyn Error>> {
    let x = crate::float::embed_from_safetensors(st, ids)?;
    Ok(quantize_f32_observed(&x, cfg))
}

pub fn di_rms_norm(
    input: &QuantTensor,
    weight: &[f32],
    rows: usize,
    cols: usize,
    cfg: DiConfig,
) -> QuantTensor {
    let x = input.dequantize();
    let y = crate::float::rms_norm(&x, weight, rows, cols);
    quantize_f32_observed(&y, cfg)
}

pub fn di_rope(input: &QuantTensor, rot: &[f32], cfg: DiConfig) -> QuantTensor {
    let x = input.dequantize();
    let y = crate::float::rope(&x, rot);
    quantize_f32_observed(&y, cfg)
}

pub fn di_score_qk(q: &QuantTensor, k: &QuantTensor, cfg: DiConfig) -> QuantTensor {
    let q = q.dequantize();
    let k = k.dequantize();
    let mut s = crate::float::score_qk(&q, &k);
    for row in s.chunks_mut(crate::SEQ) {
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
    quantize_f32_observed(&s, cfg)
}

pub fn di_attn_v(p: &QuantTensor, v: &QuantTensor, cfg: DiConfig) -> QuantTensor {
    let p = p.dequantize();
    let v = v.dequantize();
    let c = crate::float::attn_v(&p, &v);
    quantize_f32_observed(&c, cfg)
}

pub struct IllmRotary {
    pub rq: Vec<f32>,
    pub rk: Vec<f32>,
}

impl IllmRotary {
    pub fn new() -> Self {
        Self {
            rq: crate::float::rot(crate::SEQ, crate::HEADS, crate::HEAD_DIM),
            rk: crate::float::rot(crate::SEQ, crate::KV_HEADS, crate::HEAD_DIM),
        }
    }
}

pub fn layer(x: &QuantTensor, w: &IllmLayerWeights, r: &IllmRotary, cfg: DiConfig) -> QuantTensor {
    let n1 = di_rms_norm(x, &w.ln1, crate::SEQ, crate::HIDDEN, cfg);
    let mut q = di_matmul(&n1, &w.wq, crate::SEQ, crate::HIDDEN, crate::HIDDEN, cfg);
    q = di_add_rows(&q, &w.bq, crate::SEQ, crate::HIDDEN, cfg);
    let mut k = di_matmul(
        &n1,
        &w.wk,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
        cfg,
    );
    k = di_add_rows(
        &k,
        &w.bk,
        crate::SEQ,
        crate::KV_HEADS * crate::HEAD_DIM,
        cfg,
    );
    let mut v = di_matmul(
        &n1,
        &w.wv,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
        cfg,
    );
    v = di_add_rows(
        &v,
        &w.bv,
        crate::SEQ,
        crate::KV_HEADS * crate::HEAD_DIM,
        cfg,
    );
    let q = di_rope(&q, &r.rq, cfg);
    let k = di_rope(&k, &r.rk, cfg);
    let s = di_score_qk(&q, &k, cfg);
    let p = di_softmax(&s, crate::HEADS * crate::SEQ, crate::SEQ, cfg);
    let c = di_attn_v(&p, &v, cfg);
    let a = di_matmul(&c, &w.wo, crate::SEQ, crate::HIDDEN, crate::HIDDEN, cfg);
    let h = di_add(x, &a, 32, cfg);
    let n2 = di_rms_norm(&h, &w.ln2, crate::SEQ, crate::HIDDEN, cfg);
    let g = di_matmul(
        &n2,
        &w.wg,
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
        cfg,
    );
    let u = di_matmul(
        &n2,
        &w.wu,
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
        cfg,
    );
    let sg = di_silu(&g, cfg);
    let m = di_mul(&sg, &u, cfg);
    let d = di_matmul(
        &m,
        &w.wd,
        crate::SEQ,
        crate::INTERMEDIATE,
        crate::HIDDEN,
        cfg,
    );
    di_add(&h, &d, 32, cfg)
}

pub fn forward_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
    cfg: DiConfig,
) -> Result<QuantTensor, Box<dyn Error>> {
    let r = IllmRotary::new();
    let mut x = embed_from_safetensors(st, ids, cfg)?;
    for i in 0..crate::LAYERS {
        let w = load_layer_from_safetensors(st, i, cfg)?;
        x = layer(&x, &w, &r, cfg);
    }
    let w = vec_f32(st, "model.norm.weight", &[crate::HIDDEN])?;
    Ok(di_rms_norm(&x, &w, crate::SEQ, crate::HIDDEN, cfg))
}

pub fn perplexity_tied_float_lm_head_prefix_from_safetensors(
    st: &SafeTensors<'_>,
    hidden: &QuantTensor,
    ids: &[u32],
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    let hidden = hidden.dequantize();
    crate::float::perplexity_tied_lm_head_prefix_from_safetensors(st, &hidden, ids, max_targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(scale: f64, zero_point: i128) -> QuantParams {
        QuantParams {
            scale,
            zero_point,
            bits: 8,
        }
    }

    fn dequant(t: &QuantTensor) -> Vec<f64> {
        t.values
            .iter()
            .map(|&value| (value - t.params.zero_point) as f64 * t.params.scale)
            .collect()
    }

    #[test]
    fn di_matmul_rebases_observed_accumulator() {
        let lhs = QuantTensor::new(vec![1, 2, 3, 4], params(0.5, 0));
        let rhs = QuantTensor::new(vec![5, 6, 7, 8], params(0.25, 0));
        let out = di_matmul(
            &lhs,
            &rhs,
            2,
            2,
            2,
            DiConfig {
                bits: 8,
                rounding: Rounding::Floor,
                rebase: DiRebaseMethod::Exact,
            },
        );
        assert_eq!(out.values, vec![0, 24, 197, 255]);
        assert!((out.params.scale - 31.0 * 0.5 * 0.25 / 255.0).abs() < 1e-12);
    }

    #[test]
    fn di_add_uses_common_accumulator_then_rebases_once() {
        let lhs = QuantTensor::new(vec![10, 20], params(0.5, 0));
        let rhs = QuantTensor::new(vec![4, 8], params(0.25, 0));
        let out = di_add(
            &lhs,
            &rhs,
            16,
            DiConfig {
                bits: 8,
                rounding: Rounding::Floor,
                rebase: DiRebaseMethod::Exact,
            },
        );
        assert_eq!(out.values, vec![0, 255]);
    }

    #[test]
    fn di_silu_uses_lut_from_input_quantization() {
        let input = QuantTensor::new(vec![0, 128, 255], params(2.0 / 255.0, 128));
        let out = di_silu(
            &input,
            DiConfig {
                bits: 8,
                rounding: Rounding::Floor,
                rebase: DiRebaseMethod::Exact,
            },
        );
        let ys = dequant(&out);
        assert!(ys[0] < ys[1]);
        assert!(ys[1] < ys[2]);
    }

    #[test]
    fn di_mul_multiplies_centered_values_then_rebases() {
        let lhs = QuantTensor::new(vec![130, 140], params(0.5, 128));
        let rhs = QuantTensor::new(vec![129, 131], params(0.25, 128));
        let out = di_mul(
            &lhs,
            &rhs,
            DiConfig {
                bits: 8,
                rounding: Rounding::Floor,
                rebase: DiRebaseMethod::Exact,
            },
        );
        assert_eq!(out.values, vec![0, 255]);
        assert!((out.params.scale - 34.0 * 0.5 * 0.25 / 255.0).abs() < 1e-12);
    }

    #[test]
    fn di_reciprocal_quantizes_inverse_values() {
        let input = QuantTensor::new(vec![64, 128, 255], params(1.0 / 64.0, 0));
        let out = di_reciprocal(
            &input,
            DiConfig {
                bits: 8,
                rounding: Rounding::Floor,
                rebase: DiRebaseMethod::Exact,
            },
        );
        let ys = dequant(&out);
        assert!(ys[0] > ys[1]);
        assert!(ys[1] > ys[2]);
    }

    #[test]
    fn di_exp_paper_matches_expected_endpoints() {
        let one = 1i128 << 16;
        assert_eq!(di_exp_paper(0, 15, 8, 16), one);
        let clipped = di_exp_paper(-255, 15, 8, 16);
        assert_eq!(clipped, 0);
    }

    #[test]
    fn quantize_softmax_input_uses_per_row_scales() {
        let cfg = DiSoftmaxConfig::paper_default();
        let input = vec![0.0, -1.0, -2.0, 0.0, -5.0, -10.0];
        let out = quantize_softmax_input_per_row(&input, 2, 3, cfg);
        assert_eq!(out.values, vec![0, -128, -255, 0, -128, -255]);
        assert!(out.scales[0] < out.scales[1]);
    }

    #[test]
    fn di_softmax_uses_fixed_exp_and_outputs_u8_probabilities() {
        let input = QuantTensor::new(vec![10, 20, 30], params(0.1, 0));
        let out = di_softmax(
            &input,
            1,
            3,
            DiConfig {
                bits: 8,
                rounding: Rounding::Floor,
                rebase: DiRebaseMethod::Exact,
            },
        );
        let ys = dequant(&out);
        assert_eq!(ys.len(), 3);
        assert_eq!(out.params.bits, 8);
        assert_eq!(out.params.zero_point, 0);
        assert!((out.params.scale - 1.0 / 255.0).abs() < 1e-12);
        assert!(ys[0] < ys[1]);
        assert!(ys[1] < ys[2]);
    }
}
