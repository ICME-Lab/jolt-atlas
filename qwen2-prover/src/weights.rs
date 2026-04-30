use std::{error::Error, io, path::Path};

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

pub const LN1: usize = 0;
pub const LN2: usize = 1;
pub const WQ: usize = 2;
pub const BQ: usize = 3;
pub const WK: usize = 4;
pub const BK: usize = 5;
pub const WV: usize = 6;
pub const BV: usize = 7;
pub const WO: usize = 8;
pub const WG: usize = 9;
pub const WU: usize = 10;
pub const WD: usize = 11;
pub const N: usize = 12;

pub fn load_layer(path: impl AsRef<Path>, layer: usize) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    load_layer_from_bytes(&bytes, layer)
}

pub fn load_layers(path: impl AsRef<Path>) -> Result<Vec<Vec<Vec<i32>>>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    (0..crate::LAYERS)
        .map(|layer| load_layer_from_safetensors(&st, layer))
        .collect()
}

pub fn load_layer_from_bytes(bytes: &[u8], layer: usize) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let st = SafeTensors::deserialize(bytes)?;
    load_layer_from_safetensors(&st, layer)
}

pub fn load_layer_from_safetensors(
    st: &SafeTensors<'_>,
    layer: usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    load_layer_from_safetensors_with_one(st, layer, crate::lut::ONE)
}

pub fn load_layer_from_safetensors_with_one(
    st: &SafeTensors<'_>,
    layer: usize,
    one: i32,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let p = format!("model.layers.{layer}");
    let mut ws = Vec::with_capacity(N);
    ws.push(vec_q(
        st,
        &format!("{p}.input_layernorm.weight"),
        &[crate::HIDDEN],
        one,
    )?);
    ws.push(vec_q(
        st,
        &format!("{p}.post_attention_layernorm.weight"),
        &[crate::HIDDEN],
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.self_attn.q_proj.weight"),
        crate::HIDDEN,
        crate::HIDDEN,
        one,
    )?);
    ws.push(vec_q(
        st,
        &format!("{p}.self_attn.q_proj.bias"),
        &[crate::HIDDEN],
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.self_attn.k_proj.weight"),
        crate::KV_HEADS * crate::HEAD_DIM,
        crate::HIDDEN,
        one,
    )?);
    ws.push(vec_q(
        st,
        &format!("{p}.self_attn.k_proj.bias"),
        &[crate::KV_HEADS * crate::HEAD_DIM],
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.self_attn.v_proj.weight"),
        crate::KV_HEADS * crate::HEAD_DIM,
        crate::HIDDEN,
        one,
    )?);
    ws.push(vec_q(
        st,
        &format!("{p}.self_attn.v_proj.bias"),
        &[crate::KV_HEADS * crate::HEAD_DIM],
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.self_attn.o_proj.weight"),
        crate::HIDDEN,
        crate::HIDDEN,
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.mlp.gate_proj.weight"),
        crate::INTERMEDIATE,
        crate::HIDDEN,
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.mlp.up_proj.weight"),
        crate::INTERMEDIATE,
        crate::HIDDEN,
        one,
    )?);
    ws.push(linear_q(
        st,
        &format!("{p}.mlp.down_proj.weight"),
        crate::HIDDEN,
        crate::INTERMEDIATE,
        one,
    )?);
    Ok(ws)
}

pub fn vec_q8(
    st: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<i32>, Box<dyn Error>> {
    vec_q(st, name, shape, crate::lut::ONE)
}

pub fn vec_q(
    st: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
    one: i32,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, shape, name)?;
    tensor_q(&t, one)
}

pub fn linear_q8(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
) -> Result<Vec<i32>, Box<dyn Error>> {
    linear_q(st, name, out, input, crate::lut::ONE)
}

pub fn linear_q(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
    one: i32,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, &[out, input], name)?;
    Ok(transpose(&tensor_q(&t, one)?, out, input))
}

pub fn final_norm(st: &SafeTensors<'_>) -> Result<Vec<i32>, Box<dyn Error>> {
    final_norm_with_one(st, crate::lut::ONE)
}

pub fn final_norm_with_one(st: &SafeTensors<'_>, one: i32) -> Result<Vec<i32>, Box<dyn Error>> {
    vec_q(st, "model.norm.weight", &[crate::HIDDEN], one)
}

pub fn final_norm_from_bytes(bytes: &[u8]) -> Result<Vec<i32>, Box<dyn Error>> {
    let st = SafeTensors::deserialize(bytes)?;
    final_norm(&st)
}

pub fn tensor_q8(t: &TensorView<'_>) -> Result<Vec<i32>, Box<dyn Error>> {
    tensor_q(t, crate::lut::ONE)
}

pub fn tensor_q(t: &TensorView<'_>, one: i32) -> Result<Vec<i32>, Box<dyn Error>> {
    match t.dtype() {
        Dtype::BF16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| {
                q_to(
                    bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
                    one,
                )
            })
            .collect()),
        Dtype::F16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| {
                q_to(
                    f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
                    one,
                )
            })
            .collect()),
        Dtype::F32 => Ok(t
            .data()
            .chunks_exact(4)
            .map(|b| q_to(f32::from_le_bytes([b[0], b[1], b[2], b[3]]), one))
            .collect()),
        Dtype::I32 => Ok(t
            .data()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        dt => Err(err(format!("unsupported tensor dtype {dt:?}"))),
    }
}

pub fn transpose(xs: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    assert_eq!(xs.len(), rows * cols);
    let mut ys = vec![0; xs.len()];
    for r in 0..rows {
        for c in 0..cols {
            ys[c * rows + r] = xs[r * cols + c];
        }
    }
    ys
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

pub(crate) fn q_to(x: f32, one: i32) -> i32 {
    let y = (x as f64 * one as f64).round();
    if y > i32::MAX as f64 {
        i32::MAX
    } else if y < i32::MIN as f64 {
        i32::MIN
    } else {
        y as i32
    }
}

pub(crate) fn err(msg: String) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, serialize, tensor::TensorView};

    #[test]
    fn q8_and_transpose() {
        let data: Vec<u8> = [1.0f32, -0.5, 0.25, 2.0]
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let t = TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap();
        let q = tensor_q8(&t).unwrap();
        assert_eq!(q, vec![256, -128, 64, 512]);
        assert_eq!(transpose(&q, 2, 2), vec![256, 64, -128, 512]);
    }

    #[test]
    fn reads_named_tensor_from_safetensors() {
        let data: Vec<u8> = [1.0f32, 2.0]
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let t = TensorView::new(Dtype::F32, vec![2], &data).unwrap();
        let bytes = serialize([("model.layers.0.input_layernorm.weight", t)], None).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let got = vec_q8(&st, "model.layers.0.input_layernorm.weight", &[2]).unwrap();
        assert_eq!(got, vec![256, 512]);
    }

    #[test]
    fn reads_qwen_layer0_when_present() {
        let path = std::path::Path::new("atlas-onnx-tracer/models/qwen/model.safetensors");
        if !path.exists() {
            return;
        }

        let ws = load_layer(path, 0).unwrap();
        assert_eq!(ws.len(), N);
        assert_eq!(ws[LN1].len(), crate::HIDDEN);
        assert_eq!(ws[LN2].len(), crate::HIDDEN);
        assert_eq!(ws[WQ].len(), crate::HIDDEN * crate::HIDDEN);
        assert_eq!(ws[BQ].len(), crate::HIDDEN);
        assert_eq!(
            ws[WK].len(),
            crate::HIDDEN * crate::KV_HEADS * crate::HEAD_DIM
        );
        assert_eq!(ws[BK].len(), crate::KV_HEADS * crate::HEAD_DIM);
        assert_eq!(
            ws[WV].len(),
            crate::HIDDEN * crate::KV_HEADS * crate::HEAD_DIM
        );
        assert_eq!(ws[BV].len(), crate::KV_HEADS * crate::HEAD_DIM);
        assert_eq!(ws[WO].len(), crate::HIDDEN * crate::HIDDEN);
        assert_eq!(ws[WG].len(), crate::HIDDEN * crate::INTERMEDIATE);
        assert_eq!(ws[WU].len(), crate::HIDDEN * crate::INTERMEDIATE);
        assert_eq!(ws[WD].len(), crate::INTERMEDIATE * crate::HIDDEN);
    }
}
