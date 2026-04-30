use std::{error::Error, path::Path};

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use tokenizers::Tokenizer;

pub const EOS: u32 = 151_643;

pub fn tokenize(path: impl AsRef<Path>, text: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let tok = Tokenizer::from_file(path).map_err(|e| crate::weights::err(e.to_string()))?;
    let enc = tok
        .encode(text, true)
        .map_err(|e| crate::weights::err(e.to_string()))?;
    let mut ids = enc.get_ids().to_vec();
    ids.truncate(crate::SEQ);
    ids.resize(crate::SEQ, EOS);
    Ok(ids)
}

pub fn embed(safetensors_path: impl AsRef<Path>, ids: &[u32]) -> Result<Vec<i32>, Box<dyn Error>> {
    let bytes = std::fs::read(safetensors_path)?;
    embed_from_bytes(&bytes, ids)
}

pub fn embed_from_bytes(bytes: &[u8], ids: &[u32]) -> Result<Vec<i32>, Box<dyn Error>> {
    let st = SafeTensors::deserialize(bytes)?;
    embed_from_safetensors(&st, ids)
}

pub fn embed_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<i32>, Box<dyn Error>> {
    embed_from_safetensors_with_one(st, ids, crate::lut::ONE)
}

pub fn embed_from_safetensors_with_one(
    st: &SafeTensors<'_>,
    ids: &[u32],
    one: i32,
) -> Result<Vec<i32>, Box<dyn Error>> {
    if ids.len() != crate::SEQ {
        return Err(crate::weights::err(format!(
            "expected {} token ids, got {}",
            crate::SEQ,
            ids.len()
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    if t.shape() != [crate::VOCAB, crate::HIDDEN] {
        return Err(crate::weights::err(format!(
            "model.embed_tokens.weight: expected shape [{}, {}], got {:?}",
            crate::VOCAB,
            crate::HIDDEN,
            t.shape()
        )));
    }

    let mut out = Vec::with_capacity(crate::X_LEN);
    for &id in ids {
        let id = id as usize;
        if id >= crate::VOCAB {
            return Err(crate::weights::err(format!("token id {id} exceeds vocab")));
        }
        row_q(&t, id, &mut out, one)?;
    }
    Ok(out)
}

pub fn embed_text(
    tokenizer_path: impl AsRef<Path>,
    safetensors_path: impl AsRef<Path>,
    text: &str,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let ids = tokenize(tokenizer_path, text)?;
    embed(safetensors_path, &ids)
}

pub fn decode(path: impl AsRef<Path>, id: u32) -> Result<String, Box<dyn Error>> {
    let tok = Tokenizer::from_file(path).map_err(|e| crate::weights::err(e.to_string()))?;
    tok.decode(&[id], false)
        .map_err(|e| crate::weights::err(e.to_string()))
}

pub fn argmax_tied_lm_head(
    safetensors_path: impl AsRef<Path>,
    x: &[i32],
) -> Result<u32, Box<dyn Error>> {
    let bytes = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    argmax_tied_lm_head_from_safetensors(&st, x)
}

pub fn argmax_tied_lm_head_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[i32],
) -> Result<u32, Box<dyn Error>> {
    let scores = tied_lm_head_scores_from_safetensors(st, x)?;
    Ok(scores
        .iter()
        .enumerate()
        .max_by_key(|&(_, s)| s)
        .map(|(i, _)| i as u32)
        .unwrap())
}

pub fn tied_lm_head_scores_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[i32],
) -> Result<Vec<i64>, Box<dyn Error>> {
    if x.len() != crate::HIDDEN {
        return Err(crate::weights::err(format!(
            "expected hidden vector length {}, got {}",
            crate::HIDDEN,
            x.len()
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    if t.shape() != [crate::VOCAB, crate::HIDDEN] {
        return Err(crate::weights::err(format!(
            "model.embed_tokens.weight: expected shape [{}, {}], got {:?}",
            crate::VOCAB,
            crate::HIDDEN,
            t.shape()
        )));
    }

    let mut row = Vec::with_capacity(crate::HIDDEN);
    let mut scores = Vec::with_capacity(crate::VOCAB);
    for id in 0..crate::VOCAB {
        row.clear();
        row_q8(&t, id, &mut row)?;
        let mut score = 0i64;
        for (&a, &b) in x.iter().zip(&row) {
            score += a as i64 * b as i64;
        }
        scores.push(score >> crate::lut::S);
    }
    Ok(scores)
}

pub fn nll_tied_lm_head_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[i32],
    target: u32,
) -> Result<f64, Box<dyn Error>> {
    if x.len() != crate::HIDDEN {
        return Err(crate::weights::err(format!(
            "expected hidden vector length {}, got {}",
            crate::HIDDEN,
            x.len()
        )));
    }
    if target as usize >= crate::VOCAB {
        return Err(crate::weights::err(format!(
            "target token id {target} exceeds vocab"
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    if t.shape() != [crate::VOCAB, crate::HIDDEN] {
        return Err(crate::weights::err(format!(
            "model.embed_tokens.weight: expected shape [{}, {}], got {:?}",
            crate::VOCAB,
            crate::HIDDEN,
            t.shape()
        )));
    }

    let scores = tied_lm_head_scores_from_safetensors(st, x)?
        .into_iter()
        .map(|s| s as f64 / crate::lut::ONE as f64)
        .collect::<Vec<_>>();
    let target_score = scores[target as usize];
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let sum_exp = scores.iter().map(|s| (s - max_score).exp()).sum::<f64>();
    Ok(max_score + sum_exp.ln() - target_score)
}

pub fn perplexity_tied_lm_head_from_safetensors(
    st: &SafeTensors<'_>,
    hidden: &[i32],
    ids: &[u32],
) -> Result<f64, Box<dyn Error>> {
    perplexity_tied_lm_head_prefix_from_safetensors(st, hidden, ids, usize::MAX)
}

pub fn perplexity_tied_lm_head_prefix_from_safetensors(
    st: &SafeTensors<'_>,
    hidden: &[i32],
    ids: &[u32],
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    if hidden.len() != crate::X_LEN {
        return Err(crate::weights::err(format!(
            "expected hidden length {}, got {}",
            crate::X_LEN,
            hidden.len()
        )));
    }
    if ids.len() != crate::SEQ {
        return Err(crate::weights::err(format!(
            "expected {} token ids, got {}",
            crate::SEQ,
            ids.len()
        )));
    }

    let mut nll = 0.0;
    let mut n = 0usize;
    for pos in 0..crate::SEQ - 1 {
        if n >= max_targets {
            break;
        }
        if ids[pos + 1] == EOS {
            break;
        }
        let x = &hidden[pos * crate::HIDDEN..(pos + 1) * crate::HIDDEN];
        nll += nll_tied_lm_head_from_safetensors(st, x, ids[pos + 1])?;
        n += 1;
    }
    if n == 0 {
        return Err(crate::weights::err(
            "no target tokens for perplexity".to_string(),
        ));
    }
    Ok((nll / n as f64).exp())
}

pub fn nll_from_scores(scores: &[f64], target: usize) -> f64 {
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = scores.iter().map(|s| (s - max_score).exp()).sum::<f64>();
    max_score + sum_exp.ln() - scores[target]
}

pub(crate) fn row_q8(
    t: &TensorView<'_>,
    row: usize,
    out: &mut Vec<i32>,
) -> Result<(), Box<dyn Error>> {
    row_q(t, row, out, crate::lut::ONE)
}

pub(crate) fn row_q(
    t: &TensorView<'_>,
    row: usize,
    out: &mut Vec<i32>,
    one: i32,
) -> Result<(), Box<dyn Error>> {
    let n = crate::HIDDEN;
    match t.dtype() {
        Dtype::BF16 => {
            let start = row * n * 2;
            for b in t.data()[start..start + n * 2].chunks_exact(2) {
                out.push(crate::weights::q_to(
                    bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
                    one,
                ));
            }
        }
        Dtype::F16 => {
            let start = row * n * 2;
            for b in t.data()[start..start + n * 2].chunks_exact(2) {
                out.push(crate::weights::q_to(
                    f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
                    one,
                ));
            }
        }
        Dtype::F32 => {
            let start = row * n * 4;
            for b in t.data()[start..start + n * 4].chunks_exact(4) {
                out.push(crate::weights::q_to(
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    one,
                ));
            }
        }
        Dtype::I32 => {
            let start = row * n * 4;
            for b in t.data()[start..start + n * 4].chunks_exact(4) {
                out.push(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
        }
        dt => {
            return Err(crate::weights::err(format!(
                "unsupported embedding dtype {dt:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn tokenizes_to_fixed_seq() {
        let ids = tokenize(
            root().join("atlas-onnx-tracer/models/qwen/tokenizer.json"),
            "hello",
        )
        .unwrap();
        assert_eq!(ids.len(), crate::SEQ);
        assert!(ids.iter().any(|&id| id != EOS));
    }

    #[test]
    fn embeds_text_when_weights_present() {
        let root = root();
        let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");
        if !model.exists() {
            return;
        }
        let x = embed_text(
            root.join("atlas-onnx-tracer/models/qwen/tokenizer.json"),
            model,
            "hello",
        )
        .unwrap();
        assert_eq!(x.len(), crate::X_LEN);
        assert!(x.iter().any(|&x| x != 0));
    }

    #[test]
    fn decode_eos_token() {
        let s = decode(
            root().join("atlas-onnx-tracer/models/qwen/tokenizer.json"),
            EOS,
        )
        .unwrap();
        assert!(!s.is_empty());
    }

    #[test]
    fn nll_from_scores_matches_uniform() {
        let nll = nll_from_scores(&[0.0, 0.0, 0.0, 0.0], 2);
        assert!((nll - 4f64.ln()).abs() < 1e-12);
    }
}
