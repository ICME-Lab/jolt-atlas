//! Per-model input construction, shared by every bench binary. `MODEL_SCALE` is compile-time
//! (GPT-2 12, Qwen 14) — each `load_<model>` asserts its requirement. Takes a `Vec<i32>` of
//! already-decided token ids (random via [`random_input_ids`] or real text via
//! [`qwen_token_ids`]); not every binary needs every fn here, hence `allow(dead_code)`.
#![allow(dead_code)]

use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use jolt_atlas_core::onnx_proof::AtlasSharedPreprocessing;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct LoadedModel {
    pub label: &'static str,
    pub shared: AtlasSharedPreprocessing,
    pub inputs: Vec<Tensor<i32>>,
    pub seq_len: usize,
    pub scale: usize,
    pub max_num_vars: usize,
}

pub const GPT2_VOCAB_SIZE: i32 = 50257;
const GPT2_MODEL_PATH: &str = "atlas-onnx-tracer/models/gpt2/network.onnx";
const GPT2_TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/gpt2/tokenizer.json";

pub const BGE_VOCAB_SIZE: i32 = 30522;
const BGE_MODEL_PATH: &str = "atlas-onnx-tracer/models/bge-small-en-v1.5/network.onnx";

pub const QWEN_VOCAB_SIZE: i32 = 151936;
const QWEN_MODEL_PATH: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const QWEN_TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";

pub fn random_input_ids(seq_len: usize, vocab_size: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect()
}

fn assert_valid_token_ids(model: &str, input_ids: &[i32], vocab_size: i32) {
    for &id in input_ids {
        assert!(
            (0..vocab_size).contains(&id),
            "{model}: token id {id} is out of range for vocab_size {vocab_size}"
        );
    }
}

pub fn qwen_token_ids(input_text: &str) -> Vec<i32> {
    tokenize(
        QWEN_TOKENIZER_PATH,
        "run scripts/download_qwen.py first",
        input_text,
    )
}

pub fn gpt2_token_ids(input_text: &str) -> Vec<i32> {
    tokenize(
        GPT2_TOKENIZER_PATH,
        "run scripts/download_gpt2.py first",
        input_text,
    )
}

fn tokenize(tokenizer_path: &str, install_hint: &str, input_text: &str) -> Vec<i32> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|e| panic!("failed to load {tokenizer_path}: {e} - {install_hint}"));
    let encoding = tokenizer
        .encode(input_text, false)
        .expect("tokenization failed");
    encoding.get_ids().iter().map(|&id| id as i32).collect()
}

pub fn load_gpt2(input_ids_data: Vec<i32>) -> LoadedModel {
    assert_eq!(
        common::consts::MODEL_SCALE,
        12,
        "GPT-2 exports at scale 12; currently compiled with MODEL_SCALE = {}. Edit \
         common/src/consts/general.rs and recompile.",
        common::consts::MODEL_SCALE
    );
    assert_valid_token_ids("gpt2", &input_ids_data, GPT2_VOCAB_SIZE);

    let seq_len = input_ids_data.len();
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let model = Model::load(GPT2_MODEL_PATH, &run_args);
    let max_num_vars = model.max_num_vars();
    let scale = run_args.scale;

    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();
    // Cast handler de-quantizes by scale, so the mask must be supplied pre-quantized.
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    LoadedModel {
        label: "gpt2",
        shared: AtlasSharedPreprocessing::preprocess(model),
        inputs: vec![input_ids, position_ids, attention_mask],
        seq_len,
        scale: scale as usize,
        max_num_vars,
    }
}

pub fn load_bge(input_ids_data: Vec<i32>) -> LoadedModel {
    assert_valid_token_ids("bge", &input_ids_data, BGE_VOCAB_SIZE);

    let seq_len = input_ids_data.len();
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let model = Model::load(BGE_MODEL_PATH, &run_args);
    let max_num_vars = model.max_num_vars();
    let scale = run_args.scale;

    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();
    // BERT-style binary attention mask (not fixed-point quantized, unlike GPT-2/Qwen).
    let attention_mask_data: Vec<i32> = vec![1; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    LoadedModel {
        label: "bge",
        shared: AtlasSharedPreprocessing::preprocess(model),
        inputs: vec![input_ids, attention_mask],
        seq_len,
        scale: scale as usize,
        max_num_vars,
    }
}

pub fn load_qwen(input_ids_data: Vec<i32>) -> LoadedModel {
    assert_eq!(
        common::consts::MODEL_SCALE,
        14,
        "Qwen exports at scale 14; currently compiled with MODEL_SCALE = {}. Edit \
         common/src/consts/general.rs and recompile.",
        common::consts::MODEL_SCALE
    );
    assert_valid_token_ids("qwen", &input_ids_data, QWEN_VOCAB_SIZE);

    let seq_len = input_ids_data.len();
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let scale = run_args.scale;

    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    // Qwen export input 1 is `position_ids`, fixed-point (i << scale) so RoPE's
    // Einsum(inv_freq, pos)/2^scale = inv_freq * i.
    let max_pos = (i32::MAX >> scale) as usize;
    assert!(
        seq_len.saturating_sub(1) <= max_pos,
        "seq_len={seq_len} too large for scale={scale} (max position index {max_pos})"
    );
    let position_ids_data: Vec<i32> = (0..seq_len as i32).map(|i| i << scale).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    let model = Model::load(QWEN_MODEL_PATH, &run_args);
    let max_num_vars = model.max_num_vars();

    LoadedModel {
        label: "qwen",
        shared: AtlasSharedPreprocessing::preprocess(model),
        inputs: vec![input_ids, position_ids, attention_mask],
        seq_len,
        scale: scale as usize,
        max_num_vars,
    }
}
