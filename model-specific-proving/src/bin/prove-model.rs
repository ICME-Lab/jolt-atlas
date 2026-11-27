use model_specific_proving::{protocol::TopKConfig, weights::ModelWeights};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [model, bits, ids @ ..] = &args[..] else {
        return Err("usage: prove-model MODEL_DIR FRACTIONAL_BITS TOKEN_ID...".into());
    };
    let tokens: Vec<usize> = ids.iter().map(|id| id.parse()).collect::<Result<_, _>>()?;
    let weights = ModelWeights::from_safetensors(Path::new(model), bits.parse()?)?;
    let model = weights.model(tokens.len())?;
    let vocab = model.vocab_size();
    let mut proof = tempfile::tempfile()?;
    let mut stdout = std::io::stdout().lock();
    if model.is_decoder() {
        let cfg = TopKConfig {
            positions: tokens.len().saturating_sub(4)..tokens.len(),
            k: 8.min(vocab),
            rows_per_block: 4,
            vocab_per_block: 1 << 14,
        };
        let output = model.prove_top_k_to(&tokens, &cfg, &mut proof)?;
        model.verify_top_k_from(&tokens, &cfg, &output, proof)?;
        serde_json::to_writer(&mut stdout, &output)?;
    } else {
        let output = model.prove_to(&tokens, None, &mut proof)?;
        model.verify_from(&tokens, None, &output, proof)?;
        serde_json::to_writer(&mut stdout, &output)?;
    }
    println!();
    Ok(())
}
