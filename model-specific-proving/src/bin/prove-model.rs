use model_specific_proving::{protocol::TopKConfig, weights::ModelWeights};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [directory, precision, token_ids @ ..] = args.as_slice() else {
        return Err("usage: prove-model MODEL_DIR FRACTIONAL_BITS TOKEN_ID...".into());
    };
    let tokens: Vec<usize> = token_ids
        .iter()
        .map(|id| id.parse())
        .collect::<Result<_, _>>()?;
    let weights = ModelWeights::from_safetensors(Path::new(directory), precision.parse()?)?;
    let model = weights.model(tokens.len())?;
    let mut proof = tempfile::tempfile()?;
    let stdout = std::io::stdout().lock();
    if model.is_decoder() {
        let config = TopKConfig {
            positions: tokens.len().saturating_sub(4)..tokens.len(),
            k: 8,
            rows_per_block: 4,
            vocab_per_block: 1 << 14,
        };
        let candidates = model.prove_top_k_to(&tokens, &config, &mut proof)?;
        model.verify_top_k_from(&tokens, &config, &candidates, proof)?;
        serde_json::to_writer(stdout, &candidates)?;
    } else {
        let output = model.prove_to(&tokens, None, &mut proof)?;
        model.verify_from(&tokens, None, &output, proof)?;
        serde_json::to_writer(stdout, &output)?;
    }
    println!();
    Ok(())
}
