use model_specific_proving::weights::ModelWeights;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [directory, precision, token_ids @ ..] = args.as_slice() else {
        return Err("usage: prove-model MODEL_DIR FRACTIONAL_BITS TOKEN_ID...".into());
    };
    let sequence: Vec<usize> = token_ids
        .iter()
        .map(|id| id.parse())
        .collect::<Result<_, _>>()?;
    let weights = ModelWeights::from_safetensors(Path::new(directory), precision.parse()?)?;
    let instance = weights.model(sequence.len())?;
    let (logits, proof) = instance.prove(&sequence, None)?;
    instance.verify(&sequence, None, &logits, &proof)?;
    serde_json::to_writer(std::io::stdout().lock(), &logits)?;
    println!();
    Ok(())
}
