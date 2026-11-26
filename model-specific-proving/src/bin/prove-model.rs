use model_specific_proving::weights::ModelWeights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: prove-model MODEL_DIR FRACTIONAL_BITS TOKEN_ID...")?;
    let bits = args.next().ok_or("missing fractional bits")?.parse()?;
    let tokens = args
        .map(|arg| arg.parse())
        .collect::<Result<Vec<usize>, _>>()?;
    let weights = ModelWeights::from_safetensors(path.as_ref(), bits)?;
    let model = weights.model(tokens.len())?;
    let (output, proof) = model.prove(&tokens, None)?;
    model.verify(&tokens, None, &output, &proof)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
