use std::{error::Error, path::Path, time::Instant};

use jolt_akita::{AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme};
use model_specific_proving::{protocol::AkitaAdviceSetup, weights::ModelWeights};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1).peekable();
    let metal = args.peek().is_some_and(|arg| arg == "--metal");
    if metal {
        args.next();
    }
    let model_dir = args
        .next()
        .ok_or("usage: prove-gpt2 [--metal] MODEL_DIR FRACTIONAL_BITS TOKEN_ID...")?;
    let fractional_bits = args.next().ok_or("missing fractional bits")?.parse()?;
    let tokens = args
        .map(|arg| arg.parse())
        .collect::<Result<Vec<usize>, _>>()?;
    let weights = ModelWeights::from_safetensors(Path::new(&model_dir), fractional_bits)?;
    let model = weights.model(tokens.len())?;
    let schedules =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    let prover = AkitaAdviceSetup::new(schedules.clone())?;
    #[cfg(all(feature = "akita-metal", target_os = "macos"))]
    let prover = if metal { prover.with_metal()? } else { prover };
    #[cfg(not(all(feature = "akita-metal", target_os = "macos")))]
    if metal {
        return Err("--metal requires macOS and the akita-metal feature".into());
    }
    let start = Instant::now();
    let (logits, proof) = model.prove::<AkitaScheme>(&tokens, None, &prover)?;
    println!("prove: {:.3} s", start.elapsed().as_secs_f64());
    drop(prover);
    let verifier = AkitaAdviceSetup::new(schedules)?;
    let start = Instant::now();
    model.verify(&tokens, None, &logits, &proof, &verifier)?;
    println!("verify: {:.3} s", start.elapsed().as_secs_f64());
    println!("{}", serde_json::to_string(&logits)?);
    Ok(())
}
