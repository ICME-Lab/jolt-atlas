//! Benchmark: BlindFold (zk) vs non-zk proving on GPT-2.
//!
//! Runs both proving paths back-to-back on the same model and inputs and
//! reports prover time, verifier time, and proof size for each.
//!
//! Run with:
//! ```bash
//! cargo run --release --features zk --package jolt-atlas-core --example gpt2_zk_bench
//! ```

#[cfg(not(feature = "zk"))]
fn main() {
    eprintln!("This example requires the `zk` feature. Re-run with:");
    eprintln!(
        "  cargo run --release --features zk --package jolt-atlas-core --example gpt2_zk_bench"
    );
    std::process::exit(1);
}

#[cfg(feature = "zk")]
fn main() {
    // Configure the global rayon pool BEFORE any other rayon use. Two things to
    // cap:
    // - stack size: the ZK pipeline does deep MLE work that overflows rayon's
    //   default 2 MB worker stack on GPT-2-sized polynomials.
    // - thread count: the patched arkworks MSM
    //   (`ec/src/scalar_mul/variable_base/mod.rs:813-857`) builds a fresh
    //   nested 2-thread `rayon::ThreadPoolBuilder` per MSM chunk per call,
    //   spawning `current_num_threads / 2` chunks each time. With default
    //   rayon (≈ num CPUs) and many MSMs in rapid succession we hit macOS's
    //   per-process pthread limit (`pthread_create` → EAGAIN).
    //
    // Setting this programmatically here removes the need for `RUST_MIN_STACK`
    // and `RAYON_NUM_THREADS` env vars on the command line. `build_global`
    // errors if rayon was already initialised; that's fine, we just leave the
    // existing pool in place (user-set env vars win).
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .stack_size(32 * 1024 * 1024)
        .build_global();

    use atlas_onnx_tracer::{
        model::{Model, RunArgs},
        tensor::Tensor,
    };
    use common::utils::logging::setup_tracing;
    use jolt_atlas_core::onnx_proof::{
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
        Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (_guard, _tracing_enabled) = setup_tracing("gpt2 zk-vs-nonzk bench");

    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);
    let model = Model::load("atlas-onnx-tracer/models/gpt2/model.onnx", &run_args);
    println!("max num vars: {}", model.max_num_vars());

    let mut rng = StdRng::seed_from_u64(42);
    let vocab_size: i32 = 50257;
    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();
    let scale = run_args.scale;
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();
    let inputs = [input_ids, position_ids, attention_mask];

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // ── Non-ZK path ──────────────────────────────────────────────────────
    println!("\n=== non-zk ===");
    let t = std::time::Instant::now();
    let (proof, io, _dbg) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &inputs);
    let nonzk_prove = t.elapsed();
    println!("prove:  {:.2?}", nonzk_prove);

    let nonzk_proof_bytes = {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::new();
        proof
            .serialize_compressed(&mut buf)
            .expect("serialize non-zk proof");
        buf.len()
    };
    println!(
        "size:   {} bytes ({:.2} KiB)",
        nonzk_proof_bytes,
        nonzk_proof_bytes as f64 / 1024.0
    );

    let t = std::time::Instant::now();
    proof.verify(&verifier_pp, &io, None).unwrap();
    let nonzk_verify = t.elapsed();
    println!("verify: {:.2?}", nonzk_verify);

    drop(proof);

    // ── ZK path (BlindFold) ──────────────────────────────────────────────
    println!("\n=== zk (BlindFold) ===");
    let gens = joltworks::poly::commitment::pedersen::PedersenGenerators::<
        joltworks::curve::Bn254Curve,
    >::deterministic(4096);

    let t = std::time::Instant::now();
    let (bundle, zk_io) = jolt_atlas_core::onnx_proof::zk::prove_zk(&prover_pp, &inputs, &gens);
    let zk_prove = t.elapsed();
    println!("prove:  {:.2?}", zk_prove);

    let zk_proof_bytes = zk_bundle_size(&bundle);
    println!(
        "size:   {} bytes ({:.2} KiB)  [excludes verifier-reconstructable stage_configs/baked]",
        zk_proof_bytes,
        zk_proof_bytes as f64 / 1024.0
    );

    let t = std::time::Instant::now();
    jolt_atlas_core::onnx_proof::zk::verify_zk(&bundle, &verifier_pp, &zk_io, &gens)
        .expect("ZK verification should succeed");
    let zk_verify = t.elapsed();
    println!("verify: {:.2?}", zk_verify);

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n=== summary ===");
    println!(
        "prove:  non-zk {:.2?}  |  zk {:.2?}  |  ratio {:.2}x",
        nonzk_prove,
        zk_prove,
        zk_prove.as_secs_f64() / nonzk_prove.as_secs_f64()
    );
    println!(
        "verify: non-zk {:.2?}  |  zk {:.2?}  |  ratio {:.2}x",
        nonzk_verify,
        zk_verify,
        zk_verify.as_secs_f64() / nonzk_verify.as_secs_f64()
    );
    println!(
        "size:   non-zk {} B  |  zk {} B  |  ratio {:.2}x",
        nonzk_proof_bytes,
        zk_proof_bytes,
        zk_proof_bytes as f64 / nonzk_proof_bytes as f64
    );
}

/// Size of the zk proof bundle in bytes (compressed).
///
/// Counts only the wire-protocol components. Skips `stage_configs`, `baked`, and
/// `zk_sumcheck_num_instances`, which the verifier reconstructs from the
/// transcript / R1CS layout.
#[cfg(feature = "zk")]
fn zk_bundle_size(b: &jolt_atlas_core::onnx_proof::zk::ZkProofBundle) -> usize {
    use ark_serialize::CanonicalSerialize;

    let mut n = 0;
    n += b.blindfold_proof.compressed_size();

    // BlindFoldVerifierInput: three Vec<C::G1>.
    let bvi = &b.blindfold_verifier_input;
    n += bvi.round_commitments.compressed_size();
    n += bvi.output_claims_row_commitments.compressed_size();
    n += bvi.eval_commitments.compressed_size();

    // BTreeMap<usize, EvalReductionProof<F>> — sum keys + values.
    for (k, v) in &b.eval_reduction_proofs {
        n += k.compressed_size();
        n += v.compressed_size();
    }
    for (k, v) in &b.eval_reduction_h_commitments {
        n += k.compressed_size();
        n += v.compressed_size();
    }

    n += b.commitments.compressed_size();
    n += b.output_claim.compressed_size();

    // Vec<(usize, ZkSumcheckProof)> — fields are pub, inline-size.
    for (idx, zk) in &b.zk_sumcheck_proofs {
        n += idx.compressed_size();
        n += zk.round_commitments.compressed_size();
        n += zk.poly_degrees.compressed_size();
        n += zk.output_claims_commitments.compressed_size();
    }

    for (k, v) in &b.inter_stage_commitments {
        n += k.compressed_size();
        n += v.compressed_size();
    }
    for (k, v) in &b.auxiliary_claims {
        n += k.compressed_size();
        n += v.compressed_size();
    }
    n
}
