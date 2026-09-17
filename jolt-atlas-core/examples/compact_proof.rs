//! Convert a canonical compressed BN254/HyperKZG proof to the opt-in compact
//! format and compare decoding costs. This example checks byte preservation;
//! verification still requires the model preprocessing and public IO.
use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use jolt_atlas_core::onnx_proof::ONNXProof;
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
use std::{fs, hint::black_box, io::Write, time::Instant};

type Proof = ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>>;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "compact_proof INPUT NEW_OUTPUT");
    let original = fs::read(&args[1]).unwrap();
    let mut remaining = original.as_slice();
    let proof = Proof::deserialize_compressed(&mut remaining).unwrap();
    assert!(remaining.is_empty());
    let compact = proof.serialize_compact(Compress::Yes).unwrap();
    let limit = proof.opening_claims.0.len();
    let decoded = Proof::deserialize_compact(&compact, limit).unwrap();
    let mut recovered = Vec::new();
    decoded.serialize_compressed(&mut recovered).unwrap();
    assert_eq!(recovered, original);
    let mut legacy_ns = Vec::new();
    let mut compact_ns = Vec::new();
    // Alternate order to reduce systematic warm-cache/order bias. These are
    // native decode timings, not proof verification or recursive proving.
    for iteration in 0..10 {
        for compact_first in [iteration % 2 == 0, iteration % 2 != 0] {
            let start = Instant::now();
            let decoded = if compact_first {
                Proof::deserialize_compact(black_box(&compact), limit).unwrap()
            } else {
                Proof::deserialize_compressed(black_box(original.as_slice())).unwrap()
            };
            let elapsed = start.elapsed().as_nanos();
            black_box(decoded);
            if compact_first {
                compact_ns.push(elapsed);
            } else {
                legacy_ns.push(elapsed);
            }
        }
    }
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .unwrap();
    out.write_all(&compact).unwrap();
    println!("{{\"canonical_bytes\":{},\"compact_bytes\":{},\"claims\":{},\"legacy_decode_ns\":{:?},\"compact_decode_ns\":{:?},\"exact_roundtrip\":true}}", original.len(), compact.len(), limit, legacy_ns, compact_ns);
}
