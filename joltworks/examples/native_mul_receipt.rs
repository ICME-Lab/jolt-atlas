//! Save a complete native fused-Mul component and verify it in a fresh process.
//! This is an operator fixture, not a Qwen inference or generation benchmark.
#[cfg(feature = "zk")]
mod enabled {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use atlas_onnx_tracer::{
        ops::{Mul, Op},
        tensor::Tensor,
    };
    use joltworks::{
        curve::Bn254Curve,
        poly::commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{
                native_mul::{NativeMulProof, NativeMulStatement, NativeMulWitness},
                DoryScheme, DoryVerifierSetup,
            },
            pedersen::PedersenGenerators,
        },
    };
    use std::{fs, path::Path, time::Instant};

    #[derive(CanonicalSerialize, CanonicalDeserialize)]
    struct Key {
        setup: DoryVerifierSetup,
        gens: PedersenGenerators<Bn254Curve>,
    }
    fn write<T: CanonicalSerialize>(path: &Path, value: &T) -> usize {
        let mut bytes = vec![];
        value.serialize_compressed(&mut bytes).unwrap();
        fs::write(path, &bytes).unwrap();
        bytes.len()
    }
    fn read<T: CanonicalDeserialize>(path: &Path) -> T {
        let bytes = fs::read(path).unwrap();
        let mut cursor = bytes.as_slice();
        let value = T::deserialize_compressed(&mut cursor).unwrap();
        assert!(cursor.is_empty());
        value
    }
    pub fn run() {
        let args = std::env::args().collect::<Vec<_>>();
        assert!(
            args.len() >= 3,
            "native_mul_receipt prove|verify DIRECTORY [ROWS]"
        );
        let directory = Path::new(&args[2]);
        if args[1] == "prove" {
            let rows = args.get(3).map_or(8, |s| s.parse::<usize>().unwrap());
            assert!(rows.is_power_of_two());
            let a = [i32::MIN, i32::MAX, i32::MIN, -1, 1, 16385, -16385, 0];
            let b = [i32::MIN, i32::MAX, i32::MAX, 1, -1, 16384, 16384, i32::MAX];
            let left = (0..rows).map(|i| a[i % a.len()]).collect::<Vec<_>>();
            let right = (0..rows).map(|i| b[i % b.len()]).collect::<Vec<_>>();
            let timer = Instant::now();
            let pp = DoryScheme::setup_prover(rows.ilog2() as usize + 8);
            let key = Key {
                setup: DoryScheme::setup_verifier(&pp),
                gens: DoryScheme::pedersen_generators(&pp, 16),
            };
            let setup_seconds = timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let (statement, witness) = NativeMulWitness::commit(
                b"native fused Mul component fixture".to_vec(),
                &left,
                &right,
                14,
                &pp,
            )
            .unwrap();
            let commit_seconds = timer.elapsed().as_secs_f64();
            let lt = Tensor::new(Some(&left), &[rows]).unwrap();
            let rt = Tensor::new(Some(&right), &[rows]).unwrap();
            assert_eq!(witness.output(), Mul { scale: 14 }.f(vec![&lt, &rt]).data());
            let timer = Instant::now();
            let proof = NativeMulProof::prove(&statement, witness, &pp, &key.gens).unwrap();
            let prove_seconds = timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            proof.verify(&statement, &key.setup, &key.gens).unwrap();
            let verify_seconds = timer.elapsed().as_secs_f64();
            fs::create_dir_all(directory).unwrap();
            let proof_bytes = write(&directory.join("proof.bin"), &proof);
            let statement_bytes = write(&directory.join("statement.bin"), &statement);
            let key_bytes = write(&directory.join("verifier.bin"), &key);
            let record=format!("{{\n  \"scope\": \"complete native fused-Mul component, not Qwen inference or provenance\",\n  \"rows\": {rows},\n  \"shift\": 14,\n  \"setup_seconds\": {setup_seconds},\n  \"commit_seconds\": {commit_seconds},\n  \"prove_seconds\": {prove_seconds},\n  \"verify_seconds\": {verify_seconds},\n  \"proof_bytes\": {proof_bytes},\n  \"statement_bytes\": {statement_bytes},\n  \"verifier_bytes\": {key_bytes},\n  \"atlas_output_matches\": true\n}}\n");
            fs::write(directory.join("measurement.json"), &record).unwrap();
            print!("{record}");
        } else {
            assert_eq!(args[1], "verify");
            let statement: NativeMulStatement = read(&directory.join("statement.bin"));
            let proof: NativeMulProof = read(&directory.join("proof.bin"));
            let key: Key = read(&directory.join("verifier.bin"));
            assert_eq!(statement.context, b"native fused Mul component fixture");
            assert_eq!(statement.shift, 14);
            let timer = Instant::now();
            proof.verify(&statement, &key.setup, &key.gens).unwrap();
            let verify_seconds = timer.elapsed().as_secs_f64();
            let mut changed = statement.clone();
            changed.shift = 13;
            assert!(proof.verify(&changed, &key.setup, &key.gens).is_err());
            let mut changed = statement.clone();
            changed.context.push(1);
            assert!(proof.verify(&changed, &key.setup, &key.gens).is_err());
            let mut missing = proof.clone();
            missing.pcs.0.y_com = None;
            assert!(missing.verify(&statement, &key.setup, &key.gens).is_err());
            let record=format!("{{\"accepted\":true,\"verify_seconds\":{verify_seconds},\"changed_shift_rejected\":true,\"changed_context_rejected\":true,\"missing_pcs_rejected\":true}}\n");
            fs::write(directory.join("fresh-verification.json"), &record).unwrap();
            print!("{record}");
        }
    }
}
#[cfg(feature = "zk")]
fn main() {
    enabled::run();
}
#[cfg(not(feature = "zk"))]
fn main() {
    panic!("Build with --features zk");
}
