//! Complete normalization components with registered public parameters.
//! This is not full model inference, token generation or an ONNX importer.
#[cfg(feature = "zk")]
mod enabled {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use atlas_onnx_tracer::{
        ops::{Add, MeanOfSquares, Mul, Op, Rsqrt},
        tensor::Tensor,
    };
    use joltworks::{
        curve::Bn254Curve,
        poly::commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{
                native_graph::{
                    NativeGraph, NativeGraphNode, NativeGraphProof, NativeGraphStatement,
                },
                native_registration::{NativeGraphPreprocessing, NativeRegisteredGraph},
                DoryScheme,
            },
            pedersen::PedersenGenerators,
        },
    };
    use sha3::{Digest, Sha3_256};
    use std::{collections::BTreeMap, fs, path::Path, time::Instant};
    #[derive(CanonicalSerialize, CanonicalDeserialize)]
    struct Key {
        registered: NativeRegisteredGraph,
        gens: PedersenGenerators<Bn254Curve>,
    }
    fn bytes<T: CanonicalSerialize>(v: &T) -> Vec<u8> {
        let mut b = vec![];
        v.serialize_compressed(&mut b).unwrap();
        b
    }
    fn write<T: CanonicalSerialize>(p: &Path, v: &T) -> usize {
        let b = bytes(v);
        fs::write(p, &b).unwrap();
        b.len()
    }
    fn read<T: CanonicalDeserialize>(p: &Path) -> T {
        let b = fs::read(p).unwrap();
        let mut c = b.as_slice();
        let v = T::deserialize_compressed(&mut c).unwrap();
        assert!(c.is_empty());
        v
    }
    fn digest(b: &[u8]) -> String {
        Sha3_256::digest(b)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect()
    }
    fn graph(rows: usize) -> NativeGraph {
        assert!(rows.is_power_of_two() && rows <= 256);
        NativeGraph {
            context: b"registered normalization fixture".to_vec(),
            input_shapes: vec![vec![rows, 1024], vec![rows, 1], vec![rows, 1024]],
            nodes: vec![
                NativeGraphNode::mean_of_squares_with_count(0, vec![1], 14, 896),
                NativeGraphNode::add(3, 1),
                NativeGraphNode::rsqrt(4, 14),
                NativeGraphNode::broadcast(5, vec![rows, 1024]),
                NativeGraphNode::mul(0, 6, 14),
                NativeGraphNode::mul(7, 2, 14),
            ],
            outputs: vec![8],
        }
    }
    fn weights(rows: usize) -> Vec<i32> {
        (0..rows * 1024)
            .map(|i| {
                if i % 1024 < 896 {
                    16384 + ((i % 1024) % 17) as i32
                } else {
                    0
                }
            })
            .collect()
    }
    fn input(rows: usize, trial: usize) -> Vec<i32> {
        (0..rows * 1024)
            .map(|i| {
                if i % 1024 < 896 {
                    ((i * 37 + trial * 13) % 65536) as i32 - 32768
                } else {
                    0
                }
            })
            .collect()
    }
    fn reference(rows: usize, x: &[i32]) -> Vec<i32> {
        let x = Tensor::new(Some(x), &[rows, 1024]).unwrap();
        let epsilon = Tensor::new(Some(&vec![1; rows]), &[rows, 1]).unwrap();
        let w = Tensor::new(Some(&weights(rows)), &[rows, 1024]).unwrap();
        let m = MeanOfSquares {
            axes: vec![1],
            scale: 14,
            count: 896,
            padded_count: 1024,
        }
        .f(vec![&x]);
        let a = Add.f(vec![&m, &epsilon]);
        let r = Rsqrt { scale: 14 }
            .f(vec![&a])
            .expand(&[rows, 1024])
            .unwrap();
        let y = Mul { scale: 14 }.f(vec![&x, &r]);
        Mul { scale: 14 }.f(vec![&y, &w]).data().to_vec()
    }
    pub fn run() {
        let args = std::env::args().collect::<Vec<_>>();
        assert!(
            args.len() >= 4,
            "native_registered_receipt prove|verify DIRECTORY ROWS [TRUSTED_KEY_SHA3_256]"
        );
        let dir = Path::new(&args[2]);
        let rows = args[3].parse::<usize>().unwrap();
        let expected = graph(rows);
        if args[1] == "prove" {
            assert_eq!(args.len(), 4);
            assert!(!dir.exists());
            fs::create_dir_all(dir).unwrap();
            let timer = Instant::now();
            let pp = DoryScheme::setup_prover(expected.max_log_rows().unwrap() + 8);
            let gens = DoryScheme::pedersen_generators(&pp, 16);
            let setup_seconds = timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let preprocessing = NativeGraphPreprocessing::new(
                expected,
                BTreeMap::from([(1, vec![1; rows]), (2, weights(rows))]),
                &pp,
            )
            .unwrap();
            let parameter_seconds = timer.elapsed().as_secs_f64();
            let key = Key {
                registered: preprocessing.registered().clone(),
                gens,
            };
            let key_bytes = write(&dir.join("registered-key.bin"), &key);
            let key_digest = digest(&fs::read(dir.join("registered-key.bin")).unwrap());
            let info=format!("{{\"rows\":{rows},\"setup_seconds\":{setup_seconds},\"public_parameter_seconds\":{parameter_seconds},\"registered_key_bytes\":{key_bytes},\"registered_key_sha3_256\":\"{key_digest}\",\"scope\":\"complete normalization component with two public parameters, not full model inference\"}}\n");
            fs::write(dir.join("setup.json"), info).unwrap();
            let mut previous = None;
            for trial in 0..2 {
                let path = dir.join(format!("trial-{trial}"));
                fs::create_dir(&path).unwrap();
                let values = input(rows, trial);
                let expected = reference(rows, &values);
                let timer = Instant::now();
                let (st, wi) = preprocessing
                    .commit(BTreeMap::from([(0, values)]), &pp)
                    .unwrap();
                let commit_seconds = timer.elapsed().as_secs_f64();
                assert_eq!(wi.outputs(), vec![expected]);
                for id in [1, 2] {
                    assert_eq!(
                        st.input_commitment(id),
                        key.registered.public_inputs.get(&id)
                    );
                }
                if let Some(c) = previous {
                    assert_ne!(st.input_commitment(0), Some(&c));
                }
                previous = st.input_commitment(0).copied();
                let timer = Instant::now();
                let proof = NativeGraphProof::prove(&st, wi, &pp, &key.gens).unwrap();
                let prove_seconds = timer.elapsed().as_secs_f64();
                let timer = Instant::now();
                key.registered.verify(&proof, &st, &key.gens).unwrap();
                let verify_seconds = timer.elapsed().as_secs_f64();
                let proof_bytes = write(&path.join("proof.bin"), &proof);
                let statement_bytes = write(&path.join("statement.bin"), &st);
                let record=format!("{{\"rows\":{rows},\"trial\":{trial},\"commit_seconds\":{commit_seconds},\"prove_seconds\":{prove_seconds},\"verify_seconds\":{verify_seconds},\"proof_bytes\":{proof_bytes},\"statement_bytes\":{statement_bytes},\"expected_outputs_match\":true,\"public_commitments_reused\":true,\"private_commitments_fresh\":true}}\n");
                fs::write(path.join("measurement.json"), &record).unwrap();
                print!("{record}");
            }
        } else {
            assert_eq!(args[1], "verify");
            assert_eq!(
                args.len(),
                5,
                "Verification requires the registration digest from trusted setup"
            );
            let key_data = fs::read(dir.join("registered-key.bin")).unwrap();
            assert_eq!(
                digest(&key_data),
                args[4],
                "Registration digest differs from trusted setup"
            );
            let key: Key = read(&dir.join("registered-key.bin"));
            assert_eq!(bytes(&key.registered.graph), bytes(&expected));
            for trial in 0..2 {
                let path = dir.join(format!("trial-{trial}"));
                let st: NativeGraphStatement = read(&path.join("statement.bin"));
                let proof: NativeGraphProof = read(&path.join("proof.bin"));
                let timer = Instant::now();
                key.registered.verify(&proof, &st, &key.gens).unwrap();
                let seconds = timer.elapsed().as_secs_f64();
                let mut wrong = st.clone();
                wrong.graph.context.push(1);
                assert!(key.registered.verify(&proof, &wrong, &key.gens).is_err());
                let mut wrong = st.clone();
                let c = *st.input_commitment(0).unwrap();
                let id = common::CommittedPoly::DivNodeQuotient(1);
                wrong.commitments.insert(id, c);
                assert!(key.registered.verify(&proof, &wrong, &key.gens).is_err());
                let mut wrong = proof.clone();
                wrong.relations.output_claims_commitments.clear();
                assert!(key.registered.verify(&wrong, &st, &key.gens).is_err());
                let record=format!("{{\"accepted\":true,\"verify_seconds\":{seconds},\"rejections_pass\":true,\"registered_key_digest_checked\":true}}\n");
                fs::write(path.join("fresh-verification.json"), &record).unwrap();
                print!("{record}");
            }
        }
    }
}
#[cfg(feature = "zk")]
fn main() {
    enabled::run()
}
#[cfg(not(feature = "zk"))]
fn main() {
    panic!("Build with --features zk")
}
