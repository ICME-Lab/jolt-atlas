//! Matched complete gathers with flattened addresses or projected table rows.
//! This is not full model inference, token generation or an ONNX importer.
#[cfg(feature = "zk")]
mod enabled {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use atlas_onnx_tracer::{
        ops::{Add, GatherSmall, Op},
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
    fn graph(
        rows: usize,
        width: usize,
        dictionary: usize,
        logical: usize,
        flat: bool,
    ) -> NativeGraph {
        assert!(rows.is_power_of_two() && rows <= 256);
        assert!(width.is_power_of_two() && width <= 1024);
        assert!(
            dictionary.is_power_of_two()
                && dictionary <= 1024
                && logical > 0
                && logical <= dictionary
        );
        let bounded = NativeGraphNode::lookup(
            0,
            (0..dictionary)
                .map(|i| if i < logical { i as i32 } else { -1 })
                .collect(),
            2,
        );
        if flat {
            NativeGraph {
                context: b"complete gather comparison/flat".to_vec(),
                input_shapes: vec![
                    vec![rows, 1],
                    vec![dictionary, width],
                    vec![rows, width],
                    vec![rows, width],
                ],
                nodes: vec![
                    bounded,
                    NativeGraphNode::broadcast(4, vec![rows, width]),
                    NativeGraphNode::mul(5, 3, 1),
                    NativeGraphNode::add(6, 2),
                    NativeGraphNode::add(1, 1),
                    NativeGraphNode::hidden_lookup(7, 8, 2),
                ],
                outputs: vec![9],
            }
        } else {
            NativeGraph {
                context: b"complete gather comparison/rows".to_vec(),
                input_shapes: vec![vec![rows, 1], vec![dictionary, width]],
                nodes: vec![
                    bounded,
                    NativeGraphNode::add(1, 1),
                    NativeGraphNode::gather_rows(2, 3, 2),
                ],
                outputs: vec![4],
            }
        }
    }
    fn parameters(rows: usize, width: usize, flat: bool) -> BTreeMap<usize, Vec<i32>> {
        if flat {
            BTreeMap::from([
                (2, (0..rows * width).map(|i| (i % width) as i32).collect()),
                (3, vec![2 * width as i32; rows * width]),
            ])
        } else {
            BTreeMap::new()
        }
    }
    fn input(rows: usize, logical: usize, trial: usize) -> Vec<i32> {
        (0..rows)
            .map(|i| ((i * 7 + trial) % logical) as i32)
            .collect()
    }
    fn table(dictionary: usize, width: usize, trial: usize) -> Vec<i32> {
        (0..dictionary * width)
            .map(|i| ((i * 37 + trial * 13) % 512) as i32 - 256)
            .collect()
    }
    fn reference(
        rows: usize,
        width: usize,
        dictionary: usize,
        logical: usize,
        indices: &[i32],
        data: &[i32],
    ) -> Vec<i32> {
        let x = Tensor::new(Some(data), &[dictionary, width]).unwrap();
        let y = Tensor::new(Some(indices), &[rows, 1]).unwrap();
        let twice = Add.f(vec![&x, &x]);
        GatherSmall {
            axis: 0,
            dict_len: logical,
        }
        .f(vec![&twice, &y])
        .data()
        .to_vec()
    }
    pub fn run() {
        let args = std::env::args().collect::<Vec<_>>();
        assert!(args.len()>=8,"native_gather_comparison prove|verify DIRECTORY ROWS WIDTH DICTIONARY LOGICAL flat|rows [TRUSTED_KEY_SHA3_256]");
        let dir = Path::new(&args[2]);
        let rows = args[3].parse::<usize>().unwrap();
        let width = args[4].parse::<usize>().unwrap();
        let dictionary = args[5].parse::<usize>().unwrap();
        let logical = args[6].parse::<usize>().unwrap();
        let flat = match args[7].as_str() {
            "flat" => true,
            "rows" => false,
            _ => panic!("Unknown gather variant"),
        };
        let expected = graph(rows, width, dictionary, logical, flat);
        if args[1] == "prove" {
            assert_eq!(args.len(), 8);
            assert!(!dir.exists());
            fs::create_dir_all(dir).unwrap();
            let timer = Instant::now();
            let pp = DoryScheme::setup_prover(expected.max_log_rows().unwrap() + 8);
            let gens = DoryScheme::pedersen_generators(&pp, 16);
            let setup_seconds = timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let preprocessing =
                NativeGraphPreprocessing::new(expected, parameters(rows, width, flat), &pp)
                    .unwrap();
            let parameter_seconds = timer.elapsed().as_secs_f64();
            let key = Key {
                registered: preprocessing.registered().clone(),
                gens,
            };
            let key_bytes = write(&dir.join("registered-key.bin"), &key);
            let key_digest = digest(&fs::read(dir.join("registered-key.bin")).unwrap());
            let info=format!("{{\"rows\":{rows},\"setup_seconds\":{setup_seconds},\"public_parameter_seconds\":{parameter_seconds},\"registered_key_bytes\":{key_bytes},\"registered_key_sha3_256\":\"{key_digest}\",\"scope\":\"complete matched gather comparison with registered graph, not full model inference\"}}\n");
            fs::write(dir.join("setup.json"), info).unwrap();
            let mut previous = None;
            for trial in 0..2 {
                let path = dir.join(format!("trial-{trial}"));
                fs::create_dir(&path).unwrap();
                let values = input(rows, logical, trial);
                let data = table(dictionary, width, trial);
                let expected = reference(rows, width, dictionary, logical, &values, &data);
                let input_digest = digest(
                    &values
                        .iter()
                        .chain(&data)
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                let timer = Instant::now();
                let (st, wi) = preprocessing
                    .commit(BTreeMap::from([(0, values), (1, data)]), &pp)
                    .unwrap();
                let commit_seconds = timer.elapsed().as_secs_f64();
                assert_eq!(wi.outputs(), vec![expected.clone()]);
                let output_digest = digest(
                    &expected
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                for &id in key.registered.public_inputs.keys() {
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
                let record=format!("{{\"rows\":{rows},\"width\":{width},\"dictionary\":{dictionary},\"logical\":{logical},\"flat\":{flat},\"output_sha3_256\":\"{output_digest}\",\"input_sha3_256\":\"{input_digest}\",\"trial\":{trial},\"commit_seconds\":{commit_seconds},\"prove_seconds\":{prove_seconds},\"verify_seconds\":{verify_seconds},\"proof_bytes\":{proof_bytes},\"statement_bytes\":{statement_bytes},\"expected_outputs_match\":true,\"public_commitments_reused\":true,\"private_commitments_fresh\":true}}\n");
                fs::write(path.join("measurement.json"), &record).unwrap();
                print!("{record}");
            }
        } else {
            assert_eq!(args[1], "verify");
            assert_eq!(
                args.len(),
                9,
                "Verification requires the registration digest from trusted setup"
            );
            let key_data = fs::read(dir.join("registered-key.bin")).unwrap();
            assert_eq!(
                digest(&key_data),
                args[8],
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
                let id = common::CommittedPoly::DivNodeQuotient(2);
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
