//! Complete small vector graphs, not Qwen inference or provenance benchmarks.
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
                native_graph::{
                    NativeGraph, NativeGraphNode, NativeGraphProof, NativeGraphStatement,
                    NativeGraphWitness,
                },
                native_lookup::{
                    NativeLookupChainProof, NativeLookupChainStatement, NativeLookupWitness,
                },
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
    fn bytes<T: CanonicalSerialize>(v: &T) -> Vec<u8> {
        let mut b = vec![];
        v.serialize_compressed(&mut b).unwrap();
        b
    }
    fn write<T: CanonicalSerialize>(path: &Path, v: &T) -> usize {
        let b = bytes(v);
        fs::write(path, &b).unwrap();
        b.len()
    }
    fn read<T: CanonicalDeserialize>(path: &Path) -> T {
        let b = fs::read(path).unwrap();
        let mut c = b.as_slice();
        let v = T::deserialize_compressed(&mut c).unwrap();
        assert!(c.is_empty());
        v
    }
    fn graph(kind: &str, rows: usize) -> NativeGraph {
        assert!(rows >= 2 && rows.is_power_of_two());
        let (num_inputs, nodes, outputs) = match kind {
            "mixed" => (
                2,
                vec![
                    NativeGraphNode::mul(0, 1, 14),
                    NativeGraphNode::lookup(2, (0..8).rev().collect(), 2),
                    NativeGraphNode::mul(3, 1, 14),
                    NativeGraphNode::lookup(2, (0..8).map(|v| v * v).collect(), 2),
                    NativeGraphNode::mul(4, 5, 1),
                ],
                vec![6, 3],
            ),
            "table-chain" => (
                1,
                vec![
                    NativeGraphNode::lookup(0, (0..8).rev().collect(), 2),
                    NativeGraphNode::lookup(1, (0..8).map(|v| (v + 1) % 8).collect(), 2),
                    NativeGraphNode::lookup(2, (0..8).map(|v| v ^ 3).collect(), 2),
                    NativeGraphNode::lookup(3, (0..8).map(|v| (v * 3) % 8).collect(), 2),
                ],
                vec![4],
            ),
            _ => panic!("unknown fixture"),
        };
        NativeGraph {
            context: format!("native vector graph fixture/{kind}").into_bytes(),
            log_rows: rows.ilog2() as usize,
            num_inputs,
            nodes,
            outputs,
        }
    }
    fn inputs(g: &NativeGraph) -> Vec<Vec<i32>> {
        let mut inputs = vec![(0usize..1usize << g.log_rows)
            .map(|i| (i % 8) as i32)
            .collect()];
        if g.num_inputs == 2 {
            inputs.push(vec![1 << 14; 1 << g.log_rows]);
        }
        inputs
    }
    fn reference(g: &NativeGraph) -> Vec<Vec<i32>> {
        let mut values = inputs(g);
        for n in &g.nodes {
            let out = if let Some(m) = &n.mul {
                let a = Tensor::new(Some(&values[n.input]), &[1 << g.log_rows]).unwrap();
                let b = Tensor::new(Some(&values[m.right]), &[1 << g.log_rows]).unwrap();
                Mul {
                    scale: i32::from(m.shift),
                }
                .f(vec![&a, &b])
                .data()
                .to_vec()
            } else {
                values[n.input]
                    .iter()
                    .map(|v| n.lookup.as_ref().unwrap().table[*v as usize])
                    .collect()
            };
            values.push(out);
        }
        g.outputs.iter().map(|i| values[*i].clone()).collect()
    }
    pub fn run() {
        let args = std::env::args().collect::<Vec<_>>();
        assert_eq!(args.len(),5,"native_graph_receipt prove|verify|prove-chain|verify-chain DIRECTORY mixed|table-chain ROWS");
        let directory = Path::new(&args[2]);
        let kind = &args[3];
        let rows = args[4].parse::<usize>().unwrap();
        let expected = graph(kind, rows);
        let chain = args[1].ends_with("-chain");
        if chain {
            assert_eq!(kind, "table-chain");
        }
        if args[1] == "prove" || args[1] == "prove-chain" {
            let timer = Instant::now();
            let pp = DoryScheme::setup_prover(expected.log_rows + 8);
            let key = Key {
                setup: DoryScheme::setup_verifier(&pp),
                gens: DoryScheme::pedersen_generators(&pp, 16),
            };
            let setup_seconds = timer.elapsed().as_secs_f64();
            fs::create_dir_all(directory).unwrap();
            let timer = Instant::now();
            let (commit_seconds, prove_seconds, verify_seconds, proof_bytes, statement_bytes) =
                if chain {
                    let mut current = inputs(&expected)
                        .remove(0)
                        .into_iter()
                        .map(|v| v as usize)
                        .collect::<Vec<_>>();
                    let mut statement = NativeLookupChainStatement {
                        context: expected.context.clone(),
                        stages: vec![],
                    };
                    let mut witnesses = vec![];
                    for (i, node) in expected.nodes.iter().enumerate() {
                        let l = node.lookup.as_ref().unwrap();
                        let next = current.iter().map(|v| l.table[*v] as usize).collect();
                        let (s, w) = NativeLookupWitness::commit(
                            NativeLookupChainStatement::stage_context(&expected.context, i),
                            l.table.clone(),
                            current,
                            l.log_chunk,
                            &pp,
                        )
                        .unwrap();
                        statement.stages.push(s);
                        witnesses.push(w);
                        current = next;
                    }
                    assert_eq!(
                        current.iter().map(|v| *v as i32).collect::<Vec<_>>(),
                        reference(&expected)[0]
                    );
                    let commit_seconds = timer.elapsed().as_secs_f64();
                    let timer = Instant::now();
                    let proof =
                        NativeLookupChainProof::prove(&statement, witnesses, &pp, &key.gens)
                            .unwrap();
                    let prove_seconds = timer.elapsed().as_secs_f64();
                    let timer = Instant::now();
                    proof.verify(&statement, &key.setup, &key.gens).unwrap();
                    let verify_seconds = timer.elapsed().as_secs_f64();
                    (
                        commit_seconds,
                        prove_seconds,
                        verify_seconds,
                        write(&directory.join("proof.bin"), &proof),
                        write(&directory.join("statement.bin"), &statement),
                    )
                } else {
                    let (statement, witness) =
                        NativeGraphWitness::commit(expected.clone(), inputs(&expected), &pp)
                            .unwrap();
                    assert_eq!(witness.outputs(), reference(&expected));
                    let commit_seconds = timer.elapsed().as_secs_f64();
                    let timer = Instant::now();
                    let proof =
                        NativeGraphProof::prove(&statement, witness, &pp, &key.gens).unwrap();
                    let prove_seconds = timer.elapsed().as_secs_f64();
                    let timer = Instant::now();
                    proof.verify(&statement, &key.setup, &key.gens).unwrap();
                    let verify_seconds = timer.elapsed().as_secs_f64();
                    (
                        commit_seconds,
                        prove_seconds,
                        verify_seconds,
                        write(&directory.join("proof.bin"), &proof),
                        write(&directory.join("statement.bin"), &statement),
                    )
                };
            let verifier_bytes = write(&directory.join("verifier.bin"), &key);
            let nodes = expected.nodes.len();
            let variant = if chain {
                "separate proofs with required hidden equality"
            } else {
                "shared graph proof"
            };
            let record=format!("{{\"scope\":\"small native vector graph, not Qwen or provenance\",\"fixture\":\"{kind}\",\"variant\":\"{variant}\",\"rows\":{rows},\"nodes\":{nodes},\"setup_seconds\":{setup_seconds},\"commit_seconds\":{commit_seconds},\"prove_seconds\":{prove_seconds},\"verify_seconds\":{verify_seconds},\"proof_bytes\":{proof_bytes},\"statement_bytes\":{statement_bytes},\"verifier_bytes\":{verifier_bytes},\"expected_outputs_match\":true}}\n");
            fs::write(directory.join("measurement.json"), &record).unwrap();
            print!("{record}");
        } else {
            assert!(args[1] == "verify" || args[1] == "verify-chain");
            let key: Key = read(&directory.join("verifier.bin"));
            let verify_seconds = if chain {
                let statement: NativeLookupChainStatement = read(&directory.join("statement.bin"));
                let proof: NativeLookupChainProof = read(&directory.join("proof.bin"));
                assert_eq!(statement.context, expected.context);
                assert_eq!(statement.stages.len(), expected.nodes.len());
                for (s, n) in statement.stages.iter().zip(&expected.nodes) {
                    let l = n.lookup.as_ref().unwrap();
                    assert_eq!(s.table, l.table);
                    assert_eq!(s.log_rows, expected.log_rows);
                    assert_eq!(s.log_chunk, l.log_chunk);
                }
                let timer = Instant::now();
                proof.verify(&statement, &key.setup, &key.gens).unwrap();
                let seconds = timer.elapsed().as_secs_f64();
                let mut wrong = statement.clone();
                wrong.stages[0].table[0] ^= 1;
                assert!(proof.verify(&wrong, &key.setup, &key.gens).is_err());
                let mut missing = proof.clone();
                missing.edges.clear();
                assert!(missing.verify(&statement, &key.setup, &key.gens).is_err());
                let mut missing = proof;
                missing.stages.pop();
                assert!(missing.verify(&statement, &key.setup, &key.gens).is_err());
                seconds
            } else {
                let statement: NativeGraphStatement = read(&directory.join("statement.bin"));
                let proof: NativeGraphProof = read(&directory.join("proof.bin"));
                assert_eq!(bytes(&statement.graph), bytes(&expected));
                let timer = Instant::now();
                proof.verify(&statement, &key.setup, &key.gens).unwrap();
                let seconds = timer.elapsed().as_secs_f64();
                let mut wrong = statement.clone();
                wrong.graph.context.push(1);
                assert!(proof.verify(&wrong, &key.setup, &key.gens).is_err());
                let mut wrong = statement.clone();
                wrong.graph.nodes.pop();
                assert!(proof.verify(&wrong, &key.setup, &key.gens).is_err());
                let mut missing = proof;
                missing.indicators = None;
                assert!(missing.verify(&statement, &key.setup, &key.gens).is_err());
                seconds
            };
            let record=format!("{{\"accepted\":true,\"verify_seconds\":{verify_seconds},\"rejections_pass\":true}}\n");
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
