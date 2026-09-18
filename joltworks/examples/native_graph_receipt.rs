//! Complete small vector graphs, not Qwen inference or provenance benchmarks.
#[cfg(feature = "zk")]
mod enabled {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use atlas_onnx_tracer::{
        ops::{
            Add, Cos, GatherSmall, MeanOfSquares, Mul, Op, Rsqrt, ScalarConstDiv, Sigmoid, Sin,
            Sub, Sum,
        },
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
        if kind == "greedy-sequence" {
            use joltworks::poly::commitment::dory::native_generation::{
                append_greedy_sequence, GreedySequenceRule,
            };
            assert!(rows >= 4);
            let mut g = NativeGraph {
                context: format!("native tensor graph fixture/{kind}").into_bytes(),
                input_shapes: vec![vec![rows], vec![rows, 4]],
                nodes: vec![],
                outputs: vec![],
            };
            let selected = append_greedy_sequence(
                &mut g,
                0,
                1,
                GreedySequenceRule {
                    prompt_length: rows - 3,
                    score_start_position: 0,
                    response_length: 2,
                    maximum_new_tokens: 3,
                    logical_vocabulary: 3,
                    end_token: 2,
                },
            )
            .unwrap();
            g.outputs = vec![selected];
            return g;
        }
        if kind == "first-argmax" {
            return joltworks::poly::commitment::dory::native_logic::first_argmax_with_count(
                format!("native tensor graph fixture/{kind}").into_bytes(),
                vec![rows, 32],
                29,
            )
            .unwrap();
        }
        if matches!(kind, "sine" | "cosine") {
            return NativeGraph::trig(
                format!("native tensor graph fixture/{kind}").into_bytes(),
                vec![rows],
                14,
                kind == "cosine",
            )
            .unwrap();
        }
        if kind == "softmax" {
            return NativeGraph::softmax(
                format!("native tensor graph fixture/{kind}").into_bytes(),
                vec![rows, 32],
                14,
            )
            .unwrap();
        }
        if matches!(
            kind,
            "select" | "boolean-and" | "checked-neg" | "finite-select"
        ) {
            use joltworks::poly::commitment::dory::native_logic::{
                append_and, append_checked_neg, append_is_nan, append_select,
            };
            let count = match kind {
                "boolean-and" => 2,
                "checked-neg" => 1,
                _ => 3,
            };
            let mut g = NativeGraph {
                context: format!("native tensor graph fixture/{kind}").into_bytes(),
                input_shapes: vec![vec![rows]; count],
                nodes: vec![],
                outputs: vec![],
            };
            let result = match kind {
                "select" => append_select(&mut g, 0, 1, 2),
                "boolean-and" => append_and(&mut g, 0, 1),
                "checked-neg" => append_checked_neg(&mut g, 0),
                _ => {
                    let finite = append_is_nan(&mut g, 0).unwrap();
                    append_select(&mut g, finite, 1, 2)
                }
            }
            .unwrap();
            g.outputs = vec![result];
            return g;
        }
        let (num_inputs, nodes, outputs) = match kind {
            "division" => (
                1,
                vec![
                    NativeGraphNode::div_floor(0, 2470649),
                    NativeGraphNode::rem_euclid(0, 2470649),
                ],
                vec![1, 2],
            ),
            "row-gather" | "row-gather-wide" => {
                let dictionary = if kind == "row-gather-wide" { 256 } else { 16 };
                let logical = if kind == "row-gather-wide" { 251 } else { 13 };
                (
                    2,
                    vec![
                        NativeGraphNode::lookup(
                            0,
                            (0..dictionary)
                                .map(|i| if i < logical { i } else { -1 })
                                .collect(),
                            2,
                        ),
                        NativeGraphNode::add(1, 1),
                        NativeGraphNode::gather_rows(2, 3, 2),
                    ],
                    vec![4],
                )
            }
            "hidden-table" => (
                2,
                vec![
                    NativeGraphNode::add(1, 1),
                    NativeGraphNode::hidden_lookup(0, 2, 2),
                ],
                vec![3],
            ),
            "logical-mean" => (
                1,
                vec![NativeGraphNode::mean_of_squares_with_count(
                    0,
                    vec![1],
                    14,
                    896,
                )],
                vec![1],
            ),
            "slice-concat" => (
                1,
                vec![
                    NativeGraphNode::slice(0, 1, 32, 32),
                    NativeGraphNode::slice(0, 1, 0, 32),
                    NativeGraphNode::concat(1, 2, 1),
                ],
                vec![1, 2, 3],
            ),

            "layout" => (
                1,
                vec![
                    NativeGraphNode::broadcast(0, vec![rows, 8]),
                    NativeGraphNode::permute(1, vec![1, 0]),
                    NativeGraphNode::reshape(2, vec![2, 4, rows]),
                    NativeGraphNode::sum(3, vec![0, 1]),
                    NativeGraphNode::reshape(4, vec![rows]),
                ],
                vec![1, 2, 3, 5],
            ),
            "rms-normalization" => (
                3,
                vec![
                    NativeGraphNode::mean_of_squares(0, vec![1], 14),
                    NativeGraphNode::add(3, 1),
                    NativeGraphNode::rsqrt(4, 14),
                    NativeGraphNode::broadcast(5, vec![rows, 8]),
                    NativeGraphNode::mul(0, 6, 14),
                    NativeGraphNode::mul(7, 2, 14),
                ],
                vec![3, 4, 5, 7, 8],
            ),
            "rsqrt" => (
                1,
                vec![
                    NativeGraphNode::rsqrt(0, 14),
                    NativeGraphNode::mul(0, 1, 14),
                    NativeGraphNode::sum(2, vec![0]),
                ],
                vec![1, 2, 3],
            ),
            "normalization" => (
                1,
                vec![
                    NativeGraphNode::mean_of_squares(0, vec![1], 14),
                    NativeGraphNode::rsqrt(1, 14),
                ],
                vec![1, 2],
            ),
            "activation" | "activation-narrow" => {
                let bound = 1i32 << 17;
                let domain = (-bound..bound).collect::<Vec<_>>();
                let grid = Tensor::new(Some(&domain), &[domain.len()]).unwrap();
                let table = Sigmoid { scale: 14 }.f(vec![&grid]).data().to_vec();
                (
                    1,
                    vec![
                        NativeGraphNode::clamped_lookup(
                            0,
                            table,
                            -bound,
                            if kind == "activation-narrow" { 2 } else { 4 },
                        ),
                        NativeGraphNode::mul(0, 1, 14),
                        NativeGraphNode::sum(2, vec![0]),
                    ],
                    vec![1, 2, 3],
                )
            }
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
            "add-sub" => (
                2,
                vec![NativeGraphNode::add(0, 1), NativeGraphNode::sub(0, 1)],
                vec![2, 3],
            ),
            "residual" => (
                3,
                vec![
                    NativeGraphNode::add(0, 1),
                    NativeGraphNode::sub(0, 1),
                    NativeGraphNode::mul(3, 2, 14),
                    NativeGraphNode::sub(5, 3),
                    NativeGraphNode::lookup(6, vec![3, 4], 1),
                    NativeGraphNode::add(4, 7),
                    NativeGraphNode::add(8, 8),
                    NativeGraphNode::sub(9, 9),
                ],
                vec![3, 4, 5, 8, 10],
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
            "sum" => (
                1,
                vec![
                    NativeGraphNode::sum(0, vec![1]),
                    NativeGraphNode::sum(0, vec![0]),
                    NativeGraphNode::sum(0, vec![0, 1]),
                    NativeGraphNode::add(1, 1),
                ],
                vec![1, 2, 3, 4],
            ),
            "mean-squares" => (
                1,
                vec![
                    NativeGraphNode::mean_of_squares(0, vec![1], 14),
                    NativeGraphNode::mul(1, 1, 14),
                    NativeGraphNode::sum(2, vec![0]),
                ],
                vec![1, 2, 3],
            ),
            "matrix" => (
                2,
                vec![
                    NativeGraphNode::einsum(0, 1, "mk,kn->mn", 14, [16, 16]),
                    NativeGraphNode::add(2, 2),
                    NativeGraphNode::sum(3, vec![1]),
                ],
                vec![2, 4],
            ),
            "batched-matrix" => (
                2,
                vec![
                    NativeGraphNode::einsum(0, 1, "bmk,bnk->bmn", 14, [16, 16]),
                    NativeGraphNode::mean_of_squares(2, vec![2], 14),
                    NativeGraphNode::sum(3, vec![0, 1, 2]),
                ],
                vec![2, 3, 4],
            ),
            _ => panic!("unknown fixture"),
        };
        let input_shapes = match kind {
            "hidden-table" => vec![vec![rows], vec![128]],
            "row-gather" => vec![vec![rows, 1], vec![16, 8]],
            "row-gather-wide" => vec![vec![rows, 1], vec![256, 1024]],
            "logical-mean" => vec![vec![rows, 1024]],
            "slice-concat" => vec![vec![rows, 64]],
            "layout" => vec![vec![rows, 1]],
            "rms-normalization" => vec![vec![rows, 8], vec![rows, 1], vec![rows, 8]],
            "matrix" => vec![vec![rows, 32], vec![32, rows]],
            "batched-matrix" => vec![vec![2, rows, 32], vec![2, 16, 32]],
            "sum" | "mean-squares" | "normalization" => vec![vec![rows, 8]; num_inputs],
            _ => vec![vec![rows]; num_inputs],
        };
        NativeGraph {
            context: format!("native tensor graph fixture/{kind}").into_bytes(),
            input_shapes,
            nodes,
            outputs,
        }
    }
    fn inputs(g: &NativeGraph) -> Vec<Vec<i32>> {
        if g.context.ends_with(b"/greedy-sequence") {
            let rows = g.input_shapes[0][0];
            let mut tokens = vec![0; rows];
            tokens[rows - 3] = 1;
            tokens[rows - 2] = 2;
            let scores = (0..rows)
                .flat_map(|row| {
                    let index = if row == rows - 4 {
                        1
                    } else if row == rows - 3 {
                        2
                    } else {
                        0
                    };
                    (0..4).map(move |col| {
                        if col == 3 {
                            i32::MAX
                        } else if col == index {
                            9
                        } else {
                            0
                        }
                    })
                })
                .collect();
            return vec![tokens, scores];
        }

        if g.context.ends_with(b"/first-argmax") {
            return vec![(0..g.input_shapes[0][0] * 32)
                .map(|i| {
                    let row = i / 32;
                    let column = i % 32;
                    if row % 4 == 0 {
                        i32::MIN
                    } else if column == row % 32 || column == 31 {
                        i32::MAX
                    } else {
                        -(column as i32)
                    }
                })
                .collect()];
        }
        if [b"/division".as_slice(), b"/sine", b"/cosine"]
            .iter()
            .any(|suffix| g.context.ends_with(suffix))
        {
            return vec![(0..g.input_shapes[0][0])
                .map(|i| match i % 8 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    2 => -2470650,
                    3 => -2470649,
                    4 => -1,
                    5 => 0,
                    6 => 2470648,
                    _ => 2470649,
                })
                .collect()];
        }
        let kind = std::str::from_utf8(&g.context)
            .unwrap()
            .rsplit('/')
            .next()
            .unwrap();
        if matches!(
            kind,
            "select" | "boolean-and" | "checked-neg" | "finite-select"
        ) {
            let rows = g.input_shapes[0][0];
            let signed = [i32::MIN + 1, i32::MAX, -16384, -1, 0, 1, 16384, 7];
            let a = (0..rows).map(|i| signed[i % 8]).collect::<Vec<_>>();
            let b = (0..rows).map(|i| signed[(i + 3) % 8]).collect::<Vec<_>>();
            return match kind {
                "select" => vec![(0..rows).map(|i| (i % 2) as i32).collect(), a, b],
                "boolean-and" => vec![
                    (0..rows).map(|i| (i % 2) as i32).collect(),
                    (0..rows).map(|i| ((i / 2) % 2) as i32).collect(),
                ],
                "checked-neg" => vec![a],
                _ => vec![a.clone(), b, a],
            };
        }
        if g.context.ends_with(b"/row-gather") || g.context.ends_with(b"/row-gather-wide") {
            let dictionary = g.input_shapes[1][0];
            let logical = if dictionary == 256 { 251 } else { 13 };
            let count = g.input_shapes[1].iter().product::<usize>();
            return vec![
                (0..g.input_shapes[0][0])
                    .map(|i| ((i * 7) % logical) as i32)
                    .collect(),
                (0..count).map(|i| (i % 65536) as i32 * 37 - 2000).collect(),
            ];
        }
        if g.context.ends_with(b"/hidden-table") {
            return vec![
                (0..g.input_shapes[0][0])
                    .map(|i| ((i * 17) % 128) as i32)
                    .collect(),
                (0..128).map(|i| i * 37 - 2000).collect(),
            ];
        }
        if g.context.ends_with(b"/logical-mean") {
            let count = g.input_shapes[0].iter().product::<usize>();
            return vec![(0..count)
                .map(|i| {
                    if i % 1024 >= 896 {
                        0
                    } else {
                        ((i * 37 + 11) % 65536) as i32 - 32768
                    }
                })
                .collect()];
        }
        if g.context.ends_with(b"/slice-concat") {
            let count = g.input_shapes[0].iter().product::<usize>();
            return vec![(0..count)
                .map(|i| match i % 32 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    _ => i as i32 * 13 - 10000,
                })
                .collect()];
        }

        if g.context.ends_with(b"/softmax") {
            let count = g.input_shapes[0].iter().product::<usize>();
            return vec![(0..count)
                .map(|i| {
                    if i % 32 > i / 32 % 32 {
                        -(1 << 29)
                    } else {
                        ((i * 37 + 11) % 131072) as i32 - 65536
                    }
                })
                .collect()];
        }

        if g.context.ends_with(b"/layout") {
            return vec![(0..g.input_shapes[0][0])
                .map(|i| i as i32 * 13 - 9)
                .collect()];
        }
        if g.context.ends_with(b"/rms-normalization") {
            let rows = g.input_shapes[0][0];
            return vec![
                (0..rows * 8)
                    .map(|i| ((i * 37 + 11) % 65536) as i32 - 32768)
                    .collect(),
                vec![1; rows],
                (0..rows * 8).map(|i| 16384 + (i % 32) as i32 * 3).collect(),
            ];
        }
        if g.context.ends_with(b"/rsqrt") {
            let values = [i32::MIN, -1, 0, 1, 2, 16384, 32768, i32::MAX];
            return vec![(0..g.input_shapes[0][0])
                .map(|i| values[i % values.len()])
                .collect()];
        }
        if g.context.ends_with(b"/normalization") {
            return vec![(0..g.input_shapes[0][0] * 8)
                .map(|i| ((i * 37 + 11) % 65536) as i32 - 32768)
                .collect()];
        }
        if g.context.ends_with(b"/activation") || g.context.ends_with(b"/activation-narrow") {
            let values = [
                i32::MIN,
                -(1 << 17) - 1,
                -(1 << 17),
                -1,
                0,
                1,
                (1 << 17) - 1,
                i32::MAX,
            ];
            return vec![(0..g.input_shapes[0][0])
                .map(|i| values[i % values.len()])
                .collect()];
        }
        if g.context.ends_with(b"/matrix") || g.context.ends_with(b"/batched-matrix") {
            return g
                .input_shapes
                .iter()
                .enumerate()
                .map(|(side, shape)| {
                    (0..shape.iter().product::<usize>())
                        .map(|i| ((i * 37 + side * 11 + 17) % 65536) as i32 - 32768)
                        .collect()
                })
                .collect();
        }
        if g.context.ends_with(b"/sum") || g.context.ends_with(b"/mean-squares") {
            let patterns = if g.context.ends_with(b"/sum") {
                vec![
                    vec![i32::MAX; 8],
                    vec![i32::MIN; 8],
                    vec![i32::MAX, 1, -1, -i32::MAX, 0, 0, 0, 0],
                    vec![-7, 6, -5, 4, -3, 2, -1, 0],
                ]
            } else {
                vec![
                    vec![i32::MAX / 2; 8],
                    vec![1 << 14; 8],
                    vec![1 << 25, 0, 0, 0, 0, 0, 0, 0],
                    vec![-7, 6, -5, 4, -3, 2, -1, 0],
                ]
            };
            return vec![(0..g.input_shapes[0][0])
                .flat_map(|row| patterns[row % patterns.len()].clone())
                .collect()];
        }
        if g.context.ends_with(b"/add-sub") || g.context.ends_with(b"/residual") {
            let a = [i32::MIN, i32::MAX, i32::MIN, i32::MAX, -1, 0, 123, -456];
            let b = [i32::MIN, i32::MAX, i32::MAX, i32::MIN, 1, -1, -456, 123];
            let rows = 1 << g.max_log_rows().unwrap();
            let mut values = vec![
                (0..rows).map(|i| a[i % 8]).collect(),
                (0..rows).map(|i| b[i % 8]).collect(),
            ];
            if g.num_inputs() == 3 {
                values.push(vec![1 << 14; rows]);
            }
            return values;
        }
        let mut inputs = vec![(0usize..1usize << g.max_log_rows().unwrap())
            .map(|i| (i % 8) as i32)
            .collect()];
        if g.num_inputs() == 2 {
            inputs.push(vec![1 << 14; 1 << g.max_log_rows().unwrap()]);
        }
        inputs
    }
    fn reference(g: &NativeGraph) -> Vec<Vec<i32>> {
        if g.context.ends_with(b"/greedy-sequence") {
            return vec![inputs(g)[1]
                [(g.input_shapes[0][0] - 4) * 4..(g.input_shapes[0][0] - 2) * 4]
                .chunks_exact(4)
                .map(|row| {
                    let maximum = row[..3].iter().max().unwrap();
                    row[..3].iter().position(|x| x == maximum).unwrap() as i32
                })
                .collect()];
        }

        if g.context.ends_with(b"/first-argmax") {
            return vec![inputs(g)[0]
                .chunks(32)
                .map(|row| {
                    let maximum = row[..29].iter().max().unwrap();
                    row[..29].iter().position(|x| x == maximum).unwrap() as i32
                })
                .collect()];
        }

        if [b"/division".as_slice(), b"/sine", b"/cosine"]
            .iter()
            .any(|suffix| g.context.ends_with(suffix))
        {
            let values = inputs(g);
            let x = Tensor::new(Some(&values[0]), &g.input_shapes[0]).unwrap();
            if g.context.ends_with(b"/division") {
                return vec![
                    ScalarConstDiv { divisor: 2470649 }
                        .f(vec![&x])
                        .data()
                        .to_vec(),
                    atlas_onnx_tracer::tensor::ops::nonlinearities::const_rem(&x, 2470649)
                        .data()
                        .to_vec(),
                ];
            }
            return vec![if g.context.ends_with(b"/cosine") {
                Cos { scale: 14 }.f(vec![&x])
            } else {
                Sin { scale: 14 }.f(vec![&x])
            }
            .data()
            .to_vec()];
        }
        let kind = std::str::from_utf8(&g.context)
            .unwrap()
            .rsplit('/')
            .next()
            .unwrap();
        if matches!(
            kind,
            "select" | "boolean-and" | "checked-neg" | "finite-select"
        ) {
            use atlas_onnx_tracer::ops::{And, Iff, IsNan, Neg};
            let values = inputs(g);
            let tensors = values
                .iter()
                .map(|v| Tensor::new(Some(v), &g.input_shapes[0]).unwrap())
                .collect::<Vec<_>>();
            let result = match kind {
                "select" => Iff.f(tensors.iter().collect()),
                "boolean-and" => And.f(tensors.iter().collect()),
                "checked-neg" => Neg.f(tensors.iter().collect()),
                _ => {
                    let mask = IsNan {
                        out_dims: g.input_shapes[0].clone(),
                    }
                    .f(vec![&tensors[0]]);
                    Iff.f(vec![&mask, &tensors[1], &tensors[2]])
                }
            };
            return vec![result.data().to_vec()];
        }
        if g.context.ends_with(b"/row-gather") || g.context.ends_with(b"/row-gather-wide") {
            let values = inputs(g);
            let indices = Tensor::new(Some(&values[0]), &g.input_shapes[0]).unwrap();
            let data = Tensor::new(Some(&values[1]), &g.input_shapes[1]).unwrap();
            let twice = Add.f(vec![&data, &data]);
            return vec![GatherSmall {
                axis: 0,
                dict_len: if g.input_shapes[1][0] == 256 { 251 } else { 13 },
            }
            .f(vec![&twice, &indices])
            .data()
            .to_vec()];
        }
        if g.context.ends_with(b"/hidden-table") {
            let values = inputs(g);
            let indices = Tensor::new(Some(&values[0]), &g.input_shapes[0]).unwrap();
            let data = Tensor::new(Some(&values[1]), &[128]).unwrap();
            let twice = Add.f(vec![&data, &data]);
            return vec![GatherSmall {
                axis: 0,
                dict_len: 128,
            }
            .f(vec![&twice, &indices])
            .data()
            .to_vec()];
        }
        if g.context.ends_with(b"/slice-concat") {
            let values = inputs(g);
            let x = Tensor::new(Some(&values[0]), &g.input_shapes[0]).unwrap();
            let a = atlas_onnx_tracer::ops::Slice {
                axis: 1,
                start: 32,
                end: 64,
            }
            .f(vec![&x]);
            let b = atlas_onnx_tracer::ops::Slice {
                axis: 1,
                start: 0,
                end: 32,
            }
            .f(vec![&x]);
            let y = atlas_onnx_tracer::ops::Concat { axis: 1 }.f(vec![&a, &b]);
            return vec![a.data().to_vec(), b.data().to_vec(), y.data().to_vec()];
        }

        if g.context.ends_with(b"/softmax") {
            let values = inputs(g);
            let x = Tensor::new(Some(&values[0]), &g.input_shapes[0]).unwrap();
            return vec![atlas_onnx_tracer::ops::SoftmaxLastAxis { scale: 14 }
                .f(vec![&x])
                .data()
                .to_vec()];
        }

        let mut values = inputs(g);
        let shapes = g.tensor_shapes().unwrap();
        for n in &g.nodes {
            let out = if let Some(e) = &n.einsum {
                // Independent wide-integer reference. The witness uses the
                // Atlas blocked kernel; this loop checks the complete result.
                let left = &values[n.input];
                let right = &values[e.right];
                let mut output = vec![];
                let clamp = |sum: i128| {
                    sum.div_euclid(1i128 << e.shift)
                        .clamp(i128::from(i32::MIN), i128::from(i32::MAX))
                        as i32
                };
                if e.equation == "mk,kn->mn" {
                    let (m, k, n) = (shapes[n.input][0], shapes[n.input][1], shapes[e.right][1]);
                    for i in 0..m {
                        for j in 0..n {
                            output.push(clamp(
                                (0..k)
                                    .map(|h| {
                                        i128::from(left[i * k + h]) * i128::from(right[h * n + j])
                                    })
                                    .sum(),
                            ));
                        }
                    }
                } else {
                    assert_eq!(e.equation, "bmk,bnk->bmn");
                    let (batch, m, k, n) = (
                        shapes[n.input][0],
                        shapes[n.input][1],
                        shapes[n.input][2],
                        shapes[e.right][1],
                    );
                    for b in 0..batch {
                        for i in 0..m {
                            for j in 0..n {
                                output.push(clamp(
                                    (0..k)
                                        .map(|h| {
                                            i128::from(left[(b * m + i) * k + h])
                                                * i128::from(right[(b * n + j) * k + h])
                                        })
                                        .sum(),
                                ));
                            }
                        }
                    }
                }
                output
            } else if let Some(m) = &n.mul {
                let a = Tensor::new(Some(&values[n.input]), &shapes[n.input]).unwrap();
                let b = Tensor::new(Some(&values[m.right]), &shapes[m.right]).unwrap();
                Mul {
                    scale: i32::from(m.shift),
                }
                .f(vec![&a, &b])
                .data()
                .to_vec()
            } else if let Some(add) = &n.add {
                let a = Tensor::new(Some(&values[n.input]), &shapes[n.input]).unwrap();
                let b = Tensor::new(Some(&values[add.right]), &shapes[add.right]).unwrap();
                if add.subtract {
                    Sub.f(vec![&a, &b])
                } else {
                    Add.f(vec![&a, &b])
                }
                .data()
                .to_vec()
            } else if let Some(reduce) = &n.reduce {
                let a = Tensor::new(Some(&values[n.input]), &shapes[n.input]).unwrap();
                if let Some(scale) = reduce.mean_scale {
                    let count = reduce
                        .axes
                        .iter()
                        .map(|axis| shapes[n.input][*axis])
                        .product();
                    MeanOfSquares {
                        axes: reduce.axes.clone(),
                        scale: i32::from(scale),
                        count: reduce.mean_count.unwrap_or(count),
                        padded_count: count,
                    }
                    .f(vec![&a])
                } else {
                    Sum {
                        axes: reduce.axes.clone(),
                    }
                    .f(vec![&a])
                }
                .data()
                .to_vec()
            } else if let Some(layout) = &n.layout {
                let mut x = Tensor::new(Some(&values[n.input]), &shapes[n.input]).unwrap();
                match layout.kind {
                    0 => x = x.expand(&layout.shape).unwrap(),
                    1 => x.reshape(&layout.shape).unwrap(),
                    2 => {
                        let mut order = (0..layout.axes.len()).collect::<Vec<_>>();
                        for (destination, axis) in layout.axes.iter().enumerate() {
                            let source = order.iter().position(|a| a == axis).unwrap();
                            x = x.move_axis(source, destination).unwrap();
                            let moved = order.remove(source);
                            order.insert(destination, moved);
                        }
                    }
                    _ => panic!("unknown layout"),
                }
                x.data().to_vec()
            } else if let Some(scale) = n.rsqrt {
                let x = Tensor::new(Some(&values[n.input]), &shapes[n.input]).unwrap();
                Rsqrt {
                    scale: i32::from(scale),
                }
                .f(vec![&x])
                .data()
                .to_vec()
            } else if n.lookup.as_ref().unwrap().clamp_lower.is_some() {
                // Compare the complete signed input against the actual Atlas
                // activation kernel, independently of the clamp/index witness.
                assert!(
                    g.context.ends_with(b"/activation")
                        || g.context.ends_with(b"/activation-narrow")
                );
                let x = Tensor::new(Some(&values[n.input]), &shapes[n.input]).unwrap();
                Sigmoid { scale: 14 }.f(vec![&x]).data().to_vec()
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
        assert_eq!(args.len(),5,"native_graph_receipt prove|verify|prove-chain|verify-chain DIRECTORY mixed|table-chain|add-sub|residual|sum|mean-squares|matrix|batched-matrix|activation|activation-narrow|rsqrt|normalization|layout|rms-normalization|softmax|slice-concat|logical-mean|hidden-table|division|sine|cosine|row-gather|row-gather-wide|select|boolean-and|checked-neg|finite-select|first-argmax|greedy-sequence ROWS");
        let directory = Path::new(&args[2]);
        let kind = &args[3];
        let rows = args[4].parse::<usize>().unwrap();
        let registration_timer = Instant::now();
        let expected = graph(kind, rows);
        let registration_seconds = registration_timer.elapsed().as_secs_f64();
        let chain = args[1].ends_with("-chain");
        if chain {
            assert_eq!(kind, "table-chain");
        }
        if args[1] == "prove" || args[1] == "prove-chain" {
            let timer = Instant::now();
            let pp = DoryScheme::setup_prover(expected.max_log_rows().unwrap() + 8);
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
            let input_shapes = &expected.input_shapes;
            let shapes = expected.tensor_shapes().unwrap();
            let output_shapes = expected
                .outputs
                .iter()
                .map(|i| &shapes[*i])
                .collect::<Vec<_>>();
            let record=format!("{{\"input_shapes\":{input_shapes:?},\"output_shapes\":{output_shapes:?},\"scope\":\"small native tensor graph, not Qwen or provenance\",\"fixture\":\"{kind}\",\"variant\":\"{variant}\",\"rows\":{rows},\"nodes\":{nodes},\"registration_seconds\":{registration_seconds},\"setup_seconds\":{setup_seconds},\"commit_seconds\":{commit_seconds},\"prove_seconds\":{prove_seconds},\"verify_seconds\":{verify_seconds},\"proof_bytes\":{proof_bytes},\"statement_bytes\":{statement_bytes},\"verifier_bytes\":{verifier_bytes},\"expected_outputs_match\":true}}\n");
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
                    assert_eq!(s.log_rows, expected.max_log_rows().unwrap());
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
                let mut missing = proof.clone();
                missing.relations.output_claims_commitments.clear();
                assert!(missing.verify(&statement, &key.setup, &key.gens).is_err());
                let mut wrong = proof;
                wrong.indicators = if wrong.indicators.is_some() {
                    None
                } else {
                    Some(wrong.relations.clone())
                };
                assert!(wrong.verify(&statement, &key.setup, &key.gens).is_err());
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
