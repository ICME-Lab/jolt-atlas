use std::io::Cursor;

use model_specific_proving::{
    layers::{
        Activation, Attention, Mlp, NormalizationKind, NormalizationParameters, Projection,
        SoftmaxMask, SoftmaxTables,
    },
    models::{Architecture, Layer, Model, PublicEmbeddings},
    protocol::{Candidate, TopKConfig},
};

fn fixture(decoder: bool, rows: usize, layers: usize, gated: bool) -> Model<'static> {
    fixture_with_scores(decoder, rows, layers, gated, false)
}

fn fixture_with_scores(
    decoder: bool,
    rows: usize,
    layers: usize,
    gated: bool,
    varied: bool,
) -> Model<'static> {
    let zero = Projection {
        weights: &[0; 4],
        bias: &[0; 2],
    };
    let qk = if varied {
        Projection {
            weights: &[16, 0, 0, 0],
            bias: &[0, 0],
        }
    } else {
        zero
    };
    let norm = NormalizationParameters {
        kind: NormalizationKind::Layer,
        gain: &[4, 4],
        bias: &[0, 0],
        epsilon_variance_units: 4,
    };
    let layers = (0..layers)
        .map(|_| Layer {
            attention: Attention {
                queries: vec![qk],
                groups: vec![[qk, zero]],
                output: zero,
                rotary: None,
                factor: 1,
                mask: if decoder {
                    SoftmaxMask::Causal
                } else {
                    SoftmaxMask::Full
                },
                tables: SoftmaxTables {
                    probabilities: &[4, 0, 0, 0],
                    sums: &[4, 0, 0, 0],
                    limit: 4,
                },
            },
            mlp: Mlp {
                up: if gated {
                    Projection {
                        weights: &[0; 4],
                        bias: &[-12; 2],
                    }
                } else {
                    zero
                },
                gate: gated.then_some(Projection {
                    weights: &[0; 4],
                    bias: &[4; 2],
                }),
                down: if gated {
                    Projection {
                        weights: &[4, 0, 0, 0],
                        bias: &[0; 2],
                    }
                } else {
                    zero
                },
                activation: Activation {
                    lower: -2,
                    correction: if gated { &[0; 4] } else { &[1, 2, 3, 0] },
                },
            },
            attention_norm: norm,
            mlp_norm: norm,
        })
        .collect();
    let embeddings = PublicEmbeddings {
        width: 2,
        tokens: &[5, -4, -1, 8, 8, -1, 1, 3, -4, 5, 10, 1],
        positions: (rows <= 3).then_some(&[-1, 0, 1, 0, 0, 0]),
        segments: (!decoder).then_some(&[0, 0, 8, -8]),
    };
    let arch = if decoder {
        Architecture::Decoder { final_norm: norm }
    } else {
        Architecture::Encoder {
            embedding_norm: norm,
        }
    };
    Model::new(rows, 2, embeddings, layers, arch).unwrap()
}

fn config(positions: std::ops::Range<usize>) -> TopKConfig {
    TopKConfig {
        positions,
        k: 2,
        rows_per_block: 2,
        vocab_per_block: 4,
    }
}

#[test]
fn proves_two_decoder_layers_with_grouped_advice() {
    let model = fixture(true, 3, 2, false);
    let tokens = [0, 1, 2];
    let cfg = config(0..3);
    let mut proof = Cursor::new(Vec::new());
    let output = model.prove_top_k_to(&tokens, &cfg, &mut proof).unwrap();
    assert_eq!(output.len(), 3);
    model
        .verify_top_k_from(&tokens, &cfg, &output, &mut proof)
        .unwrap();
    assert!(model
        .verify_top_k_from(&[0, 2, 2], &cfg, &output, proof)
        .is_err());
}

#[test]
fn proves_causal_attention_across_lookup_tile_boundary() {
    for (rows, varied) in [(65, false), (128, false), (128, true)] {
        let model = fixture_with_scores(true, rows, 1, false, varied);
        let tokens: Vec<_> = (0..rows).map(|i| i % 6).collect();
        let cfg = config(rows - 1..rows);
        let mut proof = Cursor::new(Vec::new());
        let output = model.prove_top_k_to(&tokens, &cfg, &mut proof).unwrap();
        model
            .verify_top_k_from(&tokens, &cfg, &output, proof)
            .unwrap();
    }
}

#[test]
fn proves_encoder_with_public_segments() {
    for rows in [3, 128] {
        let model = fixture(false, rows, 2, false);
        let tokens: Vec<_> = (0..rows).map(|i| i % 6).collect();
        let mut segments = vec![0; rows];
        segments[1] = 1;
        let mut proof = Cursor::new(Vec::new());
        let output = model
            .prove_to(&tokens, Some(&segments), &mut proof)
            .unwrap();
        assert_eq!(model.cls(&output).unwrap().len(), 2);
        model
            .verify_from(&tokens, Some(&segments), &output, &mut proof)
            .unwrap();
        segments[1] = 0;
        assert!(model
            .verify_from(&tokens, Some(&segments), &output, proof)
            .is_err());
    }
}

#[test]
fn proves_top_k_and_picks_the_lower_id_on_ties() {
    let model = fixture(true, 3, 1, false);
    let tokens = [0, 1, 2];
    let cfg = config(0..3);
    let mut proof = Cursor::new(Vec::new());
    let mut candidates = model.prove_top_k_to(&tokens, &cfg, &mut proof).unwrap();
    let expected = [[(0, 9), (2, 9)], [(1, 9), (4, 9)], [(2, 11), (5, 11)]]
        .map(|row| row.map(|(id, logit)| Candidate { id, logit }));
    assert_eq!(candidates, expected.map(Vec::from));
    model
        .verify_top_k_from(&tokens, &cfg, &candidates, &mut proof)
        .unwrap();
    candidates[0][0].logit += 1;
    assert!(model
        .verify_top_k_from(&tokens, &cfg, &candidates, proof)
        .is_err());
}

#[test]
fn streams_top_k_proof_and_reads_it_back() {
    let model = fixture(true, 3, 2, true);
    let tokens = [0, 1, 2];
    let cfg = config(0..3);
    let mut archive = Cursor::new(Vec::new());
    let actual = model.prove_top_k_to(&tokens, &cfg, &mut archive).unwrap();
    model
        .verify_top_k_from(&tokens, &cfg, &actual, &mut archive)
        .unwrap();
    archive.get_mut().push(0);
    assert!(model
        .verify_top_k_from(&tokens, &cfg, &actual, archive)
        .is_err());
}

#[test]
fn activates_gate_before_multiplying_by_up_projection() {
    let model = fixture(true, 1, 1, true);
    let cfg = TopKConfig {
        k: 6,
        ..config(0..1)
    };
    let mut proof = Cursor::new(Vec::new());
    let output = model.prove_top_k_to(&[0], &cfg, &mut proof).unwrap();
    assert_eq!(output[0].iter().find(|c| c.id == 0).unwrap().logit, -9);
    model.verify_top_k_from(&[0], &cfg, &output, proof).unwrap();
}
