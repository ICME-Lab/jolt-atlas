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
                    correction: if gated { &[0; 4] } else { &[1, 2, 3, 4] },
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
        Architecture::Decoder {
            final_norm: norm,
            logits_bias: &[0; 6],
        }
    } else {
        Architecture::Encoder {
            embedding_norm: norm,
        }
    };
    Model::new(rows, 2, embeddings, layers, arch).unwrap()
}

#[test]
fn proves_two_decoder_layers_with_grouped_advice() {
    let model = fixture(true, 3, 2, false);
    let tokens = [0, 1, 2];
    let (output, proof) = model.prove(&tokens, None).unwrap();
    assert_eq!(output.len(), 18);
    model.verify(&tokens, None, &output, &proof).unwrap();
    assert!(model.verify(&[0, 2, 2], None, &output, &proof).is_err());
}

#[test]
fn proves_causal_attention_across_lookup_tile_boundary() {
    for (rows, varied) in [(65, false), (128, false), (128, true)] {
        let model = fixture_with_scores(true, rows, 1, false, varied);
        let tokens: Vec<_> = (0..rows).map(|i| i % 6).collect();
        let (output, proof) = model.prove(&tokens, None).unwrap();
        assert_eq!(output.len(), tokens.len() * 6);
        model.verify(&tokens, None, &output, &proof).unwrap();
    }
}

#[test]
fn proves_encoder_with_public_segments() {
    let model = fixture(false, 3, 2, false);
    let tokens = [0, 1, 2];
    let segments = [0, 1, 0];
    let (output, proof) = model.prove(&tokens, Some(&segments)).unwrap();
    assert_eq!(model.cls(&output).unwrap().len(), 2);
    model
        .verify(&tokens, Some(&segments), &output, &proof)
        .unwrap();
    assert!(model
        .verify(&tokens, Some(&[0, 0, 0]), &output, &proof)
        .is_err());
}

#[test]
fn proves_top_k_and_picks_the_lower_id_on_ties() {
    let model = fixture(true, 3, 1, false);
    let tokens = [0, 1, 2];
    let (logits, _) = model.prove(&tokens, None).unwrap();
    let cfg = TopKConfig {
        positions: 1..3,
        k: 2,
        rows_per_block: 2,
        vocab_per_block: 4,
    };
    let (mut candidates, proof) = model.prove_top_k(&tokens, &cfg).unwrap();
    let expected: Vec<_> = logits
        .chunks_exact(6)
        .skip(1)
        .map(|row| {
            let mut r: Vec<_> = row
                .iter()
                .enumerate()
                .map(|(id, &logit)| Candidate { id, logit })
                .collect();
            r.sort_unstable_by_key(|c| (-i64::from(c.logit), c.id));
            r.truncate(2);
            r
        })
        .collect();
    assert_eq!(candidates, expected);
    model
        .verify_top_k(&tokens, &cfg, &candidates, &proof)
        .unwrap();
    candidates[0][0].logit += 1;
    assert!(model
        .verify_top_k(&tokens, &cfg, &candidates, &proof)
        .is_err());
}

#[test]
fn activates_gate_before_multiplying_by_up_projection() {
    let model = fixture(true, 1, 1, true);
    let (output, proof) = model.prove(&[0], None).unwrap();
    assert_eq!(output[0], -9);
    model.verify(&[0], None, &output, &proof).unwrap();
}
