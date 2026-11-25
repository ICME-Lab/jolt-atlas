#![cfg(feature = "akita")]

use jolt_akita::{AkitaChunkProfile, AkitaScheduleArtifacts, AkitaScheme};
use model_specific_proving::{
    layers::{
        Activation, Attention, Mlp, NormalizationKind, NormalizationParameters, Projection,
        SoftmaxMask, SoftmaxTables,
    },
    models::{Architecture, Layer, Model, PublicEmbeddings},
    protocol::{AkitaAdviceSetup, Candidate, TopKConfig},
};

fn setup() -> AkitaAdviceSetup {
    let artifacts =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    AkitaAdviceSetup::new(artifacts).unwrap()
}

fn fixture(decoder: bool, rows: usize, layers: usize, gated: bool) -> Model<'static> {
    let zero = Projection {
        weights: &[0; 4],
        bias: &[0; 2],
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
                queries: vec![zero],
                groups: vec![[zero, zero]],
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
    let setup = setup();
    let (output, proof) = model.prove::<AkitaScheme>(&tokens, None, &setup).unwrap();
    assert_eq!(output.len(), 18);
    model
        .verify(&tokens, None, &output, &proof, &setup)
        .unwrap();
    assert!(model
        .verify(&[0, 2, 2], None, &output, &proof, &setup)
        .is_err());
}

#[test]
fn proves_causal_attention_across_lookup_tile_boundary() {
    let model = fixture(true, 65, 1, false);
    let tokens: Vec<_> = (0..65).map(|i| i % 6).collect();
    let setup = setup();
    let (output, proof) = model.prove::<AkitaScheme>(&tokens, None, &setup).unwrap();
    assert_eq!(output.len(), tokens.len() * 6);
    model
        .verify(&tokens, None, &output, &proof, &setup)
        .unwrap();
}

#[test]
fn proves_encoder_with_public_segments() {
    let model = fixture(false, 3, 2, false);
    let tokens = [0, 1, 2];
    let segments = [0, 1, 0];
    let setup = setup();
    let (output, proof) = model
        .prove::<AkitaScheme>(&tokens, Some(&segments), &setup)
        .unwrap();
    assert_eq!(model.cls(&output).unwrap().len(), 2);
    model
        .verify(&tokens, Some(&segments), &output, &proof, &setup)
        .unwrap();
    assert!(model
        .verify(&tokens, Some(&[0, 0, 0]), &output, &proof, &setup)
        .is_err());
}

#[test]
fn proves_top_k_and_picks_the_lower_id_on_ties() {
    let model = fixture(true, 2, 1, false);
    let tokens = [0, 1];
    let setup = setup();
    let (logits, _) = model.prove::<AkitaScheme>(&tokens, None, &setup).unwrap();
    let cfg = TopKConfig {
        positions: 0..2,
        k: 2,
        rows_per_block: 1,
        vocab_per_block: 4,
    };
    let (mut candidates, proof) = model
        .prove_top_k::<AkitaScheme>(&tokens, &cfg, &setup)
        .unwrap();
    let expected: Vec<_> = logits
        .chunks_exact(6)
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
        .verify_top_k(&tokens, &cfg, &candidates, &proof, &setup)
        .unwrap();
    candidates[0][0].logit += 1;
    assert!(model
        .verify_top_k(&tokens, &cfg, &candidates, &proof, &setup)
        .is_err());
}

#[test]
fn activates_gate_before_multiplying_by_up_projection() {
    let model = fixture(true, 1, 1, true);
    let setup = setup();
    let (output, proof) = model.prove::<AkitaScheme>(&[0], None, &setup).unwrap();
    assert_eq!(output[0], -9);
    model.verify(&[0], None, &output, &proof, &setup).unwrap();
}
