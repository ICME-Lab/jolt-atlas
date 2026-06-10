use ark_bn254::{Bn254, Fr};
use ark_ff::One;
use joltworks::{
    field::JoltField,
    poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
    transcripts::{Blake2bTranscript, Transcript},
};
use qwen3_common::{
    IopLayerProof, LayerRmsNormVerifierInput, LayerRopeVerifierInput, LayerSiluVerifierInput,
    LayerSoftmaxVerifierInput, LayerVerifierPublicInput,
};
use qwen3_prover::{
    commitment::{CommitLayerParams, commit_layer_hidden_openings, commit_layer_openings},
    layer::{EvalClaim, prove_iop_layer, prove_layer},
    opening::prove_layer_opening_reduction_sumcheck,
    ops::{
        prove_add, prove_add_claims, prove_matmul, prove_mul, prove_pv_matmul, prove_qk_score,
        prove_rms_norm, prove_rope, prove_silu, prove_softmax,
    },
};
use qwen3_tracer::{TraceLayerInput, layer_input_from_trace_dir};
use qwen3_verifier::{verify_iop_layer, verify_layer, verify_layer_opening_reduction};

fn layer_verifier_public_input(
    input: qwen3_prover::layer::LayerProverInput,
) -> LayerVerifierPublicInput {
    LayerVerifierPublicInput {
        seq: input.shape.seq,
        down_proj_weight: input.down_proj.witness.rhs,
        silu: LayerSiluVerifierInput {
            advice: input.silu.advice,
        },
        gate_proj_weight: input.gate_proj.witness.rhs,
        up_proj_weight: input.up_proj.witness.rhs,
        rms_norm_mlp: LayerRmsNormVerifierInput {
            advice: input.rms_norm_mlp.advice,
            weight: input.rms_norm_mlp.witness.weight,
        },
        o_proj_weight: input.o_proj.witness.rhs,
        softmax: LayerSoftmaxVerifierInput {
            advice: input.softmax.advice,
        },
        q_rope: LayerRopeVerifierInput {
            cos: input.q_rope.witness.cos,
            sin: input.q_rope.witness.sin,
        },
        k_rope: LayerRopeVerifierInput {
            cos: input.k_rope.witness.cos,
            sin: input.k_rope.witness.sin,
        },
        q_norm: LayerRmsNormVerifierInput {
            advice: input.q_norm.advice,
            weight: input.q_norm.witness.weight,
        },
        k_norm: LayerRmsNormVerifierInput {
            advice: input.k_norm.advice,
            weight: input.k_norm.witness.weight,
        },
        q_proj_weight: input.q_proj.witness.rhs,
        k_proj_weight: input.k_proj.witness.rhs,
        v_proj_weight: input.v_proj.witness.rhs,
        rms_norm_atten: LayerRmsNormVerifierInput {
            advice: input.rms_norm_atten.advice,
            weight: input.rms_norm_atten.witness.weight,
        },
    }
}

fn iop_layer_proof(output: qwen3_prover::layer::IopLayerOutput) -> IopLayerProof {
    output.proof
}

#[test]
#[ignore]
fn converts_fox_trace_layer0() {
    let traced = fox_trace_layer0();
    assert_eq!(traced.shape.seq, 197);
    assert_eq!(traced.hidden_out.len(), 197 * 1024);
    assert_eq!(traced.input.down_proj.params.output_shape.rows, 256);
    assert_eq!(traced.input.down_proj.params.inner, 4096);
}

#[test]
#[ignore]
fn commits_fox_trace_layer0() {
    let traced = fox_trace_layer0();
    type Pcs = HyperKZG<Bn254>;
    let setup = Pcs::setup_prover(20);
    let params = CommitLayerParams {
        pcs_domain_size: 1 << 20,
    };

    let commitments = step("commit_layer", || {
        Some(
            commit_layer_openings(
                &traced.input.opening_witnesses,
                traced.input.shape()?,
                params,
                &setup,
            )
            .expect("layer commitment succeeds"),
        )
    });

    eprintln!(
        "commit_layer chunks: hidden_in_a={}, hidden_in_b={}, hidden_out={}, silu_lookup_ra={}, softmax_lookup_ra={}",
        commitments.hidden_in_a.commitments.len(),
        commitments.hidden_in_b.commitments.len(),
        commitments.hidden_out.commitments.len(),
        commitments.silu_lookup_ra.commitments.len(),
        commitments.softmax_lookup_ra.commitments.len(),
    );
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_pv_matmul_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.pv_matmul.params;
    let point = vec![Fr::from(3_u64); params.context_vars()];
    let claim = EvalClaim {
        value: eval_context_values(
            &input.o_proj.witness.lhs,
            &point,
            params.seq,
            params.q_heads,
            params.head_dim,
        ),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("pv_matmul_direct", || {
        prove_pv_matmul(claim, input.pv_matmul, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_qk_score_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.qk_score.params;
    let point = vec![Fr::from(3_u64); params.score_vars()];
    let claim = EvalClaim {
        value: eval_score_values(
            &input.softmax.witness.input,
            &point,
            params.seq,
            params.q_heads,
        ),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("qk_score_direct", || {
        prove_qk_score(claim, input.qk_score, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_rms_norm_atten_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.rms_norm_atten.params;
    let point = vec![Fr::from(3_u64); params.shape.point_len()];
    let claim = EvalClaim {
        value: eval_matrix(
            &input.rms_norm_atten.witness.output,
            params.shape.rows,
            params.shape.cols,
            &point,
        ),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("rms_norm_atten_direct", || {
        prove_rms_norm(
            vec![claim.clone(), claim],
            input.rms_norm_atten,
            &mut transcript,
        )
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_q_rope_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.q_rope.params;
    let point = vec![Fr::from(3_u64); params.tensor_vars()];
    let claim = EvalClaim {
        value: eval_rope_tensor(&input.q_rope.witness.output, &params, &point),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("q_rope_direct", || {
        prove_rope(claim, input.q_rope, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_k_rope_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.k_rope.params;
    let point = vec![Fr::from(3_u64); params.tensor_vars()];
    let claim = EvalClaim {
        value: eval_rope_tensor(&input.k_rope.witness.output, &params, &point),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("k_rope_direct", || {
        prove_rope(claim, input.k_rope, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_silu_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.silu.params;
    let point = vec![Fr::from(3_u64); params.shape.point_len()];
    let claim = EvalClaim {
        value: eval_matrix(
            &input.silu.witness.output,
            params.shape.rows,
            params.shape.cols,
            &point,
        ),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("silu_direct", || {
        prove_silu(claim, input.silu, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_softmax_direct() {
    let traced = fox_trace_layer0();
    let input = traced.input;
    let params = input.softmax.params;
    let point = vec![Fr::from(3_u64); params.shape.point_len()];
    let claim = EvalClaim {
        value: eval_matrix(
            &input.softmax.witness.output,
            params.shape.rows,
            params.shape.cols,
            &point,
        ),
        point,
    };
    let mut transcript = Blake2bTranscript::default();
    step("softmax_direct", || {
        prove_softmax(claim, input.softmax, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0() {
    let traced = fox_trace_layer0();
    let hidden_out_claim = hidden_out_iop_claim(&traced);
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();
    let output = step("layer", || {
        prove_iop_layer(hidden_out_claim.value, input, &mut transcript)
    });

    let verify_input = fox_trace_layer0().input;
    let mut verifier_transcript = Blake2bTranscript::default();
    step("verify_layer", || {
        verify_iop_layer(
            layer_verifier_public_input(verify_input),
            iop_layer_proof(output),
            &mut verifier_transcript,
        )
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_opening_reduction() {
    let traced = fox_trace_layer0();
    let hidden_out_claim = hidden_out_iop_claim(&traced);
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();
    let output = step("layer", || {
        prove_iop_layer(hidden_out_claim.value, input, &mut transcript)
    });

    let verify_input = fox_trace_layer0().input;
    let shape = verify_input.shape;
    let qwen3_prover::layer::IopLayerOutput {
        proof,
        opening_witnesses,
        ..
    } = output;
    let output_proof = proof;
    let mut verifier_transcript = Blake2bTranscript::default();
    let verified_output = step("verify_layer", || {
        verify_iop_layer(
            layer_verifier_public_input(verify_input),
            output_proof,
            &mut verifier_transcript,
        )
    });

    let mut opening_transcript = Blake2bTranscript::default();
    let opening = step("opening_reduction", || {
        prove_layer_opening_reduction_sumcheck(
            &verified_output.opening_claims,
            &opening_witnesses,
            shape,
            &mut opening_transcript,
        )
    });

    let mut opening_verifier_transcript = Blake2bTranscript::default();
    step("verify_opening_reduction", || {
        verify_layer_opening_reduction(
            &verified_output.opening_claims,
            shape,
            &opening.proof,
            &mut opening_verifier_transcript,
        )
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_chunked_pcs_opening() {
    let traced = fox_trace_layer0();

    type Pcs = HyperKZG<Bn254>;
    let commit_params = CommitLayerParams {
        pcs_domain_size: 1 << 20,
    };
    let setup = step("chunked_pcs_setup", || Some(Pcs::setup_prover(20)));
    let hidden_commitments = step("commit_hidden_states", || {
        Some(
            commit_layer_hidden_openings(&traced.input.opening_witnesses, commit_params, &setup)
                .expect("hidden commitment succeeds"),
        )
    });
    let input = traced.input;

    let mut transcript = Blake2bTranscript::default();
    let output = step("layer", || {
        prove_layer(
            input,
            hidden_commitments,
            commit_params,
            &setup,
            &mut transcript,
        )
    });

    let verifier_setup = Pcs::setup_verifier(&setup);
    let verify_input = fox_trace_layer0().input;
    let commitments = output.commitments;
    let iop = iop_layer_proof(output.iop);
    let opening = output.opening;
    let pcs_opening = output.pcs_opening;
    let mut verifier_transcript = Blake2bTranscript::default();
    step("verify_layer", || {
        verify_layer(
            layer_verifier_public_input(verify_input),
            &commitments,
            iop,
            &opening.proof,
            &pcs_opening,
            &verifier_setup,
            &mut verifier_transcript,
        )
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_down_proj_matmul_only() {
    let traced = fox_trace_layer0();
    let point = vec![Fr::from(3_u64); 18];
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&traced.hidden_out, 197, 1024, 256, 1024, &point),
        point,
    };
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();

    let residual_add_mlp = step("residual_add_mlp", || {
        prove_add(hidden_out_claim, input.residual_add_mlp, &mut transcript)
    });
    eprintln!(
        "down_proj shape: output=({}, {}), inner={}, lhs_len={}, rhs_len={}, output_len={}",
        input.down_proj.params.output_shape.rows,
        input.down_proj.params.output_shape.cols,
        input.down_proj.params.inner,
        input.down_proj.witness.lhs.len(),
        input.down_proj.witness.rhs.len(),
        input.down_proj.witness.output.len(),
    );
    step("down_proj_matmul_only", || {
        prove_matmul(
            residual_add_mlp.rhs_claim.clone(),
            input.down_proj,
            &mut transcript,
        )
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_gate_proj_matmul_only() {
    let traced = fox_trace_layer0();
    let point = vec![Fr::from(3_u64); 18];
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&traced.hidden_out, 197, 1024, 256, 1024, &point),
        point,
    };
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();

    let residual_add_mlp = step("residual_add_mlp", || {
        prove_add(hidden_out_claim, input.residual_add_mlp, &mut transcript)
    });
    let down_proj = step("down_proj", || {
        prove_matmul(
            residual_add_mlp.rhs_claim.clone(),
            input.down_proj,
            &mut transcript,
        )
    });
    let silu_up = step("silu_up", || {
        prove_mul(down_proj.lhs.clone(), input.silu_up, &mut transcript)
    });
    let silu = step("silu", || {
        prove_silu(silu_up.lhs.clone(), input.silu, &mut transcript)
    });

    eprintln!(
        "gate_proj claim: point_len={}, value={:?}",
        silu.input.point.len(),
        silu.input.value
    );
    eprintln!(
        "gate_proj output eval at claim point: {:?}",
        eval_matrix(
            &input.gate_proj.witness.output,
            input.gate_proj.params.output_shape.rows,
            input.gate_proj.params.output_shape.cols,
            &silu.input.point,
        )
    );
    eprintln!(
        "gate_proj shape: output=({}, {}), inner={}, lhs_len={}, rhs_len={}, output_len={}",
        input.gate_proj.params.output_shape.rows,
        input.gate_proj.params.output_shape.cols,
        input.gate_proj.params.inner,
        input.gate_proj.witness.lhs.len(),
        input.gate_proj.witness.rhs.len(),
        input.gate_proj.witness.output.len(),
    );
    step("gate_proj_matmul_only", || {
        prove_matmul(silu.input.clone(), input.gate_proj, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_up_proj_matmul_only() {
    let traced = fox_trace_layer0();
    let point = vec![Fr::from(3_u64); 18];
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&traced.hidden_out, 197, 1024, 256, 1024, &point),
        point,
    };
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();

    let residual_add_mlp = step("residual_add_mlp", || {
        prove_add(hidden_out_claim, input.residual_add_mlp, &mut transcript)
    });
    let down_proj = step("down_proj", || {
        prove_matmul(
            residual_add_mlp.rhs_claim.clone(),
            input.down_proj,
            &mut transcript,
        )
    });
    let silu_up = step("silu_up", || {
        prove_mul(down_proj.lhs.clone(), input.silu_up, &mut transcript)
    });

    eprintln!(
        "up_proj claim: point_len={}, value={:?}",
        silu_up.rhs.point.len(),
        silu_up.rhs.value
    );
    eprintln!(
        "up_proj output eval at claim point: {:?}",
        eval_matrix(
            &input.up_proj.witness.output,
            input.up_proj.params.output_shape.rows,
            input.up_proj.params.output_shape.cols,
            &silu_up.rhs.point,
        )
    );
    eprintln!(
        "up_proj shape: output=({}, {}), inner={}, lhs_len={}, rhs_len={}, output_len={}",
        input.up_proj.params.output_shape.rows,
        input.up_proj.params.output_shape.cols,
        input.up_proj.params.inner,
        input.up_proj.witness.lhs.len(),
        input.up_proj.witness.rhs.len(),
        input.up_proj.witness.output.len(),
    );
    step("up_proj_matmul_only", || {
        prove_matmul(silu_up.rhs.clone(), input.up_proj, &mut transcript)
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_o_proj_matmul_only() {
    let traced = fox_trace_layer0();
    let point = vec![Fr::from(3_u64); 18];
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&traced.hidden_out, 197, 1024, 256, 1024, &point),
        point,
    };
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();

    let residual_add_mlp = step("residual_add_mlp", || {
        prove_add(hidden_out_claim, input.residual_add_mlp, &mut transcript)
    });
    let down_proj = step("down_proj", || {
        prove_matmul(
            residual_add_mlp.rhs_claim.clone(),
            input.down_proj,
            &mut transcript,
        )
    });
    let silu_up = step("silu_up", || {
        prove_mul(down_proj.lhs.clone(), input.silu_up, &mut transcript)
    });
    let silu = step("silu", || {
        prove_silu(silu_up.lhs.clone(), input.silu, &mut transcript)
    });
    let gate_proj = step("gate_proj", || {
        prove_matmul(silu.input.clone(), input.gate_proj, &mut transcript)
    });
    let up_proj = step("up_proj", || {
        prove_matmul(silu_up.rhs.clone(), input.up_proj, &mut transcript)
    });
    let rms_norm_mlp = step("rms_norm_mlp", || {
        prove_rms_norm(
            vec![gate_proj.lhs.clone(), up_proj.lhs.clone()],
            input.rms_norm_mlp,
            &mut transcript,
        )
    });
    let residual_add_attn = step("residual_add_attn", || {
        prove_add_claims(
            vec![
                residual_add_mlp.lhs_claim.clone(),
                rms_norm_mlp.input.clone(),
            ],
            input.residual_add_attn,
            &mut transcript,
        )
    });

    eprintln!(
        "o_proj claim: point_len={}, value={:?}",
        residual_add_attn.rhs_claim.point.len(),
        residual_add_attn.rhs_claim.value
    );
    eprintln!(
        "o_proj output eval at claim point: {:?}",
        eval_matrix(
            &input.o_proj.witness.output,
            input.o_proj.params.output_shape.rows,
            input.o_proj.params.output_shape.cols,
            &residual_add_attn.rhs_claim.point,
        )
    );
    eprintln!(
        "o_proj shape: output=({}, {}), inner={}, lhs_len={}, rhs_len={}, output_len={}",
        input.o_proj.params.output_shape.rows,
        input.o_proj.params.output_shape.cols,
        input.o_proj.params.inner,
        input.o_proj.witness.lhs.len(),
        input.o_proj.witness.rhs.len(),
        input.o_proj.witness.output.len(),
    );
    step("o_proj_matmul_only", || {
        prove_matmul(
            residual_add_attn.rhs_claim.clone(),
            input.o_proj,
            &mut transcript,
        )
    });
}

#[test]
#[ignore]
fn proves_fox_trace_layer0_pv_matmul_only() {
    let traced = fox_trace_layer0();
    let point = vec![Fr::from(3_u64); 18];
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&traced.hidden_out, 197, 1024, 256, 1024, &point),
        point,
    };
    let input = traced.input;
    let mut transcript = Blake2bTranscript::default();

    let residual_add_mlp = step("residual_add_mlp", || {
        prove_add(hidden_out_claim, input.residual_add_mlp, &mut transcript)
    });
    let down_proj = step("down_proj", || {
        prove_matmul(
            residual_add_mlp.rhs_claim.clone(),
            input.down_proj,
            &mut transcript,
        )
    });
    let silu_up = step("silu_up", || {
        prove_mul(down_proj.lhs.clone(), input.silu_up, &mut transcript)
    });
    let silu = step("silu", || {
        prove_silu(silu_up.lhs.clone(), input.silu, &mut transcript)
    });
    let gate_proj = step("gate_proj", || {
        prove_matmul(silu.input.clone(), input.gate_proj, &mut transcript)
    });
    let up_proj = step("up_proj", || {
        prove_matmul(silu_up.rhs.clone(), input.up_proj, &mut transcript)
    });
    let rms_norm_mlp = step("rms_norm_mlp", || {
        prove_rms_norm(
            vec![gate_proj.lhs.clone(), up_proj.lhs.clone()],
            input.rms_norm_mlp,
            &mut transcript,
        )
    });
    let residual_add_attn = step("residual_add_attn", || {
        prove_add_claims(
            vec![
                residual_add_mlp.lhs_claim.clone(),
                rms_norm_mlp.input.clone(),
            ],
            input.residual_add_attn,
            &mut transcript,
        )
    });
    let context_values = input.o_proj.witness.lhs.clone();
    let o_proj = step("o_proj", || {
        prove_matmul(
            residual_add_attn.rhs_claim.clone(),
            input.o_proj,
            &mut transcript,
        )
    });

    eprintln!(
        "pv context claim: point_len={}, value={:?}",
        o_proj.lhs.point.len(),
        o_proj.lhs.value
    );
    eprintln!(
        "pv context eval at claim point: {:?}",
        eval_context_values(
            &context_values,
            &o_proj.lhs.point,
            input.pv_matmul.params.seq,
            input.pv_matmul.params.q_heads,
            input.pv_matmul.params.head_dim,
        )
    );
    let pv_acc = eval_pv_acc(
        &input.pv_matmul.witness.p,
        &input.pv_matmul.witness.v,
        &o_proj.lhs.point,
        input.pv_matmul.params.seq,
        input.pv_matmul.params.q_heads,
        input.pv_matmul.params.kv_heads,
        input.pv_matmul.params.head_dim,
    );
    let pv_round_rhs = Fr::from(256_u64) * o_proj.lhs.value
        + eval_context_remainder(
            &input.pv_matmul.witness.context_remainder_bits,
            &o_proj.lhs.point,
            input.pv_matmul.params.seq,
            input.pv_matmul.params.q_heads,
            input.pv_matmul.params.head_dim,
        )
        - Fr::from(256_u64)
            * eval_context_msb(
                &input.pv_matmul.witness.context_remainder_bits[7],
                &o_proj.lhs.point,
                input.pv_matmul.params.seq,
                input.pv_matmul.params.q_heads,
                input.pv_matmul.params.head_dim,
            );
    eprintln!("pv acc direct: {pv_acc:?}");
    eprintln!("pv round rhs: {pv_round_rhs:?}");
    step("pv_matmul_only", || {
        prove_pv_matmul(o_proj.lhs.clone(), input.pv_matmul, &mut transcript)
    });
}

fn fox_trace_layer0() -> TraceLayerInput {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("qwen3-prover has a workspace parent");
    layer_input_from_trace_dir(
        workspace.join("traces/qwen3-0.6b/fox_eos_full_awy"),
        workspace.join("models/qwen3-0.6b/model.q8.bin"),
        0,
    )
    .expect("fox trace layer 0 converts")
}

fn eval_i32(
    values: &[i32],
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    point: &[Fr],
) -> Fr {
    assert_eq!(point.len(), 18);
    let mut value = Fr::from(0_u64);
    for row in 0..rows {
        for col in 0..cols {
            let padded_index = col * padded_rows + row;
            let source_index = row * cols + col;
            value += eq_eval(padded_index, point) * Fr::from_i32(values[source_index]);
        }
    }
    assert_eq!(padded_rows * padded_cols, 1 << point.len());
    value
}

fn eq_eval(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, point)| {
            if ((index >> bit) & 1) == 1 {
                *point
            } else {
                Fr::one() - point
            }
        })
        .product()
}

fn eval_matrix(values: &[i32], rows: usize, cols: usize, point: &[Fr]) -> Fr {
    let row_vars = rows.ilog2() as usize;
    let (row_point, col_point) = point.split_at(row_vars);
    (0..rows)
        .map(|row| {
            eq_eval(row, row_point)
                * (0..cols)
                    .map(|col| eq_eval(col, col_point) * Fr::from_i32(values[row * cols + col]))
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_context_values(
    values: &[i32],
    point: &[Fr],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Fr {
    let q_vars = seq.ilog2() as usize;
    let d_vars = head_dim.ilog2() as usize;
    let (q_point, rest) = point.split_at(q_vars);
    let (d_point, h_point) = rest.split_at(d_vars);
    (0..seq)
        .map(|q| {
            eq_eval(q, q_point)
                * (0..heads)
                    .map(|h| {
                        eq_eval(h, h_point)
                            * (0..head_dim)
                                .map(|d| {
                                    eq_eval(d, d_point)
                                        * Fr::from_i32(values[(q * heads + h) * head_dim + d])
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_rope_tensor(
    values: &[i32],
    params: &qwen3_prover::ops::rope::RopeParams,
    point: &[Fr],
) -> Fr {
    let seq_vars = params.seq.ilog2() as usize;
    let head_vars = params.heads.ilog2() as usize;
    let (seq_point, rest) = point.split_at(seq_vars);
    let (head_point, dim_point) = rest.split_at(head_vars);
    (0..params.seq)
        .map(|seq| {
            eq_eval(seq, seq_point)
                * (0..params.heads)
                    .map(|head| {
                        eq_eval(head, head_point)
                            * (0..params.head_dim)
                                .map(|dim| {
                                    eq_eval(dim, dim_point)
                                        * Fr::from_i32(
                                            values[(seq * params.heads + head) * params.head_dim
                                                + dim],
                                        )
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_score_values(values: &[i32], point: &[Fr], seq: usize, heads: usize) -> Fr {
    let q_vars = seq.ilog2() as usize;
    let h_vars = heads.ilog2() as usize;
    let (q_point, rest) = point.split_at(q_vars);
    let (h_point, k_point) = rest.split_at(h_vars);
    (0..heads)
        .map(|h| {
            eq_eval(h, h_point)
                * (0..seq)
                    .map(|q| {
                        eq_eval(q, q_point)
                            * (0..seq)
                                .map(|k| {
                                    eq_eval(k, k_point)
                                        * Fr::from_i32(values[(h * seq + q) * seq + k])
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_pv_acc(
    p: &[i32],
    v: &[i32],
    point: &[Fr],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Fr {
    let q_vars = seq.ilog2() as usize;
    let d_vars = head_dim.ilog2() as usize;
    let (q_point, rest) = point.split_at(q_vars);
    let (d_point, h_point) = rest.split_at(d_vars);
    let group = heads / kv_heads;
    (0..seq)
        .map(|q| {
            eq_eval(q, q_point)
                * (0..heads)
                    .map(|h| {
                        let kh = h / group;
                        eq_eval(h, h_point)
                            * (0..head_dim)
                                .map(|d| {
                                    eq_eval(d, d_point)
                                        * (0..seq)
                                            .map(|k| {
                                                Fr::from_i32(p[(h * seq + q) * seq + k])
                                                    * Fr::from_i32(
                                                        v[(k * kv_heads + kh) * head_dim + d],
                                                    )
                                            })
                                            .sum::<Fr>()
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_context_remainder(
    bits: &[Vec<bool>; 8],
    point: &[Fr],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, table)| {
            Fr::from(1_u64 << bit) * eval_context_msb(table, point, seq, heads, head_dim)
        })
        .sum()
}

fn eval_context_msb(bits: &[bool], point: &[Fr], seq: usize, heads: usize, head_dim: usize) -> Fr {
    let q_vars = seq.ilog2() as usize;
    let d_vars = head_dim.ilog2() as usize;
    let (q_point, rest) = point.split_at(q_vars);
    let (d_point, h_point) = rest.split_at(d_vars);
    (0..seq)
        .map(|q| {
            eq_eval(q, q_point)
                * (0..heads)
                    .map(|h| {
                        eq_eval(h, h_point)
                            * (0..head_dim)
                                .filter(|d| bits[(q * heads + h) * head_dim + *d])
                                .map(|d| eq_eval(d, d_point))
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn step<T>(name: &str, f: impl FnOnce() -> Option<T>) -> T {
    let started = std::time::Instant::now();
    eprintln!("start {name}");
    let result = f().unwrap_or_else(|| panic!("{name} failed"));
    eprintln!("finish {name}: {:.3}s", started.elapsed().as_secs_f64());
    result
}

fn hidden_out_iop_claim(traced: &TraceLayerInput) -> EvalClaim {
    let mut transcript = Blake2bTranscript::default();
    let point = transcript.challenge_vector::<Fr>(18);
    hidden_out_claim_at(traced, point)
}

fn hidden_out_claim_at(traced: &TraceLayerInput, point: Vec<Fr>) -> EvalClaim {
    EvalClaim {
        value: eval_i32(&traced.hidden_out, 197, 1024, 256, 1024, &point),
        point,
    }
}
