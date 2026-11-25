use jolt_transcript::{prover_transcript, verifier_transcript, Keccak};

use super::*;
use crate::protocol::pcs::ProofSection;

impl Model<'_> {
    fn checkpoint_domain(&self) -> [u8; 32] {
        match self.head {
            Head::Decoder { .. } => protocol_tag(b"atlas/decoder/layer-openings/v2"),
            Head::Encoder(_) => protocol_tag(b"atlas/encoder/layer-openings/v2"),
        }
    }

    pub(super) fn prove_checkpointed<C: ModelPcs>(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        input: &[i32],
        setup: &C::AdviceProverSetup,
    ) -> Result<(Vec<i32>, Proof<C>)> {
        let mut layout = self.layout.clone();
        let pre = matches!(self.head, Head::Decoder { .. });
        let mut inputs = Vec::with_capacity(self.layers.len() + 1);
        let mut specs = Vec::with_capacity(self.layers.len());
        let mut clips = Vec::with_capacity(self.layers.len());
        let mut layer_commitments = Vec::with_capacity(self.layers.len());

        let mut head_advice = layout.witness();
        inputs.push(match &self.head {
            Head::Decoder { .. } => input.to_vec(),
            Head::Encoder(norm) => norm.execute::<C::Field>(input, &mut head_advice)?.output,
        });
        for (i, layer) in self.layers.iter().enumerate() {
            layout.section = Some(i + 1);
            let mut advice = layout.witness();
            let mut trace = layer.execute::<C::Field>(&inputs[i], pre, &mut advice)?;
            let spec = layer.attention.clip_specs(&trace.attention);
            let clip = layer.attention.clip_plan(&mut layout, &spec)?;
            advice.extend(&layout);
            layer
                .attention
                .record_clip(&clip, &trace.attention, &mut advice)?;
            let next = if pre {
                std::mem::take(&mut trace.mlp_sum)
            } else {
                std::mem::take(&mut trace.mlp_norm.output)
            };
            drop(trace);
            specs.push(spec);
            clips.push(clip);
            layer_commitments.push(C::commit_section(
                &layout,
                &mut advice,
                &advice_plan::<C>(&layout)?,
                setup,
            )?);
            inputs.push(next);
        }

        layout.section = Some(0);
        head_advice.extend(&layout);
        let last = &inputs[self.layers.len()];
        let output = match &self.head {
            Head::Decoder { norm, logits } => {
                let norm = norm.execute::<C::Field>(last, &mut head_advice)?;
                logits.execute(&norm.output, &mut head_advice)?.output
            }
            Head::Encoder(_) => last.clone(),
        };
        let head = C::commit_section(
            &layout,
            &mut head_advice,
            &advice_plan::<C>(&layout)?,
            setup,
        )?;

        let mut t = prover_transcript::<Keccak>(&self.checkpoint_domain(), b"");
        self.bind(&mut t, tokens, segments, &output);
        bind_clips(&mut t, &specs);
        C::bind(&head, &mut t);
        for commitments in &layer_commitments {
            C::bind(commitments, &mut t);
        }

        let mut sections = Vec::with_capacity(inputs.len());
        let mut claims = Vec::new();
        let mut head = Some(head);
        if let Head::Decoder { norm, logits } = &self.head {
            sections.push(ProofSection::prove(
                &layout,
                head.take().unwrap(),
                setup,
                &mut t,
                |io| {
                    let mut advice = layout.witness();
                    let trace = norm.execute(last, &mut advice)?;
                    let rounded = logits.execute(&trace.output, &mut advice)?;
                    let claim = logits.reverse(
                        point(logits.shape(), &output, io)?,
                        || (&trace.output, &rounded),
                        io,
                    )?;
                    claims.push(norm.reverse(&[claim], || (last, &trace), io)?);
                    Ok(advice)
                },
            )?);
        } else {
            let mut messages = Default::default();
            let mut io = Session::prove(C::Field::prover(&mut t), &mut messages);
            // The point draws challenges only; no proof messages precede the first layer.
            claims.push(point(self.shape, &output, &mut io)?);
        }

        for (i, commitments) in layer_commitments.into_iter().enumerate().rev() {
            layout.section = Some(i + 1);
            let layer = &self.layers[i];
            let x = &inputs[i];
            sections.push(ProofSection::prove(
                &layout,
                commitments,
                setup,
                &mut t,
                |io| {
                    let mut advice = layout.witness();
                    let trace = layer.execute(x, pre, &mut advice)?;
                    layer
                        .attention
                        .record_clip(&clips[i], &trace.attention, &mut advice)?;
                    let mut lookups = Batches::default();
                    claims = layer.reverse(
                        self.shape,
                        pre,
                        std::mem::take(&mut claims),
                        &clips[i],
                        || &trace,
                        || x,
                        io,
                        &mut lookups,
                    )?;
                    lookups.finish(io)?;
                    Ok(advice)
                },
            )?);
        }
        if let Head::Encoder(norm) = &self.head {
            layout.section = Some(0);
            sections.push(ProofSection::prove(
                &layout,
                head.take().unwrap(),
                setup,
                &mut t,
                |io| {
                    let mut advice = layout.witness();
                    let trace = norm.execute(input, &mut advice)?;
                    norm.reverse(&claims, || (input, &trace), io)?;
                    Ok(advice)
                },
            )?);
        }
        // Store commitments and proofs in model order, independent of proof execution order.
        sections.reverse();
        if pre {
            sections.rotate_right(1);
        }
        Ok((
            output,
            Proof {
                clipping: specs,
                sections,
                transcript: t.narg_string().to_vec(),
            },
        ))
    }

    pub(super) fn verify_checkpointed<C: ModelPcs>(
        &self,
        tokens: &[usize],
        segments: Option<&[usize]>,
        input: &[i32],
        output: &[i32],
        proof: &Proof<C>,
        layout: &Layout,
        clips: &ClipPlans,
        setup: &C::AdviceVerifierSetup,
    ) -> Result<()> {
        let [head, layers @ ..] = proof.sections.as_slice() else {
            return Err("missing model proof sections");
        };
        if layers.len() != self.layers.len() {
            return Err("invalid model proof section count");
        }
        let mut layout = layout.clone();
        let mut t =
            verifier_transcript::<Keccak>(&self.checkpoint_domain(), b"", &proof.transcript);
        self.bind(&mut t, tokens, segments, output);
        bind_clips(&mut t, &proof.clipping);
        for section in &proof.sections {
            C::bind(&section.commitments, &mut t);
        }

        let mut claims = Vec::new();
        let pre = matches!(self.head, Head::Decoder { .. });
        if let Head::Decoder { norm, logits } = &self.head {
            layout.section = Some(0);
            head.verify(&layout, setup, &mut t, |io| {
                let claim =
                    logits.reverse(point(logits.shape(), output, io)?, || unreachable!(), io)?;
                claims.push(norm.reverse(&[claim], || unreachable!(), io)?);
                Ok(())
            })?;
        } else {
            let messages = crate::protocol::session::Messages::default();
            let mut io = messages.read(&mut t);
            claims.push(point(self.shape, output, &mut io)?);
            io.finish()?;
        }
        for (i, section) in layers.iter().enumerate().rev() {
            layout.section = Some(i + 1);
            section.verify(&layout, setup, &mut t, |io| {
                let mut lookups = Batches::default();
                claims = self.layers[i].reverse(
                    self.shape,
                    pre,
                    std::mem::take(&mut claims),
                    &clips[i],
                    || unreachable!(),
                    || unreachable!(),
                    io,
                    &mut lookups,
                )?;
                lookups.finish(io)
            })?;
        }
        if let Head::Encoder(norm) = &self.head {
            layout.section = Some(0);
            head.verify(&layout, setup, &mut t, |io| {
                claims = vec![norm.reverse(&claims, || unreachable!(), io)?];
                Ok(())
            })?;
        }
        if claims
            .iter()
            .any(|q| q.value != self.shape.eval(input, &q.point))
        {
            return Err("invalid public embedding evaluation");
        }
        t.check_eof().map_err(|_| "unused transcript bytes")
    }
}
