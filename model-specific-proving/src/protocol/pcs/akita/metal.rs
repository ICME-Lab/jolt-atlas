#[cfg(test)]
#[test]
fn native_metal_commitment_and_opening_verify_on_cpu() {
    use std::sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    };

    use super::F;
    use jolt_akita::{AkitaChunkProfile, AkitaNativeBatching, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_field::Ring;
    use jolt_openings::BatchOpeningScheme;
    use jolt_poly::{MultilinearPoly, OneHotPolynomial};
    use jolt_transcript::{prover_transcript, verifier_transcript, Keccak};
    use tracing_subscriber::{layer::Context, prelude::*, registry::LookupSpan, Layer};

    use crate::protocol::pcs::akita::{opening_statement, AkitaAdviceSetup, Key, K};

    // The fork sets `gpu_s` on these spans after a successful dispatch.
    struct Dispatches(Arc<AtomicU8>);
    impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Dispatches {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _: &tracing::Id,
            _: Context<'_, S>,
        ) {
            if attrs.metadata().name() == "TraceOneHotColumn::commit_inner" {
                self.0.fetch_or(4, Ordering::Relaxed);
            }
        }

        fn on_record(&self, id: &tracing::Id, _: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                let bit = match span.name() {
                    "metal_trace_commit_dispatch" => 1,
                    "metal_trace_packing_dispatch" | "metal_trace_fold_dispatch" => 2,
                    _ => return,
                };
                self.0.fetch_or(bit, Ordering::Relaxed);
            }
        }
    }
    let dispatches = Arc::new(AtomicU8::new(0));
    let subscriber = tracing_subscriber::registry().with(Dispatches(Arc::clone(&dispatches)));
    tracing::subscriber::set_global_default(subscriber).unwrap();
    let artifacts =
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
    let cpu = AkitaAdviceSetup::new(artifacts.clone()).unwrap();
    let gpu = AkitaAdviceSetup::new(artifacts)
        .unwrap()
        .with_metal()
        .unwrap();
    let key = Key {
        rounds: 21,
        count: 34,
    };
    assert!(gpu.native_metal(key));
    let polynomials: Arc<[OneHotPolynomial]> = (0..key.count)
        .map(|column| {
            let mut rows = vec![None; 1 << (key.rounds - K.ilog2() as usize)];
            for (row, value) in rows.iter_mut().take(64).enumerate() {
                *value = (row % 3 != 1).then_some(((row + column) % K) as u8);
            }
            OneHotPolynomial::new(K, rows)
        })
        .collect::<Vec<_>>()
        .into();
    let (expected, _) =
        AkitaScheme::commit_one_hot_group(&cpu.setup(key).unwrap(), key.digest(), &polynomials)
            .unwrap();
    let vk = cpu.verification_setup(key).unwrap();
    let point = vec![F::from_u64(3); key.rounds];
    let values: Vec<_> = polynomials
        .iter()
        .map(|poly| poly.evaluate(&point))
        .collect();
    let packed: Arc<[_]> = polynomials
        .iter()
        .map(|poly| {
            let mut indices = super::Indices::new(poly.num_rows());
            for (row, &column) in poly.indices().iter().enumerate() {
                if let Some(column) = column {
                    indices.insert(row, column);
                }
            }
            indices
        })
        .collect::<Vec<_>>()
        .into();
    for (setup, expected_dispatches) in [(&cpu, 4), (&gpu, 3)] {
        dispatches.store(0, Ordering::Relaxed);
        let pk = setup.setup(key).unwrap();
        let (commitment, hint) = setup.commit_group(key, &packed).unwrap();
        assert_eq!(commitment, expected);
        let statement = opening_statement(&commitment, &point, &values);
        let mut prover = prover_transcript::<Keccak>(&[1; 32], b"");
        let proof = AkitaNativeBatching::prove_batch(
            &pk,
            statement.clone(),
            polynomials
                .iter()
                .map(|poly| poly as &dyn MultilinearPoly<F>)
                .collect(),
            hint,
            &mut prover,
        )
        .unwrap();
        let mut verifier = verifier_transcript::<Keccak>(&[1; 32], b"", prover.narg_string());
        AkitaNativeBatching::verify_batch(&vk, &statement, &proof, &mut verifier).unwrap();
        assert_eq!(dispatches.load(Ordering::Relaxed), expected_dispatches);
    }
}
