use crate::akita::backend::AkitaScheduleArtifacts;
use crate::openings::{
    CommitmentGroupRole, GroupOpeningClaim, GroupOpeningWithHint, OpeningsError,
    TaggedGroupOpeningClaim, VerifierOpeningClaim,
};
use akita_params::{PolynomialGroupLayout, PrecommittedGroupProfiles};
use akita_pcs::custom_source::RootPolyMeta;
use akita_pcs::{AkitaError, CommitOutput, CpuBackend, GroupContext, SourceHandle};
use atlas_algebra::poly::point::EvaluationClaim;
use atlas_transcript::Transcript;
use serde::{Deserialize, Serialize};
use std::{
    collections::{btree_map::Entry, BTreeMap},
    sync::Arc,
};

use crate::akita::backend::adapters::{
    akita_error, commit_failed, dense_num_vars, dense_polynomials, invalid_batch, invalid_setup,
    jolt_to_akita_evals, one_hot_columns, one_hot_num_vars, serialize_akita, validate_one_hot_k,
    with_backend_pool, AkitaBackendCommitment, AkitaBackendDensePoly, AkitaBackendExtField,
    AkitaBackendFlavor, AkitaBackendHint, AkitaBackendOneHotPoly, AkitaBatchProof, AkitaCommitment,
    AkitaCpuBackend, AkitaField, AkitaHintSource, AkitaLayoutDigest, AkitaProverHint,
    AkitaProverSetup, AkitaSetupFlavor, AkitaSetupParams, AkitaVerifierScheduleArtifacts,
    AkitaVerifierSetup, BackendVerifierCache, FullWidthBackendSetup, AKITA_SOURCE_RING_DIMENSION,
};
use crate::akita::backend::configs::AkitaChunkProfile;
use crate::akita::backend::native_batching::AkitaNativeBatching;
use crate::akita::backend::one_hot_family::{with_one_hot_family, OneHotFamily};
use crate::akita::backend::trace_onehot::{TraceOneHotColumn, TraceOneHotRows};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaBackend;

fn split_commit_output(
    output: CommitOutput<AkitaField, AkitaBackendExtField>,
) -> (AkitaBackendCommitment, AkitaBackendHint) {
    (output.committed_group, output.private_handle)
}

/// Commits one imported one-hot source group under the setup's K-specific
/// scheme; `context` names the precommitted groups the row is keyed on.
fn commit_one_hot_source(
    setup: &AkitaProverSetup,
    backend: &AkitaCpuBackend,
    source: &SourceHandle<AkitaField, AkitaBackendExtField>,
    context: GroupContext<'_>,
) -> Result<(AkitaBackendCommitment, AkitaBackendHint), AkitaError> {
    let scheme = setup
        .verifier
        .one_hot_scheme()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    with_one_hot_family!(scheme scheme, |scheme| backend.commit(
        scheme.schedules(),
        source,
        context
    ))
    .map(split_commit_output)
}

/// Strictly ascending roles make the ordered commitment group list
/// unambiguous and forbid duplicate or permuted groups.
pub(crate) fn validate_group_order(
    roles: impl IntoIterator<Item = CommitmentGroupRole>,
) -> Result<(), OpeningsError> {
    let mut previous: Option<CommitmentGroupRole> = None;
    for role in roles {
        if let Some(previous) = previous {
            if role.order() <= previous.order() {
                return Err(invalid_batch(format!(
                    "Akita commitment groups must be in canonical ascending order, found {} after {}",
                    role.diagnostic_name(),
                    previous.diagnostic_name()
                )));
            }
        }
        previous = Some(role);
    }
    Ok(())
}

impl AkitaBackend {
    pub fn commit_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: &[Vec<AkitaField>],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = dense_num_vars(first)?;

        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        for polynomial in polynomials {
            if polynomial.len() != first.len() {
                return Err(invalid_batch(format!(
                    "Akita commitment group mixes {}-evaluation and {}-evaluation polynomials",
                    polynomial.len(),
                    first.len()
                )));
            }
        }

        let dense = dense_polynomials(polynomials)?;
        Self::commit_dense_backend(setup, layout_digest, num_vars, dense)
    }

    /// Commits a group of row-major one-hot columns (`columns[c][row]` is row `row`'s hot index)
    /// through the backend's one-hot flavor as one commitment object whose members are opened
    /// together at a shared point.
    pub fn commit_one_hot_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        columns: Vec<Vec<Option<u8>>>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = columns
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let rows = first.len();
        let num_vars = one_hot_num_vars(rows, setup.one_hot_k())?;
        Self::validate_commit_shape(setup, num_vars, columns.len())?;
        if let Some(column) = columns.iter().find(|column| column.len() != rows) {
            return Err(invalid_batch(format!(
                "Akita commitment group mixes {}-row and {rows}-row one-hot columns",
                column.len()
            )));
        }
        let poly_count = columns.len();
        let backend_polynomials = one_hot_columns(columns, setup.one_hot_k())?;
        let (backend_commitment, backend_hint) =
            Self::commit_one_hot_backend(setup, backend_polynomials)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::OneHot {
                poly_count,
                one_hot_k: setup.one_hot_k(),
            },
        )
    }

    /// Commits the native trace batch without constructing per-column
    /// index vectors or Akita's generic one-hot block representation.
    pub fn commit_trace_one_hot(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        rows: Arc<dyn TraceOneHotRows>,
        group_hints: &[&AkitaProverHint],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let profiles = if group_hints.is_empty() {
            None
        } else {
            Some(Self::group_profiles(setup, group_hints)?)
        };
        let source = TraceOneHotColumn::new(setup.one_hot_k(), AKITA_SOURCE_RING_DIMENSION, rows)
            .map_err(commit_failed)?;
        let num_vars = RootPolyMeta::num_vars(
            source
                .first()
                .ok_or_else(|| commit_failed("empty trace batch"))?,
        );
        let poly_count = source.len();
        Self::validate_commit_shape(setup, num_vars, poly_count)?;
        let one_hot_k = setup.one_hot_k();
        let (_, backend) = setup.one_hot_backend()?;
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = backend.import_source(source)?;
            let context = profiles.as_ref().map_or_else(
                GroupContext::scheduler_without_precommitted_groups,
                GroupContext::scheduler_with_precommitted_groups,
            );
            commit_one_hot_source(setup, backend, &source, context)
        })
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::TraceOneHot {
                poly_count,
                one_hot_k,
            },
        )
    }

    fn commit_one_hot_backend(
        setup: &AkitaProverSetup,
        polynomials: Vec<AkitaBackendOneHotPoly>,
    ) -> Result<(AkitaBackendCommitment, AkitaBackendHint), OpeningsError> {
        let (_, backend) = setup.one_hot_backend()?;
        with_backend_pool(|| {
            let source = backend.import_source(polynomials)?;
            commit_one_hot_source(
                setup,
                backend,
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map_err(commit_failed)
    }

    /// Freezes the ordered group profiles the final trace group commits
    /// against. Order is the caller's canonical role order; the backend keys the
    /// grouped row on this exact sequence.
    fn group_profiles(
        setup: &AkitaProverSetup,
        group_hints: &[&AkitaProverHint],
    ) -> Result<PrecommittedGroupProfiles, OpeningsError> {
        if group_hints.is_empty() {
            return Err(invalid_batch(
                "Akita grouped trace opening requires at least one auxiliary group",
            ));
        }
        let required = group_hints
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid_batch("Akita auxiliary group count overflows"))?;
        if setup.max_total_batch_polys() < required {
            return Err(invalid_batch(format!(
                "Akita grouped trace opening requires total polynomial capacity {required}, setup has {}",
                setup.max_total_batch_polys()
            )));
        }
        let mut profiles = Vec::with_capacity(group_hints.len());
        for hint in group_hints {
            if hint.commitment.backend_flavor != AkitaBackendFlavor::Dense
                || hint.commitment.poly_count != 1
                || !matches!(hint.source, AkitaHintSource::Dense { .. })
            {
                return Err(invalid_batch(
                    "Akita auxiliary group must contain one dense polynomial",
                ));
            }
            let (auxiliary_group, _) = hint.backend.as_ref().ok_or_else(|| {
                invalid_batch("Akita auxiliary group is missing backend opening data")
            })?;
            profiles.push(auxiliary_group.profile);
        }
        PrecommittedGroupProfiles::from_profiles(profiles).map_err(akita_error)
    }

    fn validate_commit_shape(
        setup: &AkitaProverSetup,
        num_vars: usize,
        poly_count: usize,
    ) -> Result<(), OpeningsError> {
        if num_vars != setup.max_num_vars() {
            return Err(invalid_batch(format!(
                "Akita commitment dimension {num_vars} does not match exact setup dimension {}",
                setup.max_num_vars()
            )));
        }
        if poly_count > setup.max_num_polys_per_commitment_group() {
            return Err(invalid_batch(format!(
                "Akita commitment group has {poly_count} polynomials but setup supports {}",
                setup.max_num_polys_per_commitment_group()
            )));
        }
        Ok(())
    }

    fn package_commitment(
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        backend_commitment: AkitaBackendCommitment,
        backend_hint: AkitaBackendHint,
        source: AkitaHintSource,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let backend_flavor = source.backend_flavor();
        let one_hot_k = match source {
            AkitaHintSource::Dense { .. } => 0,
            AkitaHintSource::OneHot { one_hot_k, .. }
            | AkitaHintSource::TraceOneHot { one_hot_k, .. } => one_hot_k,
        };
        let commitment = AkitaCommitment {
            backend_flavor,
            layout_digest,
            num_vars,
            poly_count: source.len(),
            one_hot_k,
            backend_coeff_len: backend_commitment.rows().coeff_len(),
            serialized_backend_bytes: serialize_akita(backend_commitment.commitment())?,
        };
        Ok((
            commitment.clone(),
            AkitaProverHint {
                commitment,
                backend: Some((backend_commitment, backend_hint)),
                source,
            },
        ))
    }

    fn commit_dense_backend(
        setup: &AkitaProverSetup,
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        dense: Vec<AkitaBackendDensePoly>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let poly_count = dense.len();
        let (_, backend) = setup.dense_backend()?;
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = backend.import_source(dense)?;
            backend.commit(
                setup
                    .verifier
                    .dense_scheme()
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?
                    .schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map(split_commit_output)
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::Dense { poly_count },
        )
    }
}

/// The scheme operations atlas drives: setup, single and batched commit, open and verify.
impl AkitaBackend {
    pub fn setup(
        params: AkitaSetupParams,
    ) -> Result<(AkitaProverSetup, AkitaVerifierSetup), OpeningsError> {
        if params.flavor == AkitaSetupFlavor::Dense
            && params.akita_chunk_profile != AkitaChunkProfile::default()
        {
            return Err(OpeningsError::InvalidSetup(
                "dense-only Akita setup cannot select one-hot witness chunking".to_owned(),
            ));
        }
        if params.flavor != AkitaSetupFlavor::Dense {
            OneHotFamily::from_parts(params.one_hot_k, params.akita_chunk_profile)
                .and_then(|family| {
                    family.validate_num_vars(
                        params.max_num_vars,
                        params.max_num_polys_per_commitment_group,
                    )
                })
                .map_err(invalid_setup)?;
        }
        if params.grouped_schedule.as_ref().is_some_and(|request| {
            request.final_group()
                != PolynomialGroupLayout::new(
                    params.max_num_vars,
                    params.max_num_polys_per_commitment_group,
                )
        }) {
            return Err(OpeningsError::InvalidSetup(
                "the grouped schedule request final shape must equal setup shape".to_owned(),
            ));
        }
        let artifacts = &params.schedule_artifacts;
        let dense_catalog = artifacts.dense_catalog().map_err(invalid_setup)?;
        let dense_schedule_artifact = || dense_catalog.to_artifact_bytes().map_err(invalid_setup);
        let one_hot_schedule_artifact = || {
            let base = artifacts
                .one_hot_catalog_for_profile(params.one_hot_k, params.akita_chunk_profile)
                .map_err(invalid_setup)?;
            let catalog = params
                .grouped_schedule
                .as_ref()
                .map_or_else(
                    || Ok(base.clone()),
                    |grouped_schedule| {
                        grouped_schedule.extend_catalog(
                            &dense_catalog,
                            &artifacts.full_dense_catalog()?,
                            &base,
                            params.one_hot_k,
                            params.akita_chunk_profile,
                        )
                    },
                )
                .map_err(invalid_setup)?;
            catalog.to_artifact_bytes().map_err(invalid_setup)
        };
        let profile = params.akita_chunk_profile;
        let schedule_artifacts = match params.flavor {
            AkitaSetupFlavor::Both => AkitaVerifierScheduleArtifacts::Both {
                dense_profile: artifacts.dense_profile(),
                profile,
                dense: dense_schedule_artifact()?,
                one_hot: one_hot_schedule_artifact()?,
            },
            AkitaSetupFlavor::OneHot => AkitaVerifierScheduleArtifacts::OneHot {
                profile,
                one_hot: one_hot_schedule_artifact()?,
            },
            AkitaSetupFlavor::Dense => AkitaVerifierScheduleArtifacts::Dense {
                dense_profile: artifacts.dense_profile(),
                dense: dense_schedule_artifact()?,
            },
        };
        let one_hot_log_k = validate_one_hot_k(params.one_hot_k)
            .map_err(|err| OpeningsError::InvalidSetup(err.to_string()))?;
        let verifier = AkitaVerifierSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            max_total_batch_polys: params.max_total_batch_polys,
            default_layout_digest: params.default_layout_digest,
            one_hot_k: params.one_hot_k,
            schedule_artifacts,
            backend_cache: BackendVerifierCache::default(),
        };
        let (backend_prover_setup, cpu_backend, backend_verifier_setup) =
            if params.flavor == AkitaSetupFlavor::OneHot {
                (None, None, None)
            } else {
                let scheme = verifier.dense_scheme()?;
                let backend_prover_setup = with_backend_pool(|| {
                    scheme.setup_prover(params.max_num_vars, params.max_total_batch_polys)
                })
                .map_err(invalid_setup)?;
                let cpu_backend = CpuBackend::new(Arc::clone(&backend_prover_setup.expanded))
                    .map_err(invalid_setup)?;
                let backend_verifier_setup =
                    with_backend_pool(|| scheme.setup_verifier(&backend_prover_setup))
                        .map_err(invalid_setup)?;
                (
                    Some(Arc::new(backend_prover_setup)),
                    Some(Arc::new(cpu_backend)),
                    Some(backend_verifier_setup),
                )
            };
        let (one_hot_backend_prover_setup, one_hot_cpu_backend, one_hot_backend_verifier_setup) =
            if params.max_num_vars >= one_hot_log_k && params.flavor != AkitaSetupFlavor::Dense {
                let backend_prover_setup =
                    crate::akita::backend::adapters::one_hot_setup_prover(&verifier)?;
                let cpu_backend = CpuBackend::new(Arc::clone(&backend_prover_setup.expanded))
                    .map_err(invalid_setup)?;
                let backend_verifier_setup =
                    crate::akita::backend::adapters::one_hot_setup_verifier(
                        &verifier,
                        &backend_prover_setup,
                    )?;
                (
                    Some(Arc::new(backend_prover_setup)),
                    Some(Arc::new(cpu_backend)),
                    Some(backend_verifier_setup),
                )
            } else {
                (None, None, None)
            };
        let mut full_width = BTreeMap::new();
        if let Some(request) = &params.grouped_schedule {
            let arities = request.full_width_arities().collect::<Vec<_>>();
            if !arities.is_empty() {
                let scheme = artifacts.full_dense_scheme().map_err(invalid_setup)?;
                for num_vars in arities {
                    if let Entry::Vacant(entry) = full_width.entry(num_vars) {
                        let setup = with_backend_pool(|| scheme.setup_prover(num_vars, 1))
                            .map_err(invalid_setup)?;
                        let backend =
                            CpuBackend::new(Arc::clone(&setup.expanded)).map_err(invalid_setup)?;
                        let _ = entry.insert(Arc::new(FullWidthBackendSetup {
                            scheme: scheme.clone(),
                            backend,
                        }));
                    }
                }
            }
        }
        verifier.prime_backend_cache(backend_verifier_setup, one_hot_backend_verifier_setup)?;
        let prover = AkitaProverSetup {
            full_width,
            backend_prover_setup,
            cpu_backend,
            one_hot_backend_prover_setup,
            one_hot_cpu_backend,
            verifier: verifier.clone(),
        };
        Ok((prover, verifier))
    }

    /// Commits one dense polynomial, given by its evaluations in index order (`evals[i]` at the
    /// Boolean point whose bits, most significant first, are `i`'s).
    pub fn commit(
        evals: &[AkitaField],
        setup: &AkitaProverSetup,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let num_vars = dense_num_vars(evals)?;
        Self::validate_commit_shape(setup, num_vars, 1)?;
        let dense = vec![AkitaBackendDensePoly::from_field_evals(
            num_vars,
            jolt_to_akita_evals(num_vars, evals)?,
        )
        .map_err(akita_error)?];
        Self::commit_dense_backend(setup, setup.default_layout_digest(), num_vars, dense)
    }

    /// Opens the dense polynomial `evals` at `point`, committing it first when no hint is given.
    pub fn open(
        evals: &[AkitaField],
        point: &[AkitaField],
        eval: AkitaField,
        setup: &AkitaProverSetup,
        hint: Option<AkitaProverHint>,
        transcript: &mut impl Transcript,
    ) -> Result<AkitaBatchProof, OpeningsError> {
        let hint = match hint {
            Some(hint) => hint,
            None => Self::commit(evals, setup)?.1,
        };
        let statement = vec![VerifierOpeningClaim {
            commitment: hint.commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        AkitaNativeBatching::prove_batch(setup, statement, hint, transcript)
    }

    pub fn verify(
        commitment: &AkitaCommitment,
        point: &[AkitaField],
        eval: AkitaField,
        proof: &AkitaBatchProof,
        setup: &AkitaVerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        AkitaNativeBatching::verify_batch(setup, &statement, proof, transcript)
    }

    pub fn prove_batch(
        setup: &AkitaProverSetup,
        auxiliary_groups: Vec<GroupOpeningWithHint<AkitaField, AkitaCommitment, AkitaProverHint>>,
        final_group: GroupOpeningClaim<AkitaField, AkitaCommitment>,
        final_hint: AkitaProverHint,
        transcript: &mut impl Transcript,
    ) -> Result<AkitaBatchProof, OpeningsError> {
        AkitaNativeBatching::prove_trace_batch(
            setup,
            auxiliary_groups,
            final_group,
            final_hint,
            transcript,
        )
    }

    pub fn verify_batch(
        setup: &AkitaVerifierSetup,
        auxiliary_groups: &[TaggedGroupOpeningClaim<AkitaField, AkitaCommitment>],
        final_group: &GroupOpeningClaim<AkitaField, AkitaCommitment>,
        proof: &AkitaBatchProof,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        AkitaNativeBatching::verify_trace_batch(
            setup,
            auxiliary_groups,
            final_group,
            proof,
            transcript,
        )
    }
}

impl AkitaBackend {
    /// The singleton commitment-object setup convention for advice and direct
    /// committed-program objects: one polynomial at `num_vars`, seeded by the
    /// object plan's layout digest. Every bounded-dense object commits through
    /// the dense flavor, so the costly one-hot backend setup is never built.
    pub fn transparent_object_setup(
        context: &Arc<AkitaScheduleArtifacts>,
        num_vars: usize,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaProverSetup, AkitaVerifierSetup), OpeningsError> {
        Self::setup(AkitaSetupParams::dense_only(
            num_vars,
            1,
            layout_digest,
            context.clone(),
        ))
    }

    pub fn commit_full_width_object(
        setup: &AkitaProverSetup,
        evals: &[AkitaField],
        layout_digest: [u8; 32],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let num_vars = dense_num_vars(evals)?;
        let cached = setup.full_width.get(&num_vars).ok_or_else(|| {
            invalid_setup("full-width object arity was not admitted during preprocessing")
        })?;
        let dense = vec![AkitaBackendDensePoly::from_field_evals(
            num_vars,
            jolt_to_akita_evals(num_vars, evals)?,
        )
        .map_err(akita_error)?];
        let (backend_commitment, backend_hint) = with_backend_pool(|| {
            let source = cached.backend.import_source(dense)?;
            cached.backend.commit(
                cached.scheme.schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .map(split_commit_output)
        .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintSource::Dense { poly_count: 1 },
        )
    }

    pub fn retag_transparent_object_setup(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaProverSetup, AkitaVerifierSetup), OpeningsError> {
        let _ = setup.dense_backend()?;
        if setup.max_num_polys_per_commitment_group() != 1 || setup.max_total_batch_polys() != 1 {
            return Err(invalid_batch(
                "a transparent object setup must admit one polynomial",
            ));
        }
        let mut retagged = setup.clone();
        retagged.verifier.default_layout_digest = layout_digest;
        let verifier = retagged.verifier.clone();
        Ok((retagged, verifier))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]
    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]
    #![expect(clippy::indexing_slicing, reason = "tests index fixture data")]
    #![expect(clippy::panic, reason = "tests reject impossible fixture variants")]

    use super::*;
    use crate::akita::backend::adapters::{
        append_verifier_setup, AkitaBackendFlavor, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
    };
    use crate::akita::backend::configs::JoltDenseBounded;
    use crate::akita::backend::AkitaOneHotScheduleArtifact;
    use akita_config::CommitmentConfig;
    use akita_schedules::ValidatedScheduleCatalog;
    use atlas_algebra::field::Ring;
    use atlas_algebra::poly::eq::eq_index_msb;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };

    /// `Σ_i evals[i] · eq(point, i)`, bits of `i` most significant first.
    fn dense_eval(evals: &[AkitaField], point: &[AkitaField]) -> AkitaField {
        evals
            .iter()
            .enumerate()
            .map(|(i, &e)| e * eq_index_msb(point, i as u128))
            .sum()
    }

    /// A row-major one-hot column's evaluation: row `row` hot at `k` is index `row · K + k`.
    fn one_hot_eval(one_hot_k: usize, indices: &[Option<u8>], point: &[AkitaField]) -> AkitaField {
        indices
            .iter()
            .enumerate()
            .filter_map(|(row, k)| k.map(|k| row * one_hot_k + k as usize))
            .map(|index| eq_index_msb(point, index as u128))
            .sum()
    }

    /// The sponge state after the transcript's last operation.
    fn state<T: Transcript>(transcript: &T) -> [u8; 32] {
        *transcript
            .state_history()
            .last()
            .expect("a transcript records its construction")
    }

    #[test]
    fn full_width_objects_do_not_widen_bounded_dense_commitments() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        use crate::akita::backend::schedule_registry::{DenseGroupLayout, GroupedScheduleParams};
        let mut params = AkitaSetupParams::dense_only(14, 1, [7; 32], artifacts);
        params.grouped_schedule = Some(GroupedScheduleParams::new(
            None,
            None,
            vec![DenseGroupLayout::FullWidth { num_vars: 14 }],
            PolynomialGroupLayout::new(14, 1),
        ));
        let (bounded, _) = AkitaBackend::setup(params).unwrap();
        let polynomial = vec![AkitaField::pow2(80); 1 << 14];
        assert!(AkitaBackend::commit(&polynomial, &bounded).is_err());
        let (commitment, hint) =
            AkitaBackend::commit_full_width_object(&bounded, &polynomial, [7; 32]).unwrap();
        assert_eq!(commitment, hint.commitment);
        assert_eq!(commitment.num_vars, 14);
    }

    #[test]
    fn setup_key_transcript_binds_backend_shape() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            max_total_batch_polys: 1,
            default_layout_digest: [7; 32],
            one_hot_k: AKITA_ONE_HOT_K256,
            schedule_artifacts: AkitaVerifierScheduleArtifacts::Both {
                dense_profile: artifacts.dense_profile(),
                profile: AkitaChunkProfile::default(),
                dense: artifacts
                    .dense_catalog()
                    .unwrap()
                    .to_artifact_bytes()
                    .unwrap(),
                one_hot: artifacts
                    .one_hot_catalog(AKITA_ONE_HOT_K256)
                    .unwrap()
                    .to_artifact_bytes()
                    .unwrap(),
            },
            backend_cache: Default::default(),
        };
        let mut baseline = Blake2bVerifier::new(b"akita-setup-key-test", Vec::new());
        let initial_state = state(&baseline);

        append_verifier_setup(&mut baseline, &setup, AkitaBackendFlavor::Dense).unwrap();
        assert_ne!(state(&baseline), initial_state);

        let mut same = Blake2bVerifier::new(b"akita-setup-key-test", Vec::new());
        append_verifier_setup(&mut same, &setup, AkitaBackendFlavor::Dense).unwrap();
        assert_eq!(state(&baseline), state(&same));

        let mut flavor_transcript = Blake2bVerifier::new(b"akita-setup-key-test", Vec::new());
        append_verifier_setup(&mut flavor_transcript, &setup, AkitaBackendFlavor::OneHot).unwrap();
        assert_ne!(state(&baseline), state(&flavor_transcript));

        let mut changed_shape = setup.clone();
        changed_shape.max_num_vars = 5;
        let mut shape_transcript = Blake2bVerifier::new(b"akita-setup-key-test", Vec::new());
        append_verifier_setup(
            &mut shape_transcript,
            &changed_shape,
            AkitaBackendFlavor::Dense,
        )
        .unwrap();
        assert_ne!(state(&baseline), state(&shape_transcript));

        let mut changed_digest = setup;
        changed_digest.default_layout_digest = [8; 32];
        let mut digest_transcript = Blake2bVerifier::new(b"akita-setup-key-test", Vec::new());
        append_verifier_setup(
            &mut digest_transcript,
            &changed_digest,
            AkitaBackendFlavor::Dense,
        )
        .unwrap();
        assert_ne!(state(&baseline), state(&digest_transcript));

        let mut changed_k = changed_digest;
        changed_k.one_hot_k = AKITA_ONE_HOT_K16;
        let mut k_transcript = Blake2bVerifier::new(b"akita-setup-key-test", Vec::new());
        append_verifier_setup(&mut k_transcript, &changed_k, AkitaBackendFlavor::Dense).unwrap();
        assert_ne!(state(&digest_transcript), state(&k_transcript));
    }

    #[test]
    fn transparent_object_setup_reuses_backend_for_a_new_layout() {
        let context =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (base, _) = AkitaBackend::transparent_object_setup(&context, 14, [3; 32]).unwrap();
        let (retagged, verifier) =
            AkitaBackend::retag_transparent_object_setup(&base, [4; 32]).unwrap();

        assert!(Arc::ptr_eq(
            base.backend_prover_setup.as_ref().unwrap(),
            retagged.backend_prover_setup.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(
            base.cpu_backend.as_ref().unwrap(),
            retagged.cpu_backend.as_ref().unwrap()
        ));
        assert_eq!(retagged.default_layout_digest(), [4; 32]);
        assert_eq!(verifier.default_layout_digest(), [4; 32]);
    }

    fn one_hot_roundtrip(
        one_hot_k: usize,
        profile: AkitaChunkProfile,
        num_vars: usize,
        num_polys: usize,
    ) {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let setup_params =
            AkitaSetupParams::one_hot_only(num_vars, num_polys, [4; 32], one_hot_k, artifacts)
                .with_akita_chunk_profile(profile);
        let (prover_setup, verifier_setup) = AkitaBackend::setup(setup_params).unwrap();
        let row_count = 1usize << (num_vars - one_hot_k.ilog2() as usize);
        let polynomials = (0..num_polys)
            .map(|poly| {
                (0..row_count)
                    .map(|row| {
                        if row == 2 {
                            None
                        } else {
                            Some(((row + poly) % one_hot_k) as u8)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (commitment, hint) =
            AkitaBackend::commit_one_hot_group(&prover_setup, [4; 32], polynomials.clone())
                .unwrap();
        assert_eq!(commitment.one_hot_k(), one_hot_k);

        let point = vec![AkitaField::from_u64(3); num_vars];
        let statement = polynomials
            .iter()
            .map(|polynomial| VerifierOpeningClaim {
                commitment: commitment.clone(),
                evaluation: EvaluationClaim::new(
                    point.clone(),
                    one_hot_eval(one_hot_k, polynomial, &point),
                ),
            })
            .collect::<Vec<_>>();
        let mut prover_transcript = Blake2bProver::new(b"akita-one-hot-k");
        let proof = AkitaNativeBatching::prove_batch(
            &prover_setup,
            statement.clone(),
            hint,
            &mut prover_transcript,
        )
        .unwrap();
        let mut verifier_transcript = Blake2bVerifier::new(b"akita-one-hot-k", Vec::new());
        AkitaNativeBatching::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
        assert_eq!(state(&prover_transcript), state(&verifier_transcript));

        if num_vars == 16 && num_polys == 1 {
            let one_hot = match &verifier_setup.schedule_artifacts {
                AkitaVerifierScheduleArtifacts::OneHot { one_hot, .. } => one_hot.clone(),
                AkitaVerifierScheduleArtifacts::Dense { .. }
                | AkitaVerifierScheduleArtifacts::Both { .. } => {
                    panic!("one-hot fixture must carry a one-hot-only catalog")
                }
            };
            let wrong_profile = if profile == AkitaChunkProfile::new(4, 2) {
                AkitaChunkProfile::new(8, 2)
            } else {
                AkitaChunkProfile::new(4, 2)
            };
            for schedule_artifacts in [
                AkitaVerifierScheduleArtifacts::OneHot {
                    profile: wrong_profile,
                    one_hot: one_hot.clone(),
                },
                AkitaVerifierScheduleArtifacts::OneHot {
                    profile: AkitaChunkProfile::default(),
                    one_hot,
                },
            ] {
                if schedule_artifacts == verifier_setup.schedule_artifacts {
                    continue;
                }
                let mut tampered = verifier_setup.clone();
                tampered.schedule_artifacts = schedule_artifacts;
                // Transport drops the honest setup's primed backend caches.
                let bytes = serde_json::to_vec(&tampered).unwrap();
                let transported: AkitaVerifierSetup = serde_json::from_slice(&bytes).unwrap();
                let mut transcript = Blake2bVerifier::new(b"akita-one-hot-k", Vec::new());
                let error = AkitaNativeBatching::verify_batch(
                    &transported,
                    &statement,
                    &proof,
                    &mut transcript,
                )
                .expect_err("a verifier profile must match its embedded catalog");
                assert!(matches!(error, OpeningsError::InvalidSetup(_)));
            }
        }

        let mut wrong_k_statement = statement;
        wrong_k_statement[0].commitment.one_hot_k = if one_hot_k == AKITA_ONE_HOT_K16 {
            AKITA_ONE_HOT_K256
        } else {
            AKITA_ONE_HOT_K16
        };
        let mut verifier_transcript = Blake2bVerifier::new(b"akita-one-hot-k", Vec::new());
        let _ = AkitaNativeBatching::verify_batch(
            &verifier_setup,
            &wrong_k_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("commitment K must match verifier setup K");
    }

    #[test]
    fn one_hot_k16_roundtrip() {
        for profile in [AkitaChunkProfile::new(1, 0), AkitaChunkProfile::new(8, 2)] {
            let family = OneHotFamily::from_parts(AKITA_ONE_HOT_K16, profile).unwrap();
            for num_polys in [1, 2] {
                for num_vars in family.num_vars_range(num_polys).0..=16 {
                    one_hot_roundtrip(AKITA_ONE_HOT_K16, profile, num_vars, num_polys);
                }
            }
        }
    }

    #[test]
    fn one_hot_k256_roundtrip() {
        for profile in [AkitaChunkProfile::new(1, 0), AkitaChunkProfile::new(8, 2)] {
            let family = OneHotFamily::from_parts(AKITA_ONE_HOT_K256, profile).unwrap();
            for num_polys in [1, 2] {
                for num_vars in family.num_vars_range(num_polys).0..=16 {
                    one_hot_roundtrip(AKITA_ONE_HOT_K256, profile, num_vars, num_polys);
                }
            }
        }
    }

    /// A serde roundtrip drops the primed key cache; the transported setup
    /// must re-derive the same backend key from its shape.
    #[test]
    fn serde_transported_setup_rederives_the_backend_key() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (_, verifier_setup) =
            AkitaBackend::setup(AkitaSetupParams::new(14, 1, [3; 32], artifacts)).unwrap();
        let json = serde_json::to_string(&verifier_setup).unwrap();
        let transported: AkitaVerifierSetup = serde_json::from_str(&json).unwrap();
        assert_eq!(transported, verifier_setup);
        let binary = bincode::serde::encode_to_vec(&verifier_setup, bincode::config::standard())
            .expect("verifier setup must encode with Jolt's binary transport");
        let (binary_transport, consumed): (AkitaVerifierSetup, usize) =
            bincode::serde::decode_from_slice(&binary, bincode::config::standard())
                .expect("verifier setup must decode with Jolt's binary transport");
        assert_eq!(consumed, binary.len());
        assert_eq!(binary_transport, verifier_setup);
        let rederived = transported
            .dense_verifier()
            .expect("shape-only setup re-derives its backend key");
        let primed = verifier_setup
            .dense_verifier()
            .expect("primed cache returns the built key");
        assert_eq!(
            serialize_akita(rederived.setup()).unwrap(),
            serialize_akita(primed.setup()).unwrap(),
            "re-derived backend key must match the primed one"
        );
    }

    #[test]
    fn serde_transported_recursive_grouped_setup_restores_its_schedule_rows() {
        use crate::akita::backend::schedule_registry::{
            GroupedScheduleParams, FIXTURE_TRUSTED_ADVICE_GROUP,
        };
        use crate::akita::backend::schedules::emit::{
            K16_COLUMN_VARIABLES, RECURSIVE_TRACE_LOG_T_CUTOVER,
        };

        // The narrowest production trace at the cutover; one-column scalar
        // rows at this arity stay direct.
        let final_group =
            PolynomialGroupLayout::new(RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES, 51);
        let grouped_schedule = GroupedScheduleParams::new(
            None,
            Some(FIXTURE_TRUSTED_ADVICE_GROUP.num_vars()),
            Vec::new(),
            final_group,
        );
        let (_, verifier_setup) = AkitaBackend::setup(AkitaSetupParams::one_hot_only_grouped(
            final_group.num_vars(),
            final_group.num_polynomials(),
            final_group.num_polynomials() + 1,
            [3; 32],
            AKITA_ONE_HOT_K16,
            Some(grouped_schedule),
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        ))
        .unwrap();
        let selection = with_one_hot_family!(scheme verifier_setup.one_hot_scheme().unwrap(), |scheme| {
            scheme
                .schedules()
                .rows()
                .find(|row| !row.profiles().precommitteds.is_empty())
                .unwrap()
                .selection()
        });
        let json = serde_json::to_string(&verifier_setup).unwrap();

        let transported: AkitaVerifierSetup = serde_json::from_str(&json).unwrap();
        let resolved = with_one_hot_family!(scheme transported.one_hot_scheme().unwrap(), |scheme| {
            scheme
                .schedules()
                .resolve_selection(selection)
                .expect("transported grouped setup must carry its schedule row")
        });
        assert!(!resolved.profiles().precommitteds.is_empty());
        assert!(resolved
            .schedule()
            .recursive_folds
            .iter()
            .any(|fold| fold.params.setup_prefix().is_some()));
    }

    #[test]
    fn grouped_setup_rejects_a_final_arity_different_from_the_main_setup() {
        use crate::akita::backend::schedule_registry::GroupedScheduleParams;

        let request = GroupedScheduleParams::new(
            None,
            Some(14),
            Vec::new(),
            PolynomialGroupLayout::new(15, 1),
        );
        let error = AkitaBackend::setup(AkitaSetupParams::one_hot_only_grouped(
            14,
            1,
            2,
            [3; 32],
            AKITA_ONE_HOT_K16,
            Some(request),
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        ))
        .expect_err("a grouped request for another final arity must fail during setup");
        assert!(error
            .to_string()
            .contains("final shape must equal setup shape"));
    }

    #[test]
    fn catalog_digest_prevents_cross_verification_with_the_same_selected_row() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (prover_setup, verifier_setup) = AkitaBackend::setup(AkitaSetupParams::dense_only(
            14,
            1,
            [7; 32],
            Arc::clone(&artifacts),
        ))
        .unwrap();
        let polynomial: Vec<AkitaField> = (0..(1u64 << 14))
            .map(|i| AkitaField::from_u64(2 + 5 * i))
            .collect();
        let (commitment, hint) = AkitaBackend::commit(&polynomial, &prover_setup).unwrap();
        let point = (3..17).map(AkitaField::from_u64).collect::<Vec<_>>();
        let value = dense_eval(&polynomial, &point);
        let statement = vec![VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point, value),
        }];
        let mut prover_transcript = Blake2bProver::new(b"catalog-replay");
        let proof = AkitaNativeBatching::prove_batch(
            &prover_setup,
            statement.clone(),
            hint,
            &mut prover_transcript,
        )
        .unwrap();

        let selected = proof.selection();
        let full_catalog = artifacts.dense_catalog().unwrap();
        let omitted = full_catalog
            .rows()
            .find(|row| row.selection() != selected)
            .expect("the base catalog must contain an unused row")
            .selection();
        let reduced_catalog = ValidatedScheduleCatalog::try_new(
            JoltDenseBounded::schedule_family_name(),
            full_catalog
                .rows()
                .filter(|row| row.selection() != omitted)
                .map(|row| (row.profiles().clone(), row.schedule().clone())),
            full_catalog.policy(),
            JoltDenseBounded::ring_challenge_config,
        )
        .unwrap();
        assert!(reduced_catalog.resolve_selection(selected).is_ok());
        assert_ne!(
            full_catalog.catalog_digest(),
            reduced_catalog.catalog_digest()
        );
        let alternate_artifacts = Arc::new(
            AkitaScheduleArtifacts::new(
                artifacts.dense_profile(),
                reduced_catalog.to_artifact_bytes().unwrap(),
                artifacts
                    .full_dense_catalog()
                    .unwrap()
                    .to_artifact_bytes()
                    .unwrap(),
                [AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256]
                    .into_iter()
                    .map(|k| AkitaOneHotScheduleArtifact {
                        k,
                        profile: AkitaChunkProfile::default(),
                        bytes: artifacts
                            .one_hot_catalog(k)
                            .unwrap()
                            .to_artifact_bytes()
                            .unwrap(),
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let (_, alternate_verifier_setup) = AkitaBackend::setup(AkitaSetupParams::dense_only(
            14,
            1,
            [7; 32],
            alternate_artifacts,
        ))
        .unwrap();
        let mut verifier_transcript = Blake2bVerifier::new(b"catalog-replay", Vec::new());
        let _ = AkitaNativeBatching::verify_batch(
            &alternate_verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("a proof must not replay across catalogs with different digests");

        let mut original_transcript = Blake2bVerifier::new(b"catalog-replay", Vec::new());
        AkitaNativeBatching::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut original_transcript,
        )
        .unwrap();
    }

    #[test]
    fn direct_opening_requires_statement_commitment_layout_digest() {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let setup_params = AkitaSetupParams::new(14, 1, [7; 32], artifacts);
        let (prover_setup, verifier_setup) = AkitaBackend::setup(setup_params).unwrap();
        let polynomial: Vec<AkitaField> = (0..(1u64 << 14))
            .map(|i| AkitaField::from_u64(2 + 5 * i))
            .collect();
        let commitment_digest = [9; 32];
        let (commitment, hint) = AkitaBackend::commit_group(
            &prover_setup,
            commitment_digest,
            std::slice::from_ref(&polynomial),
        )
        .expect("direct commitment may use its own layout digest");
        assert_eq!(commitment.layout_digest, commitment_digest);

        let point = (3..17).map(AkitaField::from_u64).collect::<Vec<_>>();
        let claim = dense_eval(&polynomial, &point);
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.clone(), claim),
        }];

        let mut prover_transcript = Blake2bProver::new(b"akita-direct-layout");
        let proof = AkitaNativeBatching::prove_batch(
            &prover_setup,
            statement.clone(),
            hint,
            &mut prover_transcript,
        )
        .expect("direct proof should prove");

        let mut verifier_transcript = Blake2bVerifier::new(b"akita-direct-layout", Vec::new());
        AkitaNativeBatching::verify_batch(
            &verifier_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect("direct proof should verify");
        assert_eq!(state(&prover_transcript), state(&verifier_transcript));

        let mut changed_commitment_statement = statement.clone();
        changed_commitment_statement[0].commitment.layout_digest = [15; 32];
        let mut verifier_transcript = Blake2bVerifier::new(b"akita-direct-layout", Vec::new());
        let _error = AkitaNativeBatching::verify_batch(
            &verifier_setup,
            &changed_commitment_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("changed direct commitment digest should reject");

        let mut changed_setup = verifier_setup;
        changed_setup.default_layout_digest = commitment_digest;
        let mut verifier_transcript = Blake2bVerifier::new(b"akita-direct-layout", Vec::new());
        let _error = AkitaNativeBatching::verify_batch(
            &changed_setup,
            &statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("direct commitment layout must not be accepted through setup default");
    }
}
