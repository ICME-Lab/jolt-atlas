//! Regenerate the embedded Atlas Akita catalogs from the workspace root.
//! `cargo run -p atlas-pcs --features akita-schedule-generator --bin generate_akita_schedules --profile build-fast`

use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog, ValidatedScheduleCatalog};
use akita_params::{
    CommittedGroupBatchProfile, GroupCommitPhaseParams, OpeningClaimsLayout, ScheduleLookupKey,
};
use jolt_akita::configs::{
    AkitaChunkProfile, JoltDenseBounded, JoltOneHotK16, JoltOneHotK16Direct, JoltOneHotK256,
    JoltOneHotK256Direct,
};
use jolt_akita::AkitaScheduleArtifacts;
use rayon::prelude::*;
use std::{error::Error, fs, path::Path};

fn generate<Plan: CommitmentConfig, Catalog: CommitmentConfig>(
    profile: AkitaChunkProfile,
    shapes: &[(usize, usize)],
) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let chunking = profile.witness_cfg();
    let policy = policy_of::<Plan>(chunking);
    let source = Plan::committed_source_contract()?;
    let rows = shapes
        .par_iter()
        .map(|&(num_vars, num_polynomials)| {
            let layout =
                OpeningClaimsLayout::new(num_vars, num_polynomials)?.root_final_group_layout()?;
            let plan = akita_planner::find_schedule(
                &ScheduleLookupKey::single(layout),
                source,
                &[],
                &policy,
                Plan::ring_challenge_config,
            )?;
            let final_group =
                GroupCommitPhaseParams::try_from_params(layout, &plan.schedule.root.params)?;
            Ok((
                CommittedGroupBatchProfile {
                    final_group,
                    precommitteds: Vec::new(),
                },
                plan.schedule,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error + Send + Sync>>>()?;
    let catalog = ValidatedScheduleCatalog::try_new(
        Catalog::schedule_family_name(),
        rows,
        &policy_of::<Catalog>(chunking),
        Catalog::ring_challenge_config,
    )?;
    let bytes = catalog.to_artifact_bytes()?;
    TrustedScheduleCatalog::<Catalog>::from_artifact_bytes(&bytes, chunking)?;
    Ok(bytes)
}

fn merge<Cfg: CommitmentConfig>(
    base: ValidatedScheduleCatalog,
    extra: &[u8],
    profile: AkitaChunkProfile,
) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let chunking = profile.witness_cfg();
    let extra = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(extra, chunking)?;
    let rows = base
        .rows()
        .chain(extra.catalog().rows())
        .map(|row| (row.profiles().clone(), row.schedule().clone()));
    let merged = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows,
        &policy_of::<Cfg>(chunking),
        Cfg::ring_challenge_config,
    )?;
    Ok(merged.to_artifact_bytes()?)
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let one_hot = AkitaChunkProfile::new(1, 0);
    let dense = AkitaChunkProfile::new(8, 2);
    let base = AkitaScheduleArtifacts::from_directory(
        AkitaScheduleArtifacts::packaged_directory(),
        dense,
    )?;
    // Each shape is (physical arity, polynomial count). One-hot arities
    // include the column index variables in addition to the row index.
    // At arity 10 the planner only supports groups of four and eight.
    // At arities 14 and above the base catalog already covers one and two.
    let dense_shapes = [(10, 4), (10, 8), (15, 16)]
        .into_iter()
        .chain((11..=13).flat_map(|vars| [1, 2, 4, 8].map(move |count| (vars, count))))
        .chain((14..=18).flat_map(|vars| [4, 8].map(move |count| (vars, count))))
        .collect::<Vec<_>>();
    let artifacts = [
        (
            "dense-bounded.aks",
            merge::<JoltDenseBounded>(
                base.dense_catalog()?,
                &generate::<JoltDenseBounded, JoltDenseBounded>(dense, &dense_shapes)?,
                dense,
            )?,
        ),
        (
            "dense-full.aks",
            base.full_dense_catalog()?.to_artifact_bytes()?,
        ),
        (
            "one-hot-k16.aks",
            merge::<JoltOneHotK16>(
                base.one_hot_catalog(16)?,
                &generate::<JoltOneHotK16Direct, JoltOneHotK16>(one_hot, &[(14, 16), (16, 32)])?,
                one_hot,
            )?,
        ),
        (
            "one-hot-k256.aks",
            merge::<JoltOneHotK256>(
                base.one_hot_catalog(256)?,
                &generate::<JoltOneHotK256Direct, JoltOneHotK256>(one_hot, &[(16, 8)])?,
                one_hot,
            )?,
        ),
    ];
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/akita/schedules");
    for (name, bytes) in artifacts {
        fs::write(directory.join(name), bytes)?;
    }
    Ok(())
}
