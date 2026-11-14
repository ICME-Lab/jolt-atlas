//! Jolt-owned external schedule catalog generation.
//!
//! Checked-in `.aks` artifacts are ordinary runtime data, not Rust modules.
//! Regenerate them with:
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
//! # Only the full-field dense family:
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules dense-full
//! ```

/// Emit-spec construction shared by the generator and drift tests.
pub mod emit {

    use std::path::{Path, PathBuf};

    use akita_config::recursive_commitment::{RecursiveCommitmentConfig, RecursiveScheduleConfig};
    use akita_config::{policy_of, CommitmentConfig};
    use akita_params::MultiChunkProfileId;
    use akita_params::{
        ChunkedWitnessCfg, OpeningClaimsLayout, PolynomialGroupLayout, ScheduleLookupKey,
    };
    use akita_pcs::AkitaError;
    use akita_planner::emit::bounded_parallel_filter_map;
    use akita_planner::{EmitSpec, PlannerPolicy};

    use crate::configs::{
        AkitaChunkProfile, JoltDenseBounded, JoltDenseFull, JoltOneHotK16Direct,
        JoltOneHotK256Direct,
    };
    use crate::one_hot_family::OneHotFamily;
    use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

    /// Admitted production trace exponents for K=16.
    pub const K16_TRACE_LOG_T: (usize, usize) = (12, 31);
    pub const K16_NUM_VARS: (usize, usize) = (
        K16_TRACE_LOG_T.0 + K16_COLUMN_VARIABLES,
        K16_TRACE_LOG_T.1 + K16_COLUMN_VARIABLES,
    );
    /// One- and two-polynomial rows used by adapter and grouped-planner diagnostics.
    pub const ONE_HOT_SCALAR_NUM_POLYS: &[usize] = &[1, 2];
    /// Bounded-dense advice and committed-program byte objects.
    pub const DENSE_NUM_VARS: (usize, usize) = (14, 35);

    /// Admitted K=16 trace one-hot column counts: the widths of the RV64 zkVM's committed
    /// one-hot trace over every bytecode (1–32 bits) and RAM (1–61 bits) size, as upstream's
    /// `jolt-claims` geometry enumerated them at `f3265b5a`, capped at 64 columns.
    pub const K16_TRACE_COLUMN_WIDTHS: [usize; 14] =
        [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];

    /// First logical trace exponent using setup offloading.
    pub const RECURSIVE_TRACE_LOG_T_CUTOVER: usize = 21;
    pub const K16_COLUMN_VARIABLES: usize = 4;
    pub const K256_COLUMN_VARIABLES: usize = 8;

    /// Reachable scalar keys for one family grid.
    pub fn keys(
        num_polys: &[usize],
        (min_vars, max_vars): (usize, usize),
    ) -> Vec<PolynomialGroupLayout> {
        let mut keys = Vec::new();
        for &polys in num_polys {
            for num_vars in min_vars..=max_vars {
                let layout = OpeningClaimsLayout::new(num_vars, polys)
                    .and_then(|layout| layout.root_final_group_layout());
                if let Ok(key) = layout {
                    keys.push(key);
                }
            }
        }
        keys
    }

    /// Default single-chunk shapes and the adapter's multi-chunk test fixtures.
    pub fn one_hot_keys(
        one_hot_k: usize,
        profile: AkitaChunkProfile,
    ) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
        let family = OneHotFamily::from_parts(one_hot_k, profile)?;
        if profile != AkitaChunkProfile::new(1, 0) {
            // These rows serve adapter roundtrips, grouped cutovers, and small guest
            // fixtures. Applications provision their own deployment coverage.
            let mut fixtures = ONE_HOT_SCALAR_NUM_POLYS
                .iter()
                .flat_map(|&polys| keys(&[polys], (family.num_vars_range(polys).0, 16)))
                .collect::<Vec<_>>();
            fixtures.push(PolynomialGroupLayout::singleton(20));
            if one_hot_k == AKITA_ONE_HOT_K16 {
                fixtures.extend([22, 31, 32].map(PolynomialGroupLayout::singleton));
                fixtures.extend((51..=64).map(|width| PolynomialGroupLayout::new(16, width)));
                fixtures.extend(
                    [(25, 51), (26, 51), (25, 64), (34, 51), (34, 64)]
                        .map(|(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
                );
            }
            fixtures.sort_unstable_by_key(|key| (key.num_polynomials(), key.num_vars()));
            fixtures.dedup();
            return Ok(fixtures);
        }
        let mut admitted = ONE_HOT_SCALAR_NUM_POLYS
            .iter()
            .flat_map(|&num_polys| keys(&[num_polys], family.num_vars_range(num_polys)))
            .collect::<Vec<_>>();
        if one_hot_k == AKITA_ONE_HOT_K256 {
            // K=256 is exercised only by explicit adapter, benchmark, and override fixtures.
            let fixtures: &[(usize, usize)] = if profile == AkitaChunkProfile::new(1, 0) {
                &[
                    (14, 1),
                    (15, 1),
                    (16, 1),
                    (20, 1),
                    (20, 29),
                    (25, 1),
                    (28, 27),
                    (29, 27),
                    (34, 27),
                ]
            } else {
                &[(16, 1)]
            };
            admitted.extend(
                fixtures
                    .iter()
                    .map(|&(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
            );
            admitted.sort_unstable_by_key(|key| (key.num_polynomials(), key.num_vars()));
            admitted.dedup();
            return Ok(admitted);
        }
        if one_hot_k != AKITA_ONE_HOT_K16 {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported one-hot K={one_hot_k}"
            )));
        }
        admitted.extend(keys(&K16_TRACE_COLUMN_WIDTHS, K16_NUM_VARS));
        let fixtures: &[(usize, usize)] = if profile == AkitaChunkProfile::new(1, 0) {
            &[(12, 1), (12, 2), (16, 1), (25, 1)]
        } else {
            &[(16, 1)]
        };
        admitted.extend(
            fixtures
                .iter()
                .map(|&(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
        );
        admitted.sort_unstable_by_key(|key| (key.num_polynomials(), key.num_vars()));
        admitted.dedup();
        Ok(admitted)
    }

    /// Fixture inventory and its direct-planning cutoff, separate from runtime admission.
    pub struct FixtureSpec {
        pub spec: EmitSpec,
        direct_plan: Option<(usize, PlannerPolicy)>,
    }

    impl FixtureSpec {
        /// Plan the direct-only prefix before the generic emitter fills remaining rows.
        pub fn prepare(mut self) -> Result<EmitSpec, String> {
            if let Some((cutover, direct_policy)) = self.direct_plan {
                self.spec.preplanned = bounded_parallel_filter_map(
                    &self.spec.keys,
                    akita_planner::emit::offline_planning_worker_count(self.spec.keys.len()),
                    |key| {
                        if key.num_vars() >= cutover {
                            return Ok(None);
                        }
                        let lookup = ScheduleLookupKey::single(*key);
                        let planned = akita_planner::find_schedule(
                            &lookup,
                            self.spec.source_contract,
                            &[],
                            &direct_policy,
                            self.spec.ring_challenge_config,
                        )
                        .map_err(|error| error.to_string())?;
                        Ok(Some((lookup, planned.schedule)))
                    },
                )?;
            }
            Ok(self.spec)
        }
    }

    fn spec<Cfg: CommitmentConfig>(
        chunking: ChunkedWitnessCfg,
        keys: Vec<PolynomialGroupLayout>,
        output_path: PathBuf,
        direct_plan: Option<(usize, PlannerPolicy)>,
    ) -> Result<FixtureSpec, AkitaError> {
        Ok(FixtureSpec {
            spec: EmitSpec {
                family_name: Cfg::schedule_family_name(),
                policy: policy_of::<Cfg>(chunking),
                source_contract: Cfg::committed_source_contract()?,
                keys,
                grouped_requests: Vec::new(),
                preplanned: Vec::new(),
                output_path,
                ring_challenge_config: Cfg::ring_challenge_config,
            },
            direct_plan,
        })
    }

    fn one_hot_spec<Direct: RecursiveScheduleConfig>(
        k: usize,
        profile: AkitaChunkProfile,
        output_dir: &Path,
    ) -> Result<FixtureSpec, AkitaError> {
        let chunking = profile.witness_cfg();
        let filename = if profile == AkitaChunkProfile::default() {
            format!("jolt-fp128-onehot-k{k}.aks")
        } else {
            format!(
                "jolt-fp128-onehot-k{k}-w{}r{}.aks",
                profile.chunks, profile.levels
            )
        };
        spec::<RecursiveCommitmentConfig<Direct>>(
            chunking,
            one_hot_keys(k, profile)?,
            output_dir.join(filename),
            Some((
                RECURSIVE_TRACE_LOG_T_CUTOVER + k.ilog2() as usize,
                policy_of::<Direct>(chunking),
            )),
        )
    }

    /// Default execution catalogs and the representative multichunk fixtures.
    pub fn family_specs(output_dir: PathBuf) -> Result<Vec<FixtureSpec>, AkitaError> {
        let mut specs = Vec::new();
        for profile in [AkitaChunkProfile::default(), AkitaChunkProfile::new(8, 2)] {
            specs.extend([
                one_hot_spec::<JoltOneHotK16Direct>(AKITA_ONE_HOT_K16, profile, &output_dir)?,
                one_hot_spec::<JoltOneHotK256Direct>(AKITA_ONE_HOT_K256, profile, &output_dir)?,
            ]);
        }
        specs.extend([
            spec::<JoltDenseBounded>(
                MultiChunkProfileId::W8R2.cfg(),
                keys(&[1, 2], DENSE_NUM_VARS),
                output_dir.join("jolt-fp128-dense-bounded.aks"),
                None,
            )?,
            spec::<JoltDenseFull>(
                MultiChunkProfileId::W8R2.cfg(),
                keys(&[1], DENSE_NUM_VARS),
                output_dir.join("jolt-fp128-dense-full.aks"),
                None,
            )?,
        ]);
        Ok(specs)
    }
}
