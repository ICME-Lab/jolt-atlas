use jolt_akita::{AkitaChunkProfile, AkitaOneHotScheduleArtifact, AkitaScheduleArtifacts};
use jolt_openings::OpeningsError;
use std::sync::{Arc, OnceLock};

/// Load the combined Atlas and Jolt catalogs bundled with this build.
pub fn schedule_artifacts() -> Result<Arc<AkitaScheduleArtifacts>, OpeningsError> {
    static ARTIFACTS: OnceLock<Result<Arc<AkitaScheduleArtifacts>, String>> = OnceLock::new();
    ARTIFACTS
        .get_or_init(|| {
            let one_hot = AkitaChunkProfile::new(1, 0);
            AkitaScheduleArtifacts::new(
                AkitaChunkProfile::new(8, 2),
                include_bytes!("schedules/dense-bounded.aks").to_vec(),
                include_bytes!("schedules/dense-full.aks").to_vec(),
                vec![
                    AkitaOneHotScheduleArtifact {
                        k: 16,
                        profile: one_hot,
                        bytes: include_bytes!("schedules/one-hot-k16.aks").to_vec(),
                    },
                    AkitaOneHotScheduleArtifact {
                        k: 256,
                        profile: one_hot,
                        bytes: include_bytes!("schedules/one-hot-k256.aks").to_vec(),
                    },
                ],
            )
            .map(Arc::new)
            .map_err(|error| error.to_string())
        })
        .clone()
        .map_err(OpeningsError::InvalidSetup)
}
