//! Fuzz the verifier on mutated honest proofs and mutated public IO.
//!
//! The input format and the driver are `jolt_atlas_core::onnx_proof::fuzz`: an 8-byte IO-edit
//! header, then a serialized proof of the checked-in op gauntlet. `Err` verdicts (and inputs
//! that do not deserialize) are expected; a panic, an abort, an OOM or a timeout is a finding.
//!
//! `ATLAS_FUZZ_WRITE_SEED=<path>` writes the honest input to `<path>` at startup, to seed the
//! corpus.

#![no_main]

use std::path::Path;
use std::sync::OnceLock;

use jolt_atlas_core::onnx_proof::{fuzz::VerifyTarget, AkitaScheme, Fp128};
use libfuzzer_sys::fuzz_target;

fn target() -> &'static VerifyTarget<Fp128, AkitaScheme> {
    static TARGET: OnceLock<VerifyTarget<Fp128, AkitaScheme>> = OnceLock::new();
    TARGET.get_or_init(|| {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let target = VerifyTarget::new(&repo_root);
        if let Some(path) = std::env::var_os("ATLAS_FUZZ_WRITE_SEED") {
            std::fs::write(&path, target.seed()).expect("write the corpus seed");
        }
        target
    })
}

fuzz_target!(|data: &[u8]| {
    let _ = target().run(data);
});
