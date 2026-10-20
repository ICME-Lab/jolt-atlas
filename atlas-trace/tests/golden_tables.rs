//! Golden hashes over every generated lookup table.
//!
//! These pins ARE the enforcement of [`atlas_trace::SemVersion`]: if any
//! entry of any table changes for any scale, a hash here changes, and the
//! fix is a new `SemVersion` variant with new pins — never editing a pin
//! under the same version.

use atlas_trace::{
    Scale, SemVersion,
    semantics::tables::{TableKind, exp_lut_decomposed, table},
};

/// FNV-1a 64-bit over a little-endian i32 stream (dependency-free and
/// stable across platforms and releases).
fn fnv1a(values: impl IntoIterator<Item = i32>) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for v in values {
        for byte in v.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn table_hash(kind: TableKind, scale: u8) -> u64 {
    let t = table(kind, Scale::new(scale).unwrap(), SemVersion::V1);
    fnv1a(t.entries.iter().copied())
}

fn exp_lut_hash(scale: u8) -> u64 {
    let lut = exp_lut_decomposed(Scale::new(scale).unwrap(), SemVersion::V1);
    fnv1a(
        lut.lut_hi
            .iter()
            .chain(&lut.lut_lo)
            .copied()
            .chain([lut.base as i32]),
    )
}

#[test]
fn golden_hashes_v1() {
    let cases: &[(TableKind, u8, u64)] = &[
        (TableKind::Tanh, 8, 0xdfddcf5274b3a199),
        (TableKind::Sigmoid, 8, 0x80cd4efe390434c6),
        (TableKind::Erf, 8, 0x48e3bcf919574c3a),
        (TableKind::Sin, 8, 0x74e65207a61a851f),
        (TableKind::Cos, 8, 0x8dd6774013299eaf),
        (TableKind::Tanh, 14, 0x6a4a199581d654ed),
        (TableKind::Sigmoid, 14, 0x86913e0d02894ecc),
        (TableKind::Erf, 14, 0xb920c6754634b4b8),
        (TableKind::Sin, 14, 0xba7ce4479007e751),
        (TableKind::Cos, 14, 0xaea6b28a4b95c4f9),
        (TableKind::Tanh, 15, 0x520bb1d0d7a98f9f),
        (TableKind::Sigmoid, 15, 0x12c4c81213690896),
        (TableKind::Erf, 15, 0xa3f4cf008a5722ce),
        (TableKind::Sin, 15, 0x8f8a728185ef2cd7),
        (TableKind::Cos, 15, 0x63350baab109933c),
    ];
    for &(kind, scale, expected) in cases {
        let got = table_hash(kind, scale);
        assert_eq!(
            got, expected,
            "{kind:?} table at scale {scale} changed under SemVersion::V1: \
             got {got:#018x}, pinned {expected:#018x}. Any table change \
             requires a new SemVersion variant and new pins."
        );
    }

    let exp_cases: &[(u8, u64)] = &[
        (8, 0x5790701787d7c194),
        (14, 0xea34f76e6c2fa16b),
        (15, 0x6d049d70e0342e2c),
    ];
    for &(scale, expected) in exp_cases {
        let got = exp_lut_hash(scale);
        assert_eq!(
            got, expected,
            "softmax exp LUT at scale {scale} changed under SemVersion::V1: \
             got {got:#018x}, pinned {expected:#018x}."
        );
    }
}

/// Regenerate the pins (run with `--ignored --nocapture` after a deliberate
/// SemVersion bump).
#[test]
#[ignore]
fn print_hashes() {
    for scale in [8u8, 14, 15] {
        for kind in [
            TableKind::Tanh,
            TableKind::Sigmoid,
            TableKind::Erf,
            TableKind::Sin,
            TableKind::Cos,
        ] {
            println!(
                "(TableKind::{kind:?}, {scale}, {:#018x}),",
                table_hash(kind, scale)
            );
        }
        println!("exp lut {scale}: {:#018x}", exp_lut_hash(scale));
    }
}
