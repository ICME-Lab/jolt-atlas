//! Golden hashes over every generated lookup table.
//!
//! These pins ARE the enforcement of [`atlas_trace::SemVersion`]: if any
//! entry of any table changes for any scale, a hash here changes, and the
//! fix is a new `SemVersion` variant with new pins — never editing a pin
//! under the same version.

use atlas_trace::{
    Scale, SemVersion,
    semantics::{
        softmax::softmax_tables,
        tables::{TableKind, table},
    },
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

#[test]
fn golden_hashes_v1() {
    let cases: &[(TableKind, u8, u64)] = &[
        (TableKind::Tanh, 8, 0xdfddcf5274b3a199),
        (TableKind::Sigmoid, 8, 0x80cd4efe390434c6),
        (TableKind::Erf, 8, 0x48e3bcf919574c3a),
        (TableKind::Sin, 8, 0x0805d4bdb9865c97),
        (TableKind::Cos, 8, 0x21bfcd4a891d61d3),
        (TableKind::Tanh, 14, 0x6a4a199581d654ed),
        (TableKind::Sigmoid, 14, 0x86913e0d02894ecc),
        (TableKind::Erf, 14, 0xb920c6754634b4b8),
        (TableKind::Sin, 14, 0x690f5f8d52813c1a),
        (TableKind::Cos, 14, 0x520b71c8693a7cde),
        (TableKind::Tanh, 15, 0x520bb1d0d7a98f9f),
        (TableKind::Sigmoid, 15, 0x12c4c81213690896),
        (TableKind::Erf, 15, 0xa3f4cf008a5722ce),
        (TableKind::Sin, 15, 0x937a366fa5a75e2b),
        (TableKind::Cos, 15, 0xa782099c1c21686c),
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
}

fn softmax_tables_hash(scale: u8) -> u64 {
    let t = softmax_tables(Scale::new(scale).unwrap(), SemVersion::V2);
    fnv1a(
        t.out
            .iter()
            .chain(&t.sum)
            .copied()
            .chain([t.address_bits as i32]),
    )
}

#[test]
fn golden_hashes_v2_softmax() {
    let cases: &[(u8, u64)] = &[
        (8, 0xecf911aeeb9b0dd1),
        (14, 0x0d77e4cd5f82b76e),
        (15, 0x97f5bdf088b75992),
    ];
    for &(scale, expected) in cases {
        let got = softmax_tables_hash(scale);
        assert_eq!(
            got, expected,
            "softmax tables at scale {scale} changed under SemVersion::V2: \
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
        println!(
            "softmax tables {scale}: {:#018x}",
            softmax_tables_hash(scale)
        );
    }
}
