//! BitZ (ePrint 2026/2141): a commitment to a vector of bits, packed into `F_{2^128}` and
//! Reed–Solomon encoded, against which one evaluation claim over Fp128 is proven.

pub mod gf128;
pub mod merkle;
pub mod pack;
pub mod rs;
