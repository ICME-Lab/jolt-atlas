//! Committed Sumcheck.
//!
//! This crate starts with the scalar Pedersen and Schnorr building blocks used
//! by the committed round-polynomial sumcheck protocol.

pub mod committed_round;
pub mod multiplication;
pub mod ops;
pub mod pedersen;
pub mod round;
pub mod schnorr;
pub mod sumcheck;

pub use committed_round::{
    absorb_round_poly_commitments, challenge_round_poly, challenge_round_poly_optimized,
    commit_round_poly, prove_round_consistency, verify_round_consistency, CommittedRoundPoly,
    RoundConsistencyProof,
};
pub use multiplication::{
    absorb_multiplication_response, prove_multiplication, verify_multiplication,
    MultiplicationProof,
};
pub use ops::hadamard::Hadamard;
pub use pedersen::{Commitment, Opening, PedersenParams};
pub use round::{build_round_poly, DenseMleTable, MleTable, RoundPoly, RoundRelation};
pub use schnorr::{
    prove_equality, prove_opening, verify_equality, verify_opening, EqualityProof, OpeningProof,
};
pub use sumcheck::{CommittedSumCheckProof, CommittedSumCheckRound, SumCheck};
