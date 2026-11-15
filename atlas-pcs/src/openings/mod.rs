//! Opening-claim data shared by the Akita adapter: equal-point prefix packing, commitment-group
//! claims and their protocol roles, and the adapter's error type. Taken from upstream
//! `jolt-openings` (`docs/upstream-inventory.md`); its scheme traits are not carried over, so
//! `crate::commitment_scheme::CommitmentScheme` is the one PCS interface.
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

mod prefix;

use jolt_field::JoltField;
use jolt_poly::EvaluationClaim;

pub use prefix::{PrefixPackedClaims, PrefixPackedLayout};

/// One opening claim against a commitment: the commitment and its evaluation at a point.
#[derive(Clone, Debug)]
pub struct VerifierOpeningClaim<F: JoltField, C> {
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OpeningsError {
    #[error("opening proof verification failed")]
    VerificationFailed,

    #[error("invalid setup parameters: {0}")]
    InvalidSetup(String),

    #[error("invalid batch opening: {0}")]
    InvalidBatch(String),

    #[error("commitment failed: {0}")]
    CommitFailed(String),

    #[error("opening proof generation failed: {0}")]
    ProveFailed(String),

    #[error("polynomial size {poly_size} exceeds setup max {setup_max}")]
    PolynomialTooLarge { poly_size: usize, setup_max: usize },
}

/// One physical commitment group's opening claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupOpeningClaim<F, C> {
    pub commitment: C,
    pub point: Vec<F>,
    pub evaluations: Vec<F>,
}

impl<F, C> GroupOpeningClaim<F, C> {
    pub fn new(commitment: C, point: Vec<F>, evaluations: Vec<F>) -> Self {
        Self {
            commitment,
            point,
            evaluations,
        }
    }
}

/// Protocol-supplied identity and batch order of one commitment group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentGroupRole {
    order: u64,
    transcript_label: &'static [u8],
    diagnostic_name: &'static str,
    transcript_index: Option<u64>,
}

impl CommitmentGroupRole {
    pub const fn new(
        order: u64,
        transcript_label: &'static [u8],
        diagnostic_name: &'static str,
    ) -> Self {
        Self {
            order,
            transcript_label,
            diagnostic_name,
            transcript_index: None,
        }
    }

    pub const fn new_indexed(
        order: u64,
        transcript_label: &'static [u8],
        diagnostic_name: &'static str,
        transcript_index: u64,
    ) -> Self {
        Self {
            order,
            transcript_label,
            diagnostic_name,
            transcript_index: Some(transcript_index),
        }
    }

    /// Protocol-defined canonical batch position.
    pub const fn order(self) -> u64 {
        self.order
    }

    /// Protocol-defined domain-separating transcript tag.
    pub const fn transcript_label(self) -> &'static [u8] {
        self.transcript_label
    }

    pub const fn diagnostic_name(self) -> &'static str {
        self.diagnostic_name
    }

    /// Optional semantic index for repeated protocol roles.
    pub const fn transcript_index(self) -> Option<u64> {
        self.transcript_index
    }
}

/// One commitment group's public opening claim, tagged with its protocol role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaggedGroupOpeningClaim<F, C> {
    pub role: CommitmentGroupRole,
    pub claim: GroupOpeningClaim<F, C>,
}

/// One commitment group's tagged claim paired with the prover's retained
/// opening hint.
pub type GroupOpeningWithHint<F, C, H> = (TaggedGroupOpeningClaim<F, C>, H);

impl<F, C> TaggedGroupOpeningClaim<F, C> {
    pub fn new(role: CommitmentGroupRole, claim: GroupOpeningClaim<F, C>) -> Self {
        Self { role, claim }
    }
}
