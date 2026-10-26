//! Akita (lattice, Module-SIS) commitment scheme over the 128-bit Solinas field, built on
//! upstream's `jolt-akita`.
//!
//! Akita has no additive homomorphism, so the joint RLC opening of the batch opening reduction
//! cannot be verified against a combined commitment. Instead:
//!
//! * **Commit.** [`CommitmentScheme::batch_commit_all`] groups the committed polynomials into
//!   *classes*, each committed as one Akita object. A polynomial's commitment is an [`AkitaSlot`]
//!   handle into the [`AkitaBatchCommitment`], which carries the class commitments.
//! * **Open.** [`CommitmentScheme::prove_rlc`] does not form the RLC. For every class with an
//!   opened member it evaluates every slot at the class's point (the suffix of the joint point)
//!   and opens the class at those evaluations natively.
//! * **Verify.** [`CommitmentScheme::verify_rlc`] checks that the slot evaluations reproduce the
//!   joint claim (`Σ γ_i · v_i · pad_i`, `pad_i` accounting for the index-0 overlap of smaller
//!   polynomials), then verifies each class's opening.
//!
//! There are two kinds of class:
//!
//! * **Dense**, keyed by `(encoding, num_vars)`: prefix-packed into one physical polynomial (slot
//!   `i` occupies coefficients `[i·2^n, (i+1)·2^n)`), committed under the dense-bounded flavor,
//!   and opened by reducing the slot evaluations to one claim with upstream's
//!   [`PrefixPackedLayout`]. The flavor admits coefficients in `[0, 2^64)`, so a signed class is
//!   committed shifted by a public offset `c` (`2^31` for `i32` coefficients, `2^63` for `i64`):
//!   its slots hold `f + c` over their whole logical domain, a slot's MLE is `f(x) + c`, and the
//!   verifier subtracts `c`.
//! * **One-hot**: up to [`ONE_HOT_COLUMNS`] K = 16 one-hot polynomials of one arity, committed
//!   sparsely as the columns of one trace one-hot group (unused columns empty) and opened together
//!   at one point under the one-hot flavor. Atlas's one-hot polynomials are column-major
//!   (`index = k·T + t`, address variables first); Akita's are row-major (`index = t·K + k`), the
//!   same function with the address variables moved last.
//!
//! A class whose members are all identically zero commits nothing; the verifier then requires its
//! slot evaluations to be zero.
//!
//! Akita setups are exact-shape (the commitment arity equals the setup arity). Setups are built
//! from upstream's packaged schedule catalogs and cached process-wide: one dense setup per
//! physical arity, retagged with each class's layout digest, and one one-hot setup per arity.
//! Akita runs its own Fiat-Shamir argument, bridged to ours through one full-width challenge per
//! opening.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::{Arc, Mutex, OnceLock},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_akita::{
    AkitaBatchProof, AkitaChunkProfile, AkitaCommitment, AkitaProverHint, AkitaProverSetup,
    AkitaScheduleArtifacts, AkitaScheme as Upstream, AkitaSetupParams, AkitaVerifierSetup,
    TraceOneHotRows, AKITA_ONE_HOT_K16,
};
use jolt_field::Ring;
use jolt_openings::{
    CommitmentScheme as UpstreamScheme, GroupOpeningClaim, PrefixPackedClaims, PrefixPackedLayout,
    TransparentObjectSetup,
};
use jolt_poly::{EqPolynomial, Polynomial as UpstreamDense};
use num_traits::{One, Zero};
use rayon::prelude::*;

use crate::{
    field::{
        fp128::{Fp128, Inner},
        JoltField,
    },
    poly::multilinear_polynomial::MultilinearPolynomial,
    subprotocols::upstream_sumcheck::{FacadeChallenge, FacadeTranscript},
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};

use super::commitment_scheme::{BatchCommitOutput, CommitmentScheme};

/// Smallest physical arity of the dense-bounded catalog.
pub const MIN_DENSE_NUM_VARS: usize = 14;
/// Largest physical arity of the dense-bounded catalog.
pub const MAX_DENSE_NUM_VARS: usize = 34;
/// Smallest arity committed as a one-hot class: narrower K = 16 one-hot polynomials are packed
/// densely, which costs little at that size.
pub const MIN_ONE_HOT_NUM_VARS: usize = 16;
/// Largest arity of the K = 16 one-hot catalog.
pub const MAX_ONE_HOT_NUM_VARS: usize = 35;
/// Columns of a one-hot class's trace group: the catalog's group width.
pub const ONE_HOT_COLUMNS: usize = 64;
const LOG_ONE_HOT_K: usize = 4;

/// The dense profile of upstream's packaged dense catalogs.
const PACKAGED_DENSE_PROFILE: AkitaChunkProfile = AkitaChunkProfile::new(8, 2);

/// The Akita lattice PCS over [`Fp128`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaScheme;

/// Akita's challenges: full-width field elements of the upstream field.
impl FacadeChallenge for Inner {
    fn draw<T: Transcript>(transcript: &mut T) -> Self {
        transcript.challenge_scalar::<Fp128>().0
    }
}

/// How a dense class's coefficients map into the dense-bounded range `[0, 2^64)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Encoding {
    /// Coefficients in `[0, 2^64)`, committed as they are.
    Unsigned,
    /// `i32` coefficients, committed shifted by `2^31`.
    Signed32,
    /// `i64` coefficients, committed shifted by `2^63`.
    Signed64,
}

impl Encoding {
    fn of(poly: &MultilinearPolynomial<Fp128>) -> Self {
        match poly {
            MultilinearPolynomial::BoolScalars(_)
            | MultilinearPolynomial::U8Scalars(_)
            | MultilinearPolynomial::U16Scalars(_)
            | MultilinearPolynomial::U32Scalars(_)
            | MultilinearPolynomial::U64Scalars(_)
            | MultilinearPolynomial::OneHot(_) => Encoding::Unsigned,
            MultilinearPolynomial::I32Scalars(_) => Encoding::Signed32,
            MultilinearPolynomial::I64Scalars(_) => Encoding::Signed64,
            MultilinearPolynomial::LargeScalars(_)
            | MultilinearPolynomial::U128Scalars(_)
            | MultilinearPolynomial::I128Scalars(_)
            | MultilinearPolynomial::S128Scalars(_) => {
                panic!("Akita commits coefficients of at most 64 bits")
            }
        }
    }

    fn offset(self) -> Fp128 {
        match self {
            Encoding::Unsigned => Fp128::zero(),
            Encoding::Signed32 => Fp128::from_u64(1 << 31),
            Encoding::Signed64 => Fp128::from_u64(1 << 63),
        }
    }
}

/// How a class is committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ClassKind {
    Dense(Encoding),
    OneHot,
}

impl ClassKind {
    fn of(poly: &MultilinearPolynomial<Fp128>) -> Self {
        match poly {
            MultilinearPolynomial::OneHot(oh)
                if oh.K == AKITA_ONE_HOT_K16 && poly.get_num_vars() >= MIN_ONE_HOT_NUM_VARS =>
            {
                ClassKind::OneHot
            }
            _ => ClassKind::Dense(Encoding::of(poly)),
        }
    }

    fn tag(self) -> u8 {
        match self {
            ClassKind::Dense(Encoding::Unsigned) => 0,
            ClassKind::Dense(Encoding::Signed32) => 1,
            ClassKind::Dense(Encoding::Signed64) => 2,
            ClassKind::OneHot => 3,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(ClassKind::Dense(Encoding::Unsigned)),
            1 => Some(ClassKind::Dense(Encoding::Signed32)),
            2 => Some(ClassKind::Dense(Encoding::Signed64)),
            3 => Some(ClassKind::OneHot),
            _ => None,
        }
    }

    /// The offset a slot's committed evaluation carries over its logical one.
    fn offset(self) -> Fp128 {
        match self {
            ClassKind::Dense(encoding) => encoding.offset(),
            ClassKind::OneHot => Fp128::zero(),
        }
    }
}

/// The polynomials sharing one commitment. A one-hot kind of more than [`ONE_HOT_COLUMNS`]
/// members spans several chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ClassKey {
    kind: ClassKind,
    logical_num_vars: usize,
    chunk: usize,
}

/// A class's committed shape: `slot_count` logical polynomials of `logical_num_vars` variables in
/// an object of `physical_num_vars` variables (per column, for a one-hot class).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClassShape {
    kind: ClassKind,
    logical_num_vars: usize,
    physical_num_vars: usize,
    slot_count: usize,
}

impl ClassShape {
    /// The shape of `slot_count` polynomials of `logical_num_vars` variables, if the catalogs
    /// cover it.
    fn new(kind: ClassKind, logical_num_vars: usize, slot_count: usize) -> Option<Self> {
        if slot_count == 0 {
            return None;
        }
        let physical_num_vars = match kind {
            ClassKind::Dense(_) => {
                if logical_num_vars > MAX_DENSE_NUM_VARS {
                    return None;
                }
                let selector = slot_count.checked_next_power_of_two()?.trailing_zeros() as usize;
                let physical = (logical_num_vars + selector).max(MIN_DENSE_NUM_VARS);
                (physical <= MAX_DENSE_NUM_VARS).then_some(physical)?
            }
            ClassKind::OneHot => {
                let covered = (MIN_ONE_HOT_NUM_VARS..=MAX_ONE_HOT_NUM_VARS)
                    .contains(&logical_num_vars)
                    && slot_count <= ONE_HOT_COLUMNS;
                covered.then_some(logical_num_vars)?
            }
        };
        Some(Self {
            kind,
            logical_num_vars,
            physical_num_vars,
            slot_count,
        })
    }

    /// The protocol-owned digest of the class layout; it seeds the class's Akita setup. A dense
    /// class's digest also seeds its prefix-packed claim reduction. A one-hot class's depends on
    /// its arity alone, so classes of one arity share a setup.
    fn layout_digest(&self) -> [u8; 32] {
        let mut h = Blake2b::<U32>::new();
        match self.kind {
            ClassKind::Dense(_) => {
                h.update(b"atlas-akita-dense-class-v1");
                h.update([self.kind.tag()]);
                h.update((self.logical_num_vars as u64).to_le_bytes());
                h.update((self.physical_num_vars as u64).to_le_bytes());
                h.update((self.slot_count as u64).to_le_bytes());
            }
            ClassKind::OneHot => {
                h.update(b"atlas-akita-one-hot-class-v1");
                h.update((self.physical_num_vars as u64).to_le_bytes());
            }
        }
        h.finalize().into()
    }

    /// A dense class's prefix layout.
    fn layout(&self) -> PrefixPackedLayout<u32> {
        PrefixPackedLayout::new(
            self.logical_num_vars,
            1 << (self.physical_num_vars - self.logical_num_vars),
            0..self.slot_count as u32,
        )
        .expect("a dense class shape is a valid prefix-packed layout")
    }

    /// A logical point in Akita's variable order: a one-hot class's address variables move last.
    fn upstream_point(&self, logical: &[Fp128]) -> Vec<Inner> {
        match self.kind {
            ClassKind::Dense(_) => logical.iter().map(|x| x.0).collect(),
            ClassKind::OneHot => {
                let (address, cycle) = logical.split_at(LOG_ONE_HOT_K);
                cycle.iter().chain(address).map(|x| x.0).collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Setups
// ---------------------------------------------------------------------------

/// The exact-shape Akita setup of one class.
struct ClassSetup {
    prover: AkitaProverSetup,
    verifier: AkitaVerifierSetup,
}

/// Cached setups by catalog digest, one-hot or dense, and arity. They are pure functions of the
/// catalogs and the shape, so they are shared by every setup in the process.
type SetupCache = Mutex<BTreeMap<([u8; 32], bool, usize), Arc<ClassSetup>>>;

fn setup_cache() -> &'static SetupCache {
    static CACHE: OnceLock<SetupCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The schedule catalogs, shared by the prover and the verifier. Every class setup derives from
/// them.
#[derive(Clone)]
pub struct AkitaSetup {
    artifacts: Arc<AkitaScheduleArtifacts>,
    artifacts_digest: [u8; 32],
}

impl AkitaSetup {
    /// Upstream's packaged catalogs, or the directory named by `JOLT_AKITA_SCHEDULE_DIR`.
    pub fn from_packaged_catalogs() -> Self {
        Self::new(AkitaScheduleArtifacts::shared_from_default_directory(
            PACKAGED_DENSE_PROFILE,
        ))
    }

    fn new(artifacts: Arc<AkitaScheduleArtifacts>) -> Self {
        let mut h = Blake2b::<U32>::new();
        h.update(serde_to_bytes(&*artifacts));
        Self {
            artifacts,
            artifacts_digest: h.finalize().into(),
        }
    }

    fn cached(
        &self,
        one_hot: bool,
        num_vars: usize,
        build: impl FnOnce() -> Result<(AkitaProverSetup, AkitaVerifierSetup), String>,
    ) -> Result<Arc<ClassSetup>, String> {
        let key = (self.artifacts_digest, one_hot, num_vars);
        if let Some(setup) = setup_cache().lock().unwrap().get(&key) {
            return Ok(setup.clone());
        }
        let (prover, verifier) = build()?;
        let built = Arc::new(ClassSetup { prover, verifier });
        Ok(setup_cache()
            .lock()
            .unwrap()
            .entry(key)
            .or_insert(built)
            .clone())
    }

    /// The setup of class `shape`.
    fn class_setup(&self, shape: &ClassShape) -> Result<Arc<ClassSetup>, String> {
        let failed = |e: jolt_openings::OpeningsError| format!("Akita setup for {shape:?}: {e}");
        let n = shape.physical_num_vars;
        match shape.kind {
            ClassKind::Dense(_) => {
                let base = self.cached(false, n, || {
                    Upstream::transparent_object_setup(&self.artifacts, n, [0; 32]).map_err(failed)
                })?;
                let (prover, verifier) =
                    Upstream::retag_transparent_object_setup(&base.prover, shape.layout_digest())
                        .map_err(failed)?;
                Ok(Arc::new(ClassSetup { prover, verifier }))
            }
            ClassKind::OneHot => self.cached(true, n, || {
                Upstream::setup(AkitaSetupParams::one_hot_only(
                    n,
                    ONE_HOT_COLUMNS,
                    shape.layout_digest(),
                    AKITA_ONE_HOT_K16,
                    self.artifacts.clone(),
                ))
                .map_err(failed)
            }),
        }
    }
}

impl fmt::Debug for AkitaSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AkitaSetup({:x?})", &self.artifacts_digest[..4])
    }
}

fn serde_to_bytes<T: serde::Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("Akita types serialize with bincode")
}

fn serde_from_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .ok()
        .map(|(value, _)| value)
}

impl Valid for AkitaSetup {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AkitaSetup {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        serde_to_bytes(&*self.artifacts).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        serde_to_bytes(&*self.artifacts).serialized_size(compress)
    }
}

impl CanonicalDeserialize for AkitaSetup {
    fn deserialize_with_mode<R: std::io::Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let bytes = Vec::<u8>::deserialize_with_mode(reader, compress, validate)?;
        let artifacts: AkitaScheduleArtifacts =
            serde_from_bytes(&bytes).ok_or(SerializationError::InvalidData)?;
        Ok(Self::new(Arc::new(artifacts)))
    }
}

// ---------------------------------------------------------------------------
// Commitments
// ---------------------------------------------------------------------------

/// A polynomial's commitment: a slot of a class of the batch commitment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaSlot {
    pub class: u32,
    pub slot: u32,
}

impl AppendToTranscript for AkitaSlot {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_u64(self.class as u64);
        transcript.append_u64(self.slot as u64);
    }
}

/// One class.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaClassCommitment {
    /// The class kind (dense encoding or one-hot).
    pub kind: u8,
    pub logical_num_vars: u32,
    pub slot_count: u32,
    /// The bincode-encoded [`AkitaCommitment`]; `None` when every member is identically zero.
    pub commitment: Option<Vec<u8>>,
}

impl AkitaClassCommitment {
    fn shape(&self) -> Option<ClassShape> {
        ClassShape::new(
            ClassKind::from_tag(self.kind)?,
            self.logical_num_vars as usize,
            self.slot_count as usize,
        )
    }
}

/// Every class commitment of a proof, in class order.
#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaBatchCommitment {
    pub classes: Vec<AkitaClassCommitment>,
}

impl AppendToTranscript for AkitaBatchCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_message(b"akita_batch_commitment");
        transcript.append_u64(self.classes.len() as u64);
        for class in &self.classes {
            transcript.append_u64(class.kind as u64);
            transcript.append_u64(class.logical_num_vars as u64);
            transcript.append_u64(class.slot_count as u64);
            match &class.commitment {
                None => transcript.append_u64(0),
                Some(bytes) => {
                    transcript.append_u64(1 + bytes.len() as u64);
                    transcript.append_bytes(bytes);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prover-side class state
// ---------------------------------------------------------------------------

/// A one-hot class as a row-major trace of [`ONE_HOT_COLUMNS`] columns: column `c < slot_count` is
/// member `c`, row `t` its hot index at cycle `t`; the other columns are empty.
///
/// Akita's kernels read rows many times, so the trace is materialized once as a contiguous
/// `num_rows × ONE_HOT_COLUMNS` byte buffer, plus the per-row mask telling a hot index 0 from no
/// coefficient.
struct ClassRows {
    columns: Vec<Arc<Vec<Option<u16>>>>,
    num_rows: usize,
    selected: Vec<u8>,
    zero_masks: Vec<u64>,
}

impl ClassRows {
    fn new(columns: Vec<Arc<Vec<Option<u16>>>>) -> Self {
        let num_rows = columns[0].len();
        let mut selected = vec![jolt_akita::no_selected_row(); num_rows * ONE_HOT_COLUMNS];
        let mut zero_masks = vec![0u64; num_rows];
        selected
            .par_chunks_mut(ONE_HOT_COLUMNS)
            .zip(zero_masks.par_iter_mut())
            .enumerate()
            .with_min_len(1 << 10)
            .for_each(|(row, (dst, mask))| {
                for (c, column) in columns.iter().enumerate() {
                    match column[row] {
                        Some(0) => *mask |= 1 << c,
                        Some(k) => dst[c] = k as u8,
                        None => {}
                    }
                }
            });
        Self {
            columns,
            num_rows,
            selected,
            zero_masks,
        }
    }

    /// Each member's evaluation at `point` (Akita's order: cycle variables, then address).
    fn evaluations(&self, point: &[Inner]) -> Vec<Fp128> {
        let (cycle, address) = point.split_at(point.len() - LOG_ONE_HOT_K);
        let eq_cycle = EqPolynomial::new(cycle.to_vec()).evaluations();
        let eq_address = EqPolynomial::new(address.to_vec()).evaluations();
        self.columns
            .par_iter()
            .map(|column| {
                Fp128(
                    column
                        .par_iter()
                        .zip(eq_cycle.par_iter())
                        .with_min_len(1 << 12)
                        .map(|(k, e)| k.map_or(Inner::zero(), |k| *e * eq_address[k as usize]))
                        .sum(),
                )
            })
            .collect()
    }
}

impl TraceOneHotRows for ClassRows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_columns(&self) -> usize {
        ONE_HOT_COLUMNS
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        let start = row * ONE_HOT_COLUMNS;
        selected_rows.copy_from_slice(&self.selected[start..start + ONE_HOT_COLUMNS]);
    }

    fn fill_rows(&self, row_start: usize, selected_rows: &mut [u8]) {
        let start = row_start * ONE_HOT_COLUMNS;
        selected_rows.copy_from_slice(&self.selected[start..start + selected_rows.len()]);
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        self.zero_masks[row]
    }

    fn fill_committed_digit_zero_masks(&self, row_start: usize, masks: &mut [u64]) {
        masks.copy_from_slice(&self.zero_masks[row_start..row_start + masks.len()]);
    }
}

/// What a committed class retains for its opening.
enum ClassSource {
    /// The packed physical polynomial of a dense class.
    Dense(UpstreamDense<Inner>),
    /// The trace of a one-hot class.
    OneHot(Arc<ClassRows>),
}

/// A committed class's source and Akita state, retained for the opening.
struct CommittedClass {
    source: ClassSource,
    commitment: AkitaCommitment,
    hint: AkitaProverHint,
    setup: Arc<ClassSetup>,
}

struct ClassProverData {
    index: u32,
    shape: ClassShape,
    /// `None` for an identically zero class.
    committed: Option<CommittedClass>,
}

impl ClassProverData {
    /// Every slot's logical evaluation at `point` (Akita's order).
    fn slot_evaluations(&self, point: &[Inner]) -> Vec<Fp128> {
        let Some(committed) = &self.committed else {
            return vec![Fp128::zero(); self.shape.slot_count];
        };
        match &committed.source {
            ClassSource::OneHot(rows) => rows.evaluations(point),
            ClassSource::Dense(packed) => {
                // One eq table, shared across the slots.
                let len = 1usize << self.shape.logical_num_vars;
                let eq = EqPolynomial::new(point.to_vec()).evaluations();
                let offset = self.shape.kind.offset();
                packed.evals()[..self.shape.slot_count * len]
                    .par_chunks(len)
                    .map(|slot| {
                        let shifted: Inner = slot
                            .par_iter()
                            .zip(eq.par_iter())
                            .with_min_len(1 << 12)
                            .map(|(a, b)| *a * *b)
                            .sum();
                        Fp128(shifted) - offset
                    })
                    .collect()
            }
        }
    }
}

/// A polynomial's opening hint: its class's prover state.
#[derive(Clone)]
pub struct AkitaHint(Arc<ClassProverData>);

impl fmt::Debug for AkitaHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AkitaHint(class {} {:?})", self.0.index, self.0.shape)
    }
}

impl PartialEq for AkitaHint {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

// ---------------------------------------------------------------------------
// Proofs
// ---------------------------------------------------------------------------

/// The opening of one class: every slot's evaluation at the class's logical point, and the Akita
/// proof (empty for a zero class).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaClassOpening {
    pub class: u32,
    pub slot_evals: Vec<Fp128>,
    /// The bincode-encoded [`AkitaBatchProof`].
    pub proof: Vec<u8>,
}

/// The joint opening: one class opening per class with an opened member, in class order.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaJointProof {
    pub classes: Vec<AkitaClassOpening>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_zero(poly: &MultilinearPolynomial<Fp128>) -> bool {
    match poly {
        MultilinearPolynomial::OneHot(oh) => oh.nonzero_indices.par_iter().all(|k| k.is_none()),
        _ => (0..poly.original_len())
            .into_par_iter()
            .all(|i| poly.get_coeff(i).is_zero()),
    }
}

/// The packed physical polynomial of a dense class: each member, shifted by the class offset over
/// its whole logical domain, in its slot; unused slots zero.
fn pack_dense(shape: &ClassShape, members: &[&MultilinearPolynomial<Fp128>]) -> Vec<Inner> {
    let logical_len = 1usize << shape.logical_num_vars;
    let offset = shape.kind.offset().0;
    let mut evals = vec![Inner::zero(); 1usize << shape.physical_num_vars];
    evals
        .par_chunks_mut(logical_len)
        .zip(members.par_iter())
        .for_each(|(dst, poly)| {
            dst.par_iter_mut().for_each(|d| *d = offset);
            match poly {
                MultilinearPolynomial::OneHot(oh) => {
                    // Column-major, as `OneHotPolynomial::coeffs`: index k·T + t.
                    let t_len = oh.nonzero_indices.len();
                    for (t, k) in oh.nonzero_indices.iter().enumerate() {
                        if let Some(k) = k {
                            dst[*k as usize * t_len + t] = offset + Inner::one();
                        }
                    }
                }
                _ => dst[..poly.original_len()]
                    .par_iter_mut()
                    .enumerate()
                    .with_min_len(1 << 12)
                    .for_each(|(i, d)| *d = poly.get_coeff(i).0 + offset),
            }
        });
    evals
}

/// A one-hot class's trace.
fn one_hot_rows(members: &[&MultilinearPolynomial<Fp128>]) -> ClassRows {
    ClassRows::new(
        members
            .iter()
            .map(|poly| match poly {
                MultilinearPolynomial::OneHot(oh) => oh.nonzero_indices.clone(),
                _ => unreachable!("a one-hot class member is one-hot"),
            })
            .collect(),
    )
}

/// `Π_{j < pad} (1 − r_j)`: the factor by which a polynomial of `len − pad` variables, overlapped
/// at index 0 with one of `len` variables, scales its own evaluation at the suffix of `r`.
fn pad_factor(r: &[Fp128], pad: usize) -> Fp128 {
    r[..pad].iter().map(|x| Fp128::one() - *x).product()
}

fn inner_values(values: &[Fp128]) -> Vec<Inner> {
    values.iter().map(|x| x.0).collect()
}

/// The evaluations of a one-hot class's trace group: the members', then zero for the empty
/// columns.
fn group_evaluations(slot_evals: &[Fp128]) -> Vec<Inner> {
    let mut evals = inner_values(slot_evals);
    evals.resize(ONE_HOT_COLUMNS, Inner::zero());
    evals
}

fn invalid(msg: impl Into<String>) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(msg.into())
}

// ---------------------------------------------------------------------------
// CommitmentScheme
// ---------------------------------------------------------------------------

impl CommitmentScheme for AkitaScheme {
    type Field = Fp128;
    type ProverSetup = AkitaSetup;
    type VerifierSetup = AkitaSetup;
    type Commitment = AkitaSlot;
    type Proof = AkitaJointProof;
    type BatchedProof = AkitaJointProof;
    type OpeningProofHint = AkitaHint;
    type BatchCommitment = AkitaBatchCommitment;

    fn setup_prover(_max_num_vars: usize) -> Self::ProverSetup {
        AkitaSetup::from_packaged_catalogs()
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        Self::batch_commit_all(std::slice::from_ref(poly), setup)
            .1
            .pop()
            .expect("one polynomial was committed")
    }

    fn batch_commit<U>(
        polys: &[U],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        Self::batch_commit_all(polys, setup).1
    }

    #[tracing::instrument(skip_all, name = "AkitaScheme::batch_commit_all")]
    fn batch_commit_all<U>(polys: &[U], setup: &Self::ProverSetup) -> BatchCommitOutput<Self>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        let mut grouped: BTreeMap<(ClassKind, usize), Vec<usize>> = BTreeMap::new();
        for (i, poly) in polys.iter().enumerate() {
            let poly = poly.borrow();
            grouped
                .entry((ClassKind::of(poly), poly.get_num_vars()))
                .or_default()
                .push(i);
        }
        let mut classes: BTreeMap<ClassKey, Vec<usize>> = BTreeMap::new();
        for ((kind, logical_num_vars), members) in grouped {
            let chunk_len = match kind {
                ClassKind::Dense(_) => members.len(),
                ClassKind::OneHot => ONE_HOT_COLUMNS,
            };
            for (chunk, members) in members.chunks(chunk_len).enumerate() {
                let key = ClassKey {
                    kind,
                    logical_num_vars,
                    chunk,
                };
                classes.insert(key, members.to_vec());
            }
        }

        let mut per_poly: Vec<Option<(AkitaSlot, AkitaHint)>> = vec![None; polys.len()];
        let mut class_commitments = Vec::with_capacity(classes.len());
        for (index, (key, members)) in classes.into_iter().enumerate() {
            let shape = ClassShape::new(key.kind, key.logical_num_vars, members.len())
                .unwrap_or_else(|| {
                    panic!(
                        "no Akita catalog covers a class of {} polynomials of {key:?}",
                        members.len()
                    )
                });
            let member_polys: Vec<&MultilinearPolynomial<Fp128>> =
                members.iter().map(|&i| polys[i].borrow()).collect();
            let committed = if member_polys.par_iter().all(|p| is_zero(p)) {
                None
            } else {
                let _span = tracing::info_span!(
                    "AkitaScheme::commit_class",
                    kind = ?shape.kind,
                    num_vars = shape.physical_num_vars,
                    slots = shape.slot_count
                )
                .entered();
                let setup = setup.class_setup(&shape).unwrap_or_else(|e| panic!("{e}"));
                let (source, committed) = match shape.kind {
                    ClassKind::Dense(_) => {
                        let packed = UpstreamDense::new(pack_dense(&shape, &member_polys));
                        let committed = Upstream::commit(&packed, &setup.prover);
                        (ClassSource::Dense(packed), committed)
                    }
                    ClassKind::OneHot => {
                        let rows = Arc::new(one_hot_rows(&member_polys));
                        let committed = Upstream::commit_trace_one_hot(
                            &setup.prover,
                            shape.layout_digest(),
                            rows.clone(),
                            &[],
                        );
                        (ClassSource::OneHot(rows), committed)
                    }
                };
                let (commitment, hint) =
                    committed.unwrap_or_else(|e| panic!("Akita commit for {shape:?}: {e}"));
                Some(CommittedClass {
                    source,
                    commitment,
                    hint,
                    setup,
                })
            };
            class_commitments.push(AkitaClassCommitment {
                kind: shape.kind.tag(),
                logical_num_vars: shape.logical_num_vars as u32,
                slot_count: shape.slot_count as u32,
                commitment: committed.as_ref().map(|c| serde_to_bytes(&c.commitment)),
            });
            let data = Arc::new(ClassProverData {
                index: index as u32,
                shape,
                committed,
            });
            for (slot, &i) in members.iter().enumerate() {
                per_poly[i] = Some((
                    AkitaSlot {
                        class: index as u32,
                        slot: slot as u32,
                    },
                    AkitaHint(data.clone()),
                ));
            }
        }
        (
            AkitaBatchCommitment {
                classes: class_commitments,
            },
            per_poly
                .into_iter()
                .map(|p| p.expect("every polynomial is in a class"))
                .collect(),
        )
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        unimplemented!("Akita openings go through `prove_rlc`")
    }

    #[tracing::instrument(skip_all, name = "AkitaScheme::prove_rlc")]
    fn prove_rlc<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        _polynomials: &BTreeMap<common::CommittedPoly, MultilinearPolynomial<Self::Field>>,
        _coeffs: &[Self::Field],
        hints: Vec<Self::OpeningProofHint>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        // The classes with an opened member, in class order.
        let mut classes: BTreeMap<u32, Arc<ClassProverData>> = BTreeMap::new();
        for hint in hints {
            classes.entry(hint.0.index).or_insert(hint.0);
        }
        let max_vars = opening_point.len();
        let (openings, pow) = FacadeTranscript::<_, Inner>::scope(transcript, 0, |transcript| {
            classes
                .into_values()
                .map(|data| {
                    let shape = &data.shape;
                    let point =
                        shape.upstream_point(&opening_point[max_vars - shape.logical_num_vars..]);
                    let slot_evals = data.slot_evaluations(&point);
                    let Some(committed) = &data.committed else {
                        return AkitaClassOpening {
                            class: data.index,
                            slot_evals,
                            proof: Vec::new(),
                        };
                    };
                    let proof = match &committed.source {
                        ClassSource::Dense(packed) => {
                            let offset = shape.kind.offset();
                            let shifted: Vec<Fp128> =
                                slot_evals.iter().map(|v| *v + offset).collect();
                            let claim = shape
                                .layout()
                                .reduce_claims(
                                    &PrefixPackedClaims::new(
                                        shape.layout_digest(),
                                        point,
                                        inner_values(&shifted),
                                    ),
                                    transcript,
                                )
                                .expect("the prefix-packed claim reduces");
                            Upstream::open(
                                packed,
                                claim.point.as_slice(),
                                claim.value,
                                &committed.setup.prover,
                                Some(committed.hint.clone()),
                                transcript,
                            )
                        }
                        ClassSource::OneHot(_) => Upstream::prove_batch(
                            &committed.setup.prover,
                            Vec::new(),
                            GroupOpeningClaim::new(
                                committed.commitment.clone(),
                                point,
                                group_evaluations(&slot_evals),
                            ),
                            committed.hint.clone(),
                            transcript,
                        ),
                    }
                    .unwrap_or_else(|e| panic!("Akita open for {shape:?}: {e}"));
                    AkitaClassOpening {
                        class: data.index,
                        slot_evals,
                        proof: serde_to_bytes(&proof),
                    }
                })
                .collect()
        });
        pow.expect("the Akita bridge grinds no proof of work");
        AkitaJointProof { classes: openings }
    }

    fn verify<ProofTranscript: Transcript>(
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        Err(invalid("Akita openings are verified through `verify_rlc`"))
    }

    #[tracing::instrument(skip_all, name = "AkitaScheme::verify_rlc")]
    fn verify_rlc<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        joint_claim: &Self::Field,
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
        num_vars: &[usize],
        batch: &Self::BatchCommitment,
    ) -> Result<(), ProofVerifyError> {
        let max_vars = opening_point.len();
        if !proof.classes.windows(2).all(|w| w[0].class < w[1].class) {
            return Err(invalid("Akita class openings are not in class order"));
        }
        let by_class: BTreeMap<u32, &AkitaClassOpening> =
            proof.classes.iter().map(|c| (c.class, c)).collect();

        // The classes the opened polynomials touch, and the joint claim their slots imply.
        if commitments.len() != coeffs.len() || commitments.len() != num_vars.len() {
            return Err(invalid("misaligned joint opening statement"));
        }
        let mut opened_classes = BTreeSet::new();
        let mut reconstructed = Fp128::zero();
        for ((slot, coeff), &n) in commitments.iter().zip(coeffs).zip(num_vars) {
            let class = batch
                .classes
                .get(slot.class as usize)
                .ok_or_else(|| invalid(format!("unknown Akita class {}", slot.class)))?;
            if class.logical_num_vars as usize != n || n > max_vars {
                return Err(invalid(format!(
                    "Akita class {} holds {}-variable polynomials, the opening expects {n}",
                    slot.class, class.logical_num_vars
                )));
            }
            let opening = by_class
                .get(&slot.class)
                .ok_or_else(|| invalid(format!("Akita class {} was not opened", slot.class)))?;
            let eval = opening
                .slot_evals
                .get(slot.slot as usize)
                .filter(|_| slot.slot < class.slot_count)
                .ok_or_else(|| invalid(format!("Akita slot {} out of range", slot.slot)))?;
            reconstructed += *coeff * *eval * pad_factor(opening_point, max_vars - n);
            opened_classes.insert(slot.class);
        }
        if reconstructed != *joint_claim {
            return Err(invalid(
                "Akita slot evaluations do not reproduce the joint claim",
            ));
        }
        if !opened_classes.iter().eq(by_class.keys()) {
            return Err(invalid(
                "the Akita proof opens a different class set than the opened polynomials",
            ));
        }

        let (result, pow) = FacadeTranscript::<_, Inner>::scope(transcript, 0, |transcript| {
            proof.classes.iter().try_for_each(|opening| {
                let class = &batch.classes[opening.class as usize];
                let shape = class
                    .shape()
                    .ok_or_else(|| invalid(format!("invalid Akita class {}", opening.class)))?;
                if opening.slot_evals.len() != shape.slot_count {
                    return Err(invalid(format!(
                        "Akita class {} opening has {} evaluations, expected {}",
                        opening.class,
                        opening.slot_evals.len(),
                        shape.slot_count
                    )));
                }
                let Some(commitment) = &class.commitment else {
                    if opening.slot_evals.iter().any(|v| !v.is_zero()) || !opening.proof.is_empty()
                    {
                        return Err(invalid(format!(
                            "Akita class {} is committed as zero but its opening is not",
                            opening.class
                        )));
                    }
                    return Ok(());
                };
                let commitment: AkitaCommitment = serde_from_bytes(commitment)
                    .ok_or_else(|| invalid("malformed Akita class commitment"))?;
                let akita_proof: AkitaBatchProof = serde_from_bytes(&opening.proof)
                    .ok_or_else(|| invalid("malformed Akita class proof"))?;
                let class_setup = setup.class_setup(&shape).map_err(invalid)?;
                let point =
                    shape.upstream_point(&opening_point[max_vars - shape.logical_num_vars..]);
                match shape.kind {
                    ClassKind::Dense(_) => {
                        let offset = shape.kind.offset();
                        let shifted: Vec<Fp128> =
                            opening.slot_evals.iter().map(|v| *v + offset).collect();
                        let claim = shape
                            .layout()
                            .reduce_claims(
                                &PrefixPackedClaims::new(
                                    shape.layout_digest(),
                                    point,
                                    inner_values(&shifted),
                                ),
                                transcript,
                            )
                            .map_err(|e| invalid(format!("prefix-packed claim reduction: {e}")))?;
                        Upstream::verify(
                            &commitment,
                            claim.point.as_slice(),
                            claim.value,
                            &akita_proof,
                            &class_setup.verifier,
                            transcript,
                        )
                    }
                    ClassKind::OneHot => Upstream::verify_batch(
                        &class_setup.verifier,
                        &[],
                        &GroupOpeningClaim::new(
                            commitment,
                            point,
                            group_evaluations(&opening.slot_evals),
                        ),
                        &akita_proof,
                        transcript,
                    ),
                }
                .map_err(|e| {
                    invalid(format!(
                        "Akita class {} opening rejected: {e}",
                        opening.class
                    ))
                })
            })
        });
        pow?;
        result
    }

    fn protocol_name() -> &'static [u8] {
        b"AkitaScheme"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{
            multilinear_polynomial::PolynomialEvaluation, one_hot_polynomial::OneHotPolynomial,
        },
        transcripts::{Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript},
    };
    use ark_std::test_rng;
    use common::CommittedPoly;
    use rand::Rng;

    fn random_point(rng: &mut impl Rng, n: usize) -> Vec<Fp128> {
        (0..n)
            .map(|_| <Fp128 as jolt_field::Field>::random(rng))
            .collect()
    }

    fn random_one_hot(rng: &mut impl Rng, rows: usize) -> MultilinearPolynomial<Fp128> {
        MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
            (0..rows)
                .map(|_| (rng.gen::<u8>() % 5 != 0).then(|| rng.gen_range(0..16u16)))
                .collect(),
            AKITA_ONE_HOT_K16,
        ))
    }

    /// jolt-poly and joltworks agree on the variable order of a dense polynomial, of a densely
    /// packed one-hot polynomial, and of a one-hot class's trace.
    #[test]
    fn evaluation_conventions_agree() {
        let mut rng = test_rng();
        let n = 6;
        let coeffs: Vec<i32> = (0..1 << n).map(|_| rng.gen_range(-1000..1000)).collect();
        let ours = MultilinearPolynomial::<Fp128>::from(coeffs.clone());
        let theirs =
            UpstreamDense::<Inner>::new(coeffs.iter().map(|&c| Fp128::from_i32(c).0).collect());
        let r = random_point(&mut rng, n);
        assert_eq!(ours.evaluate(&r).0, theirs.evaluate(&inner_values(&r)));

        let ours = random_one_hot(&mut rng, 32);
        let shape = ClassShape::new(ClassKind::of(&ours), ours.get_num_vars(), 1).unwrap();
        assert!(matches!(shape.kind, ClassKind::Dense(_)));
        let packed = UpstreamDense::new(pack_dense(&shape, &[&ours]));
        let r = random_point(&mut rng, ours.get_num_vars());
        let mut physical = vec![Inner::zero(); shape.physical_num_vars - r.len()];
        physical.extend(inner_values(&r));
        assert_eq!(ours.evaluate(&r).0, packed.evaluate(&physical));

        let ours = random_one_hot(&mut rng, 1 << 12);
        let shape = ClassShape::new(ClassKind::of(&ours), ours.get_num_vars(), 1).unwrap();
        assert_eq!(shape.kind, ClassKind::OneHot);
        let r = random_point(&mut rng, ours.get_num_vars());
        let rows = one_hot_rows(&[&ours]);
        assert_eq!(
            ours.evaluate(&r),
            rows.evaluations(&shape.upstream_point(&r))[0]
        );
    }

    /// Commit a mixed batch (dense classes of several arities and encodings, a zero class, and a
    /// one-hot class), open the joint RLC at a random point, verify, and reject tampering.
    #[test]
    fn joint_opening_roundtrip() {
        let mut rng = test_rng();
        let polys: Vec<MultilinearPolynomial<Fp128>> = vec![
            MultilinearPolynomial::from(
                (0..1 << 5)
                    .map(|_| rng.gen_range(-50i32..50))
                    .collect::<Vec<_>>(),
            ),
            MultilinearPolynomial::from(
                (0..1 << 5)
                    .map(|_| rng.gen_range(i32::MIN..=i32::MAX))
                    .collect::<Vec<_>>(),
            ),
            MultilinearPolynomial::from((0..1 << 5).map(|_| rng.gen::<u8>()).collect::<Vec<_>>()),
            MultilinearPolynomial::from((0..1 << 8).map(|_| rng.gen::<i64>()).collect::<Vec<_>>()),
            MultilinearPolynomial::from(vec![0u32; 1 << 9]),
            random_one_hot(&mut rng, 64),
            random_one_hot(&mut rng, 1 << 12),
            random_one_hot(&mut rng, 1 << 12),
        ];
        let ids: Vec<CommittedPoly> = (0..polys.len())
            .map(|i| CommittedPoly::NodeOutputRaD(i, 0))
            .collect();
        let num_vars: Vec<usize> = polys.iter().map(|p| p.get_num_vars()).collect();
        let max_vars = *num_vars.iter().max().unwrap();
        let setup = AkitaScheme::setup_prover(max_vars);

        let (batch, per_poly) = AkitaScheme::batch_commit_all(&polys, &setup);
        let kinds: Vec<u8> = batch.classes.iter().map(|c| c.kind).collect();
        assert_eq!(
            kinds,
            vec![0, 0, 0, 1, 2, 3],
            "dense u8 at 5 variables, zero u32 at 9 and small one-hot at 10; i32 at 5; i64 at 8; one-hot at 16"
        );
        assert_eq!(
            batch
                .classes
                .iter()
                .filter(|c| c.commitment.is_none())
                .count(),
            1
        );
        let (slots, hints): (Vec<_>, Vec<_>) = per_poly.into_iter().unzip();

        let r = random_point(&mut rng, max_vars);
        let coeffs: Vec<Fp128> = (0..polys.len())
            .map(|_| <Fp128 as jolt_field::Field>::random(&mut rng))
            .collect();
        // The joint claim of the index-0 overlap RLC, as the opening reduction defines it.
        let joint: Fp128 = polys
            .iter()
            .zip(&coeffs)
            .map(|(p, c)| {
                let n = p.get_num_vars();
                *c * p.evaluate(&r[max_vars - n..]) * pad_factor(&r, max_vars - n)
            })
            .sum();

        let map: BTreeMap<CommittedPoly, MultilinearPolynomial<Fp128>> =
            ids.iter().copied().zip(polys.iter().cloned()).collect();
        let mut pt = Blake2bProver::new(b"akita-test");
        batch.append_to_transcript(&mut pt);
        let proof = AkitaScheme::prove_rlc(&setup, &map, &coeffs, hints, &r, &mut pt);
        assert_eq!(proof.classes.len(), 6);
        let narg = pt.into_narg();

        let refs: Vec<&AkitaSlot> = slots.iter().collect();
        let verify = |proof: &AkitaJointProof, claim: Fp128, num_vars: &[usize]| {
            let mut vt = Blake2bVerifier::new(b"akita-test", narg.clone());
            batch.append_to_transcript(&mut vt);
            AkitaScheme::verify_rlc(
                proof, &setup, &mut vt, &r, &claim, &refs, &coeffs, num_vars, &batch,
            )
        };
        verify(&proof, joint, &num_vars).expect("an honest joint opening verifies");

        assert!(
            verify(&proof, joint + Fp128::one(), &num_vars).is_err(),
            "wrong joint claim"
        );
        let mut wrong_arity = num_vars.clone();
        wrong_arity[0] += 1;
        assert!(
            verify(&proof, joint, &wrong_arity).is_err(),
            "a class wider than the opening expects"
        );
        // Shifting two slot evaluations of one class by compensating amounts keeps the joint
        // claim; the class opening must catch it, for a dense and for a one-hot class.
        for (i, j) in [(0, 1), (6, 7)] {
            let (si, sj) = (slots[i], slots[j]);
            assert_eq!(si.class, sj.class);
            let mut compensated = proof.clone();
            let class = compensated
                .classes
                .iter_mut()
                .find(|c| c.class == si.class)
                .unwrap();
            let delta = Fp128::from_u64(5);
            class.slot_evals[si.slot as usize] += delta * coeffs[j];
            class.slot_evals[sj.slot as usize] -= delta * coeffs[i];
            assert!(verify(&compensated, joint, &num_vars).is_err());
        }

        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).unwrap();
        let back = AkitaJointProof::deserialize_compressed(&bytes[..]).unwrap();
        assert_eq!(back.classes[0].slot_evals, proof.classes[0].slot_evals);
        let mut bytes = Vec::new();
        batch.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(
            AkitaBatchCommitment::deserialize_compressed(&bytes[..]).unwrap(),
            batch
        );
    }
}
