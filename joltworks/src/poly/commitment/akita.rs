//! Akita (lattice, Module-SIS) commitment scheme for Atlas over the 128-bit
//! Solinas field, built on upstream Jolt's `jolt-akita` adapter.
//!
//! Akita has no additive homomorphism, so Atlas's joint RLC opening cannot be
//! verified against a combined commitment. Instead:
//!
//! * **Commit.** [`CommitmentScheme::batch_commit_all`] groups the committed
//!   polynomials into *classes* keyed by `(flavor, logical num_vars)` where
//!   flavor is dense (small-value) or one-hot (K = 16). Each class is
//!   prefix-packed into one physical polynomial — slot `i` occupies
//!   coefficients `[i·2^n, (i+1)·2^n)` — and committed as one Akita object.
//!   Per-polynomial "commitments" are [`AkitaSlot`] handles into the
//!   [`AkitaBatchCommitment`], which carries the real class commitments.
//! * **Open.** [`CommitmentScheme::prove_rlc`] does not form an RLC. For every
//!   class with an opened member it evaluates every slot at the class's
//!   logical point (the suffix of the joint point), binds the evaluations,
//!   reduces them to one claim on the packed polynomial with upstream's
//!   [`PrefixPackedLayout`], and opens that claim natively.
//! * **Verify.** [`CommitmentScheme::verify_rlc`] checks that the slot
//!   evaluations reproduce the joint claim (`Σ γ_i · v_i · pad_i`, where
//!   `pad_i` accounts for the index-0 overlap of smaller polynomials), then
//!   replays the packed reduction and verifies each class opening.
//!
//! Setups are exact-shape (Akita requires the commitment arity to equal the
//! setup arity), built lazily per `(flavor, physical num_vars)` from the
//! checked-in schedule catalogs in `joltworks/schedules/`, and shared between
//! a prover setup and the verifier setup derived from it.

use std::{
    any::Any,
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use blake2::{digest::consts::U32, Blake2b, Digest};
use common::parallel::par_enabled;
use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaProverHint, AkitaProverSetup, AkitaScheduleArtifacts,
    AkitaScheme as Upstream, AkitaSetupParams, AkitaVerifierSetup, AKITA_ONE_HOT_K16,
};
use jolt_openings::{CommitmentScheme as UpstreamPcs, PrefixPackedClaims, PrefixPackedLayout};
use jolt_poly::{MultilinearPoly, OneHotPolynomial as UpstreamOneHot, Polynomial as UpstreamDense};
use num_traits::{One, Zero};
use rayon::prelude::*;

use crate::{
    field::{
        fp128::{Fp128, Inner},
        JoltField,
    },
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::{AppendToTranscript, Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};

use super::commitment_scheme::{BatchCommitOutput, CommitmentScheme};

/// Smallest arity in the checked-in dense-bounded catalog.
pub const MIN_DENSE_NUM_VARS: usize = 14;
/// Smallest arity in the checked-in K=16 one-hot catalog.
pub const MIN_ONE_HOT_NUM_VARS: usize = 12;
/// Largest arity in the checked-in catalogs.
pub const MAX_NUM_VARS: usize = 34;
const ONE_HOT_K: usize = AKITA_ONE_HOT_K16;
const LOG_ONE_HOT_K: usize = 4;

/// The Akita lattice PCS over [`Fp128`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaScheme;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Flavor {
    Dense,
    OneHot,
}

impl Flavor {
    fn tag(self) -> u8 {
        match self {
            Flavor::Dense => 0,
            Flavor::OneHot => 1,
        }
    }
    fn from_tag(tag: u8) -> Result<Self, ProofVerifyError> {
        match tag {
            0 => Ok(Flavor::Dense),
            1 => Ok(Flavor::OneHot),
            _ => Err(ProofVerifyError::InvalidOpeningProof(format!(
                "unknown Akita class flavor {tag}"
            ))),
        }
    }
    fn min_num_vars(self) -> usize {
        match self {
            Flavor::Dense => MIN_DENSE_NUM_VARS,
            Flavor::OneHot => MIN_ONE_HOT_NUM_VARS,
        }
    }
}

/// A class of committed polynomials sharing one packed Akita commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ClassKey {
    flavor: Flavor,
    logical_num_vars: usize,
}

/// Shape of a packed class: `slot_count` logical polynomials of
/// `logical_num_vars` variables in a `2^(physical - logical)`-slot prefix layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClassShape {
    key: ClassKey,
    physical_num_vars: usize,
    slot_count: usize,
}

impl ClassShape {
    fn new(key: ClassKey, slot_count: usize) -> Self {
        let selector = slot_count.next_power_of_two().trailing_zeros() as usize;
        let physical_num_vars = (key.logical_num_vars + selector).max(key.flavor.min_num_vars());
        assert!(
            physical_num_vars <= MAX_NUM_VARS,
            "Akita class {key:?} with {slot_count} polynomials needs {physical_num_vars} variables; \
             the checked-in schedule catalogs stop at {MAX_NUM_VARS}"
        );
        Self {
            key,
            physical_num_vars,
            slot_count,
        }
    }

    fn slot_capacity(&self) -> usize {
        1 << (self.physical_num_vars - self.key.logical_num_vars)
    }

    /// Protocol-owned digest binding the class layout; the same value seeds the
    /// Akita object setup and the prefix-packed claim reduction.
    fn layout_digest(&self) -> [u8; 32] {
        let mut h = Blake2b::<U32>::new();
        h.update(b"atlas-akita-class-v1");
        h.update([self.key.flavor.tag()]);
        h.update((self.key.logical_num_vars as u64).to_le_bytes());
        h.update((self.physical_num_vars as u64).to_le_bytes());
        h.update((self.slot_count as u64).to_le_bytes());
        h.finalize().into()
    }

    fn layout(&self) -> PrefixPackedLayout<u32> {
        PrefixPackedLayout::new(
            self.key.logical_num_vars,
            self.slot_capacity(),
            0..self.slot_count as u32,
        )
        .expect("class shape is a valid prefix-packed layout")
    }

    /// Maps a logical opening point in Atlas's variable order to jolt-poly's
    /// order for the class flavor. Atlas one-hot polynomials are column-major
    /// (`index = k·T + t`, address variables first); Akita commits them
    /// row-major (`index = t·K + k`), which is the same function with the
    /// address variables moved last.
    fn upstream_point(&self, logical: &[Inner]) -> Vec<Inner> {
        match self.key.flavor {
            Flavor::Dense => logical.to_vec(),
            Flavor::OneHot => {
                let (addr, cycle) = logical.split_at(LOG_ONE_HOT_K);
                let mut p = Vec::with_capacity(logical.len());
                p.extend_from_slice(cycle);
                p.extend_from_slice(addr);
                p
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Setups
// ---------------------------------------------------------------------------

/// Exact-shape Akita setup for one class shape.
struct ClassSetup {
    prover: AkitaProverSetup,
    verifier: AkitaVerifierSetup,
}

/// Class setups are pure functions of the catalogs and the class shape, so
/// they are cached process-wide (keyed by a digest of the catalog bytes) and
/// shared by every prover and verifier setup in the process.
type SetupCache = Mutex<BTreeMap<([u8; 32], u8, usize, [u8; 32]), Arc<ClassSetup>>>;

fn global_setup_cache() -> &'static SetupCache {
    static CACHE: std::sync::OnceLock<SetupCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn artifacts_digest(artifacts: &AkitaScheduleArtifacts) -> [u8; 32] {
    let mut h = Blake2b::<U32>::new();
    h.update(serde_to_bytes(artifacts));
    h.finalize().into()
}

fn build_class_setup(artifacts: &Arc<AkitaScheduleArtifacts>, shape: &ClassShape) -> ClassSetup {
    let digest = shape.layout_digest();
    let params = match shape.key.flavor {
        Flavor::Dense => {
            AkitaSetupParams::dense_only(shape.physical_num_vars, 1, digest, artifacts.clone())
        }
        Flavor::OneHot => AkitaSetupParams::one_hot_only(
            shape.physical_num_vars,
            1,
            digest,
            ONE_HOT_K,
            artifacts.clone(),
        ),
    };
    let (prover, verifier) = Upstream::setup(params)
        .unwrap_or_else(|e| panic!("Akita setup for class {shape:?} failed: {e}"));
    ClassSetup { prover, verifier }
}

fn cached_class_setup(
    artifacts_digest: [u8; 32],
    artifacts: &Arc<AkitaScheduleArtifacts>,
    shape: &ClassShape,
) -> Arc<ClassSetup> {
    let cache = global_setup_cache();
    let key = (
        artifacts_digest,
        shape.key.flavor.tag(),
        shape.physical_num_vars,
        shape.layout_digest(),
    );
    if let Some(s) = cache.lock().unwrap().get(&key) {
        return s.clone();
    }
    let built = Arc::new(build_class_setup(artifacts, shape));
    cache.lock().unwrap().entry(key).or_insert(built).clone()
}

/// Prover setup: the schedule catalogs plus a lazily filled cache of
/// exact-shape class setups.
#[derive(Clone)]
pub struct AkitaAtlasProverSetup {
    artifacts: Arc<AkitaScheduleArtifacts>,
    artifacts_digest: [u8; 32],
}

/// Verifier setup: the same catalogs.
#[derive(Clone)]
pub struct AkitaAtlasVerifierSetup {
    artifacts: Arc<AkitaScheduleArtifacts>,
    artifacts_digest: [u8; 32],
}

impl AkitaAtlasProverSetup {
    /// Loads the checked-in catalogs (`joltworks/schedules/`), or the directory
    /// named by `JOLT_AKITA_SCHEDULE_DIR`.
    pub fn from_default_catalogs() -> Self {
        let dir = std::env::var_os("JOLT_AKITA_SCHEDULE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("schedules"));
        let artifacts = AkitaScheduleArtifacts::from_directory(&dir)
            .unwrap_or_else(|e| panic!("loading Akita schedule catalogs from {dir:?}: {e}"));
        Self {
            artifacts_digest: artifacts_digest(&artifacts),
            artifacts: Arc::new(artifacts),
        }
    }

    fn class_setup(&self, shape: &ClassShape) -> Arc<ClassSetup> {
        cached_class_setup(self.artifacts_digest, &self.artifacts, shape)
    }
}

impl AkitaAtlasVerifierSetup {
    fn class_setup(&self, shape: &ClassShape) -> Arc<ClassSetup> {
        cached_class_setup(self.artifacts_digest, &self.artifacts, shape)
    }
}

impl fmt::Debug for AkitaAtlasProverSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AkitaAtlasProverSetup({:x?})",
            &self.artifacts_digest[..4]
        )
    }
}

impl fmt::Debug for AkitaAtlasVerifierSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AkitaAtlasVerifierSetup({:x?})",
            &self.artifacts_digest[..4]
        )
    }
}

fn serde_to_bytes<T: serde::Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("Akita types serialize with bincode")
}

fn serde_from_bytes<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, ark_serialize::SerializationError> {
    bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map(|(v, _)| v)
        .map_err(|_| ark_serialize::SerializationError::InvalidData)
}

macro_rules! impl_setup_serialization {
    ($ty:ident) => {
        impl ark_serialize::Valid for $ty {
            fn check(&self) -> Result<(), ark_serialize::SerializationError> {
                Ok(())
            }
        }
        impl CanonicalSerialize for $ty {
            fn serialize_with_mode<W: std::io::Write>(
                &self,
                writer: W,
                compress: ark_serialize::Compress,
            ) -> Result<(), ark_serialize::SerializationError> {
                serde_to_bytes(&*self.artifacts).serialize_with_mode(writer, compress)
            }
            fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
                serde_to_bytes(&*self.artifacts).serialized_size(compress)
            }
        }
        impl CanonicalDeserialize for $ty {
            fn deserialize_with_mode<R: std::io::Read>(
                reader: R,
                compress: ark_serialize::Compress,
                validate: ark_serialize::Validate,
            ) -> Result<Self, ark_serialize::SerializationError> {
                let bytes = Vec::<u8>::deserialize_with_mode(reader, compress, validate)?;
                let artifacts: AkitaScheduleArtifacts = serde_from_bytes(&bytes)?;
                Ok(Self {
                    artifacts_digest: artifacts_digest(&artifacts),
                    artifacts: Arc::new(artifacts),
                })
            }
        }
    };
}
impl_setup_serialization!(AkitaAtlasProverSetup);
impl_setup_serialization!(AkitaAtlasVerifierSetup);

// ---------------------------------------------------------------------------
// Commitments
// ---------------------------------------------------------------------------

/// Per-polynomial commitment: a handle to a slot of a class in the batch
/// commitment.
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

/// One packed class commitment.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaClassCommitment {
    flavor: u8,
    logical_num_vars: u32,
    physical_num_vars: u32,
    slot_count: u32,
    /// bincode-encoded [`AkitaCommitment`].
    commitment: Vec<u8>,
}

impl AkitaClassCommitment {
    fn shape(&self) -> Result<ClassShape, ProofVerifyError> {
        let shape = ClassShape::new(
            ClassKey {
                flavor: Flavor::from_tag(self.flavor)?,
                logical_num_vars: self.logical_num_vars as usize,
            },
            self.slot_count as usize,
        );
        if shape.physical_num_vars != self.physical_num_vars as usize {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "Akita class declares {} physical variables, layout needs {}",
                self.physical_num_vars, shape.physical_num_vars
            )));
        }
        Ok(shape)
    }
}

/// All class commitments of a proof, in class order.
#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaBatchCommitment {
    pub classes: Vec<AkitaClassCommitment>,
}

impl AppendToTranscript for AkitaBatchCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_message(b"akita_batch_commitment");
        transcript.append_u64(self.classes.len() as u64);
        for c in &self.classes {
            transcript.append_u64(c.flavor as u64);
            transcript.append_u64(c.logical_num_vars as u64);
            transcript.append_u64(c.physical_num_vars as u64);
            transcript.append_u64(c.slot_count as u64);
            transcript.append_bytes(&c.commitment);
        }
    }
}

// ---------------------------------------------------------------------------
// Hints (prover-side class state)
// ---------------------------------------------------------------------------

enum Packed {
    Dense(UpstreamDense<Inner>),
    OneHot {
        poly: UpstreamOneHot,
        /// Row-major hot indices, `slot_capacity · T` rows.
        indices: Vec<Option<u8>>,
        rows_per_slot: usize,
    },
}

struct ClassProverData {
    class_index: u32,
    shape: ClassShape,
    packed: Packed,
    hint: AkitaProverHint,
    setup: Arc<ClassSetup>,
}

impl ClassProverData {
    /// Evaluates every used slot at the logical point (in jolt-poly order).
    fn slot_evaluations(&self, upstream_point: &[Inner]) -> Vec<Inner> {
        let n = self.shape.slot_count;
        match &self.packed {
            Packed::Dense(poly) => {
                let len = 1usize << self.shape.key.logical_num_vars;
                let evals = poly.evaluations();
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        UpstreamDense::<Inner>::new(evals[i * len..(i + 1) * len].to_vec())
                            .evaluate(upstream_point)
                    })
                    .collect()
            }
            Packed::OneHot {
                indices,
                rows_per_slot,
                ..
            } => (0..n)
                .into_par_iter()
                .map(|i| {
                    let slot = indices[i * rows_per_slot..(i + 1) * rows_per_slot].to_vec();
                    <UpstreamOneHot as MultilinearPoly<Inner>>::evaluate(
                        &UpstreamOneHot::new(ONE_HOT_K, slot),
                        upstream_point,
                    )
                })
                .collect(),
        }
    }
}

/// Per-polynomial opening hint: a shared handle to its class's prover data.
#[derive(Clone)]
pub struct AkitaHint(Arc<ClassProverData>);

impl fmt::Debug for AkitaHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AkitaHint(class {} {:?})",
            self.0.class_index, self.0.shape
        )
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

/// Opening of one class: every used slot's evaluation at the class's logical
/// point, and the native Akita proof of the packed claim.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaClassOpening {
    pub class: u32,
    pub slot_evals: Vec<Fp128>,
    /// bincode-encoded [`AkitaBatchProof`].
    pub proof: Vec<u8>,
}

/// The joint opening proof: one class opening per class with an opened member.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaJointProof {
    pub classes: Vec<AkitaClassOpening>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn native_transcript<T: Transcript>(transcript: &mut T) -> &mut Blake2bTranscript {
    (transcript as &mut dyn Any)
        .downcast_mut::<Blake2bTranscript>()
        .expect("the Akita adapter requires Blake2bTranscript")
}

fn class_key(poly: &MultilinearPolynomial<Fp128>) -> ClassKey {
    match poly {
        // Akita's one-hot flavor is specialized to K = 16 (and 256, which this
        // adapter does not set up yet). Other chunk widths go through the
        // dense-bounded flavor: their coefficients are 0/1, and Atlas's
        // column-major layout is used verbatim, so no point rotation applies.
        MultilinearPolynomial::OneHot(oh) if oh.K == ONE_HOT_K => ClassKey {
            flavor: Flavor::OneHot,
            logical_num_vars: oh.get_num_vars(),
        },
        _ => ClassKey {
            flavor: Flavor::Dense,
            logical_num_vars: poly.get_num_vars(),
        },
    }
}

/// Builds the packed physical polynomial of a class from its member
/// polynomials (in slot order).
fn pack_class(shape: &ClassShape, members: &[&MultilinearPolynomial<Fp128>]) -> Packed {
    let logical_len = 1usize << shape.key.logical_num_vars;
    match shape.key.flavor {
        Flavor::Dense => {
            let mut evals = vec![Inner::default(); 1usize << shape.physical_num_vars];
            evals
                .par_chunks_mut(logical_len)
                .zip(members.par_iter())
                .for_each(|(dst, poly)| match poly {
                    MultilinearPolynomial::OneHot(oh) => {
                        // Column-major, as `OneHotPolynomial::coeffs`: index k·T + t.
                        let t_len = oh.nonzero_indices.len();
                        for (t, k) in oh.nonzero_indices.iter().enumerate() {
                            if let Some(k) = k {
                                dst[*k as usize * t_len + t] = Inner::one();
                            }
                        }
                    }
                    _ => {
                        let len = poly.original_len();
                        dst[..len]
                            .par_iter_mut()
                            .enumerate()
                            .with_min_len(par_enabled())
                            .for_each(|(i, d)| *d = poly.get_coeff(i).0);
                    }
                });
            Packed::Dense(UpstreamDense::new(evals))
        }
        Flavor::OneHot => {
            let rows_per_slot = logical_len / ONE_HOT_K;
            let mut indices = vec![None; rows_per_slot * shape.slot_capacity()];
            for (slot, poly) in members.iter().enumerate() {
                let MultilinearPolynomial::OneHot(oh) = poly else {
                    unreachable!("one-hot class member is not one-hot")
                };
                let dst = &mut indices[slot * rows_per_slot..(slot + 1) * rows_per_slot];
                for (d, k) in dst.iter_mut().zip(oh.nonzero_indices.iter()) {
                    *d = k.map(|k| k as u8);
                }
            }
            Packed::OneHot {
                poly: UpstreamOneHot::new(ONE_HOT_K, indices.clone()),
                indices,
                rows_per_slot,
            }
        }
    }
}

/// `Π_{j < pad} (1 − r_j)`: the factor by which a zero-padded polynomial of
/// `len − pad` variables evaluated at `r` differs from its own evaluation at
/// the suffix of `r`.
fn pad_factor(r: &[Fp128], pad: usize) -> Fp128 {
    r[..pad].iter().map(|x| Fp128::one() - *x).product()
}

// ---------------------------------------------------------------------------
// CommitmentScheme
// ---------------------------------------------------------------------------

impl CommitmentScheme for AkitaScheme {
    type Field = Fp128;
    type ProverSetup = AkitaAtlasProverSetup;
    type VerifierSetup = AkitaAtlasVerifierSetup;
    type Commitment = AkitaSlot;
    type Proof = AkitaJointProof;
    type BatchedProof = AkitaJointProof;
    type OpeningProofHint = AkitaHint;
    type BatchCommitment = AkitaBatchCommitment;

    const REQUIRES_MATERIALIZED_POLYS: bool = false;

    fn setup_prover(_max_num_vars: usize) -> Self::ProverSetup {
        AkitaAtlasProverSetup::from_default_catalogs()
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        AkitaAtlasVerifierSetup {
            artifacts: setup.artifacts.clone(),
            artifacts_digest: setup.artifacts_digest,
        }
    }

    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        Self::batch_commit_all(std::slice::from_ref(poly), setup)
            .1
            .pop()
            .unwrap()
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        Self::batch_commit_all(polys, gens).1
    }

    #[tracing::instrument(skip_all, name = "AkitaScheme::batch_commit_all")]
    fn batch_commit_all<U>(polys: &[U], setup: &Self::ProverSetup) -> BatchCommitOutput<Self>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        // Group polynomials into classes, remembering each one's slot.
        let mut classes: BTreeMap<ClassKey, Vec<usize>> = BTreeMap::new();
        for (i, poly) in polys.iter().enumerate() {
            classes.entry(class_key(poly.borrow())).or_default().push(i);
        }

        let mut per_poly: Vec<Option<(AkitaSlot, AkitaHint)>> = vec![None; polys.len()];
        let mut class_commitments = Vec::with_capacity(classes.len());
        for (class_index, (key, members)) in classes.into_iter().enumerate() {
            let shape = ClassShape::new(key, members.len());
            let member_polys: Vec<&MultilinearPolynomial<Fp128>> =
                members.iter().map(|&i| polys[i].borrow()).collect();
            let packed = pack_class(&shape, &member_polys);
            let class_setup = setup.class_setup(&shape);
            let (commitment, hint) = match &packed {
                Packed::Dense(poly) => Upstream::commit(poly, &class_setup.prover),
                Packed::OneHot { poly, .. } => Upstream::commit(poly, &class_setup.prover),
            }
            .unwrap_or_else(|e| panic!("Akita commit for class {shape:?} failed: {e}"));
            class_commitments.push(AkitaClassCommitment {
                flavor: key.flavor.tag(),
                logical_num_vars: key.logical_num_vars as u32,
                physical_num_vars: shape.physical_num_vars as u32,
                slot_count: members.len() as u32,
                commitment: serde_to_bytes(&commitment),
            });
            let data = Arc::new(ClassProverData {
                class_index: class_index as u32,
                shape,
                packed,
                hint,
                setup: class_setup,
            });
            for (slot, &i) in members.iter().enumerate() {
                per_poly[i] = Some((
                    AkitaSlot {
                        class: class_index as u32,
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
            per_poly.into_iter().map(Option::unwrap).collect(),
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
        // Distinct classes with at least one opened member, in class order.
        let mut classes: BTreeMap<u32, Arc<ClassProverData>> = BTreeMap::new();
        for h in hints {
            classes.entry(h.0.class_index).or_insert(h.0);
        }
        let transcript = native_transcript(transcript);
        let max_vars = opening_point.len();
        let mut openings = Vec::with_capacity(classes.len());
        for (class_index, data) in classes {
            let n = data.shape.key.logical_num_vars;
            let logical = Fp128::as_inner_slice(&opening_point[max_vars - n..]);
            let upstream_point = data.shape.upstream_point(logical);
            let slot_evals = data.slot_evaluations(&upstream_point);
            let claim = data
                .shape
                .layout()
                .reduce_claims(
                    &PrefixPackedClaims::new(
                        data.shape.layout_digest(),
                        upstream_point.clone(),
                        slot_evals.clone(),
                    ),
                    transcript,
                )
                .expect("prefix-packed claim reduction");
            let proof = match &data.packed {
                Packed::Dense(poly) => Upstream::open(
                    poly,
                    claim.point.as_slice(),
                    claim.value,
                    &data.setup.prover,
                    Some(data.hint.clone()),
                    transcript,
                ),
                Packed::OneHot { poly, .. } => Upstream::open(
                    poly,
                    claim.point.as_slice(),
                    claim.value,
                    &data.setup.prover,
                    Some(data.hint.clone()),
                    transcript,
                ),
            }
            .unwrap_or_else(|e| panic!("Akita open for class {:?} failed: {e}", data.shape));
            openings.push(AkitaClassOpening {
                class: class_index,
                slot_evals: slot_evals.into_iter().map(Fp128).collect(),
                proof: serde_to_bytes(&proof),
            });
        }
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
        unimplemented!("Akita openings are verified through `verify_rlc`")
    }

    #[tracing::instrument(skip_all, name = "AkitaScheme::verify_rlc")]
    #[allow(clippy::too_many_arguments)]
    fn verify_rlc<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        joint_claim: &Self::Field,
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
        batch: &Self::BatchCommitment,
    ) -> Result<(), ProofVerifyError> {
        let bad = |msg: String| ProofVerifyError::InvalidOpeningProof(msg);
        let max_vars = opening_point.len();
        let by_class: BTreeMap<u32, &AkitaClassOpening> =
            proof.classes.iter().map(|c| (c.class, c)).collect();
        if by_class.len() != proof.classes.len() {
            return Err(bad("duplicate Akita class opening".into()));
        }

        // Which classes the opened polynomials touch, and the joint claim they imply.
        let mut opened_classes = std::collections::BTreeSet::new();
        let mut reconstructed = Fp128::zero();
        for (slot, coeff) in commitments.iter().zip(coeffs) {
            let class = batch
                .classes
                .get(slot.class as usize)
                .ok_or_else(|| bad(format!("Akita slot refers to unknown class {}", slot.class)))?;
            let opening = by_class
                .get(&slot.class)
                .ok_or_else(|| bad(format!("Akita class {} was not opened", slot.class)))?;
            let eval = opening
                .slot_evals
                .get(slot.slot as usize)
                .ok_or_else(|| bad(format!("Akita slot {} out of range", slot.slot)))?;
            let n = class.logical_num_vars as usize;
            if n > max_vars {
                return Err(bad("Akita class is wider than the opening point".into()));
            }
            reconstructed += *coeff * *eval * pad_factor(opening_point, max_vars - n);
            opened_classes.insert(slot.class);
        }
        if reconstructed != *joint_claim {
            return Err(bad(
                "Akita slot evaluations do not reproduce the joint claim".into(),
            ));
        }
        if !opened_classes.iter().eq(by_class.keys()) {
            return Err(bad(
                "Akita proof opens a different class set than the opened polynomials".into(),
            ));
        }

        let transcript = native_transcript(transcript);
        for opening in &proof.classes {
            let class = &batch.classes[opening.class as usize];
            let shape = class.shape()?;
            if opening.slot_evals.len() != shape.slot_count {
                return Err(bad(format!(
                    "Akita class {} opening has {} evaluations, expected {}",
                    opening.class,
                    opening.slot_evals.len(),
                    shape.slot_count
                )));
            }
            let n = shape.key.logical_num_vars;
            let logical = Fp128::as_inner_slice(&opening_point[max_vars - n..]);
            let upstream_point = shape.upstream_point(logical);
            let claim = shape
                .layout()
                .reduce_claims(
                    &PrefixPackedClaims::new(
                        shape.layout_digest(),
                        upstream_point,
                        Fp128::as_inner_slice(&opening.slot_evals).to_vec(),
                    ),
                    transcript,
                )
                .map_err(|e| bad(format!("prefix-packed claim reduction: {e}")))?;
            let commitment: AkitaCommitment = serde_from_bytes(&class.commitment)
                .map_err(|_| bad("malformed Akita class commitment".into()))?;
            let akita_proof: AkitaBatchProof = serde_from_bytes(&opening.proof)
                .map_err(|_| bad("malformed Akita class proof".into()))?;
            let class_setup = setup.class_setup(&shape);
            Upstream::verify(
                &commitment,
                claim.point.as_slice(),
                claim.value,
                &akita_proof,
                &class_setup.verifier,
                transcript,
            )
            .map_err(|e| {
                bad(format!(
                    "Akita class {} opening rejected: {e}",
                    opening.class
                ))
            })?;
        }
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"AkitaScheme"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        multilinear_polynomial::PolynomialEvaluation, one_hot_polynomial::OneHotPolynomial,
    };
    use ark_std::test_rng;
    use rand::Rng;

    fn random_point(rng: &mut impl Rng, n: usize) -> Vec<Fp128> {
        (0..n).map(|_| Fp128::random(rng)).collect()
    }

    /// jolt-poly and joltworks agree on the variable order of dense and
    /// one-hot polynomials, up to the documented one-hot rotation.
    #[test]
    fn evaluation_conventions_agree() {
        let mut rng = test_rng();
        let n = 6;
        let coeffs: Vec<i32> = (0..1 << n).map(|_| rng.gen_range(-1000..1000)).collect();
        let ours = MultilinearPolynomial::<Fp128>::from(coeffs.clone());
        let theirs =
            UpstreamDense::<Inner>::new(coeffs.iter().map(|&c| Fp128::from_i32(c).0).collect());
        let r = random_point(&mut rng, n);
        assert_eq!(
            ours.evaluate(&r).0,
            theirs.evaluate(Fp128::as_inner_slice(&r))
        );

        let t = 32;
        let indices: Vec<Option<u16>> = (0..t)
            .map(|_| (rng.gen::<u8>() % 5 != 0).then(|| rng.gen_range(0..16u16)))
            .collect();
        let ours = MultilinearPolynomial::<Fp128>::OneHot(OneHotPolynomial::from_indices(
            indices.clone(),
            ONE_HOT_K,
        ));
        let theirs = UpstreamOneHot::new(
            ONE_HOT_K,
            indices.iter().map(|k| k.map(|k| k as u8)).collect(),
        );
        let shape = ClassShape::new(
            ClassKey {
                flavor: Flavor::OneHot,
                logical_num_vars: ours.get_num_vars(),
            },
            1,
        );
        let r = random_point(&mut rng, ours.get_num_vars());
        let rotated = shape.upstream_point(Fp128::as_inner_slice(&r));
        assert_eq!(
            ours.evaluate(&r).0,
            <UpstreamOneHot as MultilinearPoly<Inner>>::evaluate(&theirs, &rotated)
        );
    }

    /// Commit a mixed batch (several arities, dense and one-hot), open the
    /// joint RLC at a random point through the Atlas seams, and verify.
    #[test]
    fn joint_opening_roundtrip() {
        use common::CommittedPoly;
        let mut rng = test_rng();
        let polys: Vec<MultilinearPolynomial<Fp128>> = vec![
            MultilinearPolynomial::from(
                (0..1 << 5)
                    .map(|_| rng.gen_range(-50i32..50))
                    .collect::<Vec<_>>(),
            ),
            MultilinearPolynomial::from((0..1 << 5).map(|_| rng.gen::<u8>()).collect::<Vec<_>>()),
            MultilinearPolynomial::from((0..1 << 8).map(|_| rng.gen::<i64>()).collect::<Vec<_>>()),
            MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                (0..64).map(|_| Some(rng.gen_range(0..16u16))).collect(),
                ONE_HOT_K,
            )),
            MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                (0..64)
                    .map(|i| (i % 7 != 0).then(|| rng.gen_range(0..16u16)))
                    .collect(),
                ONE_HOT_K,
            )),
        ];
        let ids: Vec<CommittedPoly> = (0..polys.len())
            .map(|i| CommittedPoly::NodeOutputRaD(i, 0))
            .collect();
        let max_vars = polys.iter().map(|p| p.get_num_vars()).max().unwrap();
        let setup = AkitaScheme::setup_prover(max_vars);
        let vsetup = AkitaScheme::setup_verifier(&setup);

        let (batch, per_poly) = AkitaScheme::batch_commit_all(&polys, &setup);
        assert_eq!(
            batch.classes.len(),
            3,
            "two dense arities and one one-hot arity"
        );
        let (slots, hints): (Vec<_>, Vec<_>) = per_poly.into_iter().unzip();

        let r = random_point(&mut rng, max_vars);
        let coeffs: Vec<Fp128> = (0..polys.len()).map(|_| Fp128::random(&mut rng)).collect();
        // Joint claim of the index-0 overlap RLC, exactly as Atlas defines it.
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
        let mut pt = Blake2bTranscript::new(b"akita-test");
        pt.append_serializable(&batch);
        let proof = AkitaScheme::prove_rlc(&setup, &map, &coeffs, hints, &r, &mut pt);
        assert_eq!(proof.classes.len(), 3);

        let mut vt = Blake2bTranscript::new(b"akita-test");
        vt.append_serializable(&batch);
        let refs: Vec<&AkitaSlot> = slots.iter().collect();
        AkitaScheme::verify_rlc(&proof, &vsetup, &mut vt, &r, &joint, &refs, &coeffs, &batch)
            .expect("honest joint opening verifies");
        assert_eq!(pt.state, vt.state, "prover and verifier transcripts agree");

        // Tampered evaluation is rejected.
        let mut bad = proof.clone();
        bad.classes[0].slot_evals[0] += Fp128::one();
        let mut vt = Blake2bTranscript::new(b"akita-test");
        vt.append_serializable(&batch);
        assert!(AkitaScheme::verify_rlc(
            &bad, &vsetup, &mut vt, &r, &joint, &refs, &coeffs, &batch
        )
        .is_err());

        // Wrong joint claim is rejected.
        let mut vt = Blake2bTranscript::new(b"akita-test");
        vt.append_serializable(&batch);
        let wrong = joint + Fp128::one();
        assert!(AkitaScheme::verify_rlc(
            &proof, &vsetup, &mut vt, &r, &wrong, &refs, &coeffs, &batch
        )
        .is_err());

        // Serialization roundtrip of the proof and batch commitment.
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
