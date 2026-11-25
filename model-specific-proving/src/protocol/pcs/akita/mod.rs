//! Grouped sparse Akita opening for the advice slots of a model.
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::protocol::{
    session::{bind_halves, instance},
    tensor::{eq, weights},
};
use akita_config::CommitmentConfig;
use akita_schedules::ValidatedScheduleCatalog;
use jolt_akita::{
    configs::JoltOneHotK16, AkitaChunkProfile, AkitaCommitment, AkitaField as F,
    AkitaNativeBatching, AkitaOneHotScheduleArtifact, AkitaProverHint, AkitaProverSetup,
    AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
};
use jolt_field::{One, Ring, Zero};
use jolt_openings::{BatchOpeningScheme, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, UnivariatePoly};
use jolt_sumcheck::{
    prove_single, round_pow_bits, verify_single, ProveRounds, SingleSumcheckProof, SumcheckError,
};
use jolt_transcript::{FieldTranscript, ProverTranscript, VerifierTranscript};

use rayon::prelude::*;

use super::*;
#[cfg(all(test, feature = "akita-metal", target_os = "macos"))]
mod metal;
mod packed;
use packed::{Indices, PackedFold};

fn equation(v: &[F]) -> F {
    v[0] * (v[1] + v[2] * (v[0] - F::one()))
}

fn pow<T: FieldTranscript<F> + ProverTranscript>(loss: usize, t: &mut T) -> Result<()> {
    t.grind(round_pow_bits::<F, T>(loss).map_err(|_| "unsupported Akita PoW")?)
        .map_err(|_| "Akita proof of work failed")
}

fn check_pow<T: FieldTranscript<F> + VerifierTranscript>(loss: usize, t: &mut T) -> Result<()> {
    t.check_grind(round_pow_bits::<F, T>(loss).map_err(|_| "unsupported Akita PoW")?)
        .map_err(|_| "invalid Akita proof of work")
}

const ADDRESS_BITS: usize = 4;
const K: usize = 1 << ADDRESS_BITS;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    rounds: usize,
    count: usize,
}

impl Key {
    fn digest(self) -> [u8; 32] {
        let mut bytes = [0; 32];
        bytes[..8].copy_from_slice(b"mdl-spv2");
        bytes[9] = self.rounds as u8;
        bytes[16..24].copy_from_slice(&(self.count as u64).to_le_bytes());
        bytes
    }
}

/// Schedule artifacts and reusable setup data for grouped model advice.
/// The same public configuration is used for both proving and verifying.
pub struct AkitaAdviceSetup {
    #[cfg(all(feature = "akita-metal", target_os = "macos"))]
    metal: bool,
    dense_profile: AkitaChunkProfile,
    full_dense: Vec<u8>,
    dense_catalog: Vec<u8>,
    one_hot_catalog: ValidatedScheduleCatalog,
    sparse: BTreeMap<usize, Vec<usize>>,
    max_group_rows: usize,
    prover: Mutex<BTreeMap<Key, Arc<AkitaProverSetup>>>,
    verifier: Mutex<BTreeMap<Key, Arc<AkitaVerifierSetup>>>,
}

impl AkitaAdviceSetup {
    pub fn new(artifacts: Arc<AkitaScheduleArtifacts>) -> Result<Self> {
        let dense_catalog = artifacts
            .dense_catalog()
            .and_then(|catalog| catalog.to_artifact_bytes())
            .map_err(|_| "invalid Akita dense schedules")?;
        let sparse = artifacts
            .one_hot_catalog(K)
            .map_err(|_| "invalid Akita one-hot schedules")?;
        let mut shapes = BTreeMap::<usize, Vec<usize>>::new();
        for row in sparse.rows() {
            let profile = row.profiles();
            if profile.precommitteds.is_empty() {
                let shape = profile.final_group.group;
                shapes
                    .entry(shape.num_vars())
                    .or_default()
                    .push(shape.num_polynomials());
            }
        }
        for widths in shapes.values_mut() {
            widths.sort_unstable_by_key(|&n| std::cmp::Reverse(n));
            widths.dedup();
        }
        Ok(Self {
            #[cfg(all(feature = "akita-metal", target_os = "macos"))]
            metal: false,
            dense_profile: artifacts.dense_profile(),
            full_dense: artifacts
                .full_dense_catalog()
                .and_then(|catalog| catalog.to_artifact_bytes())
                .map_err(|_| "invalid Akita full-width dense schedules")?,
            sparse: shapes,
            max_group_rows: usize::MAX,
            dense_catalog,
            one_hot_catalog: sparse,
            prover: Mutex::default(),
            verifier: Mutex::default(),
        })
    }

    /// Cap the sum of padded one-hot rows in each group. Prover and verifier
    /// must use the same limit with the same schedule catalog.
    pub fn with_group_row_limit(mut self, rows: usize) -> Result<Self> {
        if rows == 0 {
            return Err("empty Akita group row limit");
        }
        self.max_group_rows = rows;
        Ok(self)
    }

    /// Use Jolt’s Metal kernels for supported Akita commitment and opening operations.
    #[cfg(all(feature = "akita-metal", target_os = "macos"))]
    pub fn with_metal(mut self) -> std::result::Result<Self, jolt_metal::MetalError> {
        jolt_metal::runtime::Device::system_default()?;
        self.metal = true;
        Ok(self)
    }

    fn setup(&self, key: Key) -> Result<Arc<AkitaProverSetup>> {
        cached(&self.prover, key, || {
            let (pk, _) =
                AkitaScheme::setup(self.params(key)?).map_err(|_| "Akita prover setup failed")?;
            #[cfg(all(feature = "akita-metal", target_os = "macos"))]
            if self.native_metal(key) {
                return pk
                    .with_trace_compute_kernel(|expanded| {
                        Ok(Arc::new(jolt_akita::MetalTraceCompute::new(expanded)?))
                    })
                    .map_err(|_| "Akita Metal commitment setup failed");
            }
            Ok(pk)
        })
    }

    #[cfg(all(feature = "akita-metal", target_os = "macos"))]
    fn native_metal(&self, key: Key) -> bool {
        self.metal
            && key.count <= 64
            && self.one_hot_catalog.rows().any(|row| {
                let profiles = row.profiles();
                let group = profiles.final_group;
                profiles.precommitteds.is_empty()
                    && group.group.num_vars() == key.rounds
                    && group.group.num_polynomials() == key.count
                    && group.inner.matrix.ring_dimension() == 512
                    && group.blocks.live_ring_elements_per_claim >= group.blocks.positions_per_block
            })
    }

    fn commit_group(
        &self,
        key: Key,
        polynomials: &Arc<[Indices]>,
    ) -> Result<(AkitaCommitment, AkitaProverHint)> {
        let pk = self.setup(key)?;
        AkitaScheme::commit_trace_one_hot(
            &pk,
            key.digest(),
            Arc::new(GroupRows::new(Arc::clone(polynomials))),
            &[],
        )
        .map_err(|_| "Akita advice commitment failed")
    }

    fn verification_setup(&self, key: Key) -> Result<Arc<AkitaVerifierSetup>> {
        cached(&self.verifier, key, || {
            AkitaScheme::verifier_setup_from_params(&self.params(key)?)
                .map_err(|_| "Akita verifier setup failed")
        })
    }

    fn params(&self, key: Key) -> Result<AkitaSetupParams> {
        // The plan fixes the opening shape. Admit only matching standalone
        // rows so each key decodes and prepares its own schedule.
        let one_hot = vec![AkitaOneHotScheduleArtifact {
            k: K,
            profile: AkitaChunkProfile::default(),
            bytes: shape_catalog::<JoltOneHotK16>(&self.one_hot_catalog, key)?,
        }];
        let artifacts = Arc::new(
            AkitaScheduleArtifacts::new(
                self.dense_profile,
                self.dense_catalog.clone(),
                self.full_dense.clone(),
                one_hot,
            )
            .map_err(|_| "invalid shape-specific Akita artifacts")?,
        );
        Ok(AkitaSetupParams::one_hot_only(
            key.rounds,
            key.count,
            key.digest(),
            K,
            artifacts,
        ))
    }
}

fn cached<T>(
    cache: &Mutex<BTreeMap<Key, Arc<T>>>,
    key: Key,
    build: impl FnOnce() -> Result<T>,
) -> Result<Arc<T>> {
    let mut cache = cache.lock().map_err(|_| "Akita setup cache unavailable")?;
    match cache.entry(key) {
        std::collections::btree_map::Entry::Occupied(entry) => Ok(entry.get().clone()),
        std::collections::btree_map::Entry::Vacant(entry) => {
            Ok(entry.insert(Arc::new(build()?)).clone())
        }
    }
}

fn shape_catalog<C: CommitmentConfig>(
    catalog: &ValidatedScheduleCatalog,
    key: Key,
) -> Result<Vec<u8>> {
    let rows = catalog.rows().filter_map(|row| {
        let profile = row.profiles();
        let shape = profile.final_group.group;
        (profile.precommitteds.is_empty()
            && shape.num_vars() == key.rounds
            && shape.num_polynomials() == key.count)
            .then(|| (profile.clone(), row.schedule().clone()))
    });
    ValidatedScheduleCatalog::try_new(
        catalog.family_name(),
        rows,
        catalog.policy(),
        C::ring_challenge_config,
    )
    .and_then(|selected| selected.to_artifact_bytes())
    .map_err(|_| "invalid Akita schedule for advice shape")
}

struct Block {
    leaf: usize,
    start: usize,
    cycles: usize,
}
struct Group {
    key: Key,
    polynomials: Vec<Vec<Block>>,
}

impl Slot {
    fn read_digits(self, start: usize, q: Opening<F>, scale: F, reads: &mut Vec<Read>) {
        let (plane, cell) = q.claim.point.split_at(q.claim.point.len() - self.cell_vars);
        let lane_vars = self.digit_vars();
        if let Some(weights) = q.planes {
            // All digit terms share the opening's one random coefficient.
            for (i, lane) in weights.chunks_exact(1 << lane_vars).enumerate() {
                reads.push(Read::new(
                    start + (i << self.cell_vars),
                    cell,
                    digit_weights(lane, scale),
                ));
            }
        } else {
            let (digit, lane) = plane.split_at(plane.len() - lane_vars);
            reads.push(Read::new(
                start,
                &[digit, cell].concat(),
                digit_weights(&weights(lane), scale),
            ));
        }
    }
}

struct Plan(Vec<Group>);
impl Plan {
    fn new(layout: &Layout, setup: &AkitaAdviceSetup) -> Result<Self> {
        let mut buckets = BTreeMap::<_, Vec<_>>::new();
        for &slot in &layout.slots {
            let local = slot.vars()
                - match slot.kind {
                    Kind::Lookup { address_vars } => address_vars,
                    Kind::Bits { .. } => slot.digit_vars(),
                };
            let (&minimum, _) = setup
                .sparse
                .range(local + ADDRESS_BITS..)
                .next()
                .ok_or("missing Akita domain schedule")?;
            buckets
                .entry(minimum)
                .or_default()
                .push((slot.id, 1 << local));
        }
        let mut groups = Vec::new();
        for (minimum, slots) in buckets {
            let (rounds, packed, widths) = setup
                .sparse
                .range(minimum..)
                .find_map(|(&rounds, widths)| {
                    let rows = 1 << (rounds - ADDRESS_BITS);
                    let limit = setup.max_group_rows / rows;
                    if limit == 0 {
                        return None;
                    }
                    let packed = pack_cycles(slots.clone(), rows);
                    partition(packed.len(), widths, limit).map(|widths| (rounds, packed, widths))
                })
                .ok_or("missing Akita group schedule within row limit")?;
            let mut packed = packed.into_iter();
            for count in widths {
                groups.push(Group {
                    key: Key { rounds, count },
                    polynomials: packed.by_ref().take(count).collect(),
                });
            }
        }
        Ok(Self(groups))
    }

    fn requests(&self, layout: &Layout, requests: Vec<Opening<F>>) -> Vec<Vec<Vec<Request>>> {
        let mut slots: Vec<Vec<_>> = layout.slots.iter().map(|_| Vec::new()).collect();
        for opening in requests {
            slots[opening.slot].push(opening);
        }
        self.0
            .iter()
            .map(|group| {
                group
                    .polynomials
                    .iter()
                    .map(|blocks| {
                        blocks
                            .iter()
                            .flat_map(|block| {
                                std::mem::take(&mut slots[block.leaf])
                                    .into_iter()
                                    .map(|opening| Request {
                                        start: block.start,
                                        opening,
                                    })
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

pub struct AkitaState {
    plan: Plan,
    groups: Vec<CommittedGroup>,
}
struct CommittedGroup {
    polynomials: Arc<[Indices]>,
    hint: AkitaProverHint,
}

struct GroupRows {
    polynomials: Arc<[Indices]>,
    prefix: usize,
}

impl GroupRows {
    fn new(polynomials: Arc<[Indices]>) -> Self {
        let prefix = polynomials
            .iter()
            .map(|poly| poly.prefix)
            .max()
            .unwrap_or(0);
        Self {
            polynomials,
            prefix,
        }
    }
}

impl jolt_akita::TraceOneHotRows for GroupRows {
    fn num_rows(&self) -> usize {
        self.polynomials[0].len()
    }

    fn num_columns(&self) -> usize {
        self.polynomials.len()
    }

    fn nonzero_prefix_len(&self) -> usize {
        self.prefix
    }

    fn fill_row(&self, row: usize, selected: &mut [u8]) {
        for (value, poly) in selected.iter_mut().zip(self.polynomials.iter()) {
            *value = poly.get(row).unwrap_or(jolt_akita::no_selected_row());
        }
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        self.polynomials
            .iter()
            .enumerate()
            .fold(0, |mask, (column, poly)| {
                mask | (u64::from(poly.get(row) == Some(0)) << column)
            })
    }
}

pub struct GroupProof {
    reduction: SingleSumcheckProof<F>,
    evaluations: Vec<F>,
    opening: <AkitaScheme as CommitmentScheme>::Proof,
}

impl ModelPcs for AkitaScheme {
    type Field = F;
    type AdvicePlan = ();
    type AdviceProverSetup = AkitaAdviceSetup;
    type AdviceVerifierSetup = AkitaAdviceSetup;
    type AdviceCommitments = Vec<AkitaCommitment>;
    type AdviceState = AkitaState;
    type AdviceOpening = Vec<GroupProof>;

    fn plan(_: &Layout) -> Result<()> {
        Ok(())
    }

    fn commit(
        layout: &Layout,
        advice: &mut Advice,
        _: &(),
        setup: &AkitaAdviceSetup,
    ) -> Result<(Self::AdviceCommitments, AkitaState)> {
        let plan = Plan::new(layout, setup)?;
        let mut commitments = Vec::with_capacity(plan.0.len());
        let mut groups = Vec::with_capacity(plan.0.len());
        for group in &plan.0 {
            let mut polynomials = Vec::with_capacity(group.key.count);
            for blocks in &group.polynomials {
                let mut indices = Indices::new(1 << (group.key.rounds - ADDRESS_BITS));
                for block in blocks {
                    let slot = layout.slots[block.leaf];
                    match slot.kind {
                        Kind::Bits { .. } => {
                            for (row, &digit) in advice.digits(slot)?.iter().enumerate() {
                                indices.insert(block.start + row, digit);
                            }
                        }
                        Kind::Lookup { .. } => {
                            for (row, column) in advice.take_lookup(slot)?.enumerate() {
                                indices.insert(block.start + row, column);
                            }
                        }
                    }
                }
                polynomials.push(indices);
            }
            let polynomials: Arc<[_]> = polynomials.into();
            let (commitment, hint) = setup.commit_group(group.key, &polynomials)?;
            commitments.push(commitment);
            groups.push(CommittedGroup { polynomials, hint });
        }
        Ok((commitments, AkitaState { plan, groups }))
    }

    fn bind(commitments: &Self::AdviceCommitments, t: &mut impl Transcript) {
        t.append(&(commitments.len() as u64));
        for c in commitments {
            t.append(c);
        }
    }

    fn open(
        layout: &Layout,
        claims: Vec<Opening<F>>,
        commitments: &Self::AdviceCommitments,
        state: AkitaState,
        _: &(),
        setup: &AkitaAdviceSetup,
        t: &mut ProverState<Keccak>,
    ) -> Result<Vec<GroupProof>> {
        let reads = state.plan.requests(layout, claims);
        state
            .plan
            .0
            .iter()
            .zip(state.groups)
            .zip(reads)
            .zip(commitments)
            .map(|(((group, witness), reads), commitment)| {
                prove_group(layout, group, reads, witness, commitment, setup, t)
            })
            .collect()
    }

    fn verify(
        layout: &Layout,
        claims: Vec<Opening<F>>,
        commitments: &Self::AdviceCommitments,
        proof: &Self::AdviceOpening,
        _: &(),
        setup: &AkitaAdviceSetup,
        t: &mut VerifierState<'_, Keccak>,
    ) -> Result<()> {
        let plan = Plan::new(layout, setup)?;
        if commitments.len() != plan.0.len() || proof.len() != plan.0.len() {
            return Err("invalid Akita group count");
        }
        let reads = plan.requests(layout, claims);
        for (((group, reads), commitment), proof) in
            plan.0.iter().zip(reads).zip(commitments).zip(proof)
        {
            let vk = setup.verification_setup(group.key)?;
            verify_group(layout, group, reads, commitment, proof, &vk, t)?;
        }
        Ok(())
    }
}

fn pack_cycles(mut leaves: Vec<(usize, usize)>, capacity: usize) -> Vec<Vec<Block>> {
    leaves.sort_by_key(|&(leaf, cycles)| (std::cmp::Reverse(cycles), leaf));
    let mut packed = Vec::new();
    let mut blocks = Vec::new();
    let mut start = 0;
    for (leaf, cycles) in leaves {
        if start + cycles > capacity {
            packed.push(std::mem::take(&mut blocks));
            start = 0;
        }
        debug_assert_eq!(start % cycles, 0);
        blocks.push(Block {
            leaf,
            start,
            cycles,
        });
        start += cycles;
    }
    if !blocks.is_empty() {
        packed.push(blocks);
    }
    packed
}

fn fixed_prefix(rounds: usize, offset: usize, free: usize) -> Vec<F> {
    (free..rounds)
        .rev()
        .map(|bit| F::from_bool(offset & (1 << bit) != 0))
        .collect()
}

fn address_weights(point: &[F]) -> [F; K] {
    weights(point).try_into().expect("K=16 address")
}

struct Request {
    start: usize,
    opening: Opening<F>,
}

fn digit_weights(lanes: &[F], scale: F) -> [F; K] {
    std::array::from_fn(|address| {
        scale
            * lanes
                .iter()
                .enumerate()
                .filter(|(i, _)| address & (1 << i) != 0)
                .map(|(_, &w)| w)
                .sum::<F>()
    })
}

/// A separable public weight on an aligned interval of the committed row axis.
struct Read {
    start: usize,
    rows: Vec<F>,
    address: [F; K],
}

impl Read {
    fn new(start: usize, rows: &[F], address: [F; K]) -> Self {
        Self {
            start,
            rows: rows.to_vec(),
            address,
        }
    }

    fn evaluate(&self, address: &[F; K], rows: &[F]) -> F {
        let (prefix, suffix) = rows.split_at(rows.len() - self.rows.len());
        let coefficient: F = self.address.iter().zip(address).map(|(&a, &b)| a * b).sum();
        coefficient
            * eq(
                prefix,
                &fixed_prefix(rows.len(), self.start, self.rows.len()),
            )
            * eq(&self.rows, suffix)
    }
}

fn partition(count: usize, widths: &[usize], limit: usize) -> Option<Vec<usize>> {
    let mut best = vec![None; count + 1];
    best[0] = Some((0, 0));
    for remaining in 1..=count {
        best[remaining] = widths
            .iter()
            .filter(|&&width| width <= remaining.min(limit))
            .filter_map(|&width| best[remaining - width].map(|(groups, _)| (groups + 1, width)))
            .min_by_key(|&(groups, _)| groups);
    }
    let mut widths = Vec::new();
    let mut remaining = count;
    while remaining != 0 {
        let (_, width) = best[remaining]?;
        widths.push(width);
        remaining -= width;
    }
    Some(widths)
}

struct GroupQueries {
    bit_check: Vec<F>,
    bit_mix: Vec<F>,
    reads: Vec<Vec<Read>>,
    sum: F,
}

impl GroupQueries {
    fn dimension(key: Key, claims: &[Vec<Request>]) -> usize {
        3 * key.rounds - ADDRESS_BITS + 3 * key.count + claims.iter().map(Vec::len).sum::<usize>()
    }

    fn read(
        layout: &Layout,
        group: &Group,
        claims: Vec<Vec<Request>>,
        t: &mut impl FieldTranscript<F>,
    ) -> Result<Self> {
        let key = group.key;
        let row_vars = key.rounds - ADDRESS_BITS;
        let random: Vec<F> = t
            .challenge_vector(Self::dimension(key, &claims))
            .map_err(|_| "invalid Akita group reduction challenge")?;
        let (bit_check, random) = random.split_at(key.rounds);
        let (valid_check, random) = random.split_at(key.rounds);
        let (mass_check, random) = random.split_at(row_vars);
        let (bit_mix, random) = random.split_at(key.count);
        let (valid_mix, random) = random.split_at(key.count);
        let (mass_mix, mut weights) = random.split_at(key.count);
        let valid_address = address_weights(&valid_check[..ADDRESS_BITS]);
        let valid_rows = &valid_check[ADDRESS_BITS..];

        let mut groups = Vec::with_capacity(key.count);
        let mut sum = F::zero();
        for (i, (requests, blocks)) in claims.into_iter().zip(&group.polynomials).enumerate() {
            let (mix, rest) = weights.split_at(requests.len());
            weights = rest;
            let mut reads = Vec::new();
            for (Request { start, opening }, &mix) in requests.into_iter().zip(mix) {
                sum += opening.claim.value * mix;
                let slot = layout.slots[opening.slot];
                match slot.kind {
                    Kind::Bits { .. } => slot.read_digits(start, opening, mix, &mut reads),
                    Kind::Lookup { address_vars } => {
                        let (address, rows) = opening.claim.point.split_at(address_vars);
                        let mut padded = vec![F::zero(); ADDRESS_BITS - address_vars];
                        padded.extend_from_slice(address);
                        reads.push(Read::new(
                            start,
                            rows,
                            address_weights(&padded).map(|a| mix * a),
                        ));
                    }
                }
            }

            let occupied: F = blocks
                .iter()
                .map(|block| {
                    let prefix = fixed_prefix(row_vars, block.start, block.cycles.ilog2() as usize);
                    eq(&mass_check[..prefix.len()], &prefix)
                })
                .sum();
            sum += mass_mix[i] * occupied;
            reads.push(Read::new(0, mass_check, [mass_mix[i]; K]));

            for block in blocks {
                let leaf = layout.slots[block.leaf];
                let Kind::Bits { planes } = leaf.kind else {
                    continue;
                };
                let cells = 1 << leaf.cell_vars;
                let lanes = 1 << leaf.digit_vars();
                for digit in 0..block.cycles / cells {
                    let bits = planes.saturating_sub(digit * lanes).min(lanes);
                    if bits == ADDRESS_BITS {
                        continue;
                    }
                    let start = block.start + digit * cells;
                    let prefix = fixed_prefix(row_vars, start, leaf.cell_vars);
                    let mix = valid_mix[i] * eq(&valid_rows[..prefix.len()], &prefix);
                    let address = std::array::from_fn(|a| {
                        if a < 1 << bits {
                            F::zero()
                        } else {
                            valid_address[a] * mix
                        }
                    });
                    reads.push(Read::new(start, &valid_rows[prefix.len()..], address));
                }
            }
            groups.push(reads);
        }
        Ok(Self {
            bit_check: bit_check.to_vec(),
            bit_mix: bit_mix.to_vec(),
            reads: groups,
            sum,
        })
    }
}

struct GroupFold<'a> {
    rounds: usize,
    packed: Vec<PackedFold<'a>>,
}

impl ProveRounds<F> for GroupFold<'_> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        _: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        let samples = self
            .packed
            .par_iter_mut()
            .map(|fold| {
                if let Some(r) = bind {
                    if round <= ADDRESS_BITS {
                        fold.bind_address(r, ADDRESS_BITS - round);
                    } else {
                        fold.bind_row(r);
                    }
                }
                if round < ADDRESS_BITS {
                    fold.address_samples(round)
                } else {
                    fold.row_samples()
                }
            })
            .reduce(
                || [F::zero(); 4],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );
        Ok(UnivariatePoly::from_evals(&samples))
    }

    fn finish_rounds(&mut self, last: F) -> std::result::Result<(), SumcheckError<F>> {
        for fold in &mut self.packed {
            fold.bind_row(last);
        }
        Ok(())
    }
}

fn group_point(point: &[F]) -> Vec<F> {
    let (address, rows) = point.split_at(ADDRESS_BITS);
    [rows, address].concat()
}

fn opening_statement(
    commitment: &AkitaCommitment,
    point: &[F],
    values: &[F],
) -> Vec<VerifierOpeningClaim<F, AkitaCommitment>> {
    values
        .iter()
        .map(|&value| VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), value),
        })
        .collect()
}

fn prove_group(
    layout: &Layout,
    group: &Group,
    claims: Vec<Vec<Request>>,
    witness: CommittedGroup,
    commitment: &AkitaCommitment,
    setup: &AkitaAdviceSetup,
    t: &mut (impl FieldTranscript<F> + ProverTranscript),
) -> Result<GroupProof> {
    let key = group.key;
    pow(GroupQueries::dimension(key, &claims), t)?;
    let queries = GroupQueries::read(layout, group, claims, t)?;
    let mut fold = GroupFold {
        rounds: key.rounds,
        packed: witness
            .polynomials
            .iter()
            .zip(&queries.reads)
            .zip(&queries.bit_mix)
            .map(|((poly, reads), &mix)| PackedFold::new(poly, reads, &queries.bit_check, mix))
            .collect(),
    };
    let (reduction, reduced) = prove_single(&mut fold, &instance(queries.sum, key.rounds, 3), t)
        .map_err(|_| "Akita lookup group reduction failed")?;
    let evaluations: Vec<_> = fold.packed.iter().map(|fold| fold.value(0)).collect();
    drop(fold);
    let point = group_point(&reduced.challenges);
    t.append_values(&evaluations);
    let polynomials = witness
        .polynomials
        .iter()
        .map(|poly| poly as &dyn MultilinearPoly<F>)
        .collect();
    let pk = setup.setup(key)?;
    let opening = AkitaNativeBatching::prove_batch(
        &pk,
        opening_statement(commitment, &point, &evaluations),
        polynomials,
        witness.hint,
        t,
    )
    .map_err(|_| "Akita lookup group opening failed")?;
    Ok(GroupProof {
        reduction,
        evaluations,
        opening,
    })
}

fn verify_group(
    layout: &Layout,
    group: &Group,
    claims: Vec<Vec<Request>>,
    commitment: &AkitaCommitment,
    proof: &GroupProof,
    setup: &jolt_akita::AkitaVerifierSetup,
    t: &mut (impl FieldTranscript<F> + VerifierTranscript),
) -> Result<()> {
    let key = group.key;
    if proof.evaluations.len() != key.count || commitment.layout_digest() != key.digest() {
        return Err("invalid Akita advice group layout");
    }
    check_pow(GroupQueries::dimension(key, &claims), t)?;
    let queries = GroupQueries::read(layout, group, claims, t)?;
    let reduced = verify_single(&instance(queries.sum, key.rounds, 3), &proof.reduction, t)
        .map_err(|_| "invalid Akita group Sumcheck")?;
    let p = &reduced.challenges;
    let (address, rows) = p.split_at(ADDRESS_BITS);
    let address = address_weights(address);
    let bit_weight = eq(&queries.bit_check, p);
    let expected = queries
        .reads
        .iter()
        .zip(&proof.evaluations)
        .zip(&queries.bit_mix)
        .map(|((reads, &value), &bit_mix)| {
            let weight = reads.iter().map(|read| read.evaluate(&address, rows)).sum();
            equation(&[value, weight, bit_mix * bit_weight])
        })
        .sum();
    if reduced.output_claim != expected {
        return Err("invalid Akita lookup group evaluation");
    }
    t.append_values(&proof.evaluations);
    let statement = opening_statement(commitment, &group_point(p), &proof.evaluations);
    AkitaNativeBatching::verify_batch(setup, &statement, &proof.opening, t)
        .map_err(|_| "invalid Akita lookup group opening")
}
