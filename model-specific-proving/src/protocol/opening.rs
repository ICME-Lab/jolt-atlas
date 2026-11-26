use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    ops::Range,
    sync::{Arc, OnceLock},
};

use bincode::Options;
use jolt_field::{Ring, Zero};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Keccak, ProverState, Transcript, VerifierState,
};
use world_common::{shape::MIN_LOG_BITS, BitZParams, LinearClaim, Root, SecurityLevel, Shape};
use world_field::{gf128::smallest_generator, F128};
use world_pcs::Pcs;
use world_transcript::{build_prover, build_verifier, Proof as BitProof};

use super::{
    field::{Scalar, F},
    lookup::Addresses,
    storage::Codes,
    sumcheck::{Column, Messages, Session},
    tensor::{equality, inner, weights, Claim, Shape as TensorShape},
    Result,
};

const ROW_BITS: usize = 9;
const WORD_BITS: usize = u128::BITS as usize;
const WORDS_PER_COLUMN: usize = ROWS / WORD_BITS;
const ROWS: usize = 1 << ROW_BITS;
const GROUP_LIMIT: usize = 32 << 23;
const DOMAIN: &[u8] = b"atlas/transformer/witness";

#[derive(Clone, Copy)]
pub(crate) struct Slot(usize);

#[derive(Clone, Copy)]
struct Region {
    bits: usize,
    capacity: usize,
    variable: bool,
    signed: bool,
}

#[derive(Clone, Default)]
pub(crate) struct Layout(Vec<Region>);

impl Region {
    fn words_per_plane(self) -> usize {
        (1 << self.bits.max(ROW_BITS)) / WORD_BITS
    }
}

#[derive(Clone, Copy)]
struct Placement {
    group: usize,
    offset: usize,
    region: Region,
    planes: usize,
}

pub(crate) struct Plan {
    slots: Vec<Placement>,
    groups: Vec<(BitZParams<Scalar>, Pcs)>,
}

pub(crate) struct Request {
    pub slot: Slot,
    pub claim: Claim,
    pub planes: Vec<F>,
    pub one_hot_sum: bool,
}

pub struct Proof {
    pub(crate) metadata: ProofMetadata,
    bit_proofs: Vec<BitProof>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ProofMetadata {
    roots: Vec<[u8; 32]>,
    pub(crate) widths: Vec<usize>,
    messages: Messages,
    openings: Messages,
    #[serde(with = "super::storage::bytes")]
    transcript: Vec<u8>,
}

impl Layout {
    pub(crate) fn reserve(&mut self, bits: usize, capacity: usize) -> Slot {
        let slot = Slot(self.0.len());
        self.0.push(Region {
            bits,
            capacity,
            variable: false,
            signed: false,
        });
        slot
    }

    pub(crate) fn variable(&mut self, slot: Slot, signed: bool) {
        self.0[slot.0].variable = true;
        self.0[slot.0].signed = signed;
    }

    pub(crate) fn packed(&self) -> Packed<'_> {
        Packed {
            layout: self,
            widths: self.0.iter().map(|r| r.capacity).collect(),
            blocks: (0..self.0.len()).map(|_| Vec::new()).collect(),
            spool: None,
        }
    }

    pub(crate) fn streamed(&self) -> Result<Packed<'_>> {
        let mut packed = self.packed();
        packed.spool = Some(Spool {
            file: tempfile::tempfile().map_err(|_| "could not create witness file")?,
            slots: (0..self.0.len()).map(|_| SlotSource::default()).collect(),
        });
        Ok(packed)
    }

    pub(crate) fn finish(&self, widths: &[usize]) -> Result<Plan> {
        static GENERATOR: OnceLock<F128> = OnceLock::new();
        if widths.len() != self.0.len() {
            return Err("invalid witness layout");
        }
        let mut spans = Vec::new();
        let mut slots = Vec::with_capacity(widths.len());
        for (id, (region, &planes)) in self.0.iter().copied().zip(widths).enumerate() {
            if planes > region.capacity || (!region.variable && planes != region.capacity) {
                return Err("unsupported witness width");
            }
            slots.push(Placement {
                group: 0,
                offset: 0,
                region,
                planes,
            });
            if planes != 0 {
                let span = 1usize
                    .checked_shl(region.bits.max(ROW_BITS) as u32)
                    .and_then(|n| n.checked_mul(planes))
                    .ok_or("range layout overflow")?;
                spans.push((id, span));
            }
        }
        let total = spans
            .iter()
            .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
            .ok_or("range layout overflow")?;
        let mut sizes = Vec::<(usize, usize)>::new();
        if total <= GROUP_LIMIT {
            sizes.push((GROUP_LIMIT, 0));
        } else {
            spans.sort_unstable_by_key(|&(id, n)| (std::cmp::Reverse(n), id));
        }
        for (id, n) in spans {
            let group = sizes
                .iter()
                .enumerate()
                .filter(|&(_, &(limit, used))| n <= limit - used)
                .min_by_key(|&(_, &(limit, used))| limit - used)
                .map(|(i, _)| i)
                .unwrap_or_else(|| {
                    sizes.push((n.next_power_of_two().max(GROUP_LIMIT), 0));
                    sizes.len() - 1
                });
            let used = &mut sizes[group].1;
            slots[id].group = group;
            slots[id].offset = *used / ROWS;
            *used += n;
        }
        let groups = sizes
            .into_iter()
            .map(|(_, used)| {
                let vars = (used.next_power_of_two().ilog2() as usize).max(MIN_LOG_BITS);
                let shape =
                    Shape::new(ROW_BITS, vars - ROW_BITS).map_err(|_| "invalid BitZ shape")?;
                let params = BitZParams::new(shape, *GENERATOR.get_or_init(smallest_generator))
                    .map_err(|_| "invalid BitZ parameters")?;
                let pcs = Pcs::new(&shape, SecurityLevel::Bits128)
                    .map_err(|_| "unsupported BitZ shape")?;
                Ok((params, pcs))
            })
            .collect::<Result<_>>()?;
        Ok(Plan { slots, groups })
    }
}

pub(crate) struct Packed<'a> {
    layout: &'a Layout,
    pub(crate) widths: Vec<usize>,
    blocks: Vec<Vec<F128>>,
    spool: Option<Spool>,
}

struct Spool {
    file: File,
    slots: Vec<SlotSource>,
}

#[derive(Default)]
struct SlotSource {
    ranges: Vec<RangeRecord>,
    digit: Option<Digit>,
}

#[derive(Clone, Copy)]
struct RangeRecord {
    offset: u64,
    plane: usize,
    width: usize,
    shape: TensorShape,
    base: u128,
    padding: u128,
}

impl RangeRecord {
    fn code(self, codes: &Codes, i: usize) -> u128 {
        let stride = self.shape.width.next_power_of_two();
        let (row, col) = (i / stride, i % stride);
        if row < self.shape.rows && col < self.shape.width {
            self.base + codes.get(row * self.shape.width + col)
        } else {
            self.padding
        }
    }
}

#[derive(Clone)]
struct Digit {
    addresses: Arc<Addresses>,
    shift: usize,
    width: usize,
}

enum Source {
    Ranges(Vec<(RangeRecord, Codes)>),
    Lookup(Digit),
}

type Sources = HashMap<usize, Source>;

impl Spool {
    fn save(&mut self, codes: &Codes) -> Result<u64> {
        let offset = self
            .file
            .stream_position()
            .map_err(|_| "could not locate witness")?;
        bincode::serialize_into(&mut self.file, codes).map_err(|_| "could not write witness")?;
        Ok(offset)
    }

    fn group(
        &mut self,
        slots: &[(usize, Placement)],
        len: usize,
        retain: bool,
    ) -> Result<(Vec<F128>, Sources)> {
        let mut group = F128::zeroed_vec(len);
        let mut sources = Sources::new();
        let mut lookups: HashMap<_, Vec<_>> = HashMap::new();
        for &(slot, placement) in slots {
            let src = &mut self.slots[slot];
            if let Some(digit) = if retain {
                src.digit.take()
            } else {
                src.digit.clone()
            } {
                lookups
                    .entry(Arc::as_ptr(&digit.addresses))
                    .or_default()
                    .push((placement, digit.clone()));
                if retain {
                    sources.insert(slot, Source::Lookup(digit));
                }
                continue;
            }
            let stride = placement.region.words_per_plane();
            let base = placement.offset * WORDS_PER_COLUMN;
            let mut records = Vec::new();
            for &record in &src.ranges {
                self.file
                    .seek(SeekFrom::Start(record.offset))
                    .map_err(|_| "could not locate witness")?;
                let codes: Codes = bincode::deserialize_from(&mut self.file)
                    .map_err(|_| "could not read witness")?;
                pack_range(
                    &mut group[base..],
                    stride,
                    record.plane,
                    record.width,
                    1 << record.shape.bits(),
                    |i| record.code(&codes, i),
                )?;
                if retain {
                    records.push((record, codes));
                }
            }
            if retain {
                sources.insert(slot, Source::Ranges(records));
            }
        }
        for digits in lookups.into_values() {
            let addresses = &digits[0].1.addresses;
            let mut rows = vec![[0u128; 16]; digits.len()];
            for word in 0..addresses.len().div_ceil(WORD_BITS) {
                rows.fill([0; 16]);
                let start = word * WORD_BITS;
                for i in start..(start + WORD_BITS).min(addresses.len()) {
                    let value = addresses.get(i);
                    for (bits, (_, digit)) in rows.iter_mut().zip(&digits) {
                        bits[(value >> digit.shift) & (digit.width - 1)] |= 1 << (i - start);
                    }
                }
                for (bits, (placement, digit)) in rows.iter().zip(&digits) {
                    let base = placement.offset * WORDS_PER_COLUMN;
                    let stride = placement.region.words_per_plane();
                    for (plane, &bits) in bits[..digit.width].iter().enumerate() {
                        group[base + plane * stride + word] = F128::from(bits);
                    }
                }
            }
        }
        Ok((group, sources))
    }
}

fn pack_range(
    words: &mut [F128],
    stride: usize,
    plane: usize,
    width: usize,
    len: usize,
    code: impl Fn(usize) -> u128,
) -> Result<()> {
    let mut rows = [0u128; 128];
    for word in 0..len.div_ceil(WORD_BITS) {
        rows[..width].fill(0);
        let start = word * WORD_BITS;
        for i in start..(start + WORD_BITS).min(len) {
            let mut bits = code(i);
            if bits >> width != 0 {
                return Err("integer outside the witness range");
            }
            while bits != 0 {
                rows[bits.trailing_zeros() as usize] |= 1 << (i - start);
                bits &= bits - 1;
            }
        }
        for (bit, &row) in rows[..width].iter().enumerate() {
            words[(plane + bit) * stride + word] = F128::from(row);
        }
    }
    Ok(())
}

impl Packed<'_> {
    pub(crate) fn is_streamed(&self) -> bool {
        self.spool.is_some()
    }

    pub(crate) fn shrink(&mut self, slot: Slot, width: usize) -> Result<()> {
        let region = self.layout.0[slot.0];
        if !region.variable
            || width > region.capacity
            || !self.blocks[slot.0].is_empty()
            || self
                .spool
                .as_ref()
                .is_some_and(|spool| !spool.slots[slot.0].ranges.is_empty())
        {
            return Err("invalid witness width");
        }
        self.widths[slot.0] = width;
        Ok(())
    }

    pub(crate) fn one_hot(
        &mut self,
        digits: &[(Slot, usize, usize)],
        addresses: &Arc<Addresses>,
    ) -> Result<()> {
        if let Some(spool) = &mut self.spool {
            for &(slot, shift, width) in digits {
                spool.slots[slot.0].digit = Some(Digit {
                    addresses: Arc::clone(addresses),
                    shift,
                    width: 1 << width,
                });
            }
            return Ok(());
        }
        let len = addresses.len();
        let stride = len.max(ROWS).div_ceil(WORD_BITS);
        for &(slot, _, width) in digits {
            self.blocks[slot.0] = F128::zeroed_vec(stride << width);
        }

        let mut rows = vec![[0u128; 16]; digits.len()];
        for word in 0..stride {
            rows.fill([0; 16]);
            let start = word * WORD_BITS;
            for i in start..(start + WORD_BITS).min(len) {
                let code = addresses.get(i);
                for (bits, &(_, shift, width)) in rows.iter_mut().zip(digits) {
                    let digit = (code >> shift) & ((1 << width) - 1);
                    bits[digit] |= 1 << (i - start);
                }
            }
            for (bits, &(slot, _, width)) in rows.iter().zip(digits) {
                let block = &mut self.blocks[slot.0];
                for (plane, &bits) in bits[..1 << width].iter().enumerate() {
                    block[plane * stride + word] = F128::from(bits);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn write<T: Copy + Into<i128>>(
        &mut self,
        slot: Slot,
        planes: Range<usize>,
        shape: TensorShape,
        offset: i128,
        values: &[T],
    ) -> Result<()> {
        let region = self.layout.0[slot.0];
        let width = planes.len();
        if planes.end > self.widths[slot.0] || width >= 128 {
            return Err("invalid witness plane");
        }
        if width == 0 {
            return Ok(());
        }
        if values.len() != shape.len() {
            return Err("invalid range witness dimensions");
        }
        let encoded = |x: T| (x.into() + offset) as u128;
        if let Some(spool) = &mut self.spool {
            let (min, max) = values
                .iter()
                .copied()
                .map(encoded)
                .fold((u128::MAX, 0), |(min, max), value| {
                    (min.min(value), max.max(value))
                });
            if max >> width != 0 || (offset as u128) >> width != 0 {
                return Err("integer outside the witness range");
            }
            let bits = (128 - (max - min).leading_zeros()) as usize;
            let codes = Codes::pack(
                values.len(),
                bits,
                values.iter().copied().map(|x| encoded(x) - min),
            )
            .ok_or("integer outside the witness range")?;
            let start = spool.save(&codes)?;
            spool.slots[slot.0].ranges.push(RangeRecord {
                offset: start,
                plane: planes.start,
                width,
                shape,
                base: min,
                padding: offset as u128,
            });
            return Ok(());
        }
        let stride = region.words_per_plane();
        let words = &mut self.blocks[slot.0];
        if words.is_empty() {
            *words = F128::zeroed_vec(self.widths[slot.0] * stride);
        }
        let cols = shape.width.next_power_of_two();
        pack_range(words, stride, planes.start, width, 1 << shape.bits(), |i| {
            let (row, col) = (i / cols, i % cols);
            if row < shape.rows && col < shape.width {
                encoded(values[row * shape.width + col])
            } else {
                offset as u128
            }
        })
    }
}

struct Query<'a> {
    slot: usize,
    offset: usize,
    cols: Arc<[F]>,
    row: Vec<F>,
    planes: &'a [F],
    mix: F,
    lanes: Option<usize>,
}

enum Coefficients<'a> {
    Empty,
    Powers { start: usize, scale: F },
    Arbitrary(&'a [F]),
}

impl<'a> Coefficients<'a> {
    fn new(planes: &'a [F]) -> Self {
        let Some(start) = planes.iter().position(|p| !p.is_zero()) else {
            return Self::Empty;
        };
        let scale = planes[start];
        let mut expected = scale;
        for &coefficient in &planes[start..] {
            if coefficient != expected {
                return Self::Arbitrary(planes);
            }
            expected += expected;
        }
        Self::Powers { start, scale }
    }

    fn evaluate(&self, mut code: u128) -> F {
        match self {
            Self::Empty => F::zero(),
            Self::Powers { start, scale } => *scale * F::from_u128(code >> start),
            Self::Arbitrary(weights) => {
                let mut sum = F::zero();
                while code != 0 {
                    sum += weights[code.trailing_zeros() as usize];
                    code &= code - 1;
                }
                sum
            }
        }
    }
}

fn project(queries: &[Query<'_>], packed: &[F128], sources: &Sources) -> Vec<Column<'static>> {
    let mut by_row = HashMap::<&[F], usize>::new();
    let mut points = Vec::new();
    let mut projections = Vec::new();
    let mut by_slot = HashMap::<usize, Vec<(&Query<'_>, usize)>>::new();
    for query in queries {
        let row = *by_row.entry(&query.row).or_insert_with(|| {
            let i = projections.len();
            projections.push(vec![F::zero(); ROWS]);
            points.push(query.row.as_slice());
            i
        });
        if let Some(lanes) = query.lanes {
            // The honest witness has exactly one set bit per cell.
            for value in &mut projections[row][..lanes] {
                *value += query.mix;
            }
        } else {
            by_slot.entry(query.slot).or_default().push((query, row));
        }
    }
    for (slot, requests) in by_slot {
        let query = requests[0].0;
        let mut factors = vec![F::zero(); requests.len()];
        match sources.get(&slot) {
            Some(Source::Ranges(ranges)) => {
                for (record, codes) in ranges {
                    let coefficients: Vec<_> = requests
                        .iter()
                        .map(|(q, _)| {
                            Coefficients::new(&q.planes[record.plane..record.plane + record.width])
                        })
                        .collect();
                    let len = 1 << record.shape.bits();
                    for col in 0..query.cols.len() {
                        for (factor, (q, _)) in factors.iter_mut().zip(&requests) {
                            *factor = q.mix * q.cols[col];
                        }
                        let start = col * ROWS;
                        for i in start..(start + ROWS).min(len) {
                            let code = record.code(codes, i);
                            if code != 0 {
                                for (((_, row), coefficient), &factor) in
                                    requests.iter().zip(&coefficients).zip(&factors)
                                {
                                    projections[*row][i - start] +=
                                        coefficient.evaluate(code) * factor;
                                }
                            }
                        }
                    }
                }
            }
            Some(Source::Lookup(digit)) => {
                for col in 0..query.cols.len() {
                    for (factor, (q, _)) in factors.iter_mut().zip(&requests) {
                        *factor = q.mix * q.cols[col];
                    }
                    let start = col * ROWS;
                    for i in start..(start + ROWS).min(digit.addresses.len()) {
                        let lane = (digit.addresses.get(i) >> digit.shift) & (digit.width - 1);
                        for ((q, row), &factor) in requests.iter().zip(&factors) {
                            projections[*row][i - start] += q.planes[lane] * factor;
                        }
                    }
                }
            }
            None => {
                let start = query.offset * WORDS_PER_COLUMN;
                let cols = query.cols.len();
                for (i, words) in packed[start..]
                    .chunks_exact(WORDS_PER_COLUMN)
                    .take(query.planes.len() * cols)
                    .enumerate()
                {
                    for (factor, (q, _)) in factors.iter_mut().zip(&requests) {
                        *factor = q.planes[i / cols] * q.mix * q.cols[i % cols];
                    }
                    for (word, &value) in words.iter().enumerate() {
                        let mut bits = value.to_u128();
                        while bits != 0 {
                            let lane = word * WORD_BITS + bits.trailing_zeros() as usize;
                            for ((_, row), &factor) in requests.iter().zip(&factors) {
                                projections[*row][lane] += factor;
                            }
                            bits &= bits - 1;
                        }
                    }
                }
            }
        }
    }
    projections
        .into_iter()
        .zip(points)
        .flat_map(|(values, point)| [Column::Dense(values), Column::Dense(weights(point))])
        .collect()
}

impl Plan {
    fn requests(&self, requests: Vec<Request>) -> Result<Vec<Vec<Request>>> {
        let mut groups: Vec<Vec<Request>> = (0..self.groups.len()).map(|_| Vec::new()).collect();
        let mut used = vec![false; self.slots.len()];
        for mut request in requests {
            let placement = self
                .slots
                .get(request.slot.0)
                .ok_or("invalid witness slot")?;
            if request.claim.point.len() != placement.region.bits
                || request.planes.len() != placement.region.capacity
            {
                return Err("invalid witness evaluation shape");
            }
            used[request.slot.0] = true;
            if placement.planes == 0 {
                if !request.claim.value.is_zero() {
                    return Err("invalid evaluation of omitted bits");
                }
                continue;
            }
            if placement.region.signed {
                // The missing planes select the signed offset before the batch is reduced.
                request.claim.value += F::from_i128(1 << (placement.planes - 1));
            }
            request.planes.truncate(placement.planes);
            groups[placement.group].push(request);
        }
        if used.contains(&false) {
            return Err("unverified witness slot");
        }
        Ok(groups)
    }

    fn reduce(
        &self,
        requests: &[Request],
        params: &BitZParams<Scalar>,
        witness: Option<(&[F128], &Sources)>,
        io: &mut Session<'_, '_>,
    ) -> Result<LinearClaim<Scalar>> {
        let mix = io.random(requests.len(), requests.len())?;
        let sum = inner(requests.iter().map(|r| r.claim.value), mix.iter().copied());
        let mut column_weights: HashMap<&[F], Arc<[F]>> = HashMap::new();
        let queries: Vec<_> = requests
            .iter()
            .zip(mix)
            .map(|(r, mix)| {
                let offset = self.slots[r.slot.0].offset;
                let (column, row) = r
                    .claim
                    .point
                    .split_at(r.claim.point.len().saturating_sub(ROW_BITS));
                let mut row_point = vec![F::zero(); ROW_BITS - row.len()];
                row_point.extend_from_slice(row);
                let weights = column_weights
                    .entry(column)
                    .or_insert_with(|| weights(column).into())
                    .clone();
                Query {
                    slot: r.slot.0,
                    offset,
                    cols: weights,
                    row: row_point,
                    planes: &r.planes,
                    mix,
                    lanes: r.one_hot_sum.then_some(1 << row.len()),
                }
            })
            .collect();
        let (end, _) = io.fold(
            sum,
            ROW_BITS,
            2,
            || {
                let (packed, sources) = witness.expect("prover witness");
                project(&queries, packed, sources)
            },
            [],
            |row| row.chunks_exact(2).map(|pair| pair[0] * pair[1]).sum(),
        )?;

        let mut columns = vec![Scalar::from(0u64); params.shape().columns()];
        for q in queries {
            let scale = q.mix * equality(&q.row, &end.challenges);
            let factors = q.planes.iter().flat_map(|&p| {
                let scale = p * scale;
                q.cols.iter().map(move |&c| scale * c)
            });
            for (dst, weight) in columns[q.offset..].iter_mut().zip(factors) {
                *dst += weight.0;
            }
        }
        let rows = weights(&end.challenges).into_iter().map(|v| v.0).collect();
        LinearClaim::new(params, rows, columns, end.output_claim.0)
            .map_err(|_| "invalid BitZ evaluation")
    }

    pub(crate) fn prove(
        &self,
        domain: &[u8; 32],
        packed: Packed<'_>,
        public: impl FnOnce(&mut ProverState<Keccak>),
        check: impl FnOnce(&mut Session<'_, '_>) -> Result<Vec<Request>>,
    ) -> Result<Proof> {
        let mut bit_proofs = Vec::new();
        let metadata = self.prove_groups(domain, packed, public, check, |proof| {
            bit_proofs.push(proof);
            Ok(())
        })?;
        Ok(Proof {
            metadata,
            bit_proofs,
        })
    }

    pub(crate) fn prove_streamed(
        &self,
        domain: &[u8; 32],
        packed: Packed<'_>,
        public: impl FnOnce(&mut ProverState<Keccak>),
        check: impl FnOnce(&mut Session<'_, '_>) -> Result<Vec<Request>>,
        mut writer: impl Write + Seek,
    ) -> Result<()> {
        if writer
            .seek(SeekFrom::End(0))
            .map_err(|_| "could not locate proof destination")?
            != 0
        {
            return Err("proof destination is not empty");
        }
        // Publish the header only after the group proofs and metadata have been written.
        writer
            .write_all(&[0; 16])
            .map_err(|_| "could not write proof header")?;
        let proof = self.prove_groups(domain, packed, public, check, |proof| {
            for bytes in [&proof.narg_string, &proof.hints] {
                writer
                    .write_all(&(bytes.len() as u64).to_le_bytes())
                    .and_then(|_| writer.write_all(bytes))
                    .map_err(|_| "could not write witness proof")?;
            }
            Ok(())
        })?;
        let metadata = writer
            .stream_position()
            .map_err(|_| "could not locate proof metadata")?;
        proof_codec()
            .serialize_into(&mut writer, &proof)
            .map_err(|_| "could not write proof metadata")?;
        writer
            .rewind()
            .and_then(|_| writer.write_all(FILE_MAGIC))
            .and_then(|_| writer.write_all(&metadata.to_le_bytes()))
            .and_then(|_| writer.flush())
            .map_err(|_| "could not finalize proof")
    }

    fn prove_groups(
        &self,
        domain: &[u8; 32],
        packed: Packed<'_>,
        public: impl FnOnce(&mut ProverState<Keccak>),
        check: impl FnOnce(&mut Session<'_, '_>) -> Result<Vec<Request>>,
        mut emit: impl FnMut(BitProof) -> Result<()>,
    ) -> Result<ProofMetadata> {
        let Packed {
            widths,
            mut blocks,
            mut spool,
            ..
        } = packed;
        let streamed = spool.is_some();
        let mut groups = vec![Vec::new(); self.groups.len()];
        for (slot, &placement) in self.slots.iter().enumerate() {
            if placement.planes != 0 {
                groups[placement.group].push((slot, placement));
            }
        }
        let mut group = |slots: &[(usize, Placement)], pcs: &Pcs, retain: bool| {
            let (block, sources) = if let Some(spool) = &mut spool {
                spool.group(slots, pcs.packed_len(), retain)?
            } else {
                let mut block = F128::zeroed_vec(pcs.packed_len());
                for &(slot, placement) in slots {
                    let words = std::mem::take(&mut blocks[slot]);
                    block[placement.offset * WORDS_PER_COLUMN..][..words.len()]
                        .copy_from_slice(&words);
                }
                (block, Sources::new())
            };
            let mut channel = build_prover(DOMAIN, b"" as &[u8]);
            let (root, data) = pcs
                .commit(&block, &mut channel)
                .map_err(|_| "could not commit witness")?;
            Ok((root, (data, block, channel, sources)))
        };

        let mut transcript = prover_transcript::<Keccak>(domain, b"");
        public(&mut transcript);
        bind_widths(&mut transcript, &widths);
        let mut roots = Vec::new();
        let mut commitments = Vec::new();
        for ((_, pcs), slots) in self.groups.iter().zip(&groups) {
            let (root, data) = group(slots, pcs, false)?;
            transcript.append_bytes(&root.0);
            roots.push(root.0);
            if !streamed {
                commitments.push(data);
            }
        }
        let mut commitments = commitments.into_iter();
        let mut messages = Messages::default();
        let requests =
            self.requests(check(&mut Session::Prover(&mut transcript, &mut messages))?)?;
        let mut openings = Messages::default();
        let mut io = Session::Prover(&mut transcript, &mut openings);
        for ((((params, pcs), slots), queries), expected) in
            self.groups.iter().zip(groups).zip(requests).zip(&roots)
        {
            let (data, block, mut channel, sources) = if let Some(data) = commitments.next() {
                data
            } else {
                let (root, data) = group(&slots, pcs, true)?;
                if &root.0 != expected {
                    return Err("regenerated witness root differs");
                }
                data
            };
            let claim = self.reduce(&queries, params, Some((&block, &sources)), &mut io)?;
            drop(sources);
            world_prover::BitZProver::new(*params, 8)
                .prove(&claim, pcs, &data, block, &mut channel)
                .map_err(|_| "could not prove witness evaluation")?;
            emit(channel.finish())?;
        }
        Ok(ProofMetadata {
            roots,
            widths,
            messages,
            openings,
            transcript: transcript.narg_string().to_vec(),
        })
    }

    pub(crate) fn verify(
        &self,
        domain: &[u8; 32],
        proof: &Proof,
        public: impl FnOnce(&mut VerifierState<'_, Keccak>),
        check: impl FnOnce(&mut Session<'_, '_>) -> Result<Vec<Request>>,
    ) -> Result<()> {
        if proof.bit_proofs.len() != self.groups.len() {
            return Err("invalid proof group count");
        }
        self.verify_with(domain, &proof.metadata, public, check, |i| {
            Ok(Cow::Borrowed(&proof.bit_proofs[i]))
        })
    }

    pub(crate) fn verify_streamed(
        &self,
        domain: &[u8; 32],
        proof: &ProofMetadata,
        end: u64,
        mut reader: impl Read + Seek,
        public: impl FnOnce(&mut VerifierState<'_, Keccak>),
        check: impl FnOnce(&mut Session<'_, '_>) -> Result<Vec<Request>>,
    ) -> Result<()> {
        reader
            .seek(SeekFrom::Start(16))
            .map_err(|_| "could not locate witness proofs")?;
        let mut reader = reader.take(end - 16);
        self.verify_with(domain, proof, public, check, |_| {
            Ok(Cow::Owned(BitProof {
                narg_string: read_bytes(&mut reader)?,
                hints: read_bytes(&mut reader)?,
            }))
        })?;
        if reader.limit() != 0 {
            return Err("unused witness proofs");
        }
        Ok(())
    }

    fn verify_with<'p>(
        &self,
        domain: &[u8; 32],
        proof: &ProofMetadata,
        public: impl FnOnce(&mut VerifierState<'_, Keccak>),
        check: impl FnOnce(&mut Session<'_, '_>) -> Result<Vec<Request>>,
        mut group_proof: impl FnMut(usize) -> Result<Cow<'p, BitProof>>,
    ) -> Result<()> {
        if proof.roots.len() != self.groups.len() {
            return Err("invalid proof group count");
        }
        let mut transcript = verifier_transcript::<Keccak>(domain, b"", &proof.transcript);
        public(&mut transcript);
        bind_widths(&mut transcript, &proof.widths);
        for root in &proof.roots {
            transcript.append_bytes(root);
        }
        let mut io = proof.messages.verifier(&mut transcript);
        let requests = self.requests(check(&mut io)?)?;
        io.finish()?;
        let mut io = proof.openings.verifier(&mut transcript);
        for (i, (((params, pcs), &root), requests)) in self
            .groups
            .iter()
            .zip(&proof.roots)
            .zip(requests)
            .enumerate()
        {
            let bits = group_proof(i)?;
            let mut channel = build_verifier(DOMAIN, b"" as &[u8], &bits);
            let commitment = pcs
                .receive_commitment(Root(root), &mut channel)
                .map_err(|_| "invalid commitment")?;
            let claim = self.reduce(&requests, params, None, &mut io)?;
            world_verifier::BitZVerifier::new(*params, 8)
                .verify(&claim, pcs, &commitment, channel)
                .map_err(|_| "invalid witness evaluation")?;
        }
        io.finish()?;
        transcript
            .check_eof()
            .map_err(|_| "unused transcript bytes")
    }
}

fn bind_widths(transcript: &mut impl Transcript, widths: &[usize]) {
    transcript.append(&(widths.len() as u64));
    for &width in widths {
        transcript.append(&(width as u64));
    }
}

// Header: magic and metadata offset. The group records occupy the bytes before metadata.
const FILE_MAGIC: &[u8; 8] = b"ATLSPRF1";

fn proof_codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn read_u64(mut reader: impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| "truncated proof")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_bytes(reader: &mut std::io::Take<impl Read>) -> Result<Vec<u8>> {
    let len = read_u64(&mut *reader)?;
    if len > reader.limit() {
        return Err("invalid witness proof length");
    }
    let mut part = reader.take(len);
    let mut bytes = Vec::new();
    part.read_to_end(&mut bytes)
        .map_err(|_| "could not read witness proof")?;
    if part.limit() != 0 {
        return Err("truncated witness proof");
    }
    Ok(bytes)
}

pub(crate) fn read_proof(mut reader: impl Read + Seek) -> Result<(ProofMetadata, u64)> {
    let length = reader
        .seek(SeekFrom::End(0))
        .map_err(|_| "could not locate proof end")?;
    reader
        .rewind()
        .map_err(|_| "could not locate proof header")?;
    let mut magic = [0; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|_| "truncated proof header")?;
    if &magic != FILE_MAGIC {
        return Err("invalid proof header");
    }
    let metadata = read_u64(&mut reader)?;
    if !(16..length).contains(&metadata) {
        return Err("invalid proof metadata offset");
    }
    reader
        .seek(SeekFrom::Start(metadata))
        .map_err(|_| "could not locate proof metadata")?;
    let mut tail = reader.take(length - metadata);
    let proof = proof_codec()
        .with_limit(length - metadata)
        .deserialize_from(&mut tail)
        .map_err(|_| "could not read proof metadata")?;
    if tail.limit() != 0 {
        return Err("unused proof metadata");
    }
    Ok((proof, metadata))
}
