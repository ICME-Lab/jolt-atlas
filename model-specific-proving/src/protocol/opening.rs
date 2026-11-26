use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    ops::Range,
    sync::{Arc, OnceLock},
};

use bincode::Options;
use jolt_field::{One, Ring, Zero};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Keccak, ProverState, Transcript, VerifierState,
};
use tracing::info_span;
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
        self.with_store(Store::Memory(Vec::new()))
    }

    pub(crate) fn streamed(&self) -> Result<Packed<'_>> {
        let file = tempfile::tempfile().map_err(|_| "could not create witness file")?;
        Ok(self.with_store(Store::File(file)))
    }

    fn with_store(&self, store: Store) -> Packed<'_> {
        Packed {
            layout: self,
            widths: self.0.iter().map(|r| r.capacity).collect(),
            witness: Witness {
                store,
                slots: (0..self.0.len()).map(|_| Stored::default()).collect(),
            },
        }
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
    witness: Witness,
}

struct Witness {
    store: Store,
    slots: Vec<Stored>,
}

enum Store {
    Memory(Vec<Option<Codes>>),
    File(File),
}

impl Store {
    fn save(&mut self, codes: Codes) -> Result<u64> {
        match self {
            Self::Memory(values) => {
                let index = values.len();
                values.push(Some(codes));
                Ok(index as u64)
            }
            Self::File(file) => {
                let offset = file
                    .stream_position()
                    .map_err(|_| "could not locate witness")?;
                bincode::serialize_into(file, &codes).map_err(|_| "could not write witness")?;
                Ok(offset)
            }
        }
    }

    fn load(&mut self, offset: u64) -> Result<Codes> {
        match self {
            Self::Memory(values) => values[offset as usize].take().ok_or("witness already used"),
            Self::File(file) => {
                file.seek(SeekFrom::Start(offset))
                    .map_err(|_| "could not locate witness")?;
                bincode::deserialize_from(file).map_err(|_| "could not read witness")
            }
        }
    }
}

type Stored = Input<u64>;

enum Input<T> {
    Ranges(Vec<(RangeRecord, T)>),
    Lookup(Digit),
}

impl<T> Default for Input<T> {
    fn default() -> Self {
        Self::Ranges(Vec::new())
    }
}

#[derive(Clone, Copy)]
struct RangeRecord {
    plane: usize,
    width: usize,
    shape: TensorShape,
    base: u128,
    padding: u128,
}

impl RangeRecord {
    fn read(self, codes: &Codes, start: usize, values: &mut [u128]) -> Option<u128> {
        let stride = self.shape.width.next_power_of_two();
        let row = start / stride;
        let col = start % stride;
        if row >= self.shape.rows || (stride >= values.len() && col >= self.shape.width) {
            return Some(self.padding);
        }
        if codes.width == 0 && (self.base == self.padding || col + values.len() <= self.shape.width)
        {
            return Some(self.base);
        }
        let mut index = start;
        let mut rest = values;
        while !rest.is_empty() {
            let (row, col) = (index / stride, index % stride);
            let count = (stride - col).min(rest.len());
            let (part, next) = rest.split_at_mut(count);
            let active = if row < self.shape.rows {
                self.shape.width.saturating_sub(col).min(count)
            } else {
                0
            };
            let offset = row * self.shape.width + col;
            for (i, out) in part[..active].iter_mut().enumerate() {
                *out = self.base + codes.get(offset + i);
            }
            part[active..].fill(self.padding);
            index += count;
            rest = next;
        }
        None
    }
}

#[derive(Clone)]
struct Digit {
    addresses: Arc<Addresses>,
    shift: usize,
    width: usize,
}

type Sources = HashMap<usize, Input<Codes>>;

impl Witness {
    fn group(
        &mut self,
        slots: &[(usize, Placement)],
        len: usize,
        retain: bool,
        mut projection: Option<&mut Projection<'_, '_>>,
    ) -> Result<(Vec<F128>, Sources)> {
        let mut group = F128::zeroed_vec(len);
        let mut sources = Sources::new();
        let mut lookups: HashMap<_, Vec<_>> = HashMap::new();
        for &(slot, placement) in slots {
            let records = match &self.slots[slot] {
                Input::Lookup(digit) => {
                    lookups
                        .entry(Arc::as_ptr(&digit.addresses))
                        .or_default()
                        .push((slot, placement, digit.clone()));
                    if retain {
                        sources.insert(slot, Input::Lookup(digit.clone()));
                        self.slots[slot] = Stored::default();
                    }
                    continue;
                }
                Input::Ranges(records) => records,
            };
            let stride = placement.region.words_per_plane();
            let base = placement.offset * WORDS_PER_COLUMN;
            let mut decoded = Vec::new();
            for &(record, position) in records {
                let codes = self.store.load(position)?;
                let requests = projection
                    .as_ref()
                    .map_or_else(Vec::new, |p| p.range_requests(slot, record));
                pack_range(
                    &mut group[base..],
                    stride,
                    record.plane,
                    record.width,
                    1 << record.shape.bits(),
                    record,
                    &codes,
                    |col, values, scale| {
                        if let Some(p) = &mut projection {
                            p.range(&requests, col, values, scale);
                        }
                    },
                )?;
                if retain {
                    decoded.push((record, codes));
                }
            }
            if retain {
                sources.insert(slot, Input::Ranges(decoded));
            }
        }
        for digits in lookups.into_values() {
            let addresses = &digits[0].2.addresses;
            let selectors: Vec<_> = digits.iter().map(|(s, _, d)| (*s, d)).collect();
            let mut rows = vec![[0u128; 16]; digits.len()];
            visit_columns(
                addresses.len(),
                |i| addresses.at(i),
                |col, values| {
                    if let Some(p) = &mut projection {
                        p.lookup(&selectors, col, values);
                    }
                    for (word, values) in values.chunks(WORD_BITS).enumerate() {
                        rows.fill([0; 16]);
                        for (row, &value) in values.iter().enumerate() {
                            for (bits, (_, _, digit)) in rows.iter_mut().zip(&digits) {
                                bits[(value >> digit.shift) & (digit.width - 1)] |= 1 << row;
                            }
                        }
                        for (bits, (_, placement, digit)) in rows.iter().zip(&digits) {
                            let base = placement.offset * WORDS_PER_COLUMN + col * WORDS_PER_COLUMN;
                            let stride = placement.region.words_per_plane();
                            for (plane, &bits) in bits[..digit.width].iter().enumerate() {
                                group[base + plane * stride + word] = F128::from(bits);
                            }
                        }
                    }
                },
            );
        }
        Ok((group, sources))
    }
}

fn visit_columns<T: Copy + Default>(
    len: usize,
    read: impl Fn(usize) -> T,
    mut visit: impl FnMut(usize, &[T]),
) {
    let mut values = [T::default(); ROWS];
    for col in 0..len.div_ceil(ROWS) {
        let start = col * ROWS;
        let values = &mut values[..(len - start).min(ROWS)];
        for (i, value) in values.iter_mut().enumerate() {
            *value = read(start + i);
        }
        visit(col, values);
    }
}

fn pack_range(
    words: &mut [F128],
    stride: usize,
    plane: usize,
    width: usize,
    len: usize,
    record: RangeRecord,
    codes: &Codes,
    mut project: impl FnMut(usize, &[u128], Option<u128>),
) -> Result<()> {
    let mut values = [0u128; ROWS];
    let mut planes = [0u128; 128];
    let mut valid = true;
    for col in 0..len.div_ceil(ROWS) {
        let start = col * ROWS;
        let count = (len - start).min(ROWS);
        let values = &mut values[..count];
        let constant = record.read(codes, start, values);
        project(col, values, constant);
        for (word, values) in values.chunks(WORD_BITS).enumerate() {
            planes[..width].fill(0);
            if let Some(value) = constant {
                valid &= value >> width == 0;
                let mask = u128::MAX >> (WORD_BITS - values.len());
                for (bit, out) in planes[..width].iter_mut().enumerate() {
                    if value & (1 << bit) != 0 {
                        *out = mask;
                    }
                }
            } else {
                for (row, &value) in values.iter().enumerate() {
                    valid &= value >> width == 0;
                    let mut bits = value;
                    while bits != 0 {
                        planes[bits.trailing_zeros() as usize] |= 1 << row;
                        bits &= bits - 1;
                    }
                }
            }
            for (bit, &value) in planes[..width].iter().enumerate() {
                words[(plane + bit) * stride + col * WORDS_PER_COLUMN + word] = F128::from(value);
            }
        }
    }
    if valid {
        Ok(())
    } else {
        Err("integer outside the witness range")
    }
}

impl Packed<'_> {
    pub(crate) fn is_streamed(&self) -> bool {
        matches!(self.witness.store, Store::File(_))
    }

    pub(crate) fn shrink(&mut self, slot: Slot, width: usize) -> Result<()> {
        let region = self.layout.0[slot.0];
        if !region.variable
            || width > region.capacity
            || !matches!(&self.witness.slots[slot.0], Input::Ranges(v) if v.is_empty())
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
        for &(slot, shift, width) in digits {
            self.witness.slots[slot.0] = Input::Lookup(Digit {
                addresses: Arc::clone(addresses),
                shift,
                width: 1 << width,
            });
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
        let width = planes.len();
        if planes.end > self.widths[slot.0] || width >= 128 {
            return Err("invalid witness plane");
        }
        if values.len() != shape.len() && !(width == 0 && values.is_empty()) {
            return Err("invalid range witness dimensions");
        }
        if width == 0 {
            return Ok(());
        }
        let encoded = |x: T| (x.into() + offset) as u128;
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
        let start = self.witness.store.save(codes)?;
        let Input::Ranges(records) = &mut self.witness.slots[slot.0] else {
            return Err("expected range witness");
        };
        records.push((
            RangeRecord {
                plane: planes.start,
                width,
                shape,
                base: min,
                padding: offset as u128,
            },
            start,
        ));
        Ok(())
    }
}

#[derive(Clone)]
struct Columns {
    start: usize,
    weights: Arc<[F]>,
}

impl Columns {
    fn new(point: &[F]) -> Self {
        let index = point.iter().try_fold(0, |i, x| {
            (*x == F::zero() || *x == F::one()).then(|| 2 * i + usize::from(!x.is_zero()))
        });
        match index {
            Some(start) => Self {
                start,
                weights: Arc::from([F::one()]),
            },
            None => Self {
                start: 0,
                weights: weights(point).into(),
            },
        }
    }
}

struct Query<'a> {
    slot: usize,
    offset: usize,
    cols: Columns,
    stride: usize,
    row: Vec<F>,
    planes: &'a [F],
    mix: F,
    lanes: Option<usize>,
}

fn range_coefficient(planes: &[F]) -> Option<(F, usize)> {
    // Ranges supplies powers of two, optionally preceded by zero low bits.
    let shift = planes.iter().position(|p| !p.is_zero())?;
    let scale = planes[shift];
    debug_assert!(planes[shift..]
        .iter()
        .zip(powers(scale))
        .all(|(&a, b)| a == b));
    Some((scale, shift))
}

fn powers(scale: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(scale), |v| Some(*v + *v))
}

type RangeProjection<'a, 'q> = (&'a Query<'q>, usize, F, usize);

struct Projection<'a, 'q> {
    by_slot: HashMap<usize, Vec<(&'a Query<'q>, usize)>>,
    points: Vec<&'a [F]>,
    outputs: Vec<Vec<F>>,
    ranges: Vec<(usize, F, usize)>,
    lookups: Vec<(usize, usize, usize, [F; 16])>,
}

impl<'a, 'q> Projection<'a, 'q> {
    fn new(queries: &'a [Query<'q>]) -> Self {
        let mut by_row = HashMap::<&[F], usize>::new();
        let mut projection = Self {
            by_slot: HashMap::new(),
            points: Vec::new(),
            outputs: Vec::new(),
            ranges: Vec::new(),
            lookups: Vec::new(),
        };
        for q in queries {
            let row = *by_row.entry(&q.row).or_insert_with(|| {
                projection.points.push(&q.row);
                projection.outputs.push(vec![F::zero(); ROWS]);
                projection.outputs.len() - 1
            });
            if let Some(lanes) = q.lanes {
                for value in &mut projection.outputs[row][..lanes] {
                    *value += q.mix;
                }
            } else {
                projection.by_slot.entry(q.slot).or_default().push((q, row));
            }
        }
        projection
    }

    fn requests(&self, slot: usize) -> &[(&'a Query<'q>, usize)] {
        self.by_slot.get(&slot).map_or(&[], Vec::as_slice)
    }

    fn range_requests(&self, slot: usize, record: RangeRecord) -> Vec<RangeProjection<'a, 'q>> {
        let planes = record.plane..record.plane + record.width;
        self.requests(slot)
            .iter()
            .filter_map(|&(q, row)| {
                let (scale, shift) = range_coefficient(&q.planes[planes.clone()])?;
                Some((q, row, q.mix * scale, shift))
            })
            .collect()
    }

    fn range(
        &mut self,
        requests: &[RangeProjection<'a, 'q>],
        col: usize,
        values: &[u128],
        constant: Option<u128>,
    ) {
        self.ranges.clear();
        self.ranges
            .extend(requests.iter().filter_map(|&(q, row, scale, shift)| {
                Some((row, q.column_weight(col)? * scale, shift))
            }));
        for &(row, weight, shift) in &self.ranges {
            let output = &mut self.outputs[row][..values.len()];
            if let Some(value) = constant {
                let term = F::from_u128(value >> shift) * weight;
                for dst in output {
                    *dst += term;
                }
            } else {
                for (dst, &value) in output.iter_mut().zip(values) {
                    if value != 0 {
                        *dst += F::from_u128(value >> shift) * weight;
                    }
                }
            }
        }
    }

    fn lookup(&mut self, digits: &[(usize, &Digit)], col: usize, values: &[usize]) {
        self.lookups.clear();
        for &(slot, digit) in digits {
            for &(q, row) in self.by_slot.get(&slot).into_iter().flatten() {
                if let Some(weight) = q.column_weight(col) {
                    let weight = weight * q.mix;
                    let mut table = [F::zero(); 16];
                    for (dst, &coefficient) in table.iter_mut().zip(q.planes) {
                        *dst = coefficient * weight;
                    }
                    self.lookups
                        .push((row, digit.shift, digit.width - 1, table));
                }
            }
        }
        for &(row, shift, mask, ref table) in &self.lookups {
            for (dst, &value) in self.outputs[row].iter_mut().zip(values) {
                *dst += table[(value >> shift) & mask];
            }
        }
    }

    fn read(&mut self, sources: &Sources) {
        let mut lookups: HashMap<_, Vec<_>> = HashMap::new();
        for (&slot, input) in sources {
            match input {
                Input::Ranges(ranges) => {
                    for (record, codes) in ranges {
                        let requests = self.range_requests(slot, *record);
                        let len = 1usize << record.shape.bits();
                        let mut values = [0; ROWS];
                        for col in 0..len.div_ceil(ROWS) {
                            if requests
                                .iter()
                                .all(|(q, ..)| q.column_weight(col).is_none())
                            {
                                continue;
                            }
                            let start = col * ROWS;
                            let values = &mut values[..(len - start).min(ROWS)];
                            let constant = record.read(codes, start, values);
                            self.range(&requests, col, values, constant);
                        }
                    }
                }
                Input::Lookup(digit) if !self.requests(slot).is_empty() => {
                    lookups
                        .entry(Arc::as_ptr(&digit.addresses))
                        .or_default()
                        .push((slot, digit));
                }
                Input::Lookup(_) => {}
            }
        }
        for digits in lookups.into_values() {
            let addresses = &digits[0].1.addresses;
            visit_columns(
                addresses.len(),
                |i| addresses.at(i),
                |col, values| {
                    self.lookup(&digits, col, values);
                },
            );
        }
    }

    fn columns(self) -> Vec<Column<'static>> {
        self.outputs
            .into_iter()
            .zip(self.points)
            .flat_map(|(values, point)| [Column::Dense(values), Column::Dense(weights(point))])
            .collect()
    }
}

impl Query<'_> {
    fn column_weight(&self, col: usize) -> Option<F> {
        self.cols
            .weights
            .get(col.wrapping_sub(self.cols.start))
            .copied()
    }
}

fn add_column_weights(
    columns: &mut [Scalar],
    weights: &[F],
    stride: usize,
    planes: &[F],
    scale: F,
) {
    const TILE: usize = 256;
    let mut start = 0;
    for run in planes.chunk_by(|&a, &b| a + a == b) {
        if !run[0].is_zero() {
            let scale = run[0] * scale;
            if run.len() == 1 {
                for (dst, &value) in columns[start..].iter_mut().zip(weights) {
                    *dst += (scale * value).0;
                }
            } else {
                let mut scratch = [F::zero(); TILE];
                for (tile, weights) in weights.chunks(TILE).enumerate() {
                    let scratch = &mut scratch[..weights.len()];
                    for (dst, &weight) in scratch.iter_mut().zip(weights) {
                        *dst = scale * weight;
                    }
                    let mut offset = start + tile * TILE;
                    for _ in 1..run.len() {
                        for (dst, weight) in columns[offset..].iter_mut().zip(scratch.iter_mut()) {
                            *dst += weight.0;
                            *weight += *weight;
                        }
                        offset += stride;
                    }
                    for (dst, weight) in columns[offset..].iter_mut().zip(scratch.iter()) {
                        *dst += weight.0;
                    }
                }
            }
        }
        start += run.len() * stride;
    }
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
        project: impl FnOnce(&[Query<'_>]) -> Result<Vec<Column<'static>>>,
        io: &mut Session<'_, '_>,
    ) -> Result<LinearClaim<Scalar>> {
        let mix = io.random(requests.len(), requests.len())?;
        let sum = inner(requests.iter().map(|r| r.claim.value), mix.iter().copied());
        let mut column_weights: HashMap<&[F], Columns> = HashMap::new();
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
                    .or_insert_with(|| Columns::new(column))
                    .clone();
                Query {
                    slot: r.slot.0,
                    offset,
                    cols: weights,
                    stride: 1 << column.len(),
                    row: row_point,
                    planes: &r.planes,
                    mix,
                    lanes: r.one_hot_sum.then_some(1 << row.len()),
                }
            })
            .collect();
        let projection = project(&queries)?;
        let _span = info_span!("Atlas BitZ row Sumcheck").entered();
        let (end, _) = io.fold(
            sum,
            ROW_BITS,
            2,
            || projection,
            [],
            |row| row.chunks_exact(2).map(|pair| pair[0] * pair[1]).sum(),
        )?;

        drop(_span);
        let _span = info_span!("Atlas BitZ terminal columns").entered();
        let mut columns = vec![Scalar::from(0u64); params.shape().columns()];
        let mut combined = HashMap::<_, (&Query<'_>, Vec<F>)>::new();
        for q in &queries {
            let scale = q.mix * equality(&q.row, &end.challenges);
            let key = (q.slot, q.cols.start, q.cols.weights.as_ptr());
            let (_, planes) = combined
                .entry(key)
                .or_insert_with(|| (q, vec![F::zero(); q.planes.len()]));
            for (dst, &coefficient) in planes.iter_mut().zip(q.planes) {
                *dst += coefficient * scale;
            }
        }
        for (q, planes) in combined.into_values() {
            add_column_weights(
                &mut columns[q.offset + q.cols.start..],
                &q.cols.weights,
                q.stride,
                &planes,
                F::one(),
            );
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
        #[cfg(all(feature = "metal", target_os = "macos"))]
        super::metal::enable_bitz();
        let streamed = packed.is_streamed();
        let Packed {
            widths,
            mut witness,
            ..
        } = packed;
        let mut groups = vec![Vec::new(); self.groups.len()];
        for (slot, &placement) in self.slots.iter().enumerate() {
            if placement.planes != 0 {
                groups[placement.group].push((slot, placement));
            }
        }
        let commit = |block: Vec<F128>, pcs: &Pcs| {
            let _span = info_span!("Atlas BitZ PCS commitment").entered();
            let mut channel = build_prover(DOMAIN, b"" as &[u8]);
            let (root, data) = pcs
                .commit(&block, &mut channel)
                .map_err(|_| "could not commit witness")?;
            Ok((root, (data, block, channel)))
        };

        let mut transcript = prover_transcript::<Keccak>(domain, b"");
        public(&mut transcript);
        bind_widths(&mut transcript, &widths);
        let mut roots = Vec::new();
        let mut commitments = Vec::new();
        for ((_, pcs), slots) in self.groups.iter().zip(&groups) {
            let (block, sources) = info_span!("Atlas BitZ initial pack")
                .in_scope(|| witness.group(slots, pcs.packed_len(), !streamed, None))?;
            let (root, data) = commit(block, pcs)?;
            transcript.append_bytes(&root.0);
            roots.push(root.0);
            if !streamed {
                commitments.push((data, sources));
            }
        }
        let mut commitments = commitments.into_iter();
        let mut messages = Messages::default();
        let requests =
            self.requests(check(&mut Session::Prover(&mut transcript, &mut messages))?)?;
        let mut openings = Messages::default();
        let mut io = Session::Prover(&mut transcript, &mut openings);
        let mut provers = HashMap::new();
        for ((((params, pcs), slots), queries), expected) in
            self.groups.iter().zip(groups).zip(requests).zip(&roots)
        {
            let mut committed = None;
            let claim = self.reduce(
                &queries,
                params,
                |queries| {
                    let mut projection = Projection::new(queries);
                    if let Some((data, sources)) = commitments.next() {
                        info_span!("Atlas BitZ projection").in_scope(|| projection.read(&sources));
                        committed = Some(data);
                    } else {
                        let (block, _) =
                            info_span!("Atlas BitZ repack and projection").in_scope(|| {
                                witness.group(
                                    &slots,
                                    pcs.packed_len(),
                                    false,
                                    Some(&mut projection),
                                )
                            })?;
                        let (root, data) = commit(block, pcs)?;
                        if &root.0 != expected {
                            return Err("regenerated witness root differs");
                        }
                        for &(slot, _) in &slots {
                            witness.slots[slot] = Stored::default();
                        }
                        committed = Some(data);
                    }
                    Ok(projection.columns())
                },
                &mut io,
            )?;
            let (data, block, mut channel) = committed.ok_or("missing prover commitment")?;
            provers
                .entry(params.shape().columns())
                .or_insert_with(|| world_prover::BitZProver::new(*params, 8))
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
        let mut verifiers = HashMap::new();
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
            let claim = self.reduce(&requests, params, |_| Ok(Vec::new()), &mut io)?;
            verifiers
                .entry(params.shape().columns())
                .or_insert_with(|| world_verifier::BitZVerifier::new(*params, 8))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_weights_match_per_plane_multiplication() {
        let weights: Vec<_> = (0..300).map(F::from_u64).collect();
        let planes = [0, 0, 2, 4, 8, 7, 0, 3, 6].map(F::from_u64);
        let stride = 512;
        let scale = -F::from_u64(11);

        for (offset, weights) in [(0, weights.as_slice()), (511, &weights[7..8])] {
            let mut actual = vec![F::one().0; stride * planes.len()];
            let mut expected = actual.clone();
            for (plane, &coefficient) in planes.iter().enumerate() {
                let start = offset + plane * stride;
                for (dst, &weight) in expected[start..].iter_mut().zip(weights) {
                    *dst += (coefficient * scale * weight).0;
                }
            }
            add_column_weights(&mut actual[offset..], weights, stride, &planes, scale);
            assert_eq!(actual, expected);
        }
    }
}
