use std::{ffi::c_void, ptr::NonNull, sync::OnceLock};

use jolt_field::{CanonicalBytes, One, Ring, Zero};
mod bitz;

pub(crate) use bitz::enable_bitz;

use super::{
    field::F,
    storage::Codes,
    tensor::{weights, Shape},
};

type Scalar = [u8; 16];
type Tile = [u32; 4];

unsafe extern "C" {
    fn atlas_metal_device(data: *const u8, len: usize) -> *mut c_void;
    fn atlas_metal_job(
        device: *mut c_void,
        cells: usize,
        columns: u32,
        reads: u32,
        kind: u32,
        input: *const u32,
        input_bytes: usize,
        tiles: *const Tile,
        tile_bytes: usize,
        table: *const Scalar,
        table_bytes: usize,
        axis_bytes: usize,
    ) -> *mut c_void;
    fn atlas_metal_round(
        job: *mut c_void,
        args: &Args,
        args_len: usize,
        axis: *const Scalar,
        axis_len: usize,
        count: u32,
        threads: u32,
        partial: *mut Scalar,
        partial_len: usize,
    ) -> bool;
    fn atlas_metal_snapshot(job: *mut c_void, data: *mut Scalar, len: usize);
    fn atlas_metal_drop(job: *mut c_void);
}

fn encode(value: F) -> Scalar {
    let mut bytes = [0; 16];
    value.to_bytes_le(&mut bytes);
    bytes
}

fn decode(value: Scalar) -> F {
    F::from_u128(u128::from_le_bytes(value))
}

fn device() -> Option<*mut c_void> {
    static DEVICE: OnceLock<Option<usize>> = OnceLock::new();
    if std::env::var_os("ATLAS_CPU_ONLY").is_some() {
        return None;
    }
    let address = *DEVICE.get_or_init(|| {
        let library = include_bytes!(concat!(env!("OUT_DIR"), "/round.metallib"));
        NonNull::new(unsafe { atlas_metal_device(library.as_ptr(), library.len()) })
            .map(|pointer| pointer.as_ptr() as usize)
    });
    address.map(|pointer| pointer as *mut c_void)
}

fn codes(input: &mut Vec<u32>, codes: &Codes) {
    // Metal reads two words at a time. Codes already stores a safe zero tail.
    input.extend(codes.bytes().chunks(4).map(|word| {
        let mut bytes = [0; 4];
        bytes[..word.len()].copy_from_slice(word);
        u32::from_le_bytes(bytes)
    }));
}

#[repr(C, align(16))]
#[derive(Clone, Default)]
struct Args {
    r: Scalar,
    coeff: [Scalar; 7],
    offset: [u32; 12],
    old_len: u32,
    len: u32,
    width: u32,
    stride: u32,
    rows: u32,
    cols: u32,
    kind: u32,
    flags: u32,
    bits: u32,
    size: u32,
    stage: u32,
    stored: u32,
    axis_stride: u32,
}

#[derive(Clone)]
struct Axis {
    values: Vec<F>,
    row: bool,
}

impl Axis {
    fn new(row: bool, values: Vec<F>) -> Self {
        Self { values, row }
    }

    fn equality(point: &[F], row_bits: usize) -> [Self; 2] {
        let (row, col) = point.split_at(row_bits);
        [
            Self::new(true, weights(row)),
            Self::new(false, weights(col)),
        ]
    }

    fn bind(&mut self, r: F, row: bool) {
        if self.row == row && self.values.len() > 1 {
            let half = self.values.len() / 2;
            let (low, high) = self.values.split_at_mut(half);
            for (x, y) in low.iter_mut().zip(high) {
                *x += r * (*y - *x);
            }
            self.values.truncate(half);
        }
    }
}

pub(crate) struct Round {
    job: NonNull<c_void>,
    args: Args,
    axes: Vec<Vec<Axis>>,
    columns: Vec<usize>,
    partial: Vec<Scalar>,
    pub(crate) challenges: Vec<F>,
}

impl Drop for Round {
    fn drop(&mut self) {
        unsafe { atlas_metal_drop(self.job.as_ptr()) }
    }
}

impl Round {
    fn new(
        mut args: Args,
        axes: Vec<Vec<Axis>>,
        columns: Vec<usize>,
        input: &[u32],
        tiles: &[Tile],
        table: &[Scalar],
    ) -> Option<Self> {
        let cells = args.rows.next_power_of_two().checked_mul(args.stride)?;
        let device = device()?;
        args.len = cells;
        args.old_len = cells;
        args.width = args.stride;
        args.stored = (columns.len() / axes.len()) as u32;
        let axis_bytes = axes
            .iter()
            .flatten()
            .map(|axis| axis.values.len())
            .sum::<usize>()
            * 16;
        let job = unsafe {
            atlas_metal_job(
                device,
                cells as usize,
                args.stored,
                axes.len() as u32,
                args.kind,
                input.as_ptr(),
                size_of_val(input),
                tiles.as_ptr(),
                size_of_val(tiles),
                table.as_ptr(),
                size_of_val(table),
                axis_bytes,
            )
        };
        Some(Self {
            job: NonNull::new(job)?,
            args,
            axes,
            columns,
            challenges: Vec::new(),
            partial: Vec::new(),
        })
    }

    pub(crate) fn step(&mut self, previous: Option<F>, degree: usize) -> Option<Vec<F>> {
        let next = self.args.len >> u32::from(previous.is_some());
        if next < 1024 {
            return None;
        }
        let mut args = self.args.clone();
        let mut axes = self.axes.clone();
        args.old_len = args.len;
        args.len = next;
        if let Some(r) = previous {
            args.r = encode(r);
            let row = args.old_len > args.width;
            for axis in axes.iter_mut().flatten() {
                axis.bind(r, row);
            }
            if !row {
                args.width /= 2;
            }
        }
        let mut flat = Vec::new();
        for batch in &axes {
            let start = flat.len();
            for (offset, axis) in args.offset.iter_mut().zip(batch) {
                *offset = (flat.len() - start) as u32;
                flat.extend(axis.values.iter().copied().map(encode));
            }
            args.axis_stride = (flat.len() - start) as u32;
        }
        let count = args.len / 2;
        let threads = if args.len >= 1 << 20 { 64 } else { 128 };
        self.partial.resize(
            count.div_ceil(threads) as usize * degree * axes.len(),
            [0; 16],
        );
        let success = unsafe {
            atlas_metal_round(
                self.job.as_ptr(),
                &args,
                size_of::<Args>(),
                flat.as_ptr(),
                size_of_val(flat.as_slice()),
                count,
                threads,
                self.partial.as_mut_ptr(),
                size_of_val(self.partial.as_slice()),
            )
        };
        if !success {
            return None;
        }
        self.args = args;
        self.args.stage += 1;
        self.axes = axes;
        self.challenges.extend(previous);
        let mut sums = vec![F::zero(); degree + 1];
        for share in self.partial.chunks_exact(degree) {
            let mut values = sums.iter_mut();
            *values.next().unwrap() += decode(share[0]);
            for (sum, &part) in values.skip(1).zip(&share[1..]) {
                *sum += decode(part);
            }
        }
        Some(sums)
    }

    pub(crate) fn snapshot(&self) -> Vec<(usize, Vec<F>)> {
        if self.challenges.is_empty() {
            return Vec::new();
        }
        let len = self.args.len as usize;
        let mut buffer = vec![[0; 16]; len * self.columns.len()];
        unsafe {
            atlas_metal_snapshot(
                self.job.as_ptr(),
                buffer.as_mut_ptr(),
                size_of_val(buffer.as_slice()),
            )
        };
        self.columns
            .iter()
            .copied()
            .zip(
                buffer
                    .chunks_exact(len)
                    .map(|values| values.iter().copied().map(decode).collect()),
            )
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn softmax<'w>(
    n: usize,
    stats: &[[i64; 3]],
    addresses: &[std::sync::Arc<Codes>],
    sizes: impl Iterator<Item = usize>,
    score: impl Fn(usize) -> &'w [i32],
    overflow: &[Option<Vec<i64>>],
    output: &[i32],
    sums: &[i32],
    limit: i32,
    radix: u128,
    claim: &[F],
    check: &[F],
    row: &[F],
    mix: F,
) -> Option<Round> {
    if !(256..=2048).contains(&n) || !n.is_power_of_two() || device().is_none() {
        return None;
    }
    let mut input = Vec::new();
    let mut tiles = Vec::new();
    for (i, size) in sizes.enumerate() {
        let base = input.len() as u32;
        input.extend(score(i).iter().map(|&v| v as u32));
        let address = input.len() as u32;
        codes(&mut input, &addresses[i]);
        let extra = if let Some(extra) = &overflow[i] {
            let start = input.len() as u32;
            input.extend(extra.iter().flat_map(|&v| [v as u32, (v >> 32) as u32]));
            start
        } else {
            u32::MAX
        };
        tiles.push([size as u32, base, address, extra]);
    }
    let table: Vec<_> = output
        .iter()
        .zip(sums)
        .enumerate()
        .flat_map(|(i, (&prob, &v))| {
            let previous = if i == 0 {
                i128::from(limit) + 1
            } else {
                i128::from(sums[i - 1])
            };
            [
                encode(F::from_i32(prob)),
                encode(F::from_i128(i128::from(v) - radix as i128 * previous)),
            ]
        })
        .collect();
    let mut axes: Vec<_> = (0..3)
        .map(|j| Axis::new(true, stats.iter().map(|s| F::from_i64(s[j])).collect()))
        .collect();
    let mut first = vec![F::zero(); n];
    first[0] = F::one();
    axes.push(Axis::new(false, first));
    axes.extend(Axis::equality(claim, n.ilog2() as usize));
    axes.extend(Axis::equality(check, n.ilog2() as usize));
    axes.push(Axis::new(true, row.to_vec()));
    let zero = F::zero();
    let coeff = [
        mix,
        F::from_u64((sums.len() - 1) as u64),
        F::from_u128(radix),
        F::from_i32(limit),
        zero,
        zero,
        zero,
    ];
    Round::new(
        Args {
            coeff: coeff.map(encode),
            rows: n as u32,
            stride: n as u32,
            bits: sums.len().ilog2(),
            size: sums.len() as u32,
            ..Args::default()
        },
        vec![axes],
        vec![0, 1, 2, 3, 4, 8],
        &input,
        &tiles,
        &table,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mlp(
    shape: Shape,
    addresses: &Codes,
    x: &[i32],
    up: Option<&[i32]>,
    correction: &[i32],
    symmetric: bool,
    lower: i32,
    claim: &[F],
    check: &[F],
    coefficients: &[F],
) -> Option<Round> {
    if !(1 << 17..=1 << 22).contains(&(1 << shape.bits())) || device().is_none() {
        return None;
    }
    let mut input: Vec<_> = x
        .iter()
        .chain(up.into_iter().flatten())
        .map(|&x| x as u32)
        .collect();
    codes(&mut input, addresses);
    let table: Vec<_> = correction.iter().map(|&v| encode(F::from_i32(v))).collect();
    let active = |len: usize| {
        (0..len.next_power_of_two())
            .map(|i| F::from_bool(i < len))
            .collect()
    };
    let mut axes = vec![
        Axis::new(true, active(shape.rows)),
        Axis::new(false, active(shape.width)),
    ];
    axes.extend(Axis::equality(claim, shape.row_bits()));
    axes.extend(Axis::equality(check, shape.row_bits()));
    let [a, b, c, d] = coefficients.try_into().ok()?;
    let last = F::from_u64((correction.len() - 1) as u64);
    let coeff = [
        encode(a),
        encode(b),
        encode(c),
        encode(d),
        encode(last),
        (lower as u128).to_le_bytes(),
        encode(F::from_i32(lower)),
    ];
    let mut columns: Vec<_> = (0..7).collect();
    if up.is_some() {
        columns.push(10);
    }
    Round::new(
        Args {
            coeff,
            rows: shape.rows as u32,
            cols: shape.width as u32,
            stride: shape.width.next_power_of_two() as u32,
            kind: 1,
            flags: u32::from(up.is_some()) | (u32::from(symmetric) << 1),
            bits: correction.len().ilog2(),
            size: correction.len() as u32,
            ..Args::default()
        },
        vec![axes],
        columns,
        &input,
        &[[0; 4]],
        &table,
    )
}

pub(crate) fn lookup<'a>(
    cells: usize,
    reads: impl ExactSizeIterator<Item = (&'a Codes, &'a [F], F)>,
    digits: impl Iterator<Item = (usize, usize)>,
    selectors: &[Vec<F>],
) -> Option<Round> {
    let count = reads.len();
    let digit_count = selectors.len();
    if !(15..=22).contains(&cells)
        || digit_count > 8
        || (1usize << cells) * count * digit_count * 12 > 384 * 1024 * 1024
        || device().is_none()
    {
        return None;
    }
    let width = 1 << (cells / 2);
    let mut input = Vec::new();
    let mut axes = Vec::new();
    let mut bits = 0;
    for (addresses, point, mix) in reads {
        codes(&mut input, addresses);
        let [row, col] = Axis::equality(point, cells - cells / 2);
        axes.push(vec![row, col, Axis::new(false, vec![mix])]);
        bits = addresses.width as u32;
    }
    let mut tiles = Vec::new();
    let mut table = Vec::new();
    for ((shift, width), values) in digits.zip(selectors) {
        tiles.push([shift as u32, (1 << width) - 1, table.len() as u32, 0]);
        table.extend(values.iter().copied().map(encode));
    }
    let columns = (0..count)
        .flat_map(|i| (0..digit_count).map(move |j| i * (digit_count + 1) + j))
        .collect();
    Round::new(
        Args {
            rows: 1 << (cells - cells / 2),
            cols: (input.len() / count) as u32,
            stride: width,
            kind: 2,
            bits,
            ..Args::default()
        },
        axes,
        columns,
        &input,
        &tiles,
        &table,
    )
}
