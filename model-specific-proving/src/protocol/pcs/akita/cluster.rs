//! TCP workers for the position-wise first fold of one Akita opening.

use std::{
    collections::BTreeSet,
    io::{self, BufReader, BufWriter, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    ops::Range,
    sync::Arc,
    time::Duration,
};

use akita_pcs::custom_source::CommitInnerPlan;
use akita_types::{AkitaExpandedSetup, RingVec};
use bincode::Options;
use jolt_akita::{
    no_selected_row, AkitaError, AkitaField, TraceOneHotComputeKernel, TraceOneHotFoldPlan,
    TraceOneHotPackingPlan, TraceOneHotRows,
};
use jolt_field::{Ring, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{GroupFold, GroupQueries, Indices, PackedFold, ADDRESS_BITS, F};

const K: usize = super::K;
const TIMEOUT: Duration = Duration::from_secs(900);
const TILE_RINGS: usize = 256;

#[derive(Debug)]
pub(super) struct Workers(Vec<SocketAddr>);

impl Workers {
    pub(super) fn new(addrs: Vec<SocketAddr>) -> crate::protocol::Result<Self> {
        if addrs.is_empty() || addrs.iter().collect::<BTreeSet<_>>().len() != addrs.len() {
            return Err("Akita workers must be nonempty and distinct");
        }
        Ok(Self(addrs))
    }
}

#[derive(Serialize, Deserialize)]
struct Fold {
    degree: usize,
    columns: usize,
    positions: usize,
    prefix: usize,
    chunks: Vec<Range<usize>>,
    challenges: Vec<Vec<(u32, i8)>>,
}

#[derive(Serialize, Deserialize)]
struct Task {
    range: Range<usize>,
    fold: Fold,
}

impl Fold {
    fn per_ring(&self) -> usize {
        self.degree / K
    }

    fn blocks(&self) -> usize {
        self.chunks.last().map_or(0, |chunk| chunk.end)
    }

    fn tiles<'a>(
        &'a self,
        range: &'a Range<usize>,
    ) -> impl Iterator<Item = (usize, Range<usize>)> + 'a {
        (0..self.blocks()).flat_map(move |block| {
            let base = block * self.positions;
            let end = range
                .end
                .min(self.prefix.div_ceil(self.per_ring()).saturating_sub(base));
            (range.start..end)
                .step_by(TILE_RINGS)
                .map(move |start| (block, start..(start + TILE_RINGS).min(end)))
        })
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new().with_limit(64 * 1024 * 1024)
}

fn wire_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

fn prepare(stream: &TcpStream) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))
}

fn size(factors: &[usize]) -> io::Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |product, &n| product.checked_mul(n))
        .ok_or_else(|| invalid("Akita task size overflow"))
}

fn zeros<T: Default + Clone>(len: usize) -> io::Result<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(wire_error)?;
    data.resize(len, T::default());
    Ok(data)
}

fn send_rows(
    out: &mut impl Write,
    rows: &dyn TraceOneHotRows,
    fold: &Fold,
    range: &Range<usize>,
) -> io::Result<()> {
    let per_ring = fold.per_ring();
    let mut indices = zeros::<u8>(TILE_RINGS * per_ring * fold.columns)?;
    let mut masks = zeros::<u64>(TILE_RINGS * per_ring)?;
    for (block, tile) in fold.tiles(range) {
        let first = (block * fold.positions + tile.start) * per_ring;
        let len = tile.len() * per_ring;
        let populated = (fold.prefix - first).min(len);
        let bytes = &mut indices[..len * fold.columns];
        let masks = &mut masks[..populated];
        rows.fill_rows_with_masks(first, &mut bytes[..populated * fold.columns], masks);
        for (row, &mask) in bytes.chunks_exact_mut(fold.columns).zip(masks.iter()) {
            for (column, value) in row.iter_mut().enumerate() {
                *value = if *value != no_selected_row() || mask & (1 << column) != 0 {
                    if usize::from(*value) >= K {
                        return Err(invalid("Akita one-hot index out of range"));
                    }
                    *value + 1
                } else {
                    0
                };
            }
        }
        bytes[populated * fold.columns..].fill(0);
        out.write_all(bytes)?;
    }
    Ok(())
}

fn dispatch<T: Send, U: Send>(
    jobs: impl IntoIterator<Item = T>,
    run: impl Fn(T) -> io::Result<U> + Sync,
) -> io::Result<Vec<U>> {
    std::thread::scope(|scope| {
        let run = &run;
        let jobs: Vec<_> = jobs
            .into_iter()
            .map(|job| scope.spawn(move || run(job)))
            .collect();
        jobs.into_iter()
            .map(|job| {
                job.join()
                    .map_err(|_| io::Error::other("Akita dispatch panicked"))?
            })
            .collect()
    })
}

fn request(
    addr: SocketAddr,
    rows: &dyn TraceOneHotRows,
    fold: &Fold,
    range: Range<usize>,
    output: &mut [i32],
) -> io::Result<()> {
    let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(10))?;
    prepare(&stream)?;
    let mut writer = BufWriter::new(&stream);
    codec()
        .serialize_into(&mut writer, &(1u8, &range, fold))
        .map_err(wire_error)?;
    send_rows(&mut writer, rows, fold, &range)?;
    writer.flush()?;
    let mut reader = BufReader::new(&stream);
    let mut bytes = [0u8; 4];
    for coefficient in output {
        reader.read_exact(&mut bytes)?;
        *coefficient = i32::from_le_bytes(bytes);
    }
    Ok(())
}

impl TraceOneHotComputeKernel for Workers {
    fn bind_source(
        self: Arc<Self>,
        _: Arc<dyn TraceOneHotRows>,
    ) -> Result<Arc<dyn TraceOneHotComputeKernel>, AkitaError> {
        Ok(self)
    }

    fn commit(
        &self,
        _: &AkitaExpandedSetup<AkitaField>,
        _: &dyn TraceOneHotRows,
        _: usize,
        _: CommitInnerPlan,
    ) -> Result<Option<Vec<RingVec<AkitaField>>>, AkitaError> {
        Ok(None)
    }

    fn coefficient_packing(
        &self,
        rows: &dyn TraceOneHotRows,
        k: usize,
        plan: TraceOneHotPackingPlan<'_>,
    ) -> Result<Option<Vec<Vec<F>>>, AkitaError> {
        let positions = plan.num_positions();
        let degree = plan.ring_dimension();
        let blocks = plan.blocks_per_column();
        if k != K
            || degree < K
            || positions == 0
            || blocks == 0
            || blocks.checked_mul(positions) != Some(plan.segment_rings())
        {
            return Ok(None);
        }
        let fold = Fold {
            degree,
            positions,
            columns: rows.num_columns(),
            prefix: rows.nonzero_prefix_len(),
            chunks: vec![0..blocks],
            challenges: Vec::new(),
        };
        let subring = plan.subring_dimension();
        let width = size(&[blocks, subring]).map_err(worker_error)?;
        let count = size(&[fold.columns, width]).map_err(worker_error)?;
        let chunk = positions.div_ceil(self.0.len());
        let jobs = self.0.iter().take(positions.div_ceil(chunk)).enumerate();
        let mut partials = dispatch(jobs, |(i, &address)| {
            let range = i * chunk..((i + 1) * chunk).min(positions);
            let stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))?;
            prepare(&stream)?;
            let mut out = BufWriter::new(&stream);
            send(&mut out, &(3u8, &range, &fold))?;
            send(
                &mut out,
                &(subring, plan.position_weights(), plan.packing_weights()),
            )?;
            send_rows(&mut out, rows, &fold, &range)?;
            out.flush()?;
            let mut input = BufReader::new(&stream);
            (0..count)
                .map(|_| receive::<F>(&mut input))
                .collect::<io::Result<Vec<_>>>()
        })
        .map_err(worker_error)?
        .into_iter();
        let mut parts = partials
            .next()
            .ok_or_else(|| AkitaError::InvalidInput("empty Akita packing work".into()))?;
        for partial in partials {
            for (value, next) in parts.iter_mut().zip(partial) {
                *value += next;
            }
        }
        Ok(Some(parts.chunks_exact(width).map(<[F]>::to_vec).collect()))
    }

    fn decompose_fold(
        &self,
        rows: &dyn TraceOneHotRows,
        k: usize,
        plan: TraceOneHotFoldPlan<'_>,
    ) -> Result<Option<Vec<i32>>, AkitaError> {
        let positions = plan.num_positions();
        let degree = plan.ring_dimension();
        let blocks = plan.blocks_per_column();
        if k != K
            || degree < K
            || blocks == 0
            || positions == 0
            || plan.chunk_ranges().is_empty()
            || blocks.checked_mul(positions) != Some(plan.segment_rings())
        {
            return Ok(None);
        }
        let fold = Fold {
            degree,
            columns: rows.num_columns(),
            positions,
            prefix: rows.nonzero_prefix_len(),
            chunks: plan.chunk_ranges().to_vec(),
            challenges: plan
                .challenges()
                .iter()
                .map(|challenge| {
                    challenge
                        .positions
                        .iter()
                        .copied()
                        .zip(challenge.coeffs.iter().copied())
                        .collect()
                })
                .collect(),
        };
        let width = size(&[degree, fold.chunks.len()]).map_err(worker_error)?;
        let mut output =
            zeros(size(&[positions, width]).map_err(worker_error)?).map_err(worker_error)?;
        let positions_per_worker = positions.div_ceil(self.0.len());
        let jobs = output
            .chunks_mut(positions_per_worker * width)
            .zip(&self.0)
            .enumerate();
        dispatch(jobs, |(i, (coefficients, &addr))| {
            let start = i * positions_per_worker;
            request(
                addr,
                rows,
                &fold,
                start..start + coefficients.len() / width,
                coefficients,
            )
            .map_err(|error| io::Error::other(format!("{addr}: {error}")))
        })
        .map_err(worker_error)?;
        Ok(Some(output))
    }
}

fn worker_error(error: io::Error) -> AkitaError {
    AkitaError::InvalidInput(format!("Akita cluster: {error}"))
}

fn execute(task: Task, input: &mut impl Read) -> io::Result<Vec<i32>> {
    let Task { range, fold } = task;
    if !(K..=2048).contains(&fold.degree)
        || !fold.degree.is_power_of_two()
        || !(1..=64).contains(&fold.columns)
        || range.is_empty()
        || range.end > fold.positions
        || fold.blocks() == 0
        || fold.prefix > size(&[fold.blocks(), fold.positions, fold.per_ring()])?
        || fold.challenges.len() != size(&[fold.blocks(), fold.columns])?
        || fold
            .challenges
            .iter()
            .flatten()
            .any(|&(shift, _)| shift as usize >= fold.degree)
    {
        return Err(invalid("invalid Akita fold geometry"));
    }
    let mut previous = 0;
    for chunk in &fold.chunks {
        if chunk.start != previous || chunk.is_empty() {
            return Err(invalid("invalid Akita fold chunk"));
        }
        previous = chunk.end;
    }
    let stride = fold.per_ring() * fold.columns;
    let width = size(&[fold.chunks.len(), fold.degree])?;
    let mut output = zeros(size(&[range.len(), width])?)?;
    let mut selectors = zeros::<u8>(TILE_RINGS * stride)?;
    for (block, tile) in fold.tiles(&range) {
        let count = tile.len();
        let selectors = &mut selectors[..count * stride];
        input.read_exact(selectors)?;
        if selectors.iter().any(|&value| usize::from(value) > K) {
            return Err(invalid("invalid Akita selector"));
        }
        let chunk = fold.chunks.partition_point(|r| r.end <= block);
        let start = (tile.start - range.start) * width;
        output[start..][..count * width]
            .par_chunks_exact_mut(width)
            .zip(selectors.par_chunks_exact(stride))
            .for_each(|(coefficients, selectors)| {
                let coefficients = &mut coefficients[chunk * fold.degree..][..fold.degree];
                for (row, selectors) in selectors.chunks_exact(fold.columns).enumerate() {
                    for (col, &code) in selectors.iter().enumerate() {
                        if code == 0 {
                            continue;
                        }
                        let monomial = row * K + usize::from(code - 1);
                        let challenge = &fold.challenges[col * fold.blocks() + block];
                        for &(shift, digit) in challenge {
                            let index = monomial + shift as usize;
                            let digit = i32::from(digit);
                            if index < fold.degree {
                                coefficients[index] += digit;
                            } else {
                                coefficients[index - fold.degree] -= digit;
                            }
                        }
                    }
                }
            });
    }
    Ok(output)
}

fn serve_packing(task: Task, reader: &mut impl Read, writer: &mut impl Write) -> io::Result<()> {
    let Task { range, fold } = task;
    let (subring, position, packing): (usize, Vec<F>, Vec<F>) = receive(reader)?;
    if !(K..=2048).contains(&fold.degree)
        || !fold.degree.is_power_of_two()
        || !(1..=64).contains(&fold.columns)
        || range.is_empty()
        || range.end > fold.positions
        || fold.blocks() == 0
        || subring == 0
        || position.len() != fold.positions
        || size(&[packing.len(), subring])? != fold.degree
        || fold.prefix > size(&[fold.blocks(), fold.positions, fold.per_ring()])?
    {
        return Err(invalid("invalid Akita packing geometry"));
    }
    let stride = fold.per_ring() * fold.columns;
    let mut output = zeros::<F>(size(&[fold.columns, fold.blocks(), subring])?)?;
    let mut selectors = zeros::<u8>(TILE_RINGS * stride)?;
    let mut weights = vec![F::zero(); packing.len()];
    for (block, tile) in fold.tiles(&range) {
        let selectors = &mut selectors[..tile.len() * stride];
        reader.read_exact(selectors)?;
        for (i, values) in selectors.chunks_exact(stride).enumerate() {
            for (weight, &packing) in weights.iter_mut().zip(&packing) {
                *weight = position[tile.start + i] * packing;
            }
            for (row, columns) in values.chunks_exact(fold.columns).enumerate() {
                for (column, &code) in columns.iter().enumerate() {
                    if code == 0 {
                        continue;
                    }
                    if usize::from(code) > K {
                        return Err(invalid("invalid Akita selector"));
                    }
                    let coefficient = row * K + usize::from(code - 1);
                    let offset = (column * fold.blocks() + block) * subring;
                    output[offset + coefficient / packing.len()] +=
                        weights[coefficient % packing.len()];
                }
            }
        }
    }
    for value in output {
        send(writer, &value)?;
    }
    writer.flush()
}

fn send<T: Serialize>(out: &mut impl Write, value: &T) -> io::Result<()> {
    codec().serialize_into(out, value).map_err(wire_error)
}

fn receive<T: DeserializeOwned>(input: &mut impl Read) -> io::Result<T> {
    codec().deserialize_from(input).map_err(wire_error)
}

#[derive(Serialize, Deserialize)]
struct SumJob<R, B, M> {
    rounds: usize,
    sparse_rounds: usize,
    reads: R,
    check: B,
    mix: M,
}

pub(super) struct RemoteFold {
    rounds: usize,
    nodes: Vec<(BufReader<TcpStream>, usize)>,
    pub(super) evaluations: Vec<F>,
}

impl Workers {
    pub(super) fn sumcheck(
        &self,
        rounds: usize,
        polys: &[Indices],
        queries: &GroupQueries,
        sparse_rounds: usize,
    ) -> io::Result<RemoteFold> {
        let chunk = polys.len().div_ceil(self.0.len());
        let jobs = polys.chunks(chunk).zip(&self.0).enumerate();
        let nodes = dispatch(jobs, |(i, (polys, &addr))| {
            let start = i * chunk;
            let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(10))?;
            prepare(&stream)?;
            let mut out = BufWriter::new(&stream);
            send(&mut out, &2u8)?;
            send(
                &mut out,
                &SumJob {
                    rounds,
                    sparse_rounds,
                    reads: &queries.reads[start..][..polys.len()],
                    check: &queries.bit_check,
                    mix: &queries.bit_mix[start..][..polys.len()],
                },
            )?;
            for poly in polys {
                out.write_all(&poly.wire_bytes())?;
            }
            out.flush()?;
            drop(out);
            Ok((BufReader::new(stream), polys.len()))
        })?;
        Ok(RemoteFold {
            rounds,
            nodes,
            evaluations: Vec::new(),
        })
    }
}

#[derive(Debug)]
pub(super) struct RemoteFoldError(io::Error);

impl From<SumcheckError<F>> for RemoteFoldError {
    fn from(error: SumcheckError<F>) -> Self {
        Self(wire_error(error))
    }
}

impl std::fmt::Display for RemoteFoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl ProveRounds<F, RemoteFoldError> for RemoteFold {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        previous: Option<F>,
        _: usize,
        _: F,
    ) -> Result<UnivariatePoly<F>, RemoteFoldError> {
        let parts = dispatch(&mut self.nodes, |(node, _)| {
            send(node.get_mut(), &previous)?;
            receive::<[F; 4]>(node)
        })
        .map_err(RemoteFoldError)?;
        let samples = parts
            .into_iter()
            .fold([F::zero(); 4], |a, b| std::array::from_fn(|i| a[i] + b[i]));
        Ok(UnivariatePoly::from_evals(&samples))
    }

    fn finish_rounds(&mut self, last: F) -> Result<(), RemoteFoldError> {
        let values = dispatch(&mut self.nodes, |(node, count)| {
            send(node.get_mut(), &last)?;
            let values: Vec<F> = receive(node)?;
            if values.len() != *count {
                return Err(invalid("invalid Akita worker evaluation count"));
            }
            Ok(values)
        })
        .map_err(RemoteFoldError)?;
        self.evaluations = values.into_iter().flatten().collect();
        Ok(())
    }
}

fn serve_sumcheck(
    reader: &mut BufReader<&TcpStream>,
    out: &mut BufWriter<&TcpStream>,
) -> io::Result<()> {
    type Job = SumJob<Vec<Vec<super::Read>>, Vec<F>, Vec<F>>;
    let job: Job = receive(reader)?;
    if job.rounds <= ADDRESS_BITS
        || job.rounds >= usize::BITS as usize
        || !(1..=super::packed::SPARSE_ROUNDS).contains(&job.sparse_rounds)
        || job.check.len() != job.rounds
        || job.mix.is_empty()
        || job.reads.len() != job.mix.len()
    {
        return Err(invalid("invalid Akita Sumcheck job"));
    }
    let row_vars = job.rounds - ADDRESS_BITS;
    let len = 1 << row_vars;
    let mut polys = Vec::with_capacity(job.mix.len());
    for reads in &job.reads {
        for read in reads {
            if read.rows.len() > row_vars {
                return Err(invalid("invalid Akita Sumcheck read"));
            }
            let width = 1 << read.rows.len();
            if read.start >= len || read.start % width != 0 {
                return Err(invalid("invalid Akita Sumcheck read offset"));
            }
        }
        let mut poly = Indices::new(len);
        reader.read_exact(&mut poly.bytes)?;
        polys.push(poly);
    }
    let mut fold = GroupFold {
        rounds: job.rounds,
        packed: polys
            .iter()
            .zip(&job.reads)
            .zip(&job.mix)
            .map(|((poly, reads), &mix)| {
                PackedFold::new(poly, reads, &job.check, mix).with_sparse_rounds(job.sparse_rounds)
            })
            .collect(),
    };
    for round in 0..job.rounds {
        let previous: Option<F> = receive(reader)?;
        if previous.is_some() != (round != 0) {
            return Err(invalid("invalid Akita Sumcheck round"));
        }
        let message = fold
            .prove_round(previous, round, F::zero())
            .map_err(wire_error)?;
        send(
            out,
            &std::array::from_fn::<_, 4, _>(|i| message.evaluate(F::from_u64(i as u64))),
        )?;
        out.flush()?;
    }
    fold.finish_rounds(receive(reader)?).map_err(wire_error)?;
    let values: Vec<_> = fold.packed.iter().map(|poly| poly.value(0)).collect();
    send(out, &values)?;
    out.flush()
}

fn serve_connection(stream: TcpStream) -> io::Result<u8> {
    prepare(&stream)?;
    let mut reader = BufReader::new(&stream);
    let mut writer = BufWriter::new(&stream);
    let operation = receive::<u8>(&mut reader)?;
    match operation {
        1 => {
            let task = receive(&mut reader)?;
            let output = execute(task, &mut reader)?;
            for coefficient in output {
                writer.write_all(&coefficient.to_le_bytes())?;
            }
            writer.flush()
        }
        2 => serve_sumcheck(&mut reader, &mut writer),
        3 => serve_packing(receive(&mut reader)?, &mut reader, &mut writer),
        _ => Err(invalid("unknown Akita worker request")),
    }?;
    Ok(operation)
}

/// Serve Akita proof requests on a private prover network. The worker receives
/// witness data in plaintext over this connection.
pub fn serve_worker(listener: TcpListener) -> io::Result<()> {
    for stream in listener.incoming() {
        if let Err(error) = serve_connection(stream?) {
            eprintln!("Akita worker: {error}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::pcs::akita::{opening_statement, AkitaAdviceSetup, Indices, Key, F};
    use jolt_akita::{AkitaChunkProfile, AkitaNativeBatching, AkitaScheduleArtifacts};
    use jolt_field::Ring;
    use jolt_openings::BatchOpeningScheme;
    use jolt_poly::MultilinearPoly;
    use jolt_transcript::{prover_transcript, verifier_transcript, Keccak};

    #[test]
    fn distributed_fold_preserves_the_native_opening() {
        let listeners = (0..2)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect::<Vec<_>>();
        let addresses = listeners
            .iter()
            .map(|listener| listener.local_addr().unwrap())
            .collect();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let workers = listeners
            .into_iter()
            .map(|listener| {
                listener.set_nonblocking(true).unwrap();
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    let mut operations = 0;
                    while !done.load(std::sync::atomic::Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => operations |= 1 << serve_connection(stream).unwrap(),
                            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                                std::thread::sleep(Duration::from_millis(1));
                            }
                            Err(error) => panic!("{error}"),
                        }
                    }
                    operations
                })
            })
            .collect::<Vec<_>>();

        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let cpu = AkitaAdviceSetup::new(artifacts.clone()).unwrap();
        let cluster = AkitaAdviceSetup::new(artifacts)
            .unwrap()
            .with_workers(addresses)
            .unwrap();
        let key = Key {
            rounds: 19,
            count: 2,
        };
        let indices: Arc<[_]> = (0..key.count)
            .map(|column| {
                let mut indices = Indices::new(1 << (key.rounds - K.ilog2() as usize));
                for i in 0..indices.len() - 3 {
                    if i % 7 != 3 {
                        indices.insert(i, ((i + i / 13 + column) % K) as u8);
                    }
                }
                indices
            })
            .collect::<Vec<_>>()
            .into();
        let queries = GroupQueries {
            bit_check: vec![F::from_u64(3); key.rounds],
            bit_mix: vec![F::from_u64(2); key.count],
            reads: (0..key.count).map(|_| Vec::new()).collect(),
            sum: F::zero(),
        };
        let mut local = GroupFold {
            rounds: key.rounds,
            packed: indices
                .iter()
                .zip(&queries.reads)
                .zip(&queries.bit_mix)
                .map(|((poly, reads), &mix)| PackedFold::new(poly, reads, &queries.bit_check, mix))
                .collect(),
        };
        let mut remote = cluster
            .workers
            .as_ref()
            .unwrap()
            .sumcheck(key.rounds, &indices, &queries, 1)
            .unwrap();
        let mut previous = None;
        for round in 0..key.rounds {
            let a = local.prove_round(previous, round, F::zero()).unwrap();
            let b = remote.prove_round(previous, round, F::zero()).unwrap();
            assert_eq!(a, b);
            previous = Some(F::from_u64((round + 7) as u64));
        }
        local.finish_rounds(previous.unwrap()).unwrap();
        remote.finish_rounds(previous.unwrap()).unwrap();
        assert_eq!(
            remote.evaluations,
            local.packed.iter().map(|p| p.value(0)).collect::<Vec<_>>()
        );
        drop(remote);
        let point = vec![F::from_u64(3); key.rounds];
        let values = indices
            .iter()
            .map(|indices| indices.evaluate(&point))
            .collect::<Vec<_>>();
        let (local_commitment, _) = cpu.commit_group(key, &indices).unwrap();
        let (commitment, hint) = cluster.commit_group(key, &indices).unwrap();
        assert_eq!(commitment, local_commitment);
        let statement = opening_statement(&commitment, &point, &values);
        let mut prover = prover_transcript::<Keccak>(&[1; 32], b"");
        let proof = AkitaNativeBatching::prove_batch(
            &cluster.setup(key).unwrap(),
            statement.clone(),
            indices
                .iter()
                .map(|p| p as &dyn MultilinearPoly<F>)
                .collect(),
            hint,
            &mut prover,
        )
        .unwrap();
        let mut verifier = verifier_transcript::<Keccak>(&[1; 32], b"", prover.narg_string());
        AkitaNativeBatching::verify_batch(
            &cpu.verification_setup(key).unwrap(),
            &statement,
            &proof,
            &mut verifier,
        )
        .unwrap();
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        for worker in workers {
            let operations = worker.join().unwrap();
            assert_eq!(operations, 0b1110);
        }
    }
}
