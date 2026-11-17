//! Mutation sites over a finished proof and its public IO.
//!
//! A proof is the bytes the verifier's transcript reads, and the sites are taken from the log
//! of those reads ([`Recording`](super::recording::Recording)): one site per value read, so
//! every commitment, claim, round coefficient, reduction coefficient, proof-of-work nonce and
//! the joint opening proof is mutated without this module naming any of them. Every read is
//! absorbed, so a mutated proof is usually rejected because the verifier's transcript diverges
//! rather than because an algebraic check fails. That is still the property we want: *every
//! byte is either absorbed or checked*. A site whose mutation is accepted is a dead byte
//! (proof malleability); a verifier panic is a robustness bug.
//!
//! What the format cannot express has no site: there is no key to remove or to add an orphan
//! under, and no count to raise. A missing value is a truncation, a surplus one an appended
//! byte, and parts of a second honest execution's proof are spliced in at read boundaries.

use std::collections::BTreeMap;

use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::{trace::ModelExecutionIO, Model, Tensor};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::recording::{Read, ReadKind};
use crate::onnx_proof::ONNXProof;

/// One mutation of a proof or its IO.
#[derive(Clone, Debug)]
pub(crate) enum Site {
    /// Flip the low bit of byte `byte` of read `index` (an index into the read log).
    Read {
        index: usize,
        kind: ReadKind,
        byte: usize,
    },
    /// The proof cut at the start of read `index`: that read and every later one are missing.
    Truncated { index: usize },
    /// One byte appended to the proof.
    Extended,
    /// Swap the commitments of polynomials `i` and `j` (reads `i` and `j` of the log).
    CommitmentSwap { i: usize, j: usize },
    /// Replace commitment read `i` with the same read of a second honest execution.
    CommitmentReplaced { i: usize },
    /// The opening reduction (its sumcheck and reduced claims) and the joint opening proof of
    /// a second honest execution: everything from read `from` on.
    ReducedSpliced { from: usize },
    /// `io.outputs[0][elem] += 1`.
    IoOutput { elem: usize },
    /// `io.inputs[tensor][elem] += 1`.
    IoInput { tensor: usize, elem: usize },
    /// `io.input_indices[i] += 1`.
    IoInputIndex { i: usize },
    /// No output tensor.
    IoOutputCount,
    /// The output tensor truncated by one element (a shape the model does not produce).
    IoOutputShape,
    /// The last input tensor dropped.
    IoInputCount,
}

/// A proof with the log of the verifier's reads of it.
pub(crate) struct LoggedProof<'a> {
    pub proof: &'a ONNXProof,
    pub reads: &'a [Read],
}

impl LoggedProof<'_> {
    /// The bytes of read `index`.
    fn bytes(&self, index: usize) -> &[u8] {
        let read = &self.reads[index];
        &self.proof.0[read.offset..read.offset + read.len]
    }

    /// The bytes from the start of read `index` to the end of the proof.
    fn tail(&self, index: usize) -> &[u8] {
        &self.proof.0[self.reads[index].offset..]
    }

    /// The reads cover the proof exactly, in order: every byte is read once.
    pub(crate) fn assert_covers_the_proof(&self) {
        let mut offset = 0;
        for read in self.reads {
            assert_eq!(read.offset, offset, "a gap or an overlap before {read:?}");
            offset += read.len;
        }
        assert_eq!(offset, self.proof.0.len(), "bytes after the last read");
    }

    /// The first read of the opening reduction: walking back from the joint opening proof, the
    /// reduced claims, then the reduction sumcheck's round messages and nonces. `None` for a
    /// proof with no joint opening.
    fn reduction_start(&self) -> Option<usize> {
        let (&last, rest) = self.reads.split_last()?;
        if last.kind != ReadKind::OpeningProof {
            return None;
        }
        let claims = rest
            .iter()
            .rev()
            .take_while(|read| read.kind == ReadKind::Claim)
            .count();
        let rounds = rest[..rest.len() - claims]
            .iter()
            .rev()
            .take_while(|read| matches!(read.kind, ReadKind::RoundCoeff | ReadKind::PowNonce))
            .count();
        Some(rest.len() - claims - rounds)
    }
}

/// Whether to enumerate every site (`JOLT_SOUNDNESS_EXHAUSTIVE=1`).
pub(crate) fn exhaustive() -> bool {
    std::env::var_os("JOLT_SOUNDNESS_EXHAUSTIVE").is_some_and(|v| v != "0")
}

/// Sampling seed (`JOLT_SOUNDNESS_SEED`, default fixed).
pub(crate) fn sampling_seed() -> u64 {
    std::env::var("JOLT_SOUNDNESS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x50FF_D00D)
}

fn variant_name(debug: String) -> String {
    debug
        .split(['(', '{', ' '])
        .next()
        .unwrap_or("")
        .to_string()
}

/// The stratum of an opening id: (polynomial kind, sumcheck kind, op kind
/// of the executing node). Sampled sweeps keep one id per stratum.
pub(crate) fn opening_stratum(id: &OpeningId, model: &Model) -> (String, String, String) {
    let node_kind = match id.sumcheck {
        SumcheckId::NodeExecution(idx) if idx < model.graph().nodes.len() => {
            format!(
                "{:?}",
                super::op_kind::OpKind::of(&model.graph().nodes[idx].op)
            )
        }
        _ => String::new(),
    };
    (
        variant_name(format!("{:?}", id.polynomial)),
        variant_name(format!("{:?}", id.sumcheck)),
        node_kind,
    )
}

/// How many reads of each scalar kind (claim, round coefficient, reduction coefficient,
/// nonce) a sampled sweep mutates.
const SCALAR_SITES_PER_KIND: usize = 64;

/// How many polynomial commitments a sampled sweep mutates.
const COMMITMENT_SITES: usize = 16;

/// `all`, or `cap` of them drawn with `rng`, always with the first and the last.
fn sample<T: Copy>(all: Vec<T>, cap: usize, rng: &mut StdRng) -> Vec<T> {
    if exhaustive() || all.len() <= cap {
        return all;
    }
    let (first, last) = (all[0], all[all.len() - 1]);
    let mut chosen = vec![first, last];
    chosen.extend(all[1..all.len() - 1].choose_multiple(rng, cap - 2).copied());
    chosen
}

/// The bytes of a framed read (an eight-byte length, then the value) whose low bit is flipped:
/// the low byte of the length, the value's first and last bytes, and `inner` of the bytes
/// between (256 when exhaustive).
fn framed_bytes(len: usize, inner: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut bytes = vec![0];
    if len > 8 {
        bytes.extend([8, len - 1]);
        let between: Vec<usize> = (9..len - 1).collect();
        let cap = if exhaustive() { 256 } else { inner };
        bytes.extend(between.choose_multiple(rng, cap.min(between.len())));
    }
    bytes
}

/// Every mutation site of `honest` / `io`. The verifier must reject each one with an `Err`:
/// never accept, never panic. `other` is a second honest proof of the same model on different
/// inputs (splice material); its reads are the same in number and kind.
///
/// A sampled sweep draws its sites from each kind of read with the seed: up to
/// [`SCALAR_SITES_PER_KIND`] claims, round coefficients, reduction coefficients and nonces
/// (always the first and the last of the stream), the batch commitment and the joint opening
/// proof at several bytes each, and up to [`COMMITMENT_SITES`] polynomial commitments. An
/// exhaustive sweep takes every read. Either way every read boundary is a truncation.
pub(crate) fn enumerate_sites(
    honest: &LoggedProof<'_>,
    other: &LoggedProof<'_>,
    io: &ModelExecutionIO,
) -> Vec<Site> {
    let mut rng = StdRng::seed_from_u64(sampling_seed());
    let mut sites = vec![];

    let mut by_kind: BTreeMap<ReadKind, Vec<usize>> = BTreeMap::new();
    for (index, read) in honest.reads.iter().enumerate() {
        by_kind.entry(read.kind).or_default().push(index);
    }
    for (&kind, indices) in &by_kind {
        if kind.is_framed() {
            // The first framed read of its kind is the batch commitment or the joint opening
            // proof; the commitments after the batch commitment are the polynomials', many and
            // short.
            let (first, rest) = indices.split_first().expect("a kind in the log has a read");
            let rest = sample(rest.to_vec(), COMMITMENT_SITES, &mut rng);
            let reads = std::iter::once((*first, 12)).chain(rest.into_iter().map(|i| (i, 2)));
            for (index, inner) in reads {
                for byte in framed_bytes(honest.reads[index].len, inner, &mut rng) {
                    sites.push(Site::Read { index, kind, byte });
                }
            }
        } else {
            for index in sample(indices.clone(), SCALAR_SITES_PER_KIND, &mut rng) {
                sites.push(Site::Read {
                    index,
                    kind,
                    byte: 0,
                });
            }
        }
    }

    for index in 0..honest.reads.len() {
        sites.push(Site::Truncated { index });
    }
    sites.push(Site::Extended);

    // The first commitment read is the batch commitment; the rest are the polynomials'.
    // Swapping or replacing a commitment is a no-op when the two polynomials (or the two
    // executions' versions of one polynomial) have the same commitment.
    let commitments = by_kind.remove(&ReadKind::Commitment).unwrap_or_default();
    let polys = commitments.get(1..).unwrap_or_default();
    let pairs: Vec<(usize, usize)> = polys
        .iter()
        .enumerate()
        .flat_map(|(at, &i)| polys[at + 1..].iter().map(move |&j| (i, j)))
        .filter(|&(i, j)| honest.bytes(i) != honest.bytes(j))
        .collect();
    for (i, j) in sample(pairs, 24, &mut rng) {
        sites.push(Site::CommitmentSwap { i, j });
    }
    let replaceable: Vec<usize> = commitments
        .iter()
        .copied()
        .filter(|&i| honest.bytes(i) != other.bytes(i))
        .collect();
    for i in sample(replaceable, 16, &mut rng) {
        sites.push(Site::CommitmentReplaced { i });
    }
    if let Some(from) = honest.reduction_start() {
        if honest.tail(from) != other.tail(from) {
            sites.push(Site::ReducedSpliced { from });
        }
    }

    let out_len = io.outputs[0].data().len();
    for elem in [0, out_len / 2, out_len - 1] {
        sites.push(Site::IoOutput { elem });
    }
    for (tensor, t) in io.inputs.iter().enumerate() {
        let len = t.data().len();
        for elem in [0, len / 2, len - 1] {
            sites.push(Site::IoInput { tensor, elem });
        }
    }
    for i in 0..io.input_indices.len() {
        sites.push(Site::IoInputIndex { i });
    }
    sites.push(Site::IoOutputCount);
    sites.push(Site::IoOutputShape);
    sites.push(Site::IoInputCount);

    // Dedupe sites that collapsed onto the same element.
    let mut seen = std::collections::HashSet::new();
    sites.retain(|s| seen.insert(format!("{s:?}")));
    sites
}

/// The mutation of `site` applied to `honest` / `io`: the mutated proof.
pub(crate) fn apply_site(
    honest: &LoggedProof<'_>,
    io: &mut ModelExecutionIO,
    other: &LoggedProof<'_>,
    site: &Site,
) -> ONNXProof {
    let mut proof = honest.proof.clone();
    let reads = honest.reads;
    match site {
        Site::Read { index, byte, .. } => proof.0[reads[*index].offset + byte] ^= 1,
        Site::Truncated { index } => proof.0.truncate(reads[*index].offset),
        Site::Extended => proof.0.push(0),
        Site::CommitmentSwap { i, j } => {
            // `i` is read before `j`: the bytes up to `i`, `j`'s, those between, `i`'s, the rest.
            let (first, second) = (&reads[*i], &reads[*j]);
            proof.0 = [
                &honest.proof.0[..first.offset],
                honest.bytes(*j),
                &honest.proof.0[first.offset + first.len..second.offset],
                honest.bytes(*i),
                &honest.proof.0[second.offset + second.len..],
            ]
            .concat();
        }
        Site::CommitmentReplaced { i } => {
            let read = &reads[*i];
            proof.0 = [
                &honest.proof.0[..read.offset],
                other.bytes(*i),
                &honest.proof.0[read.offset + read.len..],
            ]
            .concat();
        }
        Site::ReducedSpliced { from } => {
            proof.0.truncate(reads[*from].offset);
            proof.0.extend_from_slice(other.tail(*from));
        }
        Site::IoOutput { elem } => {
            let d = io.outputs[0].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        Site::IoInput { tensor, elem } => {
            let d = io.inputs[*tensor].data_mut();
            d[*elem] = d[*elem].wrapping_add(1);
        }
        Site::IoInputIndex { i } => io.input_indices[*i] += 1,
        Site::IoOutputCount => io.outputs.clear(),
        Site::IoOutputShape => {
            let out = &io.outputs[0];
            let len = out.data().len() - 1;
            io.outputs[0] = Tensor::from_slice(&out.data()[..len], &[len]).expect("site");
        }
        Site::IoInputCount => {
            io.inputs.pop();
        }
    }
    proof
}

/// A short label for grouping sites in reports.
pub(crate) fn site_class(site: &Site) -> &'static str {
    match site {
        Site::Read { kind, .. } => match kind {
            ReadKind::Commitment => "Read(Commitment)",
            ReadKind::Claim => "Read(Claim)",
            ReadKind::RoundCoeff => "Read(RoundCoeff)",
            ReadKind::ReductionCoeff => "Read(ReductionCoeff)",
            ReadKind::OpeningProof => "Read(OpeningProof)",
            ReadKind::PowNonce => "Read(PowNonce)",
        },
        Site::Truncated { .. } => "Truncated",
        Site::Extended => "Extended",
        Site::CommitmentSwap { .. } => "CommitmentSwap",
        Site::CommitmentReplaced { .. } => "CommitmentReplaced",
        Site::ReducedSpliced { .. } => "ReducedSpliced",
        Site::IoOutput { .. } => "IoOutput",
        Site::IoInput { .. } => "IoInput",
        Site::IoInputIndex { .. } => "IoInputIndex",
        Site::IoOutputCount => "IoOutputCount",
        Site::IoOutputShape => "IoOutputShape",
        Site::IoInputCount => "IoInputCount",
    }
}
