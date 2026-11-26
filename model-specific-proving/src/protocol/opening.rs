use std::sync::OnceLock;

use jolt_field::Zero;
use jolt_transcript::{
    prover_transcript, verifier_transcript, Keccak, ProverState, Transcript, VerifierState,
};
use world_common::{shape::MIN_LOG_BITS, BitZParams, LinearClaim, Root, SecurityLevel, Shape};
use world_field::{gf128::smallest_generator, F128};
use world_pcs::Pcs;
use world_transcript::{build_prover, build_verifier, Proof as BitProof};

use super::{
    field::{Scalar, F},
    sumcheck::{Column, Messages, Session},
    tensor::{equality, inner, weights, Claim},
    Result,
};

const ROW_BITS: usize = 7;
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
}

#[derive(Clone, Default)]
pub(crate) struct Layout(Vec<Region>);

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
}

pub struct Proof {
    roots: Vec<Root>,
    pub(crate) widths: Vec<usize>,
    messages: Messages,
    openings: Messages,
    bit_proofs: Vec<BitProof>,
    transcript: Vec<u8>,
}

impl Layout {
    pub(crate) fn reserve(&mut self, bits: usize, capacity: usize) -> Slot {
        let slot = Slot(self.0.len());
        self.0.push(Region {
            bits,
            capacity,
            variable: false,
        });
        slot
    }

    pub(crate) fn variable(&mut self, slot: Slot) {
        self.0[slot.0].variable = true;
    }

    pub(crate) fn packed(&self) -> Packed<'_> {
        Packed {
            layout: self,
            widths: self.0.iter().map(|r| r.capacity).collect(),
            blocks: (0..self.0.len()).map(|_| Vec::new()).collect(),
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
    blocks: Vec<Vec<F128>>,
}

impl Packed<'_> {
    pub(crate) fn shrink(&mut self, slot: Slot, width: usize) -> Result<()> {
        let region = self.layout.0[slot.0];
        if !region.variable || width > region.capacity || !self.blocks[slot.0].is_empty() {
            return Err("invalid witness width");
        }
        self.widths[slot.0] = width;
        Ok(())
    }

    pub(crate) fn write(
        &mut self,
        slot: Slot,
        plane: usize,
        width: usize,
        code: impl Fn(usize) -> u128,
    ) -> Result<()> {
        let region = self.layout.0[slot.0];
        if plane + width > self.widths[slot.0] || width >= 128 {
            return Err("invalid witness plane");
        }
        if width == 0 {
            return Ok(());
        }
        let stride = 1 << region.bits.saturating_sub(ROW_BITS);
        let words = &mut self.blocks[slot.0];
        if words.is_empty() {
            *words = F128::zeroed_vec(self.widths[slot.0] * stride);
        }
        let mut rows = [0u128; 128];
        for word in 0..stride {
            rows[..width].fill(0);
            let start = word * ROWS;
            for i in start..(start + ROWS).min(1 << region.bits) {
                let mut bits = code(i);
                if bits >> width != 0 {
                    return Err("integer outside the witness range");
                }
                while bits != 0 {
                    rows[bits.trailing_zeros() as usize] |= 1 << (i % ROWS);
                    bits &= bits - 1;
                }
            }
            for (bit, &row) in rows[..width].iter().enumerate() {
                words[(plane + bit) * stride + word] = F128::from(row);
            }
        }
        Ok(())
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
        packed: Option<&[F128]>,
        io: &mut Session<'_, '_>,
    ) -> Result<LinearClaim<Scalar>> {
        let mix = io.random(requests.len(), requests.len())?;
        let sum = inner(requests.iter().map(|r| r.claim.value), mix.iter().copied());
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
                (offset, weights(column), row_point, r.planes.as_slice(), mix)
            })
            .collect();
        let (end, _) = io.fold(
            sum,
            ROW_BITS,
            2,
            || {
                let packed = packed.expect("prover witness");
                queries
                    .iter()
                    .flat_map(|(offset, cols, row, planes, mix)| {
                        let mut projection = vec![F::zero(); ROWS];
                        let coeffs = planes
                            .iter()
                            .flat_map(|&p| cols.iter().map(move |&c| p * *mix * c));
                        for (&word, weight) in packed[*offset..].iter().zip(coeffs) {
                            let mut bits = word.to_u128();
                            while bits != 0 {
                                projection[bits.trailing_zeros() as usize] += weight;
                                bits &= bits - 1;
                            }
                        }
                        [Column::Dense(projection), Column::Dense(weights(row))]
                    })
                    .collect()
            },
            [],
            |row| row.chunks_exact(2).map(|pair| pair[0] * pair[1]).sum(),
        )?;

        let mut columns = vec![Scalar::from(0u64); params.shape().columns()];
        for (offset, cols, row, planes, mix) in queries {
            let scale = mix * equality(&row, &end.challenges);
            let factors = planes
                .iter()
                .flat_map(|&p| cols.iter().map(move |&c| p * scale * c));
            for (dst, weight) in columns[offset..].iter_mut().zip(factors) {
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
        let mut transcript = prover_transcript::<Keccak>(domain, b"");
        public(&mut transcript);
        bind_widths(&mut transcript, &packed.widths);
        let mut roots = Vec::new();
        let mut commitments = Vec::new();
        let mut by_group = vec![Vec::new(); self.groups.len()];
        for (placement, block) in self.slots.iter().zip(packed.blocks) {
            if placement.planes != 0 {
                by_group[placement.group].push((placement.offset, block));
            }
        }
        for ((_, pcs), chunks) in self.groups.iter().zip(by_group) {
            let mut block = F128::zeroed_vec(pcs.packed_len());
            for (offset, words) in chunks {
                block[offset..offset + words.len()].copy_from_slice(&words);
            }
            let mut channel = build_prover(DOMAIN, b"" as &[u8]);
            let (root, data) = pcs
                .commit(&block, &mut channel)
                .map_err(|_| "could not commit witness")?;
            transcript.append_bytes(&root.0);
            roots.push(root);
            commitments.push((data, block, channel));
        }
        let mut messages = Messages::default();
        let requests = check(&mut Session::Prover(&mut transcript, &mut messages))?;
        let mut openings = Messages::default();
        let mut io = Session::Prover(&mut transcript, &mut openings);
        let mut bit_proofs = Vec::new();
        for (((params, pcs), (data, block, mut channel)), queries) in self
            .groups
            .iter()
            .zip(commitments)
            .zip(self.requests(requests)?)
        {
            let claim = self.reduce(&queries, params, Some(&block), &mut io)?;
            world_prover::BitZProver::new(*params, 8)
                .prove(&claim, pcs, &data, block, &mut channel)
                .map_err(|_| "could not prove witness evaluation")?;
            bit_proofs.push(channel.finish());
        }
        Ok(Proof {
            roots,
            widths: packed.widths,
            messages,
            openings,
            bit_proofs,
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
        if proof.roots.len() != self.groups.len() || proof.bit_proofs.len() != self.groups.len() {
            return Err("invalid proof group count");
        }
        let mut transcript = verifier_transcript::<Keccak>(domain, b"", &proof.transcript);
        public(&mut transcript);
        bind_widths(&mut transcript, &proof.widths);
        for root in &proof.roots {
            transcript.append_bytes(&root.0);
        }
        let mut io = proof.messages.verifier(&mut transcript);
        let requests = self.requests(check(&mut io)?)?;
        io.finish()?;
        let mut io = proof.openings.verifier(&mut transcript);
        for ((((params, pcs), &root), bits), requests) in self
            .groups
            .iter()
            .zip(&proof.roots)
            .zip(&proof.bit_proofs)
            .zip(requests)
        {
            let mut channel = build_verifier(DOMAIN, b"" as &[u8], bits);
            let commitment = pcs
                .receive_commitment(root, &mut channel)
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
