//! Flat slots for range bits and fixed-table addresses.

use std::sync::Arc;

use jolt_field::{CanonicalEncoding, Ring, Zero};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Keccak, ProverState, Transcript, VerifierState,
};

use super::{
    lookup::Addresses,
    session::{Claim, Messages, ProofField, Session},
    Result,
};

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Bits { planes: usize },
    Lookup { address_vars: usize },
}

#[derive(Clone, Copy)]
pub(crate) struct Slot {
    pub(crate) id: usize,
    pub(crate) cell_vars: usize,
    pub(crate) kind: Kind,
}

impl Slot {
    pub(crate) fn vars(self) -> usize {
        self.cell_vars
            + match self.kind {
                Kind::Bits { planes } => planes.next_power_of_two().ilog2() as usize,
                Kind::Lookup { address_vars } => address_vars,
            }
    }

    pub(crate) fn digit_vars(self) -> usize {
        (self.vars() - self.cell_vars).min(2)
    }

    pub(crate) fn digit_count(self, planes: usize) -> usize {
        planes.div_ceil(1 << self.digit_vars()).max(1)
    }
}

#[derive(Clone, Default)]
pub struct Layout {
    slots: Vec<Slot>,
    pub(crate) integer_bits: u32,
}

impl Layout {
    pub(crate) fn add(&mut self, cell_vars: usize, kind: Kind) -> Result<Slot> {
        let slot = Slot {
            id: self.slots.len(),
            cell_vars,
            kind,
        };
        if slot.vars() >= usize::BITS as usize {
            return Err("advice exceeds address space");
        }
        self.slots.push(slot);
        Ok(slot)
    }

    pub(crate) fn replace_bits(&mut self, slot: Slot) {
        self.slots[slot.id] = slot;
    }

    pub(crate) fn remove_last(&mut self) -> Result<()> {
        self.slots.pop().map(|_| ()).ok_or("empty advice layout")
    }

    pub(crate) fn witness(&self) -> Advice {
        Advice(self.slots.iter().map(|_| None).collect())
    }

    fn validate<F>(&self, requests: &[Opening<F>]) -> Result<()> {
        let mut opened = vec![false; self.slots.len()];
        for request in requests {
            let slot = self.slots.get(request.slot).ok_or("invalid advice slot")?;
            let arity = match &request.planes {
                None => slot.vars(),
                Some(planes)
                    if matches!(slot.kind, Kind::Bits { .. })
                        && planes.len() == 1 << (slot.vars() - slot.cell_vars) =>
                {
                    slot.cell_vars
                }
                _ => return Err("invalid bit plane weights"),
            };
            if request.claim.point.len() != arity {
                return Err("invalid advice evaluation arity");
            }
            opened[slot.id] = true;
        }
        if opened.contains(&false) {
            return Err("unopened advice slot");
        }
        Ok(())
    }
}

pub(crate) enum Part {
    Digits(Arc<Vec<u8>>),
    Lookup {
        addresses: Arc<Addresses>,
        shift: usize,
    },
}

pub struct Advice(Vec<Option<Part>>);

impl Advice {
    pub(crate) fn extend(&mut self, layout: &Layout) {
        self.0.resize_with(layout.slots.len(), || None);
    }

    pub(crate) fn replace_digits(&mut self, slot: Slot, digits: Arc<Vec<u8>>) {
        self.0[slot.id] = Some(Part::Digits(digits));
    }

    pub(crate) fn range_digits(&self, slot: Slot) -> Arc<Vec<u8>> {
        let Some(Part::Digits(digits)) = &self.0[slot.id] else {
            unreachable!("recorded range")
        };
        Arc::clone(digits)
    }

    pub(crate) fn put(&mut self, slot: Slot, part: Part) -> Result<()> {
        if self.0[slot.id].is_some() {
            return Err("duplicate advice");
        }
        self.0[slot.id] = Some(part);
        Ok(())
    }

    #[cfg(feature = "bitz")]
    fn digits(&mut self, slot: Slot) -> Result<Arc<Vec<u8>>> {
        let Kind::Bits { planes } = slot.kind else {
            return Err("invalid digit slot");
        };
        match self.0[slot.id].take() {
            Some(Part::Digits(digits))
                if digits.len() == slot.digit_count(planes) << slot.cell_vars =>
            {
                Ok(digits)
            }
            _ => Err("invalid digit count"),
        }
    }

    #[cfg(feature = "bitz")]
    fn fill_bits(&mut self, slot: Slot, mut set: impl FnMut(usize)) -> Result<()> {
        match slot.kind {
            Kind::Bits { planes } => {
                let cells = 1 << slot.cell_vars;
                let lanes = 1 << slot.digit_vars();
                let digits = self.digits(slot)?;
                for (digit, rows) in digits.chunks_exact(cells).enumerate() {
                    let first = digit * lanes;
                    let used = planes.saturating_sub(first).min(lanes);
                    for (cell, &code) in rows.iter().enumerate() {
                        if code >= 1 << used {
                            return Err("invalid range digit");
                        }
                        let mut remaining = code;
                        while remaining != 0 {
                            let bit = remaining.trailing_zeros() as usize;
                            set((first + bit) * cells + cell);
                            remaining &= remaining - 1;
                        }
                    }
                }
            }
            Kind::Lookup { .. } => {
                let cells = 1 << slot.cell_vars;
                let (addresses, shift, mask) = self.take_lookup(slot)?;
                for cell in 0..addresses.len() {
                    let digit = addresses.digit(cell, shift, mask);
                    set(usize::from(digit) * cells + cell);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "bitz")]
    fn take_lookup(&mut self, slot: Slot) -> Result<(Arc<Addresses>, usize, u32)> {
        let Kind::Lookup { address_vars } = slot.kind else {
            return Err("not a lookup advice slot");
        };
        match self.0[slot.id].take() {
            Some(Part::Lookup { addresses, shift }) if addresses.len() == 1 << slot.cell_vars => {
                let mask = (1 << address_vars) - 1;
                Ok((addresses, shift, mask))
            }
            _ => Err("invalid lookup advice shape"),
        }
    }
}

pub struct Opening<F> {
    pub(crate) slot: usize,
    pub(crate) claim: Claim<F>,
    pub(crate) planes: Option<Vec<F>>,
}

#[cfg(test)]
#[test]
fn range_digits_preserve_plane_order_and_padding() {
    use crate::protocol::{
        range::{Range, Ranges},
        tensor::Shape,
    };

    let mut layout = Layout::default();
    let ranges = Ranges::new(
        &mut layout,
        Shape { rows: 3, width: 3 },
        [Range::signed(3), Range::unsigned(4)],
    )
    .unwrap();
    let signed = [-4, -3, -2, -1, 0, 1, 2, 3, 0];
    let unsigned = [0, 1, 2, 3, 4, 5, 6, 7, 15];
    let mut advice = layout.witness();
    ranges.record(&mut advice, &[&signed, &unsigned]).unwrap();
    let Some(Part::Digits(digits)) = &advice.0[ranges.slot.id] else {
        panic!("expected range digits");
    };
    let mut expected = [4u8; 16];
    for (i, (&signed, &unsigned)) in signed.iter().zip(&unsigned).enumerate() {
        expected[i / 3 * 4 + i % 3] = (signed + 4 + (unsigned << 3)) as u8;
    }
    assert_eq!(&digits[..16], &expected.map(|v| v & 15));
    assert_eq!(&digits[16..], &expected.map(|v| v >> 4));
}

impl<F> Opening<F> {
    pub(crate) fn new(slot: Slot, claim: Claim<F>) -> Self {
        Self {
            slot: slot.id,
            claim,
            planes: None,
        }
    }
}

/// Commits to bit planes and lookups. Openings must enforce that stored bit planes contain bits.
pub trait ModelPcs {
    type Field: ProofField;
    const MIN_FIELD_BITS: u32 = 128;
    type AdviceProverSetup;
    type AdviceVerifierSetup;
    type AdviceCommitments;
    type AdviceState;
    type AdviceOpening;
    type AdvicePlan;

    #[doc(hidden)]
    fn plan(layout: &Layout) -> Result<Self::AdvicePlan>;

    #[doc(hidden)]
    fn commit(
        layout: &Layout,
        advice: &mut Advice,
        plan: &Self::AdvicePlan,
        setup: &Self::AdviceProverSetup,
    ) -> Result<(Self::AdviceCommitments, Self::AdviceState)>;
    #[doc(hidden)]
    fn bind(commitments: &Self::AdviceCommitments, t: &mut impl Transcript);
    #[doc(hidden)]
    fn open(
        layout: &Layout,
        requests: Vec<Opening<Self::Field>>,
        commitments: &Self::AdviceCommitments,
        state: Self::AdviceState,
        plan: &Self::AdvicePlan,
        setup: &Self::AdviceProverSetup,
        t: &mut ProverState<Keccak>,
    ) -> Result<Self::AdviceOpening>;
    #[doc(hidden)]
    fn verify(
        layout: &Layout,
        requests: Vec<Opening<Self::Field>>,
        commitments: &Self::AdviceCommitments,
        proof: &Self::AdviceOpening,
        plan: &Self::AdvicePlan,
        setup: &Self::AdviceVerifierSetup,
        t: &mut VerifierState<'_, Keccak>,
    ) -> Result<()>;
}

pub struct Proof<C: ModelPcs> {
    pub(crate) clipping: Vec<Vec<crate::layers::softmax::ClipSpec>>,
    commitments: C::AdviceCommitments,
    messages: Messages<C::Field>,
    opening: C::AdviceOpening,
    transcript: Vec<u8>,
}

impl<C: ModelPcs> Proof<C> {
    pub(crate) fn prove<W>(
        domain: &[u8; 32],
        layout: &Layout,
        mut witness: W,
        advice: impl FnOnce(&mut W) -> &mut Advice,
        plan: &C::AdvicePlan,
        setup: &C::AdviceProverSetup,
        public: impl FnOnce(&mut ProverState<Keccak>),
        reverse: impl FnOnce(W, &mut Session<'_, '_, C::Field>) -> Result<()>,
    ) -> Result<Self> {
        let (commitments, state) =
            <C as ModelPcs>::commit(layout, advice(&mut witness), plan, setup)?;
        let mut transcript = prover_transcript::<Keccak>(domain, b"");
        public(&mut transcript);
        C::bind(&commitments, &mut transcript);
        let mut messages = Messages::default();
        let mut io = Session::prove(C::Field::prover(&mut transcript), &mut messages);
        reverse(witness, &mut io)?;
        let requests = io.finish()?;
        layout.validate(&requests)?;
        let opening = <C as ModelPcs>::open(
            layout,
            requests,
            &commitments,
            state,
            plan,
            setup,
            &mut transcript,
        )?;
        Ok(Self {
            clipping: Vec::new(),
            commitments,
            messages,
            opening,
            transcript: transcript.narg_string().to_vec(),
        })
    }

    pub(crate) fn verify(
        &self,
        domain: &[u8; 32],
        layout: &Layout,
        setup: &C::AdviceVerifierSetup,
        public: impl FnOnce(&mut VerifierState<'_, Keccak>),
        reverse: impl FnOnce(&mut Session<'_, '_, C::Field>) -> Result<()>,
    ) -> Result<()> {
        let plan = plan::<C>(layout)?;
        let mut t = verifier_transcript::<Keccak>(domain, b"", &self.transcript);
        public(&mut t);
        C::bind(&self.commitments, &mut t);
        let mut io = self.messages.read(&mut t);
        reverse(&mut io)?;
        let requests = io.finish()?;
        layout.validate(&requests)?;
        C::verify(
            layout,
            requests,
            &self.commitments,
            &self.opening,
            &plan,
            setup,
            &mut t,
        )?;
        t.check_eof().map_err(|_| "unused transcript bytes")
    }
}

pub(crate) fn plan<C: ModelPcs>(layout: &Layout) -> Result<C::AdvicePlan> {
    let bits = layout.integer_bits.max(C::MIN_FIELD_BITS);
    if C::Field::MODULUS_BITS < bits || C::Field::from_u64(2).is_zero() {
        return Err("field is too small for the model's integer relations");
    }
    C::plan(layout)
}

pub(crate) fn bind_i32(t: &mut impl Transcript, values: &[i32]) {
    t.append(&(values.len() as u64));
    let mut bytes = [0; 16 * 1024];
    for values in values.chunks(bytes.len() / 4) {
        for (dst, value) in bytes.chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        t.append_bytes(&bytes[..values.len() * 4]);
    }
}

#[cfg(feature = "bitz")]
mod bitz;
#[cfg(feature = "bitz")]
pub use bitz::BitzScheme;
