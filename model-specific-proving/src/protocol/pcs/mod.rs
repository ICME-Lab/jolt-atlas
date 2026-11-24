//! Flat slots for range bits and fixed-table addresses.

use jolt_field::{CanonicalEncoding, Ring, Zero};
use jolt_poly::OneHotPolynomial;
use jolt_transcript::{
    prover_transcript, verifier_transcript, Keccak, ProverState, Transcript, VerifierState,
};

use super::{
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

    fn place(&self, row_vars: usize, compact: bool) -> Result<(usize, Vec<usize>)> {
        let mut offsets = Vec::with_capacity(self.slots.len());
        let mut end = 0usize;
        for &slot in &self.slots {
            let axes = match slot.kind {
                Kind::Bits { planes } if compact => planes,
                _ => 1 << (slot.vars() - slot.cell_vars),
            };
            let cells = 1usize << slot.cell_vars.max(row_vars);
            let span = cells.checked_mul(axes).ok_or("advice layout overflow")?;
            offsets.push(end);
            end = end.checked_add(span).ok_or("advice layout overflow")?;
        }
        let size = end
            .checked_next_power_of_two()
            .ok_or("advice layout overflow")?;
        Ok((size.ilog2() as usize, offsets))
    }

    pub(crate) fn remove_last(&mut self) -> Result<()> {
        self.slots.pop().map(|_| ()).ok_or("empty advice layout")
    }

    pub fn num_vars(&self) -> Result<usize> {
        Ok(self.place(0, false)?.0)
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
    Bits(Vec<u8>),
    Lookup(OneHotPolynomial),
}

pub struct Advice(Vec<Option<Part>>);

impl Advice {
    pub(crate) fn put(&mut self, slot: Slot, part: Part) -> Result<()> {
        if self.0[slot.id].is_some() {
            return Err("duplicate advice");
        }
        self.0[slot.id] = Some(part);
        Ok(())
    }

    pub(crate) fn bits(&self, slot: Slot) -> &[u8] {
        match self.0[slot.id].as_ref().expect("declared advice") {
            Part::Bits(bits) => bits,
            Part::Lookup(_) => unreachable!("expected bit planes"),
        }
    }

    pub(crate) fn polynomial(&self, slot: Slot) -> &OneHotPolynomial {
        match self.0[slot.id].as_ref().expect("declared advice") {
            Part::Lookup(poly) => poly,
            Part::Bits(_) => unreachable!("expected lookup"),
        }
    }
}

pub struct Opening<F> {
    pub(crate) slot: usize,
    pub(crate) claim: Claim<F>,
    pub(crate) planes: Option<Vec<F>>,
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
        advice: &Advice,
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
    commitments: C::AdviceCommitments,
    messages: Messages<C::Field>,
    opening: C::AdviceOpening,
    transcript: Vec<u8>,
}

impl<C: ModelPcs> Proof<C> {
    pub(crate) fn prove<W>(
        domain: &[u8; 32],
        layout: &Layout,
        witness: W,
        advice: impl FnOnce(&W) -> &Advice,
        plan: &C::AdvicePlan,
        setup: &C::AdviceProverSetup,
        public: impl FnOnce(&mut ProverState<Keccak>),
        reverse: impl FnOnce(W, &mut Session<'_, '_, C::Field>) -> Result<()>,
    ) -> Result<Self> {
        let (commitments, state) = <C as ModelPcs>::commit(layout, advice(&witness), plan, setup)?;
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
        <C as ModelPcs>::verify(
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
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    t.append_bytes(&bytes);
}

#[cfg(feature = "akita")]
mod akita;
#[cfg(feature = "akita")]
pub use akita::AkitaAdviceSetup;
