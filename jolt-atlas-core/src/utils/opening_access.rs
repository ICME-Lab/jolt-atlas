//! Opening accessor and provider utilities for computation nodes.
//!
//! Provides a node-scoped accessor/provider layer over opening accumulators,
//! with a small internal builder used to derive opening IDs consistently.

use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPoly;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{
        OpeningAccumulator, OpeningId, OpeningPoint, PolynomialId, ProverOpeningAccumulator,
        SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
    },
    transcripts::Transcript,
};

/// Selects the node index source used to build opening identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Use the current node index.
    Current,
    /// Use the node index referenced by the current node input at the given position.
    Input(usize),
}

/// Struct-based opening ID factory scoped to one computation node.
#[derive(Debug, Clone, Copy)]
struct OpeningIdBuilder<'a> {
    node: &'a ComputationNode,
    sumcheck_id: SumcheckId,
}

impl<'a> OpeningIdBuilder<'a> {
    /// Create a builder scoped to one computation node.
    fn new(node: &'a ComputationNode) -> Self {
        Self {
            node,
            sumcheck_id: SumcheckId::NodeExecution(node.idx),
        }
    }

    /// Convenience method for the common case of opening an input/current node output
    /// in this builder's default sumcheck context.
    fn node_io(&self, target: Target) -> OpeningId {
        let node_idx = match target {
            Target::Current => self.node.idx,
            Target::Input(position) => self.node.inputs[position],
        };
        OpeningId::new(VirtualPoly::NodeOutput(node_idx), self.sumcheck_id)
    }

    /// Build an advice opening for the current node in this builder's default
    /// sumcheck context.
    fn advice<Poly>(&self, poly_ctor: impl FnOnce(usize) -> Poly) -> OpeningId
    where
        Poly: Into<PolynomialId>,
    {
        OpeningId::new(poly_ctor(self.node.idx), self.sumcheck_id)
    }
}

// Lightweight wrapper that lets accessors/providers work with either shared
// or mutable accumulator references without duplicating the outer types.
enum Reference<'a, V: ?Sized> {
    Ref(&'a V),
    Mut(&'a mut V),
}

impl<'a, V: ?Sized> From<&'a V> for Reference<'a, V> {
    fn from(reference: &'a V) -> Self {
        Self::Ref(reference)
    }
}

impl<'a, V: ?Sized> From<&'a mut V> for Reference<'a, V> {
    fn from(reference: &'a mut V) -> Self {
        Self::Mut(reference)
    }
}

impl<'a, V: ?Sized> Reference<'a, V> {
    fn as_ref(&self) -> &V {
        match self {
            Self::Ref(reference) => reference,
            Self::Mut(reference) => reference,
        }
    }

    fn as_mut(&mut self) -> Option<&mut V> {
        match self {
            Self::Ref(_) => None,
            Self::Mut(reference) => Some(reference),
        }
    }
}

// Read-only and mutable access layer shared by prover and verifier helpers.
/// Generic helper for recovering existing opening claims.
pub struct AccOpeningAccessor<'a, F: JoltField, Acc: OpeningAccumulator<F> + ?Sized> {
    accumulator: Reference<'a, Acc>,
    builder: OpeningIdBuilder<'a>,
    _field: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField, Acc: OpeningAccumulator<F> + ?Sized> AccOpeningAccessor<'a, F, Acc> {
    /// Build a claim accessor from an accumulator and a computation node.
    #[allow(private_bounds)]
    pub fn new<AccRef>(accumulator: AccRef, computation_node: &'a ComputationNode) -> Self
    where
        AccRef: Into<Reference<'a, Acc>>,
    {
        Self {
            accumulator: accumulator.into(),
            builder: OpeningIdBuilder::new(computation_node),
            _field: core::marker::PhantomData,
        }
    }

    /// Access the underlying accumulator reference held by this claim accessor.
    fn acc(&self) -> &Acc {
        self.accumulator.as_ref()
    }

    /// Access the underlying mutable accumulator reference held by this claim accessor, if any.
    fn acc_mut(&mut self) -> &mut Acc {
        self.accumulator
            .as_mut()
            .expect("Accumulator reference is not mutable")
    }

    /// Read the reduced opening claim for the current node.
    pub fn get_reduced_opening(&self) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        self.acc().get_node_output_opening(self.builder.node.idx)
    }

    /// Read a node I/O opening for the selected target.
    pub fn get_node_io(&self, target: Target) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let opening_id = self.builder.node_io(target);
        self.acc().get_virtual_polynomial_opening(opening_id)
    }

    /// Read an advice opening for the current node.
    pub fn get_advice<Poly>(
        &self,
        poly_ctor: impl FnOnce(usize) -> Poly,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F)
    where
        Poly: Into<PolynomialId>,
    {
        let opening_id = self.builder.advice(poly_ctor);
        self.get_custom(opening_id)
    }

    /// Read an opening claim from an explicit opening identifier.
    ///
    /// This is useful when the caller has already constructed an `OpeningId`
    /// (for example, for virtual polynomials with custom indexing schemes).
    pub fn get_custom(&self, opening_id: impl Into<OpeningId>) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let opening_id = opening_id.into();
        match opening_id.polynomial {
            PolynomialId::Virtual(_) => self.acc().get_virtual_polynomial_opening(opening_id),
            PolynomialId::Committed(_) => self.acc().get_committed_polynomial_opening(opening_id),
        }
    }
}

impl<'a, F: JoltField> AccOpeningAccessor<'a, F, ProverOpeningAccumulator<F>> {
    /// Convert this accessor into a prover provider that can append openings
    /// to the accumulator using `transcript` and `opening_point`.
    pub fn to_provider(
        self,
        transcript: &'a mut impl Transcript,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> ProverAccOpeningProvider<'a, F, impl Transcript> {
        ProverAccOpeningProvider::from_accessor(self, transcript, opening_point)
    }
}

impl<'a, F: JoltField> AccOpeningAccessor<'a, F, VerifierOpeningAccumulator<F>> {
    /// Convert this accessor into a verifier provider that can request openings
    /// from the accumulator using `transcript` and `opening_point`.
    pub fn to_provider(
        self,
        transcript: &'a mut impl Transcript,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> VerifierAccOpeningProvider<'a, F, impl Transcript> {
        VerifierAccOpeningProvider::from_accessor(self, transcript, opening_point)
    }
}

// Shared provider read helpers are simple forwards into the underlying accessor,
// so a small macro keeps the two provider impl blocks focused on append behavior.
macro_rules! impl_accessor_methods {
    ($provider:ident) => {
        impl<'a, F: JoltField, T: Transcript> $provider<'a, F, T> {
            /// Read the reduced opening claim for the current node.
            ///
            /// Returns the post-reduction evaluation from the evaluation-reduction protocol,
            /// rather than the unreduced opening from the current sumcheck.
            pub fn get_reduced_opening(&self) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
                self.accessor.get_reduced_opening()
            }

            /// Read a node I/O opening from the wrapped accumulator.
            pub fn get_node_io(&self, target: Target) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
                self.accessor.get_node_io(target)
            }

            /// Read an advice opening from the wrapped accumulator.
            pub fn get_advice<Poly>(
                &self,
                poly_ctor: impl FnOnce(usize) -> Poly,
            ) -> (OpeningPoint<BIG_ENDIAN, F>, F)
            where
                Poly: Into<PolynomialId>,
            {
                self.accessor.get_advice(poly_ctor)
            }

            /// Read an opening claim from an explicit opening identifier.
            pub fn get_custom<Poly>(&self, opening_id: Poly) -> (OpeningPoint<BIG_ENDIAN, F>, F)
            where
                Poly: Into<OpeningId>,
            {
                self.accessor.get_custom(opening_id)
            }
        }
    };
}

// Prover providers extend the shared accessor with append operations that
// also carry claimed values into the accumulator.
/// Prover-side helper that couples a node-scoped opening builder with a
/// mutable opening accumulator.
pub struct ProverAccOpeningProvider<'a, F: JoltField, T: Transcript> {
    accessor: AccOpeningAccessor<'a, F, ProverOpeningAccumulator<F>>,
    transcript: &'a mut T,
    opening_point: OpeningPoint<BIG_ENDIAN, F>,
}

impl<'a, F: JoltField, T: Transcript> ProverAccOpeningProvider<'a, F, T> {
    fn from_accessor(
        accessor: AccOpeningAccessor<'a, F, ProverOpeningAccumulator<F>>,
        transcript: &'a mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Self {
        Self {
            accessor,
            transcript,
            opening_point,
        }
    }

    /// Update the opening point used when appending openings.
    pub fn update_point(&mut self, opening_point: OpeningPoint<BIG_ENDIAN, F>) {
        self.opening_point = opening_point;
    }

    /// Append a node I/O opening with its claimed value to the prover accumulator.
    pub fn append_node_io(&mut self, target: Target, claim: F) {
        let opening_id = self.accessor.builder.node_io(target);
        self.accessor.acc_mut().append_virtual(
            self.transcript,
            opening_id,
            self.opening_point.clone(),
            claim,
        );
    }

    /// Append an advice opening with its claimed value to the prover accumulator.
    pub fn append_advice<Poly>(&mut self, poly_ctor: impl FnOnce(usize) -> Poly, claim: F)
    where
        Poly: Into<PolynomialId>,
    {
        let opening_id = self.accessor.builder.advice(poly_ctor);
        match opening_id.polynomial {
            PolynomialId::Virtual(_) => self.accessor.acc_mut().append_virtual(
                self.transcript,
                opening_id,
                self.opening_point.clone(),
                claim,
            ),
            // TODO(AntoineF4C5): split committed advice handling between dense and one-hot
            // encodings. append_dense is correct for the current dense-only callsites.
            PolynomialId::Committed(_) => self.accessor.acc_mut().append_dense(
                self.transcript,
                opening_id,
                self.opening_point.r.clone(),
                claim,
            ),
        }
    }

    /// Append an arbitrary opening with its claimed value to the prover accumulator.
    ///
    /// The opening ID can be either virtual or committed, and is appended
    /// at this provider's current opening point.
    pub fn append_custom(&mut self, opening_id: impl Into<OpeningId>, claim: F) {
        let opening_id = opening_id.into();
        match opening_id.polynomial {
            PolynomialId::Virtual(_) => self.accessor.acc_mut().append_virtual(
                self.transcript,
                opening_id,
                self.opening_point.clone(),
                claim,
            ),
            // TODO(AntoineF4C5): consider OneHot/Dense different cases
            PolynomialId::Committed(_) => self.accessor.acc_mut().append_dense(
                self.transcript,
                opening_id,
                self.opening_point.r.clone(),
                claim,
            ),
        }
    }
}

impl_accessor_methods!(ProverAccOpeningProvider);

// Verifier providers use the same accessor pattern, but append only opening
// requests because the claimed evaluations are supplied later by verification.
/// Verifier-side helper that can be created from either shared or mutable
/// accumulator references.
pub struct VerifierAccOpeningProvider<'a, F: JoltField, T: Transcript> {
    accessor: AccOpeningAccessor<'a, F, VerifierOpeningAccumulator<F>>,
    transcript: &'a mut T,
    opening_point: OpeningPoint<BIG_ENDIAN, F>,
}

impl<'a, F: JoltField, T: Transcript> VerifierAccOpeningProvider<'a, F, T> {
    fn from_accessor(
        accessor: AccOpeningAccessor<'a, F, VerifierOpeningAccumulator<F>>,
        transcript: &'a mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Self {
        Self {
            accessor,
            transcript,
            opening_point,
        }
    }

    /// Update the opening point used when appending openings.
    pub fn update_point(&mut self, opening_point: OpeningPoint<BIG_ENDIAN, F>) {
        self.opening_point = opening_point;
    }

    /// Append a node I/O opening request to the verifier accumulator.
    ///
    /// Panics if this provider was created from an immutable reference.
    pub fn append_node_io(&mut self, target: Target) {
        let opening_id = self.accessor.builder.node_io(target);
        self.accessor.acc_mut().append_virtual(
            self.transcript,
            opening_id,
            self.opening_point.clone(),
        );
    }

    /// Append an advice opening request to the verifier accumulator.
    ///
    /// Panics if this provider was created from an immutable reference.
    pub fn append_advice<Poly>(&mut self, poly_ctor: impl FnOnce(usize) -> Poly)
    where
        Poly: Into<PolynomialId>,
    {
        let opening_id = self.accessor.builder.advice(poly_ctor);
        match opening_id.polynomial {
            PolynomialId::Virtual(_) => self.accessor.acc_mut().append_virtual(
                self.transcript,
                opening_id,
                self.opening_point.clone(),
            ),
            // TODO(AntoineF4C5): split committed advice handling between dense and one-hot
            // encodings. append_dense is correct for the current dense-only callsites.
            PolynomialId::Committed(_) => self.accessor.acc_mut().append_dense(
                self.transcript,
                opening_id,
                self.opening_point.r.clone(),
            ),
        }
    }

    /// Append an arbitrary opening request to the verifier accumulator.
    ///
    /// Panics if this provider was created from an immutable reference.
    pub fn append_custom(&mut self, opening_id: impl Into<OpeningId>) {
        let opening_id = opening_id.into();
        match opening_id.polynomial {
            PolynomialId::Virtual(_) => self.accessor.acc_mut().append_virtual(
                self.transcript,
                opening_id,
                self.opening_point.clone(),
            ),
            // TODO(AntoineF4C5): consider OneHot/Dense different cases
            PolynomialId::Committed(_) => self.accessor.acc_mut().append_dense(
                self.transcript,
                opening_id,
                self.opening_point.r.clone(),
            ),
        }
    }
}

impl_accessor_methods!(VerifierAccOpeningProvider);
