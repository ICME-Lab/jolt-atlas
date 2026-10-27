//! The prover's opening provider over [`AccOpeningAccessor`]; the accessor and the verifier's
//! provider live in `jolt-atlas-verifier`.

pub use jolt_atlas_verifier::utils::opening_access::*;

use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
};
use atlas_sumcheck::opening_proof::{prover::ProverOpeningAccumulator, OpeningId, PolynomialId};
use atlas_transcript::Transcript;

/// The prover's conversion of an accessor into a [`ProverAccOpeningProvider`].
pub trait IntoProverProvider<'a, F: JoltField> {
    /// Convert this accessor into a prover provider that can append openings
    /// to the accumulator using `transcript` and `opening_point`.
    fn into_provider<T: Transcript>(
        self,
        transcript: &'a mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> ProverAccOpeningProvider<'a, F, T>;
}

impl<'a, F: JoltField> IntoProverProvider<'a, F>
    for AccOpeningAccessor<'a, F, ProverOpeningAccumulator<F>>
{
    fn into_provider<T: Transcript>(
        self,
        transcript: &'a mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> ProverAccOpeningProvider<'a, F, T> {
        ProverAccOpeningProvider::from_accessor(self, transcript, opening_point)
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
            pub fn get_nodeio(&self, target: Target) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
                self.accessor.get_nodeio(target)
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

    /// Append a node I/O opening with its claimed value to the prover accumulator at the provider's current opening point.
    pub fn append_nodeio(&mut self, target: Target, claim: F) {
        self.append_nodeio_at(target, self.opening_point.clone(), claim);
    }

    /// Append a node I/O opening with its claimed value to the prover accumulator at an explicit opening point.
    pub fn append_nodeio_at(
        &mut self,
        target: Target,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        let opening_id = self.accessor.builder().nodeio(target);
        self.accessor
            .acc_mut()
            .append_virtual(self.transcript, opening_id, opening_point, claim);
    }

    /// Append an advice opening with its claimed value to the prover accumulator at the provider's current opening point.
    pub fn append_advice<Poly>(&mut self, poly_ctor: impl FnOnce(usize) -> Poly, claim: F)
    where
        Poly: Into<PolynomialId>,
    {
        self.append_advice_at(poly_ctor, self.opening_point.clone(), claim);
    }

    /// Append an advice opening with its claimed value to the prover accumulator at an explicit opening point.
    pub fn append_advice_at<Poly>(
        &mut self,
        poly_ctor: impl FnOnce(usize) -> Poly,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) where
        Poly: Into<PolynomialId>,
    {
        let opening_id = self.accessor.builder().advice(poly_ctor);
        match opening_id.polynomial {
            PolynomialId::Virtual(_) => self.accessor.acc_mut().append_virtual(
                self.transcript,
                opening_id,
                opening_point,
                claim,
            ),
            // TODO(AntoineF4C5): split committed advice handling between dense and one-hot
            // encodings. append_dense is correct for the current dense-only callsites.
            PolynomialId::Committed(_) => self.accessor.acc_mut().append_dense(
                self.transcript,
                opening_id,
                opening_point.r.clone(),
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
