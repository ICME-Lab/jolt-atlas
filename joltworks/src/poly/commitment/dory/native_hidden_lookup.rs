//! Read a committed table through hidden indices. Both the table evaluation
//! and the returned values are openings of the original tensor commitments.
use super::native_lookup::Lookup;
use crate::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        blindfold::{InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource},
        shout::ReadRafProvider,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::{Blake2bTranscript, Transcript},
};
use ark_bn254::Fr;
use ark_std::Zero;
use rayon::prelude::*;

#[derive(Clone)]
pub(super) struct HiddenReadParams {
    lookup: Lookup,
    table: OpeningId,
    r: Vec<Fr>,
    columns: Vec<Fr>,
    gamma: Fr,
    claim: Fr,
}
impl HiddenReadParams {
    pub fn new(
        lookup: Lookup,
        table: OpeningId,
        columns: Vec<Fr>,
        a: &dyn OpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
    ) -> Self {
        let gamma: Fr = t.challenge_scalar();
        let claim = lookup.rv_claim(a) + gamma * lookup.raf_claim(a);
        let r = lookup.r(a).r;
        Self {
            lookup,
            table,
            columns,
            r,
            gamma,
            claim,
        }
    }
    fn ra(&self) -> OpeningId {
        let (poly, stage) = self.lookup.ra_poly();
        OpeningId::new(poly, stage)
    }
}
impl SumcheckInstanceParams<Fr> for HiddenReadParams {
    fn degree(&self) -> usize {
        2
    }
    fn num_rounds(&self) -> usize {
        self.lookup.log_k
    }
    fn input_claim(&self, _: &dyn OpeningAccumulator<Fr>) -> Fr {
        self.claim
    }
    fn normalize_opening_point(&self, r: &[Fr]) -> OpeningPoint<BIG_ENDIAN, Fr> {
        r.to_vec().into()
    }
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::sum_of_products(vec![
            ProductTerm::single(ValueSource::Opening(self.lookup.output)),
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(self.lookup.input)],
            ),
        ])
    }
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<Fr>) -> Vec<Fr> {
        vec![self.gamma]
    }
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::product(vec![
                ValueSource::Opening(self.ra()),
                ValueSource::Opening(self.table),
            ]),
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(self.ra())],
            ),
        ]))
    }
    fn output_constraint_challenge_values(&self, r: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
        vec![self.gamma * IdentityPolynomial::new(self.lookup.log_k).evaluate(r)]
    }
}

pub(super) struct HiddenReadProver {
    params: HiddenReadParams,
    values: MultilinearPolynomial<Fr>,
    weights: MultilinearPolynomial<Fr>,
    identity: IdentityPolynomial<Fr>,
}
impl HiddenReadProver {
    pub fn new(
        params: HiddenReadParams,
        indices: &[usize],
        table: &MultilinearPolynomial<Fr>,
    ) -> Self {
        assert_eq!(
            table.len(),
            1 << (params.lookup.log_k + params.columns.len())
        );
        // Project trailing coordinates without copying or materializing the
        // full table as field elements. Only one value per dictionary row remains.
        let values = if params.columns.is_empty() {
            table.clone()
        } else {
            let eq = EqPolynomial::<Fr>::evals(&params.columns);
            let width = eq.len();
            let projected = (0..1 << params.lookup.log_k)
                .into_par_iter()
                .map(|row| {
                    eq.iter()
                        .enumerate()
                        .map(|(column, weight)| {
                            table.get_scaled_coeff(row * width + column, *weight)
                        })
                        .sum::<Fr>()
                })
                .collect::<Vec<_>>();
            MultilinearPolynomial::from(projected)
        };
        assert_eq!(indices.len(), 1 << params.r.len());
        let eq = EqPolynomial::<Fr>::evals(&params.r);
        let mut weights = vec![Fr::zero(); values.len()];
        for (index, weight) in indices.iter().zip(eq) {
            weights[*index] += weight;
        }
        Self {
            identity: IdentityPolynomial::new(params.lookup.log_k),
            params,
            values,
            weights: MultilinearPolynomial::from(weights),
        }
    }
}
impl SumcheckInstanceProver<Fr, Blake2bTranscript> for HiddenReadProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.params
    }
    fn compute_message(&mut self, _: usize, previous: Fr) -> UniPoly<Fr> {
        let mut evaluations = [Fr::zero(); 2];
        for i in 0..self.values.len() / 2 {
            let v = self.values.sumcheck_evals(i, 2, BindingOrder::HighToLow);
            let w = self.weights.sumcheck_evals(i, 2, BindingOrder::HighToLow);
            let k = self.identity.sumcheck_evals(i, 2, BindingOrder::HighToLow);
            for j in 0..2 {
                evaluations[j] += w[j] * (v[j] + self.params.gamma * k[j]);
            }
        }
        UniPoly::from_evals_and_hint(previous, &evaluations)
    }
    fn ingest_challenge(&mut self, r: <Fr as JoltField>::Challenge, _: usize) {
        self.values.bind_parallel(r, BindingOrder::HighToLow);
        self.weights.bind_parallel(r, BindingOrder::HighToLow);
        self.identity.bind_parallel(r, BindingOrder::HighToLow);
    }
    fn cache_openings(
        &self,
        a: &mut ProverOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        let point: Vec<Fr> = r.to_vec().into_opening();
        a.append_virtual(
            t,
            self.params.ra(),
            [point.as_slice(), self.params.r.as_slice()].concat().into(),
            self.weights.final_claim(),
        );
        a.append_dense(
            t,
            self.params.table,
            [point.as_slice(), self.params.columns.as_slice()].concat(),
            self.values.final_claim(),
        );
    }
}

pub(super) struct HiddenReadVerifier(pub HiddenReadParams);
impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for HiddenReadVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fr> {
        &self.0
    }
    fn expected_output_claim(
        &self,
        a: &VerifierOpeningAccumulator<Fr>,
        r: &[<Fr as JoltField>::Challenge],
    ) -> Fr {
        let ra = a.get_virtual_polynomial_opening(self.0.ra()).1;
        let table = a.get_committed_polynomial_opening(self.0.table).1;
        ra * (table + self.0.gamma * IdentityPolynomial::new(self.0.lookup.log_k).evaluate(r))
    }
    fn cache_openings(
        &self,
        a: &mut VerifierOpeningAccumulator<Fr>,
        t: &mut Blake2bTranscript,
        r: &[<Fr as JoltField>::Challenge],
    ) {
        let point: Vec<Fr> = r.to_vec().into_opening();
        a.append_virtual(
            t,
            self.0.ra(),
            [point.as_slice(), self.0.r.as_slice()].concat().into(),
        );
        a.append_dense(
            t,
            self.0.table,
            [point.as_slice(), self.0.columns.as_slice()].concat(),
        );
    }
}
