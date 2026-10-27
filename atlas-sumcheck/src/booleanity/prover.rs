//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::upstream_sumcheck::round_check_failed;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        ra_poly::RaPolynomial,
        split_eq_poly::SplitEqUnreducedFold,
    },
    utils::{expanding_table::ExpandingTable, thread::drop_in_background_thread},
};
use common::parallel::par_enabled;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::{fmt::Debug, iter::zip, sync::Arc};

// Specialization of BooleanitySumcheckProver for cases where log_K << 64
// In those cases, we don't require d > 1 since the RA poly is not so large.
// We then use a larger type (u16) for storing indexes,
// since we store the whole lookup index in a single value.
pub type SmallBooleanitySumcheckProver<F> = BooleanitySumcheckProver<F, u16>;

/// Unified Booleanity Sumcheck implementation for RAM, Bytecode, and Instruction lookups
#[derive(Allocative)]
pub struct BooleanitySumcheckProver<
    F: JoltField,
    I: Into<usize> + Copy + Default + Send + Sync + 'static = u8,
> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    #[allocative(skip)]
    B: GruenSplitEqPolynomial<F>,
    /// D: split-eq over time/cycle variables (phase 2, LowToHigh).
    #[allocative(skip)]
    D: GruenSplitEqPolynomial<F>,
    /// G as in the Twist and Shout paper
    G: Vec<Vec<F>>,
    /// H as in the Twist and Shout paper
    H: Vec<RaPolynomial<I, F>>,
    /// F: Expanding table
    F: ExpandingTable<F>,
    /// eq_r_r
    eq_r_r: F,
    /// Indices for H polynomials
    H_indices: Vec<Vec<Option<I>>>,
    /// Linear term, phase 1: `G` per chunk as a polynomial over the address.
    lin_g: Vec<MultilinearPolynomial<F>>,
    /// Linear term, phase 1: the address `k` as a polynomial.
    lin_id: MultilinearPolynomial<F>,
    /// Linear term, phase 2: per-chunk coefficients `γ'_d + w_d·k(r_address')`.
    lin_c: Vec<F>,
    #[allocative(skip)]
    pub(super) params: BooleanitySumcheckParams<F>,
    #[allocative(skip)]
    relation: BooleanityRelation,
    #[allocative(skip)]
    setup: RelationSetup<F>,
}

impl<F: JoltField, I: Into<usize> + Copy + Default + Debug + Send + Sync + 'static>
    RelationWitness<F> for BooleanitySumcheckProver<F, I>
{
    type Relation = BooleanityRelation;

    fn relation(&self) -> &BooleanityRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.H.iter().map(|H| H.final_claim()).collect()
    }
}

crate::relation_witness_instance!(
    [F: JoltField, I: Into<usize> + Copy + Default + Debug + Send + Sync + 'static]
    BooleanitySumcheckProver<F, I>
);

impl<F: JoltField, I: Into<usize> + Copy + Default + Debug + Send + Sync + 'static> ProveRounds<F>
    for BooleanitySumcheckProver<F, I>
{
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::prove_round", fields(variant = ?self.params.sumcheck_id))]
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        if round < self.params.log_k_chunk {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round, previous_claim)
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}

impl<F: JoltField, I: Into<usize> + Copy + Default + Send + Sync + 'static>
    BooleanitySumcheckProver<F, I>
{
    pub fn gen(
        params: BooleanitySumcheckParams<F>,
        G: Vec<Vec<F>>,
        H_indices: Vec<Vec<Option<I>>>,
    ) -> Self {
        let B = GruenSplitEqPolynomial::new(&params.r_address, jolt_poly::BindingOrder::LowToHigh);
        let D_poly =
            GruenSplitEqPolynomial::new(&params.r_cycle, jolt_poly::BindingOrder::LowToHigh);

        let k_chunk = 1 << params.log_k_chunk;
        let mut F = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F.reset(F::one());

        let (lin_g, lin_id) = if params.linear.is_some() {
            (
                G.iter()
                    .map(|g| MultilinearPolynomial::from(g.clone()))
                    .collect(),
                MultilinearPolynomial::from((0..k_chunk as u64).collect::<Vec<u64>>()),
            )
        } else {
            (Vec::new(), MultilinearPolynomial::from(vec![F::zero(); 2]))
        };

        let (relation, setup) = params.clone().into_parts();
        Self {
            B,
            D: D_poly,
            G,
            H_indices,
            H: vec![],
            F,
            eq_r_r: F::zero(),
            lin_g,
            lin_id,
            lin_c: Vec::new(),
            relation,
            setup,
            params,
        }
    }

    /// Phase-1 evaluations of the linear term at `x = 0, 1, 2, 3`.
    fn linear_phase1_evals(&self) -> [F; 4] {
        let Some(lin) = &self.params.linear else {
            return [F::zero(); 4];
        };
        let half = self.lin_id.len() / 2;
        let mut evals = [F::zero(); 4];
        for g in 0..half {
            let k0 = self.lin_id.get_bound_coeff(2 * g);
            let kd = self.lin_id.get_bound_coeff(2 * g + 1) - k0;
            for ((ra, gamma), weight) in self.lin_g.iter().zip(&lin.gammas).zip(&lin.weights) {
                let r0 = ra.get_bound_coeff(2 * g);
                let rd = ra.get_bound_coeff(2 * g + 1) - r0;
                let (mut k, mut r) = (k0, r0);
                for e in evals.iter_mut() {
                    *e += r * (*gamma + *weight * k);
                    k += kd;
                    r += rd;
                }
            }
        }
        evals
    }

    fn compute_phase1_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let m = round + 1;
        let B = &self.B;

        // Compute quadratic coefficients via generic split-eq fold (handles both E_in cases).
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = B
            .par_fold_out_in_unreduced::<{ DEGREE_BOUND - 1 }>(&|k_prime| {
                let coeffs = (0..self.params.d)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|i| {
                        let G_i = &self.G[i];
                        let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .with_min_len(par_enabled())
                            .enumerate()
                            .map(|(k, &G_k)| {
                                let k_m = k >> (m - 1);
                                let F_k = self.F[k % (1 << (m - 1))];
                                let G_times_F = G_k * F_k;

                                // For c in {0, infty}:
                                // G[k] * (F(..., c)^2 - F(..., c))
                                let eval_infty = G_times_F * F_k;
                                let eval_0 = if k_m == 0 {
                                    eval_infty - G_times_F
                                } else {
                                    F::zero()
                                };
                                [eval_0, eval_infty]
                            })
                            .fold_with(
                                [F::UnreducedMulU64::zero(); DEGREE_BOUND - 1],
                                |running, new| {
                                    [
                                        running[0] + new[0].to_unreduced(),
                                        running[1] + new[1].to_unreduced(),
                                    ]
                                },
                            )
                            .reduce(
                                || [F::UnreducedMulU64::zero(); DEGREE_BOUND - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        [
                            self.params.gammas[i] * F::reduce_mul_u64(inner_sum[0]),
                            self.params.gammas[i] * F::reduce_mul_u64(inner_sum[1]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    );
                coeffs
            });

        // The Booleanity quadratic at 1: the new address bit is 1, so only `k_m = 1` survives.
        let at_one = |k_prime: usize| -> F {
            (0..self.params.d)
                .map(|i| {
                    let sum: F = self.G[i][k_prime << m..(k_prime + 1) << m]
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| k >> (m - 1) == 1)
                        .map(|(k, &G_k)| {
                            let F_k = self.F[k % (1 << (m - 1))];
                            G_k * F_k * (F_k - F::one())
                        })
                        .sum();
                    self.params.gammas[i] * sum
                })
                .sum()
        };

        // Use Gruen optimization to get cubic evaluations from quadratic coefficients
        if self.params.linear.is_none() {
            return B.gruen_round_deg_3_from(
                quadratic_coeffs[0],
                quadratic_coeffs[1],
                &at_one,
                previous_claim,
            );
        }
        let lin = self.linear_phase1_evals();
        let booleanity = B.gruen_round_deg_3_from(
            quadratic_coeffs[0],
            quadratic_coeffs[1],
            &at_one,
            previous_claim - lin[0] - lin[1],
        )?;
        Ok(&booleanity + &UnivariatePoly::from_evals(&lin))
    }

    fn compute_phase2_message(
        &self,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let D_poly = &self.D;

        // Compute quadratic coefficients via generic split-eq fold (handles both E_in cases).
        // `[c, e, l]`: constant and quadratic coefficients of the Booleanity
        // quadratic, and the constant term of the linear check.
        let lin_c = &self.lin_c;
        let coeffs: [F; 3] = D_poly.par_fold_out_in_unreduced::<3>(&|j_prime| {
            // Accumulate in unreduced form to minimize per-term reductions
            let mut acc_c = F::UnreducedProductAccum::zero();
            let mut acc_e = F::UnreducedProductAccum::zero();
            let mut acc_l = F::zero();
            for (i, (h, gamma)) in zip(&self.H, &self.params.gammas).enumerate() {
                let h_0 = h.get_bound_coeff(2 * j_prime);
                let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                let b = h_1 - h_0;

                // Compute gamma * h0, then a single unreduced multiply by (h0 - 1)
                let g_h0 = *gamma * h_0;
                let h0_minus_one = h_0 - F::one();
                let c_unr = g_h0.mul_to_product_accum(h0_minus_one);
                acc_c += c_unr;

                // Compute gamma * b, then a single unreduced multiply by b
                let g_b = *gamma * b;
                let e_unr = g_b.mul_to_product_accum(b);
                acc_e += e_unr;

                if let Some(c) = lin_c.get(i) {
                    acc_l += *c * h_0;
                }
            }
            [
                F::reduce_product_accum(acc_c),
                F::reduce_product_accum(acc_e),
                acc_l,
            ]
        });

        // s(X) = D(X)·[eq_r_r·q_bool(X) + q_lin(X)], q_lin linear.
        D_poly.gruen_round_deg_3_from(
            self.eq_r_r * coeffs[0] + coeffs[2],
            self.eq_r_r * coeffs[1],
            &|j_prime| {
                zip(&self.H, &self.params.gammas)
                    .enumerate()
                    .map(|(i, (h, gamma))| {
                        let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                        let lin = lin_c.get(i).map_or(F::zero(), |c| *c * h_1);
                        self.eq_r_r * *gamma * h_1 * (h_1 - F::one()) + lin
                    })
                    .sum()
            },
            previous_claim,
        )
    }
}

impl<F: JoltField, I: Into<usize> + Copy + Default + Debug + Send + Sync + 'static>
    BooleanitySumcheckProver<F, I>
{
    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::bind", fields(variant = ?self.params.sumcheck_id))]
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        if round < self.params.log_k_chunk {
            // Phase 1: Bind B and update F
            self.B.bind(r);

            // Update F for this round
            self.F.update(r_j);

            if let Some(lin) = &self.params.linear {
                self.lin_id.bind_parallel(r_j, BindingOrder::LowToHigh);
                self.lin_g
                    .iter_mut()
                    .for_each(|g| g.bind_parallel(r_j, BindingOrder::LowToHigh));
                if round == self.params.log_k_chunk - 1 {
                    let k = self.lin_id.final_claim();
                    self.lin_c = zip(&lin.gammas, &lin.weights)
                        .map(|(g, w)| *g + *w * k)
                        .collect();
                    self.lin_g = Vec::new();
                }
            }

            // If transitioning to phase 2, prepare H polynomials
            if round == self.params.log_k_chunk - 1 {
                self.eq_r_r = self.B.current_scalar();

                // Initialize H polynomials using RaPolynomial
                let F = std::mem::take(&mut self.F);
                let H_indices = std::mem::take(&mut self.H_indices);
                self.H = H_indices
                    .into_iter()
                    .map(|indices| RaPolynomial::new(Arc::new(indices), F.clone_values()))
                    .collect();

                // Drop G arrays as they're no longer needed
                let g = std::mem::take(&mut self.G);
                drop_in_background_thread(g);
            }
        } else {
            // Phase 2: Bind D and H
            self.D.bind(r);
            self.H
                .par_iter_mut()
                .with_min_len(par_enabled())
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }
}
