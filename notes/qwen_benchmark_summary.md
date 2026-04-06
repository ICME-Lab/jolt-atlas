# Qwen Benchmark Summary

## End-to-end

| Area | Time | Notes |
|---|---:|---|
| `prove` | `102.94s` | Prover total |
| `verify` | `6.88s` | Verifier total |
| `total` | `117.72s` | End-to-end total |

## Prove Phases

| Prove phase | Time | Notes |
|---|---:|---|
| `iop` | `38.12s` | Largest exclusive prove phase |
| `prove_reduced_openings` | `26.53s` | Opening reduction path |
| `trace` | `19.08s` | Model execution trace during prove |
| `commit_witness_polynomials` | `17.19s` | Witness generation + commit |

## IOP Ops

| IOP op | Time | Notes |
|---|---:|---|
| `SoftmaxAxes` | `22.54s` | Dominant op-level hotspot |
| `Sigmoid` | `4.28s` | Teleport/LUT-heavy |
| `Einsum` | `3.17s` | Includes final large projection |
| `ScalarConstDiv` | `2.15s` | Repeated across graph |
| `Rsqrt` | `1.76s` | LUT/range-check related |

## Reduced Openings

| Reduced openings phase | Time | Notes |
|---|---:|---|
| `prove_batch_opening_sumcheck` | `13.40s` | Largest reduced-opening subphase |
| `build_materialized_rlc` | `7.45s` | RLC polynomial construction |
| `pcs_prove` | `5.34s` | PCS proof generation |

## Batched Sumcheck

| `BatchedSumcheck::prove` substep | Time | Notes |
|---|---:|---|
| `compute_messages` | `20.26s` | Largest sumcheck substep |
| `ingest_challenges` | `14.68s` | Second largest |
| total | `36.91s` | Aggregate batched sumcheck |

## `compute_message` Hotspots

| `compute_message` hotspot | Time | Notes |
|---|---:|---|
| `OpeningProofReductionSumcheckProver::compute_message` | `7.64s` | Opening reduction path |
| `OneHotPolynomialProverOpening::compute_message` | `6.47s` | One-hot opening path |
| `SoftmaxExponentiationSumcheckProver::compute_message` | `3.61s` | Softmax exponentiation |
| `BooleanitySumcheckProver::compute_message` | `3.38s` | Booleanity checks |
| `RaSumcheckProver::compute_message` | `2.43s` | RA lookup relation |
