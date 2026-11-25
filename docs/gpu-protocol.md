# GPU backend protocol v0: statement, protocol and soundness

**Status:** draft for review, 2026-11-25. Not implemented, not audited, parameters not set.
It fills in the skeleton of `docs/gpu-backend-plan.md` section 4.2 and the hypothesis of its
section 5.6.

## 0. Summary and what review is asked for

The protocol proves that a transformer with committed integer weights maps a public prompt to
a public output. It is two commitments, one batched sumcheck and one opening.

- **The interactive part** (section 3) is sumcheck over the prime field of order
  `p = 2^127 − 1`. It contains nothing new. Its one design choice is that every tensor between
  ops is committed, as bits, so that every relation is local and all of them are proven in
  one batch.
- **The commitment** (section 4) is LaBinius (Osadnik and Seiler, ePrint 2026/2103) with its
  evaluation prime changed from 2 to `p`. LaBinius states its extraction for a general
  evaluation prime that is inert in the ring and builds only the case 2. We use a prime that
  is not inert, and take claims in the prime field, not in the residue ring.

Three things are ours and are what we ask reviewers to check:

1. **Lemma 3**: in `R = Z[X]/(X^162 + X^81 + 1)`, every nonzero element with coefficients of
   absolute value at most 7 is a unit modulo `2^127 − 1`. It replaces LaBinius's requirement
   that the evaluation prime be inert.
2. **Theorem 2**: with Lemma 3 in place of inertness, LaBinius's left expansion and fold are
   a knowledge-sound evaluation proof for multilinear polynomials over `F_p` whose 128
   coefficients per ring element are the committed values.
3. **Theorem 1 and Proposition 1**: the commitment guarantees no range, so every range in the
   statement comes from one Booleanity check, and the statement over `F_p` is the statement
   over the integers once the weights are in range.

Section 7 lists what is assumed and what is open. The largest open cost is proof size: the
opening grows as the square root of what is committed and is 18 MiB for the weights of a 7B
model (section 6).

## 1. Notation

- `p = 2^127 − 1` and `F = F_p`. All sumchecks are over `F`.
- For `r ∈ F^n`, `eq(r) ∈ F^{2^n}` is the vector with entries
  `eq(r, b) = Π_i (r_i b_i + (1 − r_i)(1 − b_i))` for `b ∈ {0,1}^n`, indexed little-endian.
- For `a ∈ F^{2^n}`, `ã(r) = ⟨eq(r), a⟩` is its multilinear extension.
- A *tensor* is an integer array whose index set is a hypercube; other shapes are padded
  with zeros.

## 2. The statement

### 2.1 The model

A model is a fixed straight-line program of ops over integer tensors. Each tensor has a
declared width `w`, a power of two with `w ≤ 64`, and holds integers in
`[−2^{w−1}, 2^{w−1})`. Version 0 has four ops; how a transformer block is written in them is
the numeric contract's concern and is not part of this document.

| Op | Semantics |
|---|---|
| Linear | `acc[t,i] = Σ_k W[i,k]·a[t,k] + b[i]`, then `acc = 2^s·q + r` with `0 ≤ r < 2^s`; the output is `q` |
| Product | the same with a second tensor in place of the weights: `acc[t,u] = Σ_k a[t,k]·c[u,k]` |
| Add | `o = a + c`, entrywise |
| Lookup | `o[i] = T[a[i]]` for a public table `T` of `K = 2^κ` entries |

An execution is valid only if every output lies in its declared width; an overflow has no
valid execution.

### 2.2 What is committed

- **Weights.** The vector `W ∈ Z^{N_W}` of every weight and bias, each matrix on an aligned
  subcube. It is committed once per model, giving `C_W`.
- **Witness.** The vector `Z ∈ {0,1}^{N_Z}`, committed in each proof, giving `C_Z`. It holds,
  each on an aligned subcube:
  - every op's output tensor, as `w` bit planes in offset binary: entry `x` of a tensor `A`
    is `A[x] = Σ_{j<w} 2^j·Z_A[j, x] − 2^{w−1}`;
  - every remainder `r`, as `s` bit planes;
  - for every lookup, a one-hot row `E[i, ·] ∈ {0,1}^K`.
- **Public.** The prompt and the output, the tables, the program. The verifier evaluates
  their extensions itself.

**Lemma 1 (recomposition).** Let `w = 2^ℓ`, `y_i = 2^{2^i} / (1 + 2^{2^i})` for `i < ℓ`, and
`y = (y_0, …, y_{ℓ−1})`. For a tensor `A` stored as above,
`Ã(x) = (2^w − 1)·Z̃_A(y, x) − 2^{w−1}` for every `x ∈ F^n`.

*Proof.* `2^j = Π_i (2^{2^i})^{j_i}` for the bits `j_i` of `j`, and
`(1 + 2^{2^i})·eq(y_i, j_i) = (1 − j_i) + 2^{2^i} j_i`. So
`Σ_j 2^j Z_A[j, x] = Π_i (1 + 2^{2^i}) · Σ_j eq(y, j) Z_A[j, x]`, and
`Π_{i<ℓ} (1 + 2^{2^i}) = 2^{2^ℓ} − 1`. The denominators `1 + 2^{2^i}` for `i ≤ 5` are
nonzero modulo `p`. ∎

So a claim about any tensor at a point is one claim about `Z̃` at a point, with no sumcheck.
A remainder has `s` planes, and `s` need not be a power of two: its planes are split into at
most `log s + 1` blocks of power-of-two size, one claim each.

### 2.3 The relation

`R_model` is the set of `((C_W, x, y); (W, Z))` with `W ∈ F^{N_W}`, `Z ∈ F^{N_Z}` such that:

- (R0) `C_W` opens to `W` (section 4.5);
- (R1) `Z[b]·(Z[b] − 1) = 0` for every `b`;
- (R2) for every Linear op, `Σ_k W[i,k]·a[t,k] + b[i] = 2^s·q[t,i] + r[t,i]` in `F` for all
  `(t, i)`, and likewise for every Product op;
- (R3) for every Add op, `o = a + c` in `F`, entrywise;
- (R4) for every Lookup op and every `i`: `Σ_k E[i,k] = 1`, `Σ_k k·E[i,k] = a[i] + 2^{κ−1}`
  and `Σ_k T[k]·E[i,k] = o[i]` in `F`, where the input `a` has width `κ` and the table is
  indexed by the input's offset value;
- (R5) the first op's input is `x` and the last op's output is `y`,

where every tensor is read from `Z` by the formula of 2.2.

`R_model` constrains `W` only to be a vector over `F`. That is deliberate: the commitment of
section 4 guarantees no range, and a range on the weights is needed once per model, not once
per proof.

**Proposition 1 (integers).** Suppose `((C_W, x, y); (W, Z)) ∈ R_model`, every entry of `W`
is the residue of an integer in `[−2^{w_W − 1}, 2^{w_W − 1})`, and for every Linear and
Product op with `2^k` summands the widths satisfy `k + w_1 + w_2 + 1 ≤ 125` and
`s + w_q + 1 ≤ 125`, with the bias no wider than the sum it is added to. Then every equation of (R2) to (R4) holds over the integers, and `y` is
the output of the integer model on `x`.

*Proof.* By (R1) every tensor read from `Z` is an integer in its declared range. Both sides
of each equation are then integers of absolute value below `2^125 < p/2`, so equality modulo
`p` is equality. For a Linear op, `0 ≤ r < 2^s` makes `(q, r)` the Euclidean quotient and
remainder of `acc`, which are unique. For a Lookup, (R1) and the first equation of (R4) make
`E[i, ·]` a unit vector, the second makes its index the input, and the third reads the
table. Induction over the program gives the claim. ∎

If the weights are not known to be in range, `R_model` is still a well-defined statement
about a model over `F`: the map `(q, r) ↦ 2^s q + r` is injective on the declared ranges, so
each op's output is determined by its inputs whenever one exists.

## 3. The interactive protocol

In this section `W` and `Z` are oracles: the prover is bound to them before the verifier's
first message, and the verifier may ask for `W̃` and `Z̃` at points of its choice. Section 4
replaces the oracles by commitments.

### 3.1 From relations to claims

For each family of equations the verifier draws a point and the equations collapse to one
claim. `τ` denotes a fresh uniform point of the right length.

- **Booleanity (R1).** `Σ_b eq(τ_B, b)·Z(b)·(Z(b) − 1) = 0`, a sumcheck of degree 3 over the
  `n_Z = log N_Z` variables of `Z`.
- **Linear (R2).** With `τ = (τ_t, τ_i)`:
  `Σ_k W̃(τ_i, k)·ã(τ_t, k) = 2^s·q̃(τ) + r̃(τ) − b̃(τ_i)`. The left side is a sumcheck of
  degree 2 over the variables of `k` alone; the right side is three evaluation claims.
  Product is the same with `c̃(τ_u, k)` in place of `W̃(τ_i, k)`.
- **Add (R3).** `õ(τ) = ã(τ) + c̃(τ)`: three evaluation claims and no sumcheck.
- **Lookup (R4).**
  - one-hot: `2^κ·Ẽ(τ, ½, …, ½) = 1`, one evaluation claim;
  - address: `Σ_{j<κ} 2^j·2^{κ−1}·Ẽ(τ, ½, …, 1, …, ½) = ã(τ) + 2^{κ−1}`, with the `1` in
    coordinate `j`: `κ + 1` evaluation claims;
  - read: `Σ_k T(k)·Ẽ(τ, k) = õ(τ)`, a sumcheck of degree 2 over the `κ` variables of `k`.
- **Public tensors (R5)** enter as values the verifier computes.

Every tensor evaluation is rewritten by Lemma 1 as an evaluation of `Z̃`, or of `W̃`.

### 3.2 The protocol

1. **Commit.** The prover sends `C_Z`. (`C_W` is fixed.)
2. **Phase 1, one batched sumcheck.** The verifier sends the points `τ` and a batching
   point `η`. Let `S_1, …, S_J` be the sumcheck claims of 3.1 (one Booleanity claim, one per
   Linear, Product and Lookup op), each written as `Σ_b f_j(b) = σ_j` over `n_Z` variables: a
   claim over fewer variables is padded with variables its summand ignores, and its sum is
   scaled by the power of two that introduces. The parties run one sumcheck for `Σ_j eq(η, j)·Σ_b f_j(b) = Σ_j eq(η, j)·σ_j`.
   The values `σ_j` that are not constants are sent by the prover before `η` is drawn.
3. **Claims.** The sumcheck ends at a point. The prover sends the value of every
   evaluation of `W̃` and `Z̃` that the final check of the sumcheck and the evaluation claims
   of 3.1 refer to. The verifier checks the sumcheck's final identity and every linear
   identity of 3.1 against those values. There remain claims `Z̃(z_j) = v_j` for `j < J_Z`
   and `W̃(w_j) = v′_j` for `j < J_W`.
4. **Phase 2, merging the points.** The verifier sends `θ`. The parties run a sumcheck of
   degree 2 for `Σ_b (Σ_j eq(θ, j)·eq(z_j, b))·Z(b) = Σ_j eq(θ, j)·v_j`, and the same for
   `W`, in shared rounds. It ends with one claim `Z̃(ρ_Z) = v_Z` and one claim
   `W̃(ρ_W) = v_W`; the verifier computes `Σ_j eq(θ, j)·eq(z_j, ρ_Z)` itself.
5. **Open.** The two claims are proven against `C_Z` and `C_W` (section 4).

### 3.3 Soundness with oracles

**Theorem 1.** Let `W` and `Z` be fixed before step 2 and suppose no `(W, Z)` with these
oracles is in `R_model` for `(x, y)`. Then the verifier accepts steps 2 to 4 with correct
final oracle answers with probability at most

`ε_1 = (n_τ + log J + 3·n_Z + log J_Z + log J_W + 2·n_Z + 2·n_W) / p`,

where `n_τ` is the largest number of coordinates in any point `τ`.

*Proof sketch.* Some equation of (R1) to (R5) fails. The corresponding family, as a function
of `τ`, is a nonzero multilinear polynomial, so its collapsed claim of 3.1 is false except
with probability `n_τ / p`. A false claim among those batched with `eq(η, ·)` leaves the
batched claim false except with probability `log J / p`. The sumcheck of phase 1 has degree
at most 3 and `n_Z` rounds, so it ends in a false final identity except with probability
`3 n_Z / p`; a false final identity, or a false evaluation claim of 3.1, means some value
sent in step 3 is wrong. Phase 2 then starts from a false batched claim except with
probability `(log J_Z + log J_W) / p`, and its sumcheck ends in a false claim on `Z̃(ρ_Z)`
or `W̃(ρ_W)` except with probability `(2 n_Z + 2 n_W) / p`. ∎

Every term is a number of rounds over `p`: nothing grows with the size of the instance
beyond its logarithm. For `n_Z = 37` and `n_W = 33` the bound is about `2^{−119}`.
Batching with `eq(η, ·)` and not with powers of one scalar is what keeps the number of ops
out of the bound.

The protocol is complete: an honest prover's claims are all true.

## 4. The commitment

### 4.1 Ring and moduli

- `R = Z[X]/(Φ)` with `Φ = X^162 + X^81 + 1`, the 243rd cyclotomic polynomial; `K` is its
  field of fractions.
- `q` is the commitment modulus, a product of small primes, and `R_q = R/qR`. It is chosen
  for the hardness of SIS and for fast arithmetic, and has nothing to do with `p`.
- `R_p = R/pR = F[X]/(Φ)`. It contains `F` as its constants. Multiplying an element of
  `R_p` by a constant multiplies each of its 162 coefficients.
- `‖·‖` is the Euclidean norm of the coefficient vector, taken over a column for a matrix,
  and `‖c‖_op` is the operator norm of multiplication by `c`.
- `SIS(n, m, β)`: given uniform `F ∈ R_q^{n×m}`, find nonzero `x ∈ R^m` with `‖x‖ ≤ β` and
  `F x = 0 mod q`.

**Lemma 2 (splitting).** Modulo `p`, `Φ = Π_{j<18} (X^9 − ω_j)`, where the `ω_j` are the 18
primitive 27th roots of unity of `F`, and each factor is irreducible.

*Proof.* `p ≡ 1 mod 27`, so `F` has 18 primitive 27th roots of unity, the roots of
`Φ_27(T) = T^18 + T^9 + 1`, and `Φ(X) = Φ_27(X^9)`. `X^9 − ω` is irreducible over `F` if and
only if `ω` is not a cube there. As `p ≢ 1 mod 81`, `(p − 1)/3` is divisible by 9 and not by
27, so `ω^{(p−1)/3}` has order 3 and `ω` is not a cube. ∎

**Lemma 3 (short elements are units).** Every `y ∈ R` whose coefficients are not all zero
and have absolute value at most 7 is a unit of `R_p`.

*Proof.* Write `y = Σ_{a<9} X^a·g_a(X^9)` with `g_a ∈ Z[T]` of degree below 18; the
coefficients of the `g_a` are those of `y`. Modulo `X^9 − ω_j`, `y` is
`Σ_a X^a·g_a(ω_j)`. By Lemma 2 each quotient is a field, so `y` is a unit of `R_p` if and
only if for every `j` some `g_a(ω_j)` is nonzero in `F`. Fix `a` with `g_a ≠ 0`. `Φ_27` is
irreducible over the rationals and has degree 18, so the resultant `ρ = Res(g_a, Φ_27)` is a
nonzero integer, and `|ρ| = Π_ζ |g_a(ζ)|` over the complex roots `ζ` of `Φ_27`, which is at
most `(18·7)^18 < 2^126 < p`. If `g_a(ω_j) = 0` in `F` for some `j`, then `g_a` and `Φ_27`
have a common root modulo `p` and `p` divides `ρ`, a contradiction. ∎

Checked numerically: `p mod 27 = 1`, `p mod 81 = 55`, the order of `p` modulo 243 is 9, and
2,000 random elements with coefficients in `[−2, 2]` were all units. The general statement of
this kind is Lyubashevsky and Seiler, EUROCRYPT 2018, Theorem 1.1; Lemma 3 does not use it.

### 4.2 Packing and challenges

- A vector `a ∈ Z^N` with `N = 128·m·r`, `m` and `r` powers of two, and entries of absolute
  value at most `B`, is packed into `W ∈ R^{m×r}`: entry `(i, j)` is
  `Σ_{k<128} a[k + 128·(i + m·j)]·X^k`. Coefficients 128 to 161 are zero.
- A point `ρ ∈ F^{log N}` splits as `ρ = (ρ_c, ρ_r, ρ_col)` with 7, `log m` and `log r`
  coordinates. Let `g = (eq(ρ_c), 0, …, 0) ∈ F^162` and, for `u ∈ R_p`, let
  `⟨g, u⟩ = Σ_{k<128} eq(ρ_c)_k·u_k` over its coefficients. Then
  `ã(ρ) = Σ_j eq(ρ_col)_j · ⟨g, (eq(ρ_r)^T W)_j⟩`.
- The challenge set `C ⊂ R` is the set of elements with exactly `h` nonzero coefficients,
  each `±1`. `|C| = C(162, h)·2^h`, which is `2^132` at `h = 28`. `γ = max_{c∈C} ‖c‖_op`.
  The difference of two distinct challenges has coefficients in `[−2, 2]`, so it is a unit
  of `R_p` by Lemma 3.

### 4.3 The scheme

- **Setup.** `F ← R_q^{n×m}`, uniform, derived from a public seed.
- **Commit.** `Y = F·W mod q ∈ R_q^{n×r}`. The opening state is `W`.
- **Evaluate**, for a claim `ã(ρ) = t`:
  1. The prover sends `u^T = eq(ρ_r)^T·W mod p ∈ R_p^{1×r}`.
  2. The verifier checks `Σ_j eq(ρ_col)_j·⟨g, u_j⟩ = t`.
  3. The verifier sends `c ← C^r`.
  4. The prover sends `v = W·c ∈ R^m`, computed over `R`.
  5. The verifier checks `‖v‖ ≤ β′`, `F·v = Y·c mod q`, and `eq(ρ_r)^T·v = u^T·c` in `R_p`.

With `β = B·√(128·m)` bounding an honest column, `β′ = r·γ·β`.

### 4.4 Completeness

For an honest `W`, `‖v‖ ≤ Σ_j ‖c_j‖_op·‖W_{:,j}‖ ≤ r·γ·β`, both linear checks hold by
linearity, and step 2 holds by the identity of 4.2. `v` is computed without reduction: it is
an element of `R^m`, and the same `v` is read modulo `q` and modulo `p`.

### 4.5 Openings and binding

A *relaxed opening* of `Y` is a pair `(W*, s)` with `W* ∈ R^{m×r}`, `s ∈ R^r`, such that
`F·W* = Y·diag(s) mod q`, `‖W*‖ ≤ 2β′`, and every `s_j` is nonzero with coefficients in
`[−2, 2]` and `‖s_j‖_op ≤ 2γ`. Its *message* is the vector `a* ∈ F^N` of coefficients 0 to
127 of the entries of `W*·diag(s)^{−1} mod p`, which exists by Lemma 3. An honest opening is
`(W, 1)`, and its message is `a mod p`.

**Lemma 4 (binding).** Under `SIS(n, m, 8γβ′)`, no efficient adversary outputs a commitment
with two relaxed openings whose messages differ.

*Proof.* Let `(W*, s)` and `(Ŵ, ŝ)` open `Y`. For each `j`,
`z_j = ŝ_j·W*_{:,j} − s_j·Ŵ_{:,j}` satisfies `F z_j = 0 mod q` and
`‖z_j‖ ≤ 2·2γ·2β′ = 8γβ′`. If some `z_j ≠ 0` it solves SIS. Otherwise
`ŝ_j·W*_{:,j} = s_j·Ŵ_{:,j}` over `R` for every `j`; reducing modulo `p` and dividing by the
units `s_j` and `ŝ_j` gives equal messages. ∎

This is LaBinius's Theorem 1 with "modulo 2" read as "modulo `p`" and "not in `2R`" read as
"a unit of `R_p`".

### 4.6 Knowledge soundness of the evaluation proof

**Theorem 2.** There is an extractor that, given a prover making the verifier of 4.3 accept
a claim `(Y, ρ, t)` with probability `ε`, runs it an expected `r + 1` times and, with
probability at least `ε − r/|C|`, outputs a relaxed opening `(W*, s)` of `Y` whose message
`a*` satisfies `ã*(ρ) = t`.

*Proof.* The prover's first message fixes `u`. By the coordinate-wise forking lemma of Fenzi,
Moghaddas and Nguyen (as used in LaBinius, Lemma 10), the extractor obtains, with probability
at least `ε − r/|C|`, accepting transcripts for challenges `c^{(0)}, …, c^{(r)}` where
`c^{(j)}` differs from `c^{(r)}` in coordinate `j` only, with answers `v^{(j)}`. For `j < r`
put `w_j = v^{(j)} − v^{(r)}` and `s_j = c^{(j)}_j − c^{(r)}_j`. Subtracting the verifier's
checks for the two transcripts:

- `F·w_j = s_j·Y_{:,j} mod q`,
- `eq(ρ_r)^T·w_j = u_j·s_j` in `R_p`,
- `‖w_j‖ ≤ 2β′`, and `s_j` is a nonzero difference of two challenges.

So `W* = (w_j)_j` with `s = (s_j)_j` is a relaxed opening. By Lemma 3 each `s_j` is a unit
of `R_p`, and the second line gives `u_j = eq(ρ_r)^T·(w_j·s_j^{−1})` in `R_p`. The entries of
`eq(ρ_r)` are constants of `R_p`, so they act on each coefficient separately, and
`⟨g, u_j⟩ = Σ_i Σ_{k<128} eq(ρ_r)_i·eq(ρ_c)_k·(w_{ij}·s_j^{−1})_k`. Summing against
`eq(ρ_col)` and using the verifier's check of step 2,
`ã*(ρ) = Σ_j eq(ρ_col)_j·⟨g, u_j⟩ = t`. ∎

Where the argument differs from LaBinius:

- LaBinius needs `R_2` to be a field in two places: to divide by `s_j mod 2`, and for the
  sumcheck of its field-switch step. We divide by units (Lemma 3) and have no field switch:
  our claim is in the constants `F` of `R_p`, where the sumchecks of section 3 run, and
  reaches the committed coefficients through the linear form `⟨g, ·⟩`.
- LaBinius packs one element of `F_{2^162}` per ring element. We pack 128 integers, each its
  own coordinate of the multilinear polynomial.
- The message of a relaxed opening is a vector over `F`. Nothing says it is small. The
  ranges the statement needs come from (R1), not from the commitment.

### 4.7 One opening for both commitments

Both commitments use the same `F` and the same `m`. Let `P = [W | Z]`, whose commitment is
`[Y_W | Y_Z]`. Phase 2 of 3.2 is run on `P` in place of `W` and `Z` separately, with one
more coordinate selecting the half, and ends in one claim `P̃(ρ) = v`. One run of 4.3 on `P`
proves it. Theorem 2 applies to `P` as it stands.

## 5. The composed argument

Replace the oracles of section 3 by commitments and the final oracle answers by the
evaluation proof of 4.7.

**Claim.** If `SIS(n, m, 8γβ′)` is hard, an efficient prover that makes the verifier accept
`(C_W, x, y)` with probability above `ε_1 + r/|C|` plus the SIS advantage yields, by
extraction, a pair in `R_model`.

*Argument.* Run the extractor of Theorem 2 on the final evaluation proof to get a relaxed
opening of `[Y_W | Y_Z]` with message `(W*, Z*)` and `P̃*(ρ) = v`. By Lemma 4 the message is
determined by the commitments, so `(W*, Z*)` plays the role of the oracles of Theorem 1,
which bounds the probability that it is outside `R_model` and the verifier accepts. ∎

This is the usual compilation of an oracle protocol with an extractable commitment, stated
without the bookkeeping. Two points need care in a full proof: `C_Z` is sent before the
challenges and `C_W` before the proof, so the message is fixed before the first challenge;
and the extractor rewinds only the last step, after the sumcheck's challenges are fixed.

**Fiat–Shamir.** The protocol is public-coin and is made non-interactive in the random
oracle model. Every error term above is a count of rounds over `p`, or `r/|C|`: none is a
bound of the kind that fails when a prover retries a round. We have not proven the
non-interactive protocol secure. What is needed is round-by-round soundness of the sumchecks,
which is standard, and a Fiat–Shamir analysis of the fold, whose extractor forks on `r`
coordinates.

## 6. Costs

**Prover**, per proof, in passes over the data:

| Step | Work |
|---|---|
| Commit the witness | one pass over its bits: a small-modulus transform and inner products per LaBinius; nothing for the weights, whose commitment is stored |
| Linear ops | one pass over the weights (`W̃(τ_i, ·)`) and one over the activations, then a sumcheck over the layer's width |
| Booleanity | a degree-3 sumcheck over every witness bit |
| Phase 2 | a degree-2 sumcheck over the weights and over the witness |
| Left opening `u` | one pass: a small value times a field element, per committed value |
| Fold `v` | one pass in machine words: a small value times a short challenge |

**Proof size.** The opening sends `u` (`r` elements of `R_p`, 2,592 bytes each) and `v` (`m`
elements of `R` with coefficients of about 18 to 25 bits). Balancing `m` against `r`, with
`h = 28`:

| Committed | Shape | `u` | `v` | Opening | Commitment |
|---|---|---|---|---|---|
| Qwen2-0.5B weights, 8 bits | `m = 2^12`, `r = 2^10` | 2.5 MiB | 1.8 MiB | 4.4 MiB | 0.6 to 1.2 MiB |
| 7B weights, 8 bits | `m = 2^14`, `r = 2^12` | 10.1 MiB | 7.9 MiB | 18 MiB | 2.3 to 4.7 MiB |
| 10^10 witness bits | `m = 2^15`, `r = 2^12` | 10.1 MiB | 11.4 MiB | 21.5 MiB | 2.3 to 4.7 MiB |
| 10^11 witness bits | `m = 2^16`, `r = 2^14` | 40.5 MiB | 25.3 MiB | 66 MiB | 9 to 19 MiB |

The commitment column assumes 600 to 1,200 bytes per column, a guess from LaBinius's reported
sizes; it is fixed by the SIS parameters, which are not set. The witness's commitment is part
of every proof. The sumchecks add a few kilobytes. These proofs are the size of the
small-field systems' (19 MB for FlashProve), not smaller, and the size counts every
committed cell, zeros included: a one-hot row of `K` cells per lookup is `K` cells.

**Verifier.** `n·m` multiplications in `R_q` and `r` by challenges for `F v = Y c`; `m`
scalar multiplications and `r` sparse multiplications in `R_p`; the sumchecks' rounds; and
hashing the proof for the challenges.

## 7. Assumed, and open

**Assumed, with a source:** sumcheck soundness; the coordinate-wise forking lemma; the
hardness of SIS over `R` at the stated norm.

**Ours, to be checked:** Lemmas 1 to 4, Theorems 1 and 2, Proposition 1.

**Open, in the order we would like a view on them:**

1. **Is non-inertness harmless?** Lemma 3 gives units where LaBinius has a field. Does any
   other step of its analysis use that `R_2` is a field in a way that does not carry over?
2. **Parameters at 120 bits.** `q`, the rank `n`, and `h`, for `B = 128` and for `B = 1`,
   against `SIS(n, m, 8γβ′)` with `β′ = r·γ·B·√(128 m)`. LaBinius targets 100 bits with a
   binary witness and estimates over a superring; we have run no estimate.
3. **Fiat–Shamir for the fold**, with `r` up to `2^14` forking coordinates.
4. **Proof size.** `u` is the larger half because its coefficients are 127 bits. Committing
   to `u` and proving the two linear forms the verifier needs of it, or wrapping the opening
   as LaBinius does with LaBRADOR, would shrink it at a cost to the verifier. Is there a
   sumcheck-based second layer that keeps the verifier fast?
5. **Lookups.** The one-hot encoding is sound and simple, and is priced by cells under a
   square-root opening. Decomposable tables as in Shout reduce the cells per lookup; the
   witness may still be better served by a commitment with a polylogarithmic opening
   (packed bits with a bridge to `F`), keeping this scheme for the weights. That would make
   two openings.
6. **The weights' range.** Proposition 1 needs it. A proof given once when the model is
   committed is enough; its form is not designed.
7. **Committing every tensor.** It is what makes one batched sumcheck suffice. The
   alternative proves the layers in sequence and commits less.

## References

- M. Osadnik, G. Seiler. LaBinius: Fast Lattice-Based Binary Polynomial Commitment Scheme.
  ePrint 2026/2103. Sections 1.3.1, 2.3, 3.3 to 3.5.
- G. Fenzi, H. Moghaddas, N. K. Nguyen. Lattice-Based Polynomial Commitments: Towards
  Asymptotic and Concrete Efficiency. Journal of Cryptology 37(3), 2024. Lemma 7.1, as
  restated in LaBinius, Lemma 3.
- V. Lyubashevsky, G. Seiler. Short, Invertible Elements in Partially Splitting Cyclotomic
  Rings and Applications to Lattice-Based Zero-Knowledge Proofs. EUROCRYPT 2018.
- S. Setty, J. Thaler. Twist and Shout. For the lookup argument that replaces the one-hot
  encoding of section 2 at scale.
- C. Lund, L. Fortnow, H. Karloff, N. Nisan. Algebraic Methods for Interactive Proof Systems.
  The sumcheck protocol.
