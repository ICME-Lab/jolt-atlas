# zk-sumcheck

This crate is a sandbox for a committed round-polynomial sumcheck.

The design being tested is:

```text
g_i(x) = c_{i,0} + c_{i,1}x + ... + c_{i,d}x^d

g_i(r_i) = g_{i+1}(0) + g_{i+1}(1)
```

Commit to the coefficient vector of each `g_i`.

```text
C_i = Com(c_i)
```

Then the middle rounds can be checked as linear relations between commitments.

```text
<C_i,     (1, r_i, r_i^2, ..., r_i^d)>
  =
<C_{i+1}, (2, 1,   1,     ..., 1)>
```

The verifier only needs endpoint openings:

```text
g_0(0) + g_0(1)
g_k(r_k)
```

The current crate only contains the skeleton.  It intentionally does not yet
implement Pedersen commitments, IPA openings, or relation-specific sumcheck
message generation.

Related code already exists in `joltworks/src/subprotocols/sumcheck.rs` under
the `zk` feature.  That path commits to round polynomials for BlindFold.  This
crate is for a smaller, easier-to-read version focused on the round-to-round
linear commitment relation above.
