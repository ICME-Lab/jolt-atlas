# I-LLM Notes

This note records the integer-only linear quantization form to evaluate and
compare against the current fixed-point Qwen2 path.

## Integer Matmul Requantization

Given quantized integer inputs `X1_int`, `X2_int` and their zero-points `zp1`,
`zp2`, compute the raw integer matmul accumulator:

```text
P_int = (X1_int - zp1) @ (X2_int - zp2)
```

For an `n`-bit output integer range:

```text
Q = 2^n - 1
```

Requantize the accumulator using the observed or calibrated accumulator range
`[p_min, p_max]`:

```text
Y_int = round((P_int - p_min) * Q / (p_max - p_min))
```

The same rebase can be implemented with integer arithmetic only:

```text
Y_int = ((P_int - p_min) * Q + (p_max - p_min) / 2) // (p_max - p_min)
```

Equivalently, with `den = p_max - p_min`:

```text
Y_int = ((P_int - p_min) * Q + den / 2) // den
```

This maps the accumulator range linearly into the output integer range:

```text
P_int = p_min  ->  Y_int = 0
P_int = p_max  ->  Y_int = Q
```

The `+ den / 2` term implements round-to-nearest before integer division. If it
is omitted, the division floors the result and introduces a downward rounding
bias.

The shared `DI-Rebase` primitive should therefore make the rounding policy
explicit:

```text
Nearest:
Y_int = ((P_int - p_min) * Q + (p_max - p_min) / 2) // (p_max - p_min)

Floor:
Y_int = ((P_int - p_min) * Q) // (p_max - p_min)
```

The implementation must not silently clamp. It may use `debug_assert` to catch
values outside `[0, Q]` while debugging calibration ranges.

For the I-LLM execution path, prefer precomputed integer multipliers and shifts
instead of runtime division. For a fixed shift precision `r`:

```text
M_rebase ~= Q * 2^r / (p_max - p_min)
Y_int = ((P_int - p_min) * M_rebase) >> r
```

The multiplier is computed offline from `p_min`, `p_max`, and `Q`. Runtime and
proof-time work is then integer subtraction, multiplication by a constant, and
shift. This is an approximation to the exact division formula unless the
multiplier is chosen with an exact magic-division scheme.

For scale alignment, use the same pattern:

```text
M_scale ~= (s_in / s_out) * 2^r
Y_int = (X_int * M_scale) >> r
```

Any rounding behavior must be explicit: floor uses no shift bias, while
round-to-nearest adds `2^(r-1)` before shifting.

## Nonlinear LUT Plan

For nonlinear functions, build the LUT from the input tensor quantization
parameters:

```text
input: X_int, s_x, zp_x
domain: q in [0, Q]
x_real(q) = (q - zp_x) * s_x
lut_real[q] = f(x_real(q))
```

For the first implementation, apply the LUT to the actual tensor values, take
the observed output range, and rebase from that measured range:

```text
Y_raw[i] = lut_real[X_int[i]]
y_min = min_i Y_raw[i]
y_max = max_i Y_raw[i]
Y_int = DI-Rebase(Y_raw, y_min, y_max)
```

This keeps the output scale tight for the current tensor and should preserve
more accuracy during early experiments.

An alternative is to choose `y_min/y_max` from the whole LUT domain or from a
calibrated range:

```text
y_min = min_q lut_real[q]
y_max = max_q lut_real[q]
```

That is easier to make public and prove, but it can reduce precision if the
current tensor only uses a narrow part of the LUT range. Keep this as a later
calibration/proof-oriented option; do not use it as the first accuracy baseline.

For softmax exponentials, prefer a nonnegative max-shift input:

```text
d = max(score) - score
exp_out = exp(-d)
```

Then the LUT input range is nonnegative, which is easier to range-check.

## Current Primitive Functions

The initial experimental primitives live in `src/illm.rs`:

```text
di_matmul(lhs, rhs, m, k, n, cfg)
di_add(lhs, rhs, multiplier_shift, cfg)
di_silu(input, cfg)
di_reciprocal(input, cfg)
di_mul(lhs, rhs, cfg)
di_softmax(input, rows, cols, cfg)
```

`di_matmul` computes the centered integer accumulator:

```text
P_int = (lhs_int - zp_lhs) @ (rhs_int - zp_rhs)
p_scale = s_lhs * s_rhs
```

then applies observed-range DI-Rebase.

`di_add` does not rebase each branch before addition. It first lifts both
branches into a common accumulator scale with precomputed multiplier+shift, adds
there, and then applies one observed-range DI-Rebase:

```text
s_acc = min(s_lhs, s_rhs)
A_acc = ((lhs_int - zp_lhs) * M_lhs) >> r
B_acc = ((rhs_int - zp_rhs) * M_rhs) >> r
P_int = A_acc + B_acc
p_scale = s_acc
```

`di_reciprocal` is the reciprocal LUT primitive used by softmax normalization.
Unlike `exp`, reciprocal is undefined at zero, so the current experimental
implementation evaluates only the actual positive input values and asserts that
they are positive. For softmax this is natural because `sum(exp(max-score))` is
strictly positive.

`di_softmax` uses a softmax-specific implementation of DI-ClippedSoftmax and
DI-Exp. It does not call a generic `di_exp` primitive:

```text
x = score - max(row)
x = max(x, -clip)
x_int = floor(x / (clip / 255))
e_int = DI-Exp(x_int, m_x=clip, k_x=8, b_y=16)
sum_e = sum(e_int)
m_sum = floor(Q8 * 2^r / sum_e)
p_int = (e_int * m_sum) >> r
```

The default clipping value is `clip = 15`, matching the paper's recommended
range. The clipped softmax input is represented as signed 8-bit values in
`[-255, 0]` with dyadic scale `clip / 2^8`. `DI-Exp` follows the paper's
shift-based approximation with `log2(e) ~= 369 / 2^8` and returns an internal
fixed-point integer scaled by `2^16`; `exp(0)` is therefore represented as
`65536`.

The final softmax probability is always unsigned 8-bit with `scale = 1 / 255`
and `zero_point = 0`, independent of the matmul activation bit width. The
implementation uses multiplier+shift for normalization and does not silently
clamp; range violations are debug assertions. The explicit clipping step belongs
to DI-ClippedSoftmax and is not the same as clamping DI-Rebase outputs.

## DI-Softmax PPL Experiment

To isolate softmax, keep the Qwen2 forward pass in float through `score_qk`,
replace only attention softmax with the DI-softmax path, dequantize the 8-bit
softmax probabilities back to float, and continue with the float `attn_v` and
remaining block computations.

The softmax input is first quantized per row/token. Each row subtracts its
floating-point row max once, then stores integer deltas plus a row-specific
scale:

```text
delta = score - max(row)
scale_row = max_abs_delta_row / 255
delta_int = floor(delta / scale_row)
```

After this conversion, DI-softmax uses integer arithmetic: optional
DI-ClippedSoftmax input clipping, DI-Exp, integer row sum, and multiplier+shift
normalization into unsigned 8-bit probabilities.

Measured on:

```text
"hello world this is a test"
```

Full-prefix PPL:

```text
float baseline:              48.46466056027114
DI-softmax, clip=true:       49.1250485600604
DI-softmax, clip=false:      49.12478684874131
```

First 3 target PPL:

```text
DI-softmax, clip=true:       104.59642040767521
DI-softmax, clip=false:      104.59656377541972
```

For this prompt, disabling the DI-ClippedSoftmax input clip has essentially no
effect on PPL. The DI-softmax-only degradation is small relative to the full
experimental DI-Qwen path, so the large PPL degradation is more likely caused by
the quantized Q/K/score path before softmax.

The output scale is:

```text
s_y = (p_max - p_min) * s_x1 * s_x2 / Q
```

where `s_x1` and `s_x2` are the input scales for `X1_int` and `X2_int`.

## Notes

- `P_int` is the integer accumulator before output requantization.
- `p_min` and `p_max` define the accumulator range used to map `P_int` into the
  `n`-bit output interval.
- `Q` is the maximum unsigned integer value representable by `n` bits.
- `s_y` carries the real-value scale of `Y_int` after requantization.
- The product `(P_int - p_min) * Q` can be large, so implementation should use a
  wider accumulator such as `i64` or `i128`.
- The degenerate case `p_max == p_min` must be handled separately.
- If `P_int` can fall outside the calibrated `[p_min, p_max]` range, the result
  can fall outside `[0, Q]`. Treat clamping as an explicit implementation-policy
  decision, not as part of the core formula.
