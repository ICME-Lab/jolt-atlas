#!/usr/bin/env python3
"""Generate the committed fixed-point constants in atlas-trace/src/semantics/coeffs.rs.

Uses only the Python standard library (decimal at 60 significant digits).
Each constant is `round(value * 2^frac_bits)` computed in arbitrary precision.

Run: python3 scripts/gen_constants.py
"""

from decimal import Decimal, getcontext

getcontext().prec = 60


def fixed(value: Decimal, frac_bits: int) -> int:
    scaled = value * (Decimal(2) ** frac_bits)
    return int(scaled.quantize(Decimal(1), rounding="ROUND_HALF_EVEN"))


def pi() -> Decimal:
    # Chudnovsky via decimal: use the well-known Machin-like formula with atan
    # series in decimal. atan(1/x) converges quickly for x >= 5.
    def atan_inv(x: int) -> Decimal:
        total = Decimal(0)
        term = Decimal(1) / x
        xx = x * x
        n = 0
        sign = 1
        while term != 0:
            total += sign * term / (2 * n + 1)
            term /= xx
            sign = -sign
            n += 1
            if n > 200:
                break
        return total

    # Machin: pi/4 = 4*atan(1/5) - atan(1/239)
    return (4 * atan_inv(5) - atan_inv(239)) * 4


def exp(x: Decimal) -> Decimal:
    return x.exp()


def ln(x: Decimal) -> Decimal:
    return x.ln()


def sqrt(x: Decimal) -> Decimal:
    return x.sqrt()


PI = pi()

constants = [
    ("PI_Q64", "u128", fixed(PI, 64), "π in Q64.64"),
    ("HALF_PI_Q64", "u128", fixed(PI / 2, 64), "π/2 in Q64.64"),
    ("EXP_NEG1_Q62", "u64", fixed(exp(Decimal(-1)), 62), "e^-1 in Q0.62"),
    ("LN2_Q62", "u64", fixed(ln(Decimal(2)), 62), "ln 2 in Q0.62"),
    (
        "TWO_OVER_SQRT_PI_Q62",
        "u64",
        fixed(Decimal(2) / sqrt(PI), 62),
        "2/√π in Q2.62",
    ),
]

for name, ty, value, doc in constants:
    print(f"/// {doc}: `round(value * 2^fractional_bits)`.")
    print(f"pub const {name}: {ty} = {value};")
    print()
