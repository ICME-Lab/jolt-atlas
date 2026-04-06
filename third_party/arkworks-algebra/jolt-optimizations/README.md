# jolt-optimizations

Optimized scalar multiplication for BN254 using GLV endomorphisms and various precomputation techniques.

## Prerequisites

**SageMath must be installed** as the build process uses Sage scripts to generate lookup tables.

Install SageMath:
- macOS: `brew install sage`
- Ubuntu/Debian: `sudo apt-get install sagemath`
- Or download from [sagemath.org](https://www.sagemath.org/)

## Features

- 2D GLV scalar multiplication for G1
- 4D GLV scalar multiplication for G2
- Precomputed tables for fixed-base operations
- Optimized Dory commitment scheme utilities
- Scalar mul vector primitives

## Building

```bash
cargo build
```

The build will fail if `sage` is not in your PATH.