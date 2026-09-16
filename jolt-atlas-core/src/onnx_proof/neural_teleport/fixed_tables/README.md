# Fixed lookup tables

These optional program constants contain the same quantized values as the
reference sine, cosine and sigmoid generators at model scale 14. They replace
runtime floating point table construction when `fixed-tables` is enabled.
Other compile-time scales use the reference generator. Table evaluation,
transcript challenges and proof checks are unchanged.

Each file is an array of signed 32-bit integers in little endian order.
Sine and cosine contain 65,536 entries, with period modulus 2,470,649 and
six downscale bits. Sigmoid contains 262,144 entries in two's complement
input order. No table comes from the prover's advice.

Regenerate from the repository root with:

```sh
rustc jolt-atlas-core/tools/generate_fixed_tables.rs -O -o /tmp/atlas-fixed-tables
/tmp/atlas-fixed-tables jolt-atlas-core/src/onnx_proof/neural_teleport/fixed_tables
cargo test --release -p jolt-atlas-core fixed_tables_match_every_reference_entry
```

The test compares every entry against Atlas's original tensor functions.
`check-fixed-tables` additionally repeats those complete comparisons at
runtime. It is intended for validation across targets, including RISC-V,
and must be disabled for the optimized measurements.
