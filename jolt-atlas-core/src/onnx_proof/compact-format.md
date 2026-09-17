# Compact BN254/HyperKZG proof encoding

`ONNXProof<Fr, T, HyperKZG<Bn254>>::serialize_compact(Compress::Yes)` encodes an ordinary
BN254/HyperKZG proof without changing its claims, equations or transcript values.
`deserialize_compact(bytes, max_claims)` constructs the usual proof directly,
validates field and group elements, and requires complete consumption. Call
`verify` with the expected preprocessing and IO after decoding.

The format is opt-in. Existing canonical serialization and stored proofs are
unchanged. `Compress::No` retains uncompressed curve points for verifier guests.
The caller chooses a claim-count limit for its model and bounds input bytes.
This is not a native ZK proof format and does not hide softmax advice.

The encoding begins with the eight ASCII bytes `ATLSCP01`, a mode byte
(`0` uncompressed, `1` compressed), and a little-endian u64 claim count.
Claims appear in strictly increasing `OpeningId` order:

| Tag | Record |
|---|---|
| 0 | Canonical `OpeningId`, then canonical BN254 scalar |
| 1 | `SoftmaxSumOutput` run |
| 2 | `SoftmaxMaxOutput` run |
| 3 | `SoftmaxMaxIndex` run |

A run contains a little-endian u64 node, first index and nonzero length,
followed by that many little-endian u32 values. Its sumcheck is
`NodeExecution(node)`. Only exact u32 field values use runs, including the
prover's two's-complement u32 representation of negative maxima. Other values
or sumcheck identifiers use ordinary records without narrowing. Runs are
maximal, and indices must fit the decoding target's `usize` without overflow.
Raw records for packable values, duplicate keys and split contiguous runs are
rejected.

`deserialize_compact_with_canonical(bytes, max_claims, writer)` additionally
streams the exact legacy canonical encoding into a writer in the same mode.
This supports existing proof-byte transcript hashes without allocating an
expanded buffer or reserializing every scalar. Discard partial writer output
if decoding fails. It does not change the cryptographic transcript.

The rest of the proof uses the existing canonical encoding in the selected
mode. Every collection length is bounded by the remaining bytes before
allocation. Maps require increasing keys, field elements retain canonical
validation, and indices and curve points are re-encoded through a comparing
writer to reject narrowing and normalization. Sumcheck scalars need no second
serialization.
Changing the format requires a new version marker.

An advice vector with n entries originally uses 59n bytes, excluding the map
count. A compact run uses 25 + 4n bytes. Ordinary records add one tag byte, so
proofs without softmax may grow. Applications should measure their own inputs.
This encoding does not remove cryptographic claims or promise faster proving.

To compare native decoding on an existing canonical compressed HyperKZG proof:

```sh
cargo run --release -p jolt-atlas-core --example compact_proof -- INPUT NEW_OUTPUT
cargo test -p jolt-atlas-core compact::tests
```

The example checks exact recovery of the original proof and reports alternating
native decode samples. It does not verify a proof without its model and IO.
The tests also generate and verify a complete softmax proof in both encoding
modes and reject altered advice.
