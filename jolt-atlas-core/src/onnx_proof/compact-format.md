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

The encoding begins with the eight ASCII bytes `ATLSCP02`, a mode byte
(`0` uncompressed, `1` compressed), and a little-endian u64 claim count.
Claims appear in strictly increasing `OpeningId` order:

| Tag | Record |
|---|---|
| 0 | Identifier prefix length (u8), suffix length (u8), suffix bytes, canonical BN254 scalar |
| 1 | `SoftmaxSumOutput` run |
| 2 | `SoftmaxMaxOutput` run |
| 3 | `SoftmaxMaxIndex` run |

An ordinary identifier shares its longest byte prefix with the preceding
ordinary identifier, initially empty. Advice runs do not change this state.
The prefix and suffix lengths sum to at most 255. Reconstruction must consume
exactly one canonical `OpeningId`; a shorter shared prefix is rejected. The
scalar is always the complete canonical field element.

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

Operator sumchecks, commitments and the final reduced opening proof retain
the existing canonical encoding in the selected mode. The evaluation-reduction
map retains its u64 count and increasing u64 node keys, followed by one tag:

| Reduction tag | Record |
|---|---|
| 0 | The original canonical coefficient vector |
| 1 | No payload; reconstruct a one-coefficient polynomial from the reference below |

For node `n`, the reference is the first opening claim in key order with
`NodeOutput(n)` and `NodeExecution(consumer)` where `consumer >= n`. Tag 1
requires that claim to exist. Encoding uses tag 1 exactly when the coefficient
vector has length one and equals the reference scalar; tag 0 rejects that case.
Other polynomials, including empty vectors, mismatching constants, and trailing
zero coefficients, are preserved verbatim. This is only a byte reference: it
neither assumes a single consumer nor skips verification of the reconstructed
proof. The streamed legacy encoding restores every original coefficient.

Every collection length is bounded by the remaining bytes before allocation. Maps require increasing keys, field elements retain canonical
validation, and curve points are re-encoded through a comparing writer to
reject normalization. Identifier encodings contain validated enum tags and
u64 indices; they are also round-tripped where `usize` is narrower than u64
to reject narrowing. An unchanged polynomial prefix reuses its previously
validated identifier. Reduction references reuse the original scalar bytes,
so neither these values nor sumcheck scalars need a second serialization.
Changing the format requires a new version marker.

An advice vector with n entries originally uses 59n bytes, excluding the map
count. A compact run uses 25 + 4n bytes. Identifier prefixes save their length
minus three framing bytes compared with canonical claims. A reduction reference
replaces a 40-byte coefficient vector with a one-byte tag. Non-reference
reductions add one tag byte. Proofs without these repetitions may grow. Applications should measure their own inputs.
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

Version 2 supersedes the experimental `ATLSCP01` encoding. That earlier codec
is retained in the PR's commit history and measured artifacts, but this decoder
requires the version 2 marker. Existing canonical serialization is unchanged.
