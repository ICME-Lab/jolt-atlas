use allocative::Allocative;

pub mod consts;
pub mod parallel;
pub mod serialize;
pub mod utils;

// ---------------------------------------------------------------------------
// CommittedPoly
// ---------------------------------------------------------------------------

canonical_serde_enum! {
    /// Identifiers for polynomials that are **committed** (i.e. their evaluations are
    /// bound via a polynomial commitment scheme).
    ///
    /// Each variant carries enough information to uniquely identify a specific
    /// committed polynomial within a proof.  The naming convention uses the
    /// following suffixes:
    ///
    /// * **`RaD`** – one-hot *read-address decomposition* polynomial (indexed by
    ///   node and decomposition index `d`).
    /// * **`Ra`** – read-address polynomial (a single polynomial per node, before
    ///   decomposition).
    ///
    /// # Grouping
    ///
    /// | Group | Purpose |
    /// |-------|---------|
    /// | `NodeOutputRaD` / `{Cos,Sin}RaD` | One-hot read-address decompositions for activation-function lookup tables |
    /// | `Softmax*` | Polynomials specific to softmax sub-protocol |
    /// | `*NodeQuotient / *NodeRemainder / *NodeInv / *NodeRsqrt` | Scalar advice polynomials for division / reciprocal-square-root |
    /// | `GatherRa` | Read-address polynomial for the Gather operator |
    /// | `SoftmaxShift` | The softmax row shift, one value per row |
    #[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
    pub enum CommittedPoly {
        // ----- One-hot read-address decompositions (node_index, d) -----
        /// One-hot read-address decomposition for a node's output lookup.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        NodeOutputRaD(usize, usize),

        /// One-hot read-address decomposition for the **Cos** lookup table.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        CosRaD(usize, usize),

        /// One-hot read-address decomposition for the **Sin** lookup table.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SinRaD(usize, usize),

        /// One-hot read-address decomposition for the trig-table downscale (right-shift) lookup
        /// that precedes the Cos/Sin table lookup. Shared by Cos and Sin nodes.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        TrigDownscaleRaD(usize, usize),

        // ----- Scalar advice polynomials (node_index) -----
        /// Advice polynomial for the **quotient** in integer division.
        ///
        /// * `0` – node index
        DivNodeQuotient(usize),

        /// Advice polynomial for the fused Rsqrt **quotient** `⌊S³ / x̂⌋`,
        /// i.e. the value fed into the integer square root.
        ///
        /// * `0` – node index
        RsqrtQuotient(usize),

        /// Advice polynomial for the teleportation division op.
        ///
        /// * `0` – node index
        TeleportNodeQuotient(usize),

        /// Read-address polynomial for the **Gather** operator.
        ///
        /// * `0` – node index
        GatherRa(usize),

        /// One-hot read-address decomposition for the **Gather** operator.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        GatherRaD(usize, usize),

        /// One-hot read-address decomposition for the softmax table read: the clamped address
        /// `min(c − x, top)` of the output and sum tables.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxAddrRaD(usize, usize),

        /// The softmax row shift `c`, one value per row.
        ///
        /// * `0` – node index
        SoftmaxShift(usize),

        /// One-hot read-address decomposition for the **Clamp** operator's
        /// symmetric-range lookup.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SymmetricClampRaD(usize, usize),

        /// One-hot read-address decomposition for the clamped Erf/Sigmoid/Tanh
        /// variants' activation-clamp lookup stage (see [`VirtualPoly::ActivationClampRa`]).
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        ActivationClampRaD(usize, usize),

        /// One-hot read-address decomposition for the clamped Erf/Sigmoid/Tanh
        /// variants' small dense activation-table execution stage (shared across
        /// all three ops; see [`VirtualPoly::ActivationSmallRa`]).
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        ActivationSmallRaD(usize, usize),

        /// One-hot read-address decomposition for softmax's clamp lookup
        /// (`u -> min(max(u, 0), top)`, see [`VirtualPoly::SoftmaxClampRa`]). The lookup index is
        /// `u = c − x`, a signed 32-bit address.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxClampRaD(usize, usize),

        /// Dense base-`b` digit of a packed range bucket's value (a remainder, a slack, or an
        /// offset output `out + 2^31`), most significant first (`U8Scalars` in `[0, b)`; see
        /// `jolt_atlas_verifier::onnx_proof::range_check`).
        ///
        /// * `0` – bucket index
        /// * `1` – digit index `j`
        GlobalRemainderDigit(usize, usize),

    }
}

// ---------------------------------------------------------------------------
// VirtualPoly
// ---------------------------------------------------------------------------

canonical_serde_enum! {
    /// Identifiers for **virtual** (non-committed) polynomials.
    ///
    /// Virtual polynomials are derived or constructed during the proof but are
    /// *not* independently committed.  They are instead verified through sumcheck
    /// relations that tie them back to committed polynomials and public inputs.
    ///
    /// The same naming conventions as [`CommittedPoly`] apply (`Ra` = read
    /// address, `RaD` = read-address decomposition, etc.).
    ///
    /// # Grouping
    ///
    /// | Group | Purpose |
    /// |-------|---------|
    /// | `NodeOutput` / `NodeOutputRa` | MLE of a node's output tensor and its read-address polynomial |
    /// | `{Cos,Sin}Ra` | Read-address polynomials for activation-function lookups |
    /// | `Softmax*` | Intermediate polynomials arising in the softmax sub-protocol |
    /// | `HammingWeight` | Polynomial used in the Hamming-weight sumcheck |
    /// | `Div* / Sqrt* / Teleport*` | Advice polynomials of the division, square-root and teleport identities |
    #[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
    pub enum VirtualPoly {
        // ----- Node output -----
        /// Multilinear extension (MLE) of a node's output tensor.
        ///
        /// * `0` – producer node index
        NodeOutput(usize),

        /// Read-address polynomial for a node's output lookup table.
        ///
        /// * `0` – node index
        NodeOutputRa(usize),

        // ----- Activation-function read-address polynomials (node_index) -----
        /// Read-address polynomial for the **Cos** lookup table.
        ///
        /// * `0` – node index
        CosRa(usize),

        /// Read-address polynomial for the **Sin** lookup table.
        ///
        /// * `0` – node index
        SinRa(usize),

        /// Read-address polynomial for the trig-table downscale (right-shift) lookup that
        /// precedes the Cos/Sin table lookup. Shared by Cos and Sin nodes.
        ///
        /// * `0` – node index
        TrigDownscaleRa(usize),

        // ----- Softmax intermediate polynomials (node_index, feature_index) -----

        // ----- Hamming weight -----
        /// Polynomial used in the Hamming-weight sumcheck.
        ///
        /// * `0` – node index. Per node: a shared identifier would alias the
        ///   openings of every node using the check, binding one accumulator
        ///   slot at a different point per node; only the constant-1 honest
        ///   value keeps that coincidence invisible.
        HammingWeight(usize),

        // ----- Advice-derived polynomials (proven via ReadRafSumcheckProver) -----

        // ----- Division / square-root remainder & quotient polynomials -----
        /// Remainder polynomial for integer division advice.
        ///
        /// * `0` – node index
        DivRemainder(usize),

        /// Remainder polynomial of a scalar-constant division.
        ///
        /// * `0` – node index
        ScalarConstDivRemainder(usize),

        /// Remainder polynomial for square-root advice.
        ///
        /// * `0` – node index
        SqrtRemainder(usize),

        /// Quotient polynomial for the **neural teleportation** division.
        ///
        /// * `0` – node index
        TeleportQuotient(usize),

        /// Remainder polynomial for the **neural teleportation** division.
        ///
        /// * `0` – node index
        TeleportRemainder(usize),

        /// The Cos/Sin trig-table downscaled value: `TeleportRemainder >> TRIG_DOWNSCALE_BITS`,
        /// evaluated at the same opening point as `TeleportRemainder`. Shared by Cos and Sin
        /// nodes; this is what the Cos/Sin table lookup actually reads.
        ///
        /// * `0` – node index
        TrigDownscaled(usize),

        /// Per-row gap of the softmax sum-table reads below the sum scale: `S' − Σ_j T_sum`.
        ///
        /// * `0` – node index
        SoftmaxSumGap(usize),

        /// Per-row excess of the shifted sum-table reads over the sum scale:
        /// `Σ_j T_sum⁻ − S' − 1`.
        ///
        /// * `0` – node index
        SoftmaxPrevSumGap(usize),

        /// The softmax output-table read at every lane's clamped address.
        ///
        /// * `0` – node index
        SoftmaxRead(usize),

        /// The softmax sum-table read at every lane's clamped address.
        ///
        /// * `0` – node index
        SoftmaxSumRead(usize),

        /// The softmax shifted sum-table read at every lane's clamped address.
        ///
        /// * `0` – node index
        SoftmaxPrevSumRead(usize),

        /// The softmax table address `min(max(c − x, 0), top)`: the address of the table read and
        /// the value of the clamp lookup.
        ///
        /// * `0` – node index
        SoftmaxClampedAddr(usize),

        /// One-hot-encoded read-address polynomial of the softmax table read.
        ///
        /// * `0` – node index
        SoftmaxAddrRa(usize),

        /// The softmax clamp lookup's operand `u = c − x`, at the softmax reads' point.
        ///
        /// * `0` – node index
        SoftmaxClampWitness(usize),

        /// One-hot read-address polynomial for softmax's saturating-clamp lookup.
        ///
        /// * `0` – node index
        SoftmaxClampRa(usize),

        // ----- Fused rescale (einsum, Mul, Square, Cube, MeanOfSquares) -----

        /// Remainder `R = acc mod D` of a fused rescaling division
        /// (`acc = out·D + R`, where `acc` is the operator's raw i64 accumulation and
        /// `D` its divisor: `2^S`, or `N·2^S` for `MeanOfSquares`). Range-checked to
        /// `[0, D)`; bound into the operator sumcheck's input claim.
        ///
        /// * `0` – node index
        RescaleRemainder(usize),

        /// Read-address polynomial for the **Clamp** operator's symmetric-range
        /// lookup. Virtualized — [`CommittedPoly::SymmetricClampRaD`] commits its
        /// decomposition.
        ///
        /// * `0` – node index
        SymmetricClampRa(usize),

        /// The clamped Erf/Sigmoid/Tanh variants' internal "clamped input" claim,
        /// shared by both of the node's sumchecks: the small-table execution stage
        /// uses it (gamma-batched) as part of its input claim, and the
        /// activation-clamp lookup stage soundly proves it equals
        /// `ActivationClampTable[raw_input]` -- this claim is *proven*, not asserted.
        ///
        /// * `0` – node index
        ActivationClampedOutput(usize),

        /// `max(x, 0)` of a smooth-ReLU node's input, at the node's reduced output point: the
        /// value of the node's ReLU read, and the part of its output the correction table does
        /// not hold.
        ///
        /// * `0` – node index
        SmoothReluRelu(usize),

        /// Read-address polynomial for the clamped Erf/Sigmoid/Tanh variants'
        /// activation-clamp lookup stage (natively-symmetric clamp table, same shape as
        /// [`Self::SymmetricClampRa`]). Virtualized --
        /// [`CommittedPoly::ActivationClampRaD`] commits its decomposition.
        ///
        /// * `0` – node index
        ActivationClampRa(usize),

        /// Read-address polynomial for the clamped Erf/Sigmoid/Tanh variants'
        /// small dense activation-table execution stage. Virtualized --
        /// [`CommittedPoly::ActivationSmallRaD`] commits its decomposition.
        ///
        /// * `0` – node index
        ActivationSmallRa(usize),

        /// A packed range bucket's value (virtual; the recomposition of
        /// [`CommittedPoly::GlobalRemainderDigit`]'s digits).
        ///
        /// * `0` – bucket index
        GlobalRemainderValue(usize),

    }
}
