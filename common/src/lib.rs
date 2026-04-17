use allocative::Allocative;

pub mod consts;
pub mod utils;

// ---------------------------------------------------------------------------
// CommittedPolynomial
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
    /// | `NodeOutputRaD` / `{Cos,Erf,Sin,Tanh}RaD` | One-hot read-address decompositions for activation-function lookup tables |
    /// | `Softmax*` | Polynomials specific to softmax sub-protocol |
    /// | `Div* / Sqrt* / Teleport*` | Range-check one-hot polynomials for integer-arithmetic advice values |
    /// | `*NodeQuotient / *NodeRemainder / *NodeInv / *NodeRsqrt` | Scalar advice polynomials for division / reciprocal-square-root |
    /// | `GatherRa` | Read-address polynomial for the Gather operator |
    /// | `SoftmaxRecipMultRemainder` | Remainder used in the reciprocal-multiplication check of softmax |
    #[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
    pub enum CommittedPolynomial {
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

        /// One-hot read-address decomposition for the **Erf** lookup table.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        ErfRaD(usize, usize),

        /// One-hot read-address decomposition for the **Sin** lookup table.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SinRaD(usize, usize),

        /// One-hot read-address decomposition for the **Tanh** lookup table.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        TanhRaD(usize, usize),

        // ----- Range-check one-hot polynomials for advice values (node_index, d) -----
        /// Interleaved remainder `R` and divisor one-hot polynomial for the
        /// **Div** range check.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        DivRangeCheckRaD(usize, usize),

        /// Interleaved remainder `R` and divisor one-hot polynomial for the
        /// **Sqrt** division range check.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SqrtDivRangeCheckRaD(usize, usize),

        /// Interleaved `r_s` and square-root one-hot polynomial for the
        /// **Sqrt** range check.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SqrtRangeCheckRaD(usize, usize),

        /// Remainder and input one-hot polynomial for the **neural teleportation**
        /// division range check.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        TeleportRangeCheckRaD(usize, usize),

        // ----- Scalar advice polynomials (node_index) -----
        /// Advice polynomial for the **quotient** in integer division.
        ///
        /// * `0` – node index
        DivNodeQuotient(usize),

        /// Advice polynomial for the **remainder** in scalar-constant division.
        ///
        /// * `0` – node index
        ScalarConstDivNodeRemainder(usize),

        /// Advice polynomial for `1 / sqrt(x)` intermediate **inverse** in Rsqrt.
        ///
        /// * `0` – node index
        RsqrtNodeInv(usize),

        /// Advice polynomial for the teleportation division op.
        ///
        /// * `0` – node index
        TeleportNodeQuotient(usize),

        /// Sigmoid Ra d polynomial
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SigmoidRaD(usize, usize),

        /// Read-address polynomial for the **Gather** operator.
        ///
        /// * `0` – node index
        GatherRa(usize),

        /// One-hot read-address decomposition for the **Gather** operator.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        GatherRaD(usize, usize),

        /// One-hot read-address decomposition for the softmax **remainder** RC.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxRemainderRaD(usize, usize),

        /// One-hot read-address decomposition for the softmax **exp-remainder** RC.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxExpRemainderRaD(usize, usize),

        /// One-hot read-address decomposition for the softmax **exp-hi** Shout lookup.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxZHiRaD(usize, usize),

        /// One-hot read-address decomposition for the softmax **exp-lo** Shout lookup.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxZLoRaD(usize, usize),

        /// One-hot read-address decomposition for the softmax **sat-diff** RC.
        ///
        /// * `0` – node index
        /// * `1` – decomposition index `d`
        SoftmaxSatDiffRaD(usize, usize),
    }
}

// ---------------------------------------------------------------------------
// VirtualPolynomial
// ---------------------------------------------------------------------------

canonical_serde_enum! {
    /// Identifiers for **virtual** (non-committed) polynomials.
    ///
    /// Virtual polynomials are derived or constructed during the proof but are
    /// *not* independently committed.  They are instead verified through sumcheck
    /// relations that tie them back to committed polynomials and public inputs.
    ///
    /// The same naming conventions as [`CommittedPolynomial`] apply (`Ra` = read
    /// address, `RaD` = read-address decomposition, etc.).
    ///
    /// # Grouping
    ///
    /// | Group | Purpose |
    /// |-------|---------|
    /// | `NodeOutput` / `NodeOutputRa` | MLE of a node's output tensor and its read-address polynomial |
    /// | `{Cos,Erf,Sin,Tanh}Ra` | Read-address polynomials for activation-function lookups |
    /// | `Softmax*` | Intermediate polynomials arising in the softmax sub-protocol |
    /// | `HammingWeight` | Polynomial used in the Hamming-weight sumcheck |
    /// | `Div* / Sqrt* / Teleport*` | Advice-derived polynomials proven via `ReadRafSumcheckProver` from committed one-hot polynomials |
    #[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
    pub enum VirtualPolynomial {
        // ----- Node output -----
        /// Multilinear extension (MLE) of a node's output tensor.
        ///
        /// * `0` – producer node index
        NodeOutput(usize),

        /// Read-address polynomial for a node's output lookup table.
        ///
        /// * `0` – node index
        NodeOutputRa(usize),

        /// Sigmoid Ra polynomial
        SigmoidRa(usize),

        // ----- Activation-function read-address polynomials (node_index) -----
        /// Read-address polynomial for the **Cos** lookup table.
        ///
        /// * `0` – node index
        CosRa(usize),

        /// Read-address polynomial for the **Erf** lookup table.
        ///
        /// * `0` – node index
        ErfRa(usize),

        /// Read-address polynomial for the **Sin** lookup table.
        ///
        /// * `0` – node index
        SinRa(usize),

        /// Read-address polynomial for the **Tanh** lookup table.
        ///
        /// * `0` – node index
        TanhRa(usize),

        // ----- Softmax intermediate polynomials (node_index, feature_index) -----
        /// Running sum used inside softmax normalisation.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxSumOutput(usize, usize),

        /// Running max used inside softmax numerical stabilisation.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxMaxOutput(usize, usize),

        /// Index of the maximum logit (used for softmax stabilisation).
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxMaxIndex(usize, usize),

        // ----- Hamming weight -----
        /// Polynomial used in the Hamming-weight sumcheck (shared across all
        /// nodes; no parameters).
        HammingWeight,

        // ----- Advice-derived polynomials (proven via ReadRafSumcheckProver) -----

        /// Read-address polynomial derived from [`CommittedPolynomial::DivRangeCheckRaD`].
        ///
        /// * `0` – node index
        DivRangeCheckRa(usize),

        /// Read-address polynomial derived from [`CommittedPolynomial::SqrtRangeCheckRaD`].
        ///
        /// * `0` – node index
        SqrtRangeCheckRa(usize),

        /// Read-address polynomial derived from
        /// [`CommittedPolynomial::TeleportRangeCheckRaD`].
        ///
        /// * `0` – node index
        TeleportRangeCheckRa(usize),

        // ----- Division / square-root remainder & quotient polynomials -----
        /// Remainder polynomial for integer division advice.
        ///
        /// * `0` – node index
        DivRemainder(usize),

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

        /// Per-feature-vector sum of exponentiated logits: `exp_sum_q[k] = Σ_j exp_q[k,j]`.
        ///
        /// * `0` – node index
        SoftmaxExpSum(usize),

        /// The rv polynomial for the softmax exponentiation lookup.
        ///
        /// * `0` – node index
        SoftmaxExpQ(usize),

        /// The one-hot encoding of the softmax R poly
        ///
        /// * `0` – node index
        SoftmaxRemainderRa(usize),

        /// The rv polynomial for the softmax exponentiation lookup.
        ///
        /// * `0` – node index
        SoftmaxExpHi(usize),

        /// The rv polynomial for the softmax exponentiation lookup.
        ///
        /// * `0` – node index
        SoftmaxExpLo(usize),

        /// Remainder polynomial for the softmax exponentiation check in softmax.
        ///
        /// * `0` – node index
        SoftmaxExpRemainder(usize),

        /// One-hot-encoded Read-address polynomial for the softmax exp-remainder range check.
        ///
        /// * `0` – node index
        SoftmaxExpRemainderRa(usize),

        /// The raf polynomial for the softmax `exp_hi` sub-table lookup.
        ///
        /// * `0` – node index
        SoftmaxZHi(usize),

        /// The raf polynomial for the softmax `exp_lo` sub-table lookup.
        ///
        /// * `0` – node index
        SoftmaxZLo(usize),

        /// One-hot-encoded read-address polynomial for the softmax `exp_hi` sub-table lookup.
        ///
        /// * `0` – node index
        SoftmaxZHiRa(usize),

        /// One-hot-encoded read-address polynomial for the softmax `exp_lo` sub-table lookup.
        ///
        /// * `0` – node index
        SoftmaxZLoRa(usize),

        /// Saturation-diff polynomial for the softmax operand link.
        /// `sat_diff[k,j] = z[k,j] − z_c[k,j]` (≥ 0).
        /// Virtualized — the identity RC commits to its one-hot encoding.
        ///
        /// * `0` – node index
        SoftmaxSatDiff(usize),

        /// One-hot-encoded Read-address polynomial for the softmax sat-diff range check.
        ///
        /// * `0` – node index
        SoftmaxSatDiffRa(usize),

        /// Remainder polynomial for the reciprocal-multiplication check in
        /// **softmax**.
        ///
        /// * `0` – node index
        SoftmaxRecipMultRemainder(usize),
    }
}
