use allocative::Allocative;

pub mod consts;
pub mod utils;

/// Generates `CanonicalSerialize`, `CanonicalDeserialize`, and `Valid` impls
/// for an enum whose variants are unit or carry only `CanonicalSerialize` /
/// `CanonicalDeserialize` fields (up to 3 fields per variant).
///
/// Tags are assigned automatically in **declaration order** (0, 1, 2, …).
/// Adding a new variant at the end is a one-line change with no additional
/// boilerplate.
///
/// # Example
/// ```ignore
/// canonical_serde_enum! {
///     #[derive(Debug, Clone)]
///     pub enum MyPoly {
///         Foo(usize),
///         Bar(usize, usize),
///         Baz,
///     }
/// }
/// ```
macro_rules! canonical_serde_enum {
    // ── entry point ───────────────────────────────────────────────────
    (
        $(#[$enum_meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$var_meta:meta])*
                $variant:ident $( ( $($fty:ty),+ $(,)? ) )?
            ),+ $(,)?
        }
    ) => {
        // 1) Emit the enum itself, forwarding all attributes & doc-comments.
        $(#[$enum_meta])*
        $vis enum $name {
            $(
                $(#[$var_meta])*
                $variant $( ( $($fty),+ ) )?,
            )+
        }

        // 2) CanonicalSerialize
        impl ::ark_serialize::CanonicalSerialize for $name {
            fn serialize_with_mode<W: ::ark_serialize::Write>(
                &self,
                mut writer: W,
                compress: ::ark_serialize::Compress,
            ) -> ::core::result::Result<(), ::ark_serialize::SerializationError> {
                let mut _tag: u8 = 0;
                $(
                    canonical_serde_enum!(
                        @try_serialize self, _tag, writer, compress,
                        $name, $variant $( ( $($fty),+ ) )?
                    );
                    #[allow(unused_assignments)]
                    { _tag += 1; }
                )+
                unreachable!()
            }

            fn serialized_size(
                &self,
                compress: ::ark_serialize::Compress,
            ) -> usize {
                $(
                    canonical_serde_enum!(
                        @try_size self, compress,
                        $name, $variant $( ( $($fty),+ ) )?
                    );
                )+
                unreachable!()
            }
        }

        // 3) Valid (trivial – nothing extra to check)
        impl ::ark_serialize::Valid for $name {
            fn check(
                &self,
            ) -> ::core::result::Result<(), ::ark_serialize::SerializationError> {
                ::core::result::Result::Ok(())
            }
        }

        // 4) CanonicalDeserialize
        impl ::ark_serialize::CanonicalDeserialize for $name {
            fn deserialize_with_mode<R: ::ark_serialize::Read>(
                mut reader: R,
                compress: ::ark_serialize::Compress,
                validate: ::ark_serialize::Validate,
            ) -> ::core::result::Result<Self, ::ark_serialize::SerializationError> {
                let tag =
                    <u8 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                        &mut reader, compress, validate,
                    )?;
                let mut _expected: u8 = 0;
                $(
                    if tag == _expected {
                        return canonical_serde_enum!(
                            @deserialize reader, compress, validate,
                            $name, $variant $( ( $($fty),+ ) )?
                        );
                    }
                    #[allow(unused_assignments)]
                    { _expected += 1; }
                )+
                ::core::result::Result::Err(
                    ::ark_serialize::SerializationError::InvalidData,
                )
            }
        }
    };

    // ── @try_serialize helpers (per field-count) ─────────────────────

    // unit
    (@try_serialize $self:ident, $tag:ident, $writer:ident, $compress:ident,
     $name:ident, $variant:ident
    ) => {
        if let $name::$variant = $self {
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                &$tag, &mut $writer, $compress,
            )?;
            return ::core::result::Result::Ok(());
        }
    };

    // 1 field
    (@try_serialize $self:ident, $tag:ident, $writer:ident, $compress:ident,
     $name:ident, $variant:ident ($fty0:ty)
    ) => {
        if let $name::$variant(_f0) = $self {
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                &$tag, &mut $writer, $compress,
            )?;
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                _f0, &mut $writer, $compress,
            )?;
            return ::core::result::Result::Ok(());
        }
    };

    // 2 fields
    (@try_serialize $self:ident, $tag:ident, $writer:ident, $compress:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty)
    ) => {
        if let $name::$variant(_f0, _f1) = $self {
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                &$tag, &mut $writer, $compress,
            )?;
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                _f0, &mut $writer, $compress,
            )?;
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                _f1, &mut $writer, $compress,
            )?;
            return ::core::result::Result::Ok(());
        }
    };

    // 3 fields
    (@try_serialize $self:ident, $tag:ident, $writer:ident, $compress:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty, $fty2:ty)
    ) => {
        if let $name::$variant(_f0, _f1, _f2) = $self {
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                &$tag, &mut $writer, $compress,
            )?;
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                _f0, &mut $writer, $compress,
            )?;
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                _f1, &mut $writer, $compress,
            )?;
            ::ark_serialize::CanonicalSerialize::serialize_with_mode(
                _f2, &mut $writer, $compress,
            )?;
            return ::core::result::Result::Ok(());
        }
    };

    // ── @try_size helpers (per field-count) ──────────────────────────

    // unit
    (@try_size $self:ident, $compress:ident,
     $name:ident, $variant:ident
    ) => {
        if let $name::$variant = $self {
            return ::ark_serialize::CanonicalSerialize::serialized_size(&0u8, $compress);
        }
    };

    // 1 field
    (@try_size $self:ident, $compress:ident,
     $name:ident, $variant:ident ($fty0:ty)
    ) => {
        if let $name::$variant(_f0) = $self {
            return ::ark_serialize::CanonicalSerialize::serialized_size(&0u8, $compress)
                 + ::ark_serialize::CanonicalSerialize::serialized_size(_f0, $compress);
        }
    };

    // 2 fields
    (@try_size $self:ident, $compress:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty)
    ) => {
        if let $name::$variant(_f0, _f1) = $self {
            return ::ark_serialize::CanonicalSerialize::serialized_size(&0u8, $compress)
                 + ::ark_serialize::CanonicalSerialize::serialized_size(_f0, $compress)
                 + ::ark_serialize::CanonicalSerialize::serialized_size(_f1, $compress);
        }
    };

    // 3 fields
    (@try_size $self:ident, $compress:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty, $fty2:ty)
    ) => {
        if let $name::$variant(_f0, _f1, _f2) = $self {
            return ::ark_serialize::CanonicalSerialize::serialized_size(&0u8, $compress)
                 + ::ark_serialize::CanonicalSerialize::serialized_size(_f0, $compress)
                 + ::ark_serialize::CanonicalSerialize::serialized_size(_f1, $compress)
                 + ::ark_serialize::CanonicalSerialize::serialized_size(_f2, $compress);
        }
    };

    // ── @deserialize helpers (per field-count) ──────────────────────

    // unit
    (@deserialize $reader:ident, $compress:ident, $validate:ident,
     $name:ident, $variant:ident
    ) => {
        ::core::result::Result::Ok($name::$variant)
    };

    // 1 field
    (@deserialize $reader:ident, $compress:ident, $validate:ident,
     $name:ident, $variant:ident ($fty0:ty)
    ) => {
        ::core::result::Result::Ok($name::$variant(
            <$fty0 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                &mut $reader, $compress, $validate,
            )?,
        ))
    };

    // 2 fields
    (@deserialize $reader:ident, $compress:ident, $validate:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty)
    ) => {
        ::core::result::Result::Ok($name::$variant(
            <$fty0 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                &mut $reader, $compress, $validate,
            )?,
            <$fty1 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                &mut $reader, $compress, $validate,
            )?,
        ))
    };

    // 3 fields
    (@deserialize $reader:ident, $compress:ident, $validate:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty, $fty2:ty)
    ) => {
        ::core::result::Result::Ok($name::$variant(
            <$fty0 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                &mut $reader, $compress, $validate,
            )?,
            <$fty1 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                &mut $reader, $compress, $validate,
            )?,
            <$fty2 as ::ark_serialize::CanonicalDeserialize>::deserialize_with_mode(
                &mut $reader, $compress, $validate,
            )?,
        ))
    };
}

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

        // ----- Softmax-specific committed polynomials -----
        /// Remainder polynomial used in the softmax reciprocal division check.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxRemainder(usize, usize),

        /// One-hot read-address decomposition for the softmax exponentiation
        /// lookup table.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        /// * `2` – decomposition index `d`
        SoftmaxExponentiationRaD(usize, usize, usize),

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

        /// Advice polynomial for the **reciprocal square root** result in Rsqrt.
        ///
        /// * `0` – node index
        RsqrtNodeRsqrt(usize),

        /// Read-address polynomial for the **Gather** operator.
        ///
        /// * `0` – node index
        GatherRa(usize),

        /// Remainder polynomial for the reciprocal-multiplication check in
        /// **softmax**.
        ///
        /// * `0` – node index
        SoftmaxRecipMultRemainder(usize),

        /// Diff polynomial for the inv-sum remainder range check in **softmax**.
        /// `diff[k] = exp_sum_q[k] − 1 − r_inv[k]`, proved to lie in `[0, N·S)`.
        ///
        /// * `0` – node index
        SoftmaxInvSumDiff(usize),

        /// Remainder polynomial for the softmax exponentiation check in softmax.
        ///
        /// * `0` – node index
        SoftmaxExpRemainder(usize),
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
        /// Per-feature output of softmax.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxFeatureOutput(usize, usize),

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

        /// Exponentiation intermediate in softmax (`exp(x - max)`).
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxExponentiationOutput(usize, usize),

        /// Raw input logits fed into softmax.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxInputLogitsOutput(usize, usize),

        /// `|logit - max|` (absolute centred logits) used for exponentiation
        /// lookup bounds.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxAbsCenteredLogitsOutput(usize, usize),

        /// Read-address polynomial for the softmax exponentiation lookup table.
        ///
        /// * `0` – node index
        /// * `1` – feature index
        SoftmaxExponentiationRa(usize, usize),

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

        /// Per-feature-vector reciprocal of the exp sum: `inv_sum[k] = ⌊S² / exp_sum_q[k]⌋`.
        /// Used in the reciprocal-multiply optimization to replace variable-divisor
        /// division with a constant-divisor multiplication.
        ///
        /// * `0` – node index
        SoftmaxInvSum(usize),
        SoftmaxExpSum(usize),
        SoftmaxRInv(usize),

        /// The rv polynomial for the softmax exponentiation lookup.
        ///
        /// * `0` – node index
        SoftmaxExpQ(usize),

        /// The one-hot encoding of the softmax R poly
        ///
        /// * `0` – node index
        SoftmaxRemainderRa(usize),

        /// Read-address polynomial for the softmax inv-sum diff range check.
        ///
        /// * `0` – node index
        SoftmaxInvSumDiffRa(usize),

        /// The rv polynomial for the softmax exponentiation lookup.
        ///
        /// * `0` – node index
        SoftmaxExpHi(usize),

        /// The rv polynomial for the softmax exponentiation lookup.
        ///
        /// * `0` – node index
        SoftmaxExpLo(usize),

        /// Read-address polynomial for the softmax exp-remainder range check.
        ///
        /// * `0` – node index
        SoftmaxExpRemainderRa(usize),
    }
}
