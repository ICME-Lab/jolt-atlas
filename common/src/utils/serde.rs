#[macro_export]
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
