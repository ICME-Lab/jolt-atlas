#[macro_export]
/// Generates a [`Canonical`](crate::serialize::Canonical) impl for an enum whose variants are
/// unit or carry only `Canonical` fields (up to 3 fields per variant): a `u8` tag, then the
/// fields.
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

        // 2) Canonical
        impl $crate::serialize::Canonical for $name {
            fn encode<W: ::std::io::Write>(
                &self,
                mut writer: W,
            ) -> ::core::result::Result<(), $crate::serialize::Error> {
                let mut _tag: u8 = 0;
                $(
                    canonical_serde_enum!(
                        @try_serialize self, _tag, writer,
                        $name, $variant $( ( $($fty),+ ) )?
                    );
                    #[allow(unused_assignments)]
                    { _tag += 1; }
                )+
                unreachable!()
            }

            fn encoded_len(&self) -> usize {
                $(
                    canonical_serde_enum!(
                        @try_size self,
                        $name, $variant $( ( $($fty),+ ) )?
                    );
                )+
                unreachable!()
            }

            fn decode<R: ::std::io::Read>(
                mut reader: R,
            ) -> ::core::result::Result<Self, $crate::serialize::Error> {
                let tag = <u8 as $crate::serialize::Canonical>::decode(&mut reader)?;
                let mut _expected: u8 = 0;
                $(
                    if tag == _expected {
                        return canonical_serde_enum!(
                            @deserialize reader,
                            $name, $variant $( ( $($fty),+ ) )?
                        );
                    }
                    #[allow(unused_assignments)]
                    { _expected += 1; }
                )+
                ::core::result::Result::Err($crate::serialize::Error::InvalidData)
            }
        }
    };

    // ── @try_serialize helpers (per field-count) ─────────────────────

    // unit
    (@try_serialize $self:ident, $tag:ident, $writer:ident,
     $name:ident, $variant:ident
    ) => {
        if let $name::$variant = $self {
            $crate::serialize::Canonical::encode(&$tag, &mut $writer)?;
            return ::core::result::Result::Ok(());
        }
    };

    // 1 field
    (@try_serialize $self:ident, $tag:ident, $writer:ident,
     $name:ident, $variant:ident ($fty0:ty)
    ) => {
        if let $name::$variant(_f0) = $self {
            $crate::serialize::Canonical::encode(&$tag, &mut $writer)?;
            $crate::serialize::Canonical::encode(_f0, &mut $writer)?;
            return ::core::result::Result::Ok(());
        }
    };

    // 2 fields
    (@try_serialize $self:ident, $tag:ident, $writer:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty)
    ) => {
        if let $name::$variant(_f0, _f1) = $self {
            $crate::serialize::Canonical::encode(&$tag, &mut $writer)?;
            $crate::serialize::Canonical::encode(_f0, &mut $writer)?;
            $crate::serialize::Canonical::encode(_f1, &mut $writer)?;
            return ::core::result::Result::Ok(());
        }
    };

    // 3 fields
    (@try_serialize $self:ident, $tag:ident, $writer:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty, $fty2:ty)
    ) => {
        if let $name::$variant(_f0, _f1, _f2) = $self {
            $crate::serialize::Canonical::encode(&$tag, &mut $writer)?;
            $crate::serialize::Canonical::encode(_f0, &mut $writer)?;
            $crate::serialize::Canonical::encode(_f1, &mut $writer)?;
            $crate::serialize::Canonical::encode(_f2, &mut $writer)?;
            return ::core::result::Result::Ok(());
        }
    };

    // ── @try_size helpers (per field-count) ──────────────────────────

    // unit
    (@try_size $self:ident,
     $name:ident, $variant:ident
    ) => {
        if let $name::$variant = $self {
            return $crate::serialize::Canonical::encoded_len(&0u8);
        }
    };

    // 1 field
    (@try_size $self:ident,
     $name:ident, $variant:ident ($fty0:ty)
    ) => {
        if let $name::$variant(_f0) = $self {
            return $crate::serialize::Canonical::encoded_len(&0u8)
                 + $crate::serialize::Canonical::encoded_len(_f0);
        }
    };

    // 2 fields
    (@try_size $self:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty)
    ) => {
        if let $name::$variant(_f0, _f1) = $self {
            return $crate::serialize::Canonical::encoded_len(&0u8)
                 + $crate::serialize::Canonical::encoded_len(_f0)
                 + $crate::serialize::Canonical::encoded_len(_f1);
        }
    };

    // 3 fields
    (@try_size $self:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty, $fty2:ty)
    ) => {
        if let $name::$variant(_f0, _f1, _f2) = $self {
            return $crate::serialize::Canonical::encoded_len(&0u8)
                 + $crate::serialize::Canonical::encoded_len(_f0)
                 + $crate::serialize::Canonical::encoded_len(_f1)
                 + $crate::serialize::Canonical::encoded_len(_f2);
        }
    };

    // ── @deserialize helpers (per field-count) ──────────────────────

    // unit
    (@deserialize $reader:ident,
     $name:ident, $variant:ident
    ) => {
        ::core::result::Result::Ok($name::$variant)
    };

    // 1 field
    (@deserialize $reader:ident,
     $name:ident, $variant:ident ($fty0:ty)
    ) => {
        ::core::result::Result::Ok($name::$variant(
            <$fty0 as $crate::serialize::Canonical>::decode(&mut $reader)?,
        ))
    };

    // 2 fields
    (@deserialize $reader:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty)
    ) => {
        ::core::result::Result::Ok($name::$variant(
            <$fty0 as $crate::serialize::Canonical>::decode(&mut $reader)?,
            <$fty1 as $crate::serialize::Canonical>::decode(&mut $reader)?,
        ))
    };

    // 3 fields
    (@deserialize $reader:ident,
     $name:ident, $variant:ident ($fty0:ty, $fty1:ty, $fty2:ty)
    ) => {
        ::core::result::Result::Ok($name::$variant(
            <$fty0 as $crate::serialize::Canonical>::decode(&mut $reader)?,
            <$fty1 as $crate::serialize::Canonical>::decode(&mut $reader)?,
            <$fty2 as $crate::serialize::Canonical>::decode(&mut $reader)?,
        ))
    };
}
