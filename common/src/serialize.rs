//! Canonical binary encoding of proofs, commitments, setups and transcript messages.
//!
//! One encoding, byte for byte the arkworks `CanonicalSerialize` compressed encoding the proof
//! format was defined with: integers little-endian at their width (`usize` as `u64`), `bool` as
//! one byte, a `Vec` as its `u64` length then its elements, arrays and tuples as their elements,
//! an `Option` as a `bool` tag then the value. The methods are `encode`/`encoded_len`/`decode`, named
//! apart from serde's (every field implements both). Decoding always validates and returns an
//! [`Error`] on short, trailing-free-but-malformed or non-canonical input; it never panics, and a
//! decoded length never allocates more than the elements actually read need.

use std::io::{Read, Write};
use std::marker::PhantomData;

/// Why bytes did not decode (or an encoding could not be written).
#[derive(Debug)]
pub enum Error {
    /// The bytes are not a canonical encoding of the type.
    InvalidData,
    /// The reader or writer failed (for a slice reader: the input ended early).
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidData => write!(f, "invalid data"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// A type with one canonical binary encoding.
pub trait Canonical: Sized {
    /// Writes the encoding of `self`.
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error>;

    /// The number of bytes [`encode`](Self::encode) writes.
    fn encoded_len(&self) -> usize;

    /// Reads one encoded value, rejecting anything that is not a canonical encoding.
    fn decode<R: Read>(reader: R) -> Result<Self, Error>;
}

macro_rules! impl_int {
    ($($t:ty),*) => {$(
        impl Canonical for $t {
            #[inline]
            fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
                Ok(writer.write_all(&self.to_le_bytes())?)
            }

            #[inline]
            fn encoded_len(&self) -> usize {
                std::mem::size_of::<$t>()
            }

            #[inline]
            fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
                let mut bytes = [0u8; std::mem::size_of::<$t>()];
                reader.read_exact(&mut bytes)?;
                Ok(<$t>::from_le_bytes(bytes))
            }
        }
    )*};
}

impl_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl Canonical for usize {
    #[inline]
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        (*self as u64).encode(writer)
    }

    #[inline]
    fn encoded_len(&self) -> usize {
        8
    }

    #[inline]
    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        usize::try_from(u64::decode(reader)?).map_err(|_| Error::InvalidData)
    }
}

impl Canonical for bool {
    #[inline]
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        (*self as u8).encode(writer)
    }

    #[inline]
    fn encoded_len(&self) -> usize {
        1
    }

    #[inline]
    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        match u8::decode(reader)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::InvalidData),
        }
    }
}

impl Canonical for () {
    fn encode<W: Write>(&self, _writer: W) -> Result<(), Error> {
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        0
    }

    fn decode<R: Read>(_reader: R) -> Result<Self, Error> {
        Ok(())
    }
}

impl<T> Canonical for PhantomData<T> {
    fn encode<W: Write>(&self, _writer: W) -> Result<(), Error> {
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        0
    }

    fn decode<R: Read>(_reader: R) -> Result<Self, Error> {
        Ok(PhantomData)
    }
}

/// The most elements a decoded length pre-allocates: a length is prover-supplied, so a few forged
/// bytes must not request an arbitrarily large allocation before any element is read.
const PREALLOCATE_LIMIT: usize = 1 << 10;

impl<T: Canonical> Canonical for Vec<T> {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        self.len().encode(&mut writer)?;
        for x in self {
            x.encode(&mut writer)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        8 + self.iter().map(Canonical::encoded_len).sum::<usize>()
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let len = usize::decode(&mut reader)?;
        let mut values = Vec::with_capacity(len.min(PREALLOCATE_LIMIT));
        for _ in 0..len {
            values.push(T::decode(&mut reader)?);
        }
        Ok(values)
    }
}

impl Canonical for String {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        self.len().encode(&mut writer)?;
        Ok(writer.write_all(self.as_bytes())?)
    }

    fn encoded_len(&self) -> usize {
        8 + self.len()
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        String::from_utf8(Vec::<u8>::decode(reader)?).map_err(|_| Error::InvalidData)
    }
}

impl<T: Canonical, const N: usize> Canonical for [T; N] {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        for x in self {
            x.encode(&mut writer)?;
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.iter().map(Canonical::encoded_len).sum()
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let values = (0..N)
            .map(|_| T::decode(&mut reader))
            .collect::<Result<Vec<_>, _>>()?;
        values.try_into().map_err(|_| Error::InvalidData)
    }
}

impl<T: Canonical> Canonical for Option<T> {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        self.is_some().encode(&mut writer)?;
        match self {
            Some(x) => x.encode(writer),
            None => Ok(()),
        }
    }

    fn encoded_len(&self) -> usize {
        1 + self.as_ref().map_or(0, Canonical::encoded_len)
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        if bool::decode(&mut reader)? {
            Ok(Some(T::decode(reader)?))
        } else {
            Ok(None)
        }
    }
}

macro_rules! impl_tuple {
    ($($name:ident),+) => {
        impl<$($name: Canonical),+> Canonical for ($($name,)+) {
            #[allow(non_snake_case)]
            fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
                let ($($name,)+) = self;
                $($name.encode(&mut writer)?;)+
                Ok(())
            }

            #[allow(non_snake_case)]
            fn encoded_len(&self) -> usize {
                let ($($name,)+) = self;
                0 $(+ $name.encoded_len())+
            }

            fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
                Ok(($($name::decode(&mut reader)?,)+))
            }
        }
    };
}

impl_tuple!(A);
impl_tuple!(A, B);
impl_tuple!(A, B, C);
impl_tuple!(A, B, C, D);

/// Writes a slice in the `Vec` layout (its `u64` length, then the elements), for fields held as
/// slices.
pub fn encode_slice<T: Canonical, W: Write>(values: &[T], mut writer: W) -> Result<(), Error> {
    values.len().encode(&mut writer)?;
    for x in values {
        x.encode(&mut writer)?;
    }
    Ok(())
}

/// The number of bytes [`encode_slice`] writes.
pub fn slice_encoded_len<T: Canonical>(values: &[T]) -> usize {
    8 + values.iter().map(Canonical::encoded_len).sum::<usize>()
}

/// Implements [`Canonical`] for a struct as its fields' encodings in the order listed (the
/// declaration order, which is the layout the former derive wrote):
/// `impl_canonical_struct!([F: JoltField] Proof<F> { a, b });`.
#[macro_export]
macro_rules! impl_canonical_struct {
    ([$($generics:tt)*] $ty:ty { $($field:ident),* $(,)? }) => {
        impl<$($generics)*> $crate::serialize::Canonical for $ty {
            fn encode<W: ::std::io::Write>(
                &self,
                mut writer: W,
            ) -> ::core::result::Result<(), $crate::serialize::Error> {
                $($crate::serialize::Canonical::encode(&self.$field, &mut writer)?;)*
                let _ = &mut writer;
                ::core::result::Result::Ok(())
            }

            fn encoded_len(&self) -> usize {
                0 $(+ $crate::serialize::Canonical::encoded_len(&self.$field))*
            }

            fn decode<R: ::std::io::Read>(
                mut reader: R,
            ) -> ::core::result::Result<Self, $crate::serialize::Error> {
                let _ = &mut reader;
                ::core::result::Result::Ok(Self {
                    $($field: $crate::serialize::Canonical::decode(&mut reader)?,)*
                })
            }
        }
    };
}

/// Encodes `value` into a fresh buffer.
pub fn to_bytes<T: Canonical>(value: &T) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::with_capacity(value.encoded_len());
    value.encode(&mut bytes)?;
    Ok(bytes)
}

/// Decodes exactly one value from `bytes`, rejecting trailing bytes.
pub fn from_bytes<T: Canonical>(mut bytes: &[u8]) -> Result<T, Error> {
    let value = T::decode(&mut bytes)?;
    if bytes.is_empty() {
        Ok(value)
    } else {
        Err(Error::InvalidData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T: Canonical + PartialEq + std::fmt::Debug>(value: T, expected: &[u8]) {
        let bytes = to_bytes(&value).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(value.encoded_len(), expected.len());
        assert_eq!(from_bytes::<T>(&bytes).unwrap(), value);
    }

    /// The layouts the arkworks compressed encoding writes, which the proof format is defined by.
    #[test]
    fn layouts_match_the_proof_format() {
        round_trip(0x0102_0304u32, &[4, 3, 2, 1]);
        round_trip(7usize, &[7, 0, 0, 0, 0, 0, 0, 0]);
        round_trip(true, &[1]);
        round_trip(vec![1u8, 2], &[2, 0, 0, 0, 0, 0, 0, 0, 1, 2]);
        round_trip([5u16, 6], &[5, 0, 6, 0]);
        round_trip((1u8, 2u8), &[1, 2]);
        round_trip(Some(3u8), &[1, 3]);
        round_trip(None::<u8>, &[0]);
        round_trip(String::from("ab"), &[2, 0, 0, 0, 0, 0, 0, 0, b'a', b'b']);
    }

    #[test]
    fn malformed_input_is_rejected() {
        assert!(from_bytes::<bool>(&[2]).is_err());
        assert!(from_bytes::<u32>(&[1, 2]).is_err());
        assert!(from_bytes::<u8>(&[1, 2]).is_err(), "trailing bytes");
        assert!(from_bytes::<Option<u8>>(&[2, 0]).is_err());
        assert!(from_bytes::<String>(&[1, 0, 0, 0, 0, 0, 0, 0, 0xff]).is_err());
    }

    /// A length of 2⁶⁴ − 1 with nothing behind it fails at the end of the input instead of
    /// requesting that capacity up front.
    #[test]
    fn forged_length_fails_without_allocating_it() {
        let bytes = u64::MAX.to_le_bytes();
        assert!(from_bytes::<Vec<u128>>(&bytes).is_err());
        assert!(from_bytes::<Vec<u8>>(&bytes).is_err());
    }
}
