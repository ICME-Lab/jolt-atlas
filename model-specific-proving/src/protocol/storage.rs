use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(super) struct Codes {
    len: usize,
    width: usize,
    #[serde(with = "super::storage::bytes")]
    bytes: Vec<u8>,
}

impl Codes {
    pub(super) fn pack(
        len: usize,
        width: usize,
        values: impl Iterator<Item = u128>,
    ) -> Option<Self> {
        let size = len.checked_mul(width)?.div_ceil(8);
        let mut bytes = Vec::with_capacity(size.checked_add(16)?);
        let mut word = 0u128;
        let mut used = 0;
        for value in values {
            if value >> width != 0 {
                return None;
            }
            word |= value << used;
            used += width;
            while used >= 64 {
                bytes.extend_from_slice(&(word as u64).to_le_bytes());
                used -= 64;
                word = if used == 0 {
                    0
                } else {
                    value >> (width - used)
                };
            }
        }
        if used != 0 {
            bytes.extend_from_slice(&word.to_le_bytes()[..used.div_ceil(8)]);
        }
        debug_assert_eq!(bytes.len(), size);
        bytes.resize(size + 16, 0);
        Some(Self { len, width, bytes })
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn get(&self, i: usize) -> u128 {
        let bit = i * self.width;
        let start = bit / 8;
        let mut value =
            u128::from_le_bytes(self.bytes[start..start + 16].try_into().unwrap()) >> (bit % 8);
        if self.width + bit % 8 > 128 {
            value |= u128::from(self.bytes[start + 16]) << (128 - bit % 8);
        }
        value & ((1u128 << self.width) - 1)
    }
}

#[derive(Serialize, Deserialize)]
struct Integers {
    minimum: i128,
    codes: Codes,
}

impl Integers {
    fn pack<T: Copy + Into<i128>>(values: &[T]) -> Self {
        let (minimum, maximum) = values
            .iter()
            .map(|&x| x.into())
            .fold((0, 0), |(low, high), x| (low.min(x), high.max(x)));
        let width = 128 - ((maximum - minimum) as u128).leading_zeros() as usize;
        let codes = Codes::pack(
            values.len(),
            width,
            values.iter().map(|&v| (v.into() - minimum) as u128),
        )
        .expect("integer width computed from the values");
        Self { minimum, codes }
    }

    fn unpack<T: TryFrom<i128>, E: serde::de::Error>(&self) -> Result<Vec<T>, E> {
        (0..self.codes.len())
            .map(|i| {
                T::try_from(self.minimum + self.codes.get(i) as i128)
                    .map_err(|_| E::custom("invalid compressed integer"))
            })
            .collect()
    }
}

macro_rules! column {
    ($module:ident, $type:ty) => {
        pub(crate) mod $module {
            use super::*;
            pub(crate) fn serialize<S: serde::Serializer>(
                values: &[$type],
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                Integers::pack(values).serialize(serializer)
            }
            pub(crate) fn deserialize<'a, D: serde::Deserializer<'a>>(
                deserializer: D,
            ) -> Result<Vec<$type>, D::Error> {
                Integers::deserialize(deserializer)?.unpack()
            }
        }
    };
}

column!(i32s, i32);
column!(i64s, i64);
column!(u16s, u16);
column!(u32s, u32);

pub(crate) mod two_i32 {
    use super::*;
    pub(crate) fn serialize<S: serde::Serializer>(
        values: &[Vec<i32>; 2],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        values
            .each_ref()
            .map(|v| Integers::pack(v))
            .serialize(serializer)
    }
    pub(crate) fn deserialize<'a, D: serde::Deserializer<'a>>(
        deserializer: D,
    ) -> Result<[Vec<i32>; 2], D::Error> {
        let [a, b] = <[Integers; 2]>::deserialize(deserializer)?;
        Ok([a.unpack()?, b.unpack()?])
    }
}

pub(crate) mod optional_i64 {
    use super::*;
    pub(crate) fn serialize<S: serde::Serializer>(
        values: &[Option<Vec<i64>>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(values.iter().map(|v| v.as_ref().map(|v| Integers::pack(v))))
    }
    pub(crate) fn deserialize<'a, D: serde::Deserializer<'a>>(
        deserializer: D,
    ) -> Result<Vec<Option<Vec<i64>>>, D::Error> {
        Vec::<Option<Integers>>::deserialize(deserializer)?
            .iter()
            .map(|v| v.as_ref().map(Integers::unpack).transpose())
            .collect()
    }
}

pub(crate) mod bytes {
    use serde::de::{Error, Visitor};

    pub(crate) fn serialize<S: serde::Serializer>(
        value: &[u8],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(value)
    }

    pub(crate) fn deserialize<'a, D: serde::Deserializer<'a>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        struct Bytes;
        impl<'a> Visitor<'a> for Bytes {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a byte buffer")
            }
            fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
                Ok(value.to_vec())
            }
            fn visit_byte_buf<E: Error>(self, value: Vec<u8>) -> Result<Self::Value, E> {
                Ok(value)
            }
        }
        deserializer.deserialize_byte_buf(Bytes)
    }
}
