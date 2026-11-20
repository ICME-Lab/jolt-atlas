//! Protobuf layer: prost-generated ONNX types (checked in; regenerate with
//! `scripts/regen-onnx-proto.sh` against the vendored `proto/onnx.proto3`,
//! pinned to ONNX v1.20.0) plus tensor payload extraction with
//! external-data support.

#[allow(missing_docs, clippy::all)]
#[rustfmt::skip]
mod onnx;

pub use onnx::*;

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use prost::Message;

use crate::error::ImportError;

/// Decode a model file.
pub fn load_model(path: &Path) -> Result<ModelProto, ImportError> {
    let bytes = std::fs::read(path).map_err(|source| ImportError::Io {
        path: path.display().to_string(),
        source,
    })?;
    ModelProto::decode(bytes.as_slice()).map_err(|source| ImportError::Decode {
        path: path.display().to_string(),
        source,
    })
}

/// The raw little-endian payload of a tensor, resolving the ONNX
/// external-data convention (weights in side files, resolved relative to the
/// model's directory; absolute paths and `..` traversal are rejected).
pub fn tensor_payload(tensor: &TensorProto, model_dir: &Path) -> Result<Vec<u8>, ImportError> {
    let external = tensor.data_location == tensor_proto::DataLocation::External as i32;
    if !external {
        return Ok(tensor.raw_data.to_vec());
    }

    let mut location: Option<&str> = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;
    for kv in &tensor.external_data {
        match kv.key.as_str() {
            "location" => location = Some(&kv.value),
            "offset" => {
                offset = kv.value.parse().map_err(|_| ImportError::ExternalData {
                    tensor: tensor.name.clone(),
                    reason: format!("bad offset {:?}", kv.value),
                })?
            }
            "length" => {
                length = Some(kv.value.parse().map_err(|_| ImportError::ExternalData {
                    tensor: tensor.name.clone(),
                    reason: format!("bad length {:?}", kv.value),
                })?)
            }
            // `checksum` and unknown keys are ignored, per convention.
            _ => {}
        }
    }
    let location = location.ok_or_else(|| ImportError::ExternalData {
        tensor: tensor.name.clone(),
        reason: "missing location".into(),
    })?;

    let rel = PathBuf::from(location);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(ImportError::ExternalData {
            tensor: tensor.name.clone(),
            reason: format!("unsafe path {location:?}"),
        });
    }
    let full = model_dir.join(rel);
    let mut file = std::fs::File::open(&full).map_err(|source| ImportError::Io {
        path: full.display().to_string(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ImportError::Io {
            path: full.display().to_string(),
            source,
        })?;
    // Read via a length-capped stream instead of pre-allocating a buffer of
    // the declared size: `length` is untrusted metadata, and allocating it
    // up front would let a tiny model file demand arbitrary memory.
    let mut buf = Vec::new();
    match length {
        Some(len) => {
            file.by_ref()
                .take(len)
                .read_to_end(&mut buf)
                .map_err(|source| ImportError::Io {
                    path: full.display().to_string(),
                    source,
                })?;
            if buf.len() as u64 != len {
                return Err(ImportError::ExternalData {
                    tensor: tensor.name.clone(),
                    reason: format!("{full:?} holds {} of {len} bytes", buf.len()),
                });
            }
        }
        None => {
            file.read_to_end(&mut buf)
                .map_err(|source| ImportError::Io {
                    path: full.display().to_string(),
                    source,
                })?;
        }
    }
    Ok(buf)
}
