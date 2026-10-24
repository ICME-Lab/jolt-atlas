#!/usr/bin/env bash
# Regenerate src/proto/onnx.rs from the vendored proto (requires protoc).
# The vendored proto is pinned to ONNX v1.20.0:
#   https://raw.githubusercontent.com/onnx/onnx/v1.20.0/onnx/onnx.proto3
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cargo new --bin "$tmp/protogen" >/dev/null
cat > "$tmp/protogen/Cargo.toml" <<'TOML'
[package]
name = "protogen"
version = "0.1.0"
edition = "2021"
[dependencies]
prost-build = "0.13"
TOML
cat > "$tmp/protogen/src/main.rs" <<'RS'
fn main() {
    let mut config = prost_build::Config::new();
    config.out_dir(std::env::args().nth(1).unwrap());
    config.bytes(["."]);
    config
        .compile_protos(&[std::env::args().nth(2).unwrap()], &[std::env::args().nth(3).unwrap()])
        .unwrap();
}
RS
here=$(pwd)
(cd "$tmp/protogen" && cargo run --quiet -- "$here/src/proto" "$here/proto/onnx.proto3" "$here/proto")
echo "regenerated src/proto/onnx.rs"
