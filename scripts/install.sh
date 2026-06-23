#!/usr/bin/env sh
set -eu

REPO="ICME-Lab/jolt-atlas"
BIN="jolt-atlas"
VERSION="${JOLT_ATLAS_VERSION:-latest}"
INSTALL_DIR="${JOLT_ATLAS_INSTALL_DIR:-$HOME/.local/bin}"

case "$(uname -s)" in
  Linux) os="linux" ;;
  Darwin) os="macos" ;;
  MINGW*|MSYS*|CYGWIN*) os="windows" ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) arch="x86_64" ;;
  arm64|aarch64) arch="aarch64" ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [ "$VERSION" = "latest" ]; then
  url="https://github.com/$REPO/releases/latest/download/${BIN}-${os}-${arch}.tar.gz"
else
  url="https://github.com/$REPO/releases/download/$VERSION/${BIN}-${os}-${arch}.tar.gz"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$INSTALL_DIR"

echo "Downloading $url"
curl -fsSL "$url" | tar -xz -C "$tmp"
install -m 0755 "$tmp/$BIN" "$INSTALL_DIR/$BIN"
echo "Installed $BIN to $INSTALL_DIR/$BIN"
