#!/usr/bin/env sh
set -eu

REPO="ICME-Lab/jolt-atlas"
BIN="jolt-atlas"
VERSION="${JOLT_ATLAS_VERSION:-latest}"
INSTALL_DIR="${JOLT_ATLAS_INSTALL_DIR:-$HOME/.local/bin}"

# Name of the executable inside the archive. Overridden to jolt-atlas.exe on Windows.
BIN_FILE="$BIN"

case "$(uname -s)" in
  Linux) os="linux" ;;
  Darwin) os="macos" ;;
  MINGW*|MSYS*|CYGWIN*) os="windows"; BIN_FILE="${BIN}.exe" ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) arch="x86_64" ;;
  arm64|aarch64) arch="aarch64" ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

# Guard against (os, arch) combinations the release workflow does not build,
# so users get a clear message instead of a confusing 404 from the download.
supported="linux-x86_64 macos-x86_64 macos-aarch64 windows-x86_64"
case " $supported " in
  *" ${os}-${arch} "*) ;;
  *)
    echo "no prebuilt binary for ${os}-${arch}." >&2
    echo "Prebuilt targets: ${supported}." >&2
    echo "Build from source with: cargo install --git https://github.com/$REPO jolt-atlas" >&2
    exit 1
    ;;
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
install -m 0755 "$tmp/$BIN_FILE" "$INSTALL_DIR/$BIN_FILE"
echo "Installed $BIN_FILE to $INSTALL_DIR/$BIN_FILE"

# The CLI needs its fixture models. If the archive bundled a models/ directory,
# install it next to the binary so 'jolt-atlas prove <model>' works out of the box.
if [ -d "$tmp/models" ]; then
  dest_models="$INSTALL_DIR/models"
  rm -rf "$dest_models"
  cp -R "$tmp/models" "$dest_models"
  echo "Installed bundled models to $dest_models"
fi
