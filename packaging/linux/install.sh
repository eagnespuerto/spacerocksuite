#!/bin/sh
# Install SpaceRockSuite on Linux or Raspberry Pi OS from a release bundle.
# Run from the unpacked bundle folder:  sh install.sh
# Uses the distribution's NumPy, Astropy, Matplotlib and Tk (prebuilt and light,
# which matters on a Pi 3B+), in a venv that can see them.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${SRS_PREFIX:-$HOME/.local/share/spacerocksuite}"
BIN="$HOME/.local/bin"

if command -v apt-get >/dev/null 2>&1; then
  NEED=""
  for p in python3-venv python3-tk python3-numpy python3-astropy python3-matplotlib; do
    dpkg -s "$p" >/dev/null 2>&1 || NEED="$NEED $p"
  done
  if [ -n "$NEED" ]; then
    echo "Installing system packages:$NEED"
    sudo apt-get update && sudo apt-get install -y $NEED
  fi
fi

python3 -m venv --system-site-packages "$PREFIX/venv"
"$PREFIX/venv/bin/python" -m pip install --upgrade --no-deps "$HERE"/*.whl
# fill in anything the system packages did not provide
"$PREFIX/venv/bin/python" -m pip install "$HERE"/spacerocksuite-*.whl

mkdir -p "$BIN"
ln -sf "$PREFIX/venv/bin/srs" "$BIN/srs"
ln -sf "$PREFIX/venv/bin/spacerocksuite-gui" "$BIN/spacerocksuite-gui"

ICON="$PREFIX/icon.png"
cp "$HERE/icon.png" "$ICON" 2>/dev/null || true
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/spacerocksuite.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SpaceRockSuite
Comment=Precovery toolkit for small bodies
Exec=$PREFIX/venv/bin/spacerocksuite-gui
Icon=$ICON
Categories=Science;Astronomy;Education;
Terminal=false
EOF

echo
echo "SpaceRockSuite installed. Start it from the menu (Education/Science) or run: spacerocksuite-gui"
"$PREFIX/venv/bin/srs" info | head -3
