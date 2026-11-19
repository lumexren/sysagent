#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
CONFIG_DIR="$HOME/.config/sysagent"
DESKTOP_DIR="$HOME/.local/share/applications"

BIN_NAME="sysagent"
INSTALL_SYSTEMD=1
UNINSTALL=0

usage() {
  cat <<EOF
Usage: bash install.sh [options]

Options:
  --bin-name <name>   CLI binary name to install in ~/.local/bin (default: sysagent)
  --no-systemd        Skip installing/reloading the systemd user service
  --uninstall         Remove installed binary, desktop entry, and systemd unit
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin-name)
      [[ $# -ge 2 ]] || { echo "Error: --bin-name requires a value" >&2; exit 1; }
      BIN_NAME="$2"
      shift 2
      ;;
    --no-systemd)
      INSTALL_SYSTEMD=0
      shift
      ;;
    --uninstall)
      UNINSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$UNINSTALL" -eq 1 ]]; then
  echo "Uninstalling $BIN_NAME..."
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user disable --now sysagent.service 2>/dev/null || true
  fi
  rm -f "$BIN_DIR/$BIN_NAME" \
        "$DESKTOP_DIR/sysagent-quick.desktop" \
        "$SYSTEMD_USER_DIR/sysagent.service"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user daemon-reload 2>/dev/null || true
  fi
  echo "Removed launcher, desktop entry, and systemd unit."
  echo "Note: Your config ($CONFIG_DIR) and data (~/.local/share/sysagent) were preserved."
  exit 0
fi

# 1. Verify Python >= 3.11
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required (>= 3.11)." >&2
  exit 1
fi
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Error: Python >= 3.11 required, found {sys.version.split()[0]}")
PY

# 2. Create virtualenv & install dependencies
VENV_DIR="$PROJECT_DIR/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

echo "Installing dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q

# 3. Create config directory & default .env (chmod 600)
mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$DESKTOP_DIR"
if [[ ! -f "$CONFIG_DIR/.env" ]]; then
  cat > "$CONFIG_DIR/.env" <<'EOF'
# Add your Gemini API key here (or set SYSAGENT_AUTH_MODE=agy_cli):
GEMINI_API_KEY=
SYSAGENT_NAME=SysAgent
SYSAGENT_MODEL=gemini-4-argon
SYSAGENT_EVOLVE_MODEL=gemini-4-argon
SYSAGENT_WORKER_MODEL=gemini-4-hydrogen
SYSAGENT_LIGHT_MODEL=gemini-3.5-flash-lite
SYSAGENT_LOCAL_BACKEND=auto
SYSAGENT_LOCAL_FIRST=1
SYSAGENT_APPROVAL_MODE=smart
SYSAGENT_LOCALE=auto
SYSAGENT_HOST=127.0.0.1
SYSAGENT_PORT=8765
EOF
  chmod 600 "$CONFIG_DIR/.env"
  echo "Created $CONFIG_DIR/.env (chmod 600 — remember to set GEMINI_API_KEY or run '$BIN_NAME login')."
fi

# 4. Install CLI wrapper
cat > "$BIN_DIR/$BIN_NAME" <<EOF
#!/usr/bin/env bash
PYTHONPATH="$PROJECT_DIR" exec "$VENV_DIR/bin/python" -m sysagent.cli "\$@"
EOF
chmod +x "$BIN_DIR/$BIN_NAME"

# 5. Install desktop quick launcher
cat > "$DESKTOP_DIR/sysagent-quick.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SysAgent Quick
Comment=Quick command launcher (bindable to a global shortcut)
Icon=$PROJECT_DIR/assets/sysagent.svg
Exec=$BIN_DIR/$BIN_NAME quick
Terminal=false
Categories=Utility;
EOF

# 6. Install systemd user service (optional)
if [[ "$INSTALL_SYSTEMD" -eq 1 ]]; then
  mkdir -p "$SYSTEMD_USER_DIR"
  sed -e "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" \
      -e "s|^ExecStart=.*|ExecStart=$VENV_DIR/bin/python -m sysagent.cli serve|" \
      "$PROJECT_DIR/systemd/sysagent.service" > "$SYSTEMD_USER_DIR/sysagent.service"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user daemon-reload 2>/dev/null || true
  fi
fi

echo ""
echo "Installation complete! ($BIN_DIR/$BIN_NAME)"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "Warning: $BIN_DIR is not in your PATH. Add: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
echo "Next steps:"
echo "  1) Configure credentials in $CONFIG_DIR/.env (or run: $BIN_NAME login)"
if [[ "$INSTALL_SYSTEMD" -eq 1 ]]; then
  echo "  2) Enable & start daemon : systemctl --user enable --now sysagent"
fi
echo "  3) Check status / TUI    : $BIN_NAME status   |   $BIN_NAME top"
