#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
CONFIG_DIR="$HOME/.config/sysagent"

mkdir -p "$BIN_DIR" "$SYSTEMD_USER_DIR" "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/.env" ]; then
  cat > "$CONFIG_DIR/.env" <<'EOF'
# Add your Gemini API key here:
GEMINI_API_KEY=
SYSAGENT_NAME=SysAgent
SYSAGENT_MODEL=gemini-4-argon
SYSAGENT_EVOLVE_MODEL=gemini-4-argon
SYSAGENT_WORKER_MODEL=gemini-4-hydrogen
SYSAGENT_LIGHT_MODEL=gemini-3.5-flash-lite
SYSAGENT_APPROVAL_MODE=smart
SYSAGENT_LOCALE=auto
SYSAGENT_HOST=127.0.0.1
SYSAGENT_PORT=8765
EOF
  echo "Created: $CONFIG_DIR/.env (remember to set GEMINI_API_KEY)"
fi

cat > "$BIN_DIR/sysagent" <<EOF
#!/usr/bin/env bash
PYTHONPATH="$PROJECT_DIR" exec "$PROJECT_DIR/.venv/bin/python" -m sysagent.cli "\$@"
EOF
chmod +x "$BIN_DIR/sysagent"

sed -e "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" \
    -e "s|^ExecStart=.*|ExecStart=$PROJECT_DIR/.venv/bin/python -m sysagent.cli serve|" \
    "$PROJECT_DIR/systemd/sysagent.service" > "$SYSTEMD_USER_DIR/sysagent.service"
systemctl --user daemon-reload

mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/sysagent-quick.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SysAgent Quick
Comment=Quick command launcher (bindable to a global shortcut)
Icon=$PROJECT_DIR/assets/sysagent.svg
Exec=$BIN_DIR/sysagent quick
Terminal=false
Categories=Utility;
EOF

echo "Installation complete!"
echo "1) Set GEMINI_API_KEY in $CONFIG_DIR/.env"
echo "2) Start the service: systemctl --user enable --now sysagent"
echo "3) Usage: sysagent \"summarize system status\"  or  http://127.0.0.1:8765"
