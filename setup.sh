#!/usr/bin/env bash
set -e

# Synopsis: Automated One-Click Setup Script for KnowledgeBase MCP
# Usage: ./setup.sh [agent] [--project]
# Example: ./setup.sh claude

AGENT="${1:-all}"
SCOPE_FLAG=""

for arg in "$@"; do
    if [ "$arg" == "--project" ] || [ "$arg" == "--local" ]; then
        SCOPE_FLAG="--project"
    fi
done

echo ""
echo "==============================================="
echo "    KnowledgeBase MCP - Automated Setup"
echo "==============================================="
echo ""

# 1. Verify Python 3
if ! command -v python3 &> /dev/null; then
    echo "[Error] python3 is not installed or not in PATH."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# 2. Virtual Environment
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Using existing virtual environment (.venv)..."
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# 3. Install Package & CLI
echo "[2/4] Installing dependencies & registering 'knowledge-mcp' CLI..."
"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -e "$SCRIPT_DIR" --quiet

# 4. Install Agent Presets
echo "[3/4] Installing preset for '$AGENT'..."
if [ -n "$SCOPE_FLAG" ]; then
    "$VENV_PY" -m knowledge_mcp.main preset "$AGENT" --install --project
else
    "$VENV_PY" -m knowledge_mcp.main preset "$AGENT" --install
fi

# 5. Diagnostics
echo ""
echo "[4/4] Verifying database and workspace status..."
"$VENV_PY" -m knowledge_mcp.main sync-file --check-status

echo ""
echo "==============================================="
echo " [OK] Setup completed successfully!"
echo " You can now launch your AI agent and start coding."
echo "==============================================="
echo ""
