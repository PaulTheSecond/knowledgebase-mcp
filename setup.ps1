<#
.SYNOPSIS
    Automated One-Click Setup Script for KnowledgeBase MCP
.DESCRIPTION
    Creates virtual environment, installs dependencies, registers global CLI,
    and installs agent presets (Claude Code, Antigravity, Cursor, Codex).
.PARAMETER Agent
    Target agent: 'claude', 'antigravity', 'cursor', 'codex', or 'all' (default: 'all')
.PARAMETER ProjectScope
    If set, installs preset into current workspace root instead of user global profile.
#>
param(
    [string]$Agent = "all",
    [switch]$ProjectScope
)

$ErrorActionPreference = "Stop"

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "    KnowledgeBase MCP - Automated Setup" -ForegroundColor Cyan
Write-Host "===============================================`n" -ForegroundColor Cyan

# 1. Verify Python 3.10+
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[Error] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# 2. Virtual Environment
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv $venvPath
} else {
    Write-Host "[1/4] Using existing virtual environment (.venv)..." -ForegroundColor Green
}

# 3. Install Package & CLI
Write-Host "[2/4] Installing dependencies & registering 'knowledge-mcp' CLI..." -ForegroundColor Yellow
& $venvPython -m pip install -e . --quiet

# 4. Install Agent Presets
$scopeName = if ($ProjectScope) { "Project (Workspace)" } else { "User (Global Zero-Footprint)" }
Write-Host "[3/4] Installing preset for '$Agent' ($scopeName scope)..." -ForegroundColor Yellow

$installArgs = @("preset", $Agent, "--install")
if ($ProjectScope) {
    $installArgs += "--project"
}

& $venvPython -m knowledge_mcp.main @installArgs

# 5. Diagnostics
Write-Host "`n[4/4] Verifying database and workspace status..." -ForegroundColor Yellow
& $venvPython -m knowledge_mcp.main sync-file --check-status

Write-Host "`n===============================================" -ForegroundColor Green
Write-Host " [OK] Setup completed successfully!" -ForegroundColor Green
Write-Host " You can now launch your AI agent and start coding." -ForegroundColor Green
Write-Host " Global CLI available: 'knowledge-mcp'" -ForegroundColor Green
Write-Host "===============================================`n" -ForegroundColor Green
