#!/usr/bin/env bash
# Compile paper_full.tex into paper_full.pdf.
#
# Run from the repo root:
#   chmod +x render.sh   (first time only)
#   ./render.sh
#
# Supports: Ubuntu/Debian, macOS (Homebrew), Windows WSL.
# If pdflatex is not installed, this script installs TeX Live first.

set -euo pipefail

TEX=paper_full

# Detect the operating system so we know which package manager to use.
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"

# If pdflatex is missing, install TeX Live before doing anything else.
# This only runs once — subsequent renders skip straight to compilation.
if ! command -v pdflatex &>/dev/null; then
    echo ""
    echo "pdflatex not found. Installing TeX Live now."
    echo "This is a ~4 GB download and may take 10-20 minutes. It only happens once."
    echo ""

    if [[ "$OS" == "debian" || "$OS" == "wsl" ]]; then
        sudo apt-get update -qq
        sudo apt-get install -y texlive-full

    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &>/dev/null; then
            echo "Homebrew is not installed. Install it from https://brew.sh then re-run this script."
            exit 1
        fi
        brew install --cask mactex
        # MacTeX installs to a non-standard path; add it for this session.
        export PATH="/Library/TeX/texbin:$PATH"

    else
        echo "Cannot auto-install on this OS ($OS)."
        echo "Install manually: https://www.tug.org/texlive/"
        exit 1
    fi

    echo ""
    echo "TeX Live installed."
else
    echo "pdflatex found: $(command -v pdflatex)"
fi

# Make sure we are in the right directory.
if [[ ! -f "$TEX.tex" ]]; then
    echo "Error: $TEX.tex not found in $(pwd). Run this script from the repo root."
    exit 1
fi

echo ""
echo "Compiling $TEX.tex — three passes are needed to resolve all"
echo "cross-references, figure numbers, and citations correctly."
echo ""

# Pass 1: build the document structure and generate auxiliary files.
echo "Pass 1 of 3: initial build"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX.tex"

# BibTeX: read the .aux file and resolve all \cite{} keys against the .bib file.
echo ""
echo "BibTeX: resolving citations"
bibtex "$TEX"

# Pass 2: insert the resolved bibliography into the document.
echo ""
echo "Pass 2 of 3: inserting citations"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX.tex"

# Pass 3: finalise all cross-references (section numbers, figure labels, etc.).
echo ""
echo "Pass 3 of 3: finalising cross-references"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX.tex"

echo ""
echo "Done. PDF saved to: $(pwd)/$TEX.pdf"
echo "Run ./render.sh again any time you edit the paper."
echo ""

# Open the PDF if a viewer is available.
if command -v xdg-open &>/dev/null; then
    xdg-open "$TEX.pdf" &   # Linux / WSL with a desktop environment
elif command -v open &>/dev/null; then
    open "$TEX.pdf"          # macOS
fi
