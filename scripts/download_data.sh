#!/usr/bin/env bash
# Download MOCHEG and MR2 datasets.
# Run this on a machine that can reach Google Drive / GitHub Releases.
# Mainland-China users: see scripts/download_data_cn.md for Baidu AI Studio
# mirrors and manual instructions.
set -euo pipefail

ROOT=${ROOT:-data/raw}
mkdir -p "$ROOT"

# -------- prerequisites --------
if ! command -v gdown >/dev/null 2>&1; then
    echo "[deps] installing gdown ..."
    pip install --quiet gdown
fi

# ============================================================
# MOCHEG
# ============================================================
echo "[1/2] MOCHEG"
MOCHEG_DIR="$ROOT/mocheg"
mkdir -p "$MOCHEG_DIR"

# Primary: GitHub Releases (preferred — versioned)
# As of 2025 the Mocheg v1 release is published at:
#   https://github.com/PLUM-Lab/Mocheg/releases
# Pick the latest asset URL and uncomment ONE of the lines below.

# Example (release asset, replace TAG and FILENAME):
#   wget -c -O "$MOCHEG_DIR/mocheg.zip" \
#     https://github.com/PLUM-Lab/Mocheg/releases/download/<TAG>/<FILENAME>

# Fallback: Google Drive folder (the repo README usually links one).
#   The folder id may change; check the README first.
# gdown --folder --output "$MOCHEG_DIR" https://drive.google.com/drive/folders/<FOLDER_ID>

if [ ! -d "$MOCHEG_DIR/train" ]; then
    echo "  -> please open the README at https://github.com/PLUM-Lab/Mocheg"
    echo "     copy the current Google Drive / Release link, then run one of:"
    echo "       gdown --folder --output $MOCHEG_DIR <DRIVE_FOLDER_URL>"
    echo "       wget -O $MOCHEG_DIR/mocheg.zip <RELEASE_ASSET_URL>"
    echo "       unzip $MOCHEG_DIR/mocheg.zip -d $MOCHEG_DIR"
fi

# ============================================================
# MR2
# ============================================================
echo "[2/2] MR2"
MR2_DIR="$ROOT/mr2"
mkdir -p "$MR2_DIR"

MR2_GDRIVE_ID="14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4"
if [ ! -f "$MR2_DIR/MR2.zip" ] && [ ! -f "$MR2_DIR/dataset_items_train.json" ]; then
    echo "  -> downloading MR2 from Google Drive ..."
    gdown --id "$MR2_GDRIVE_ID" -O "$MR2_DIR/MR2.zip" || {
        echo "     gdown failed. Try the Baidu mirror:"
        echo "       https://aistudio.baidu.com/datasetdetail/230144"
        echo "     and manually unzip into $MR2_DIR/"
        exit 1
    }
    unzip -q "$MR2_DIR/MR2.zip" -d "$MR2_DIR"
fi

echo "done. raw data lives under $ROOT"
ls -la "$ROOT"
