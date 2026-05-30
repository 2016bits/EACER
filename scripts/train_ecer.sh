#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/ecer_mocheg.yaml}
python -m src.train --config "$CONFIG"
