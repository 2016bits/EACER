#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/mr2_ecer.yaml}
python -m src.train --config "$CONFIG"
