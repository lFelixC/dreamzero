#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH=joint exec "${SCRIPT_DIR}/droid_wan22_fseq200.sh" "$@"
