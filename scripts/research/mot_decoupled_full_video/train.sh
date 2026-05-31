#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

exec bash "${REPO_ROOT}/scripts/train/droid_wan22_mot_decoupled_full_video.sh" "$@"
