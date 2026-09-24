#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode=${1:---help}
case "$mode" in
  --build) docker build -t flyarena-lab:local backend ;;
  --cpu|--cuda)
    lab_token=${LAB_TOKEN:-}
    [[ ${#lab_token} -ge 16 ]] || { echo 'Set LAB_TOKEN to at least 16 characters.' >&2; exit 1; }
    gpu=()
    [[ "$mode" != --cuda ]] || gpu=(--gpus all)
    exec docker run --rm --init --name "${LAB_CONTAINER_NAME:-flyarena-lab}" \
      --memory 6g --cpus 4 --cap-drop ALL --security-opt no-new-privileges \
      --read-only --tmpfs /tmp:rw,noexec,nosuid,size=128m \
      -p "127.0.0.1:${LAB_PORT:-8765}:8000" "${gpu[@]}" \
      -e LAB_TOKEN -e "LAB_ORIGINS=${LAB_ORIGINS:-http://127.0.0.1:5173,http://127.0.0.1:4173}" flyarena-lab:local ;;
  --help) echo 'Usage: scripts/lab.sh --build|--cpu|--cuda'
    echo 'Local only. Set LAB_TOKEN (16+ characters). Optional LAB_PORT, LAB_ORIGINS, LAB_CONTAINER_NAME.' ;;
  *) echo 'Unknown option; use --help.' >&2; exit 1 ;;
esac
