#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 [TASK] [NAME] [CONFIG_ROOT] [CODE_ROOT]"
  echo "Example:"
  echo "  $0 NeedlePassing PeterBlack \\"
  echo "     /home/zijianwu/Codes/surg_gui/configs \\"
  echo "     /home/zijianwu/Codes/surg_gui"
  exit 1
fi

TASK="$1"
NAME="$2"
CONFIG_ROOT="$3"   # where mm_timestamp.json lives
CODE_ROOT="$4"     # where sync_adaptive.py lives

BASE_CFG="${CONFIG_ROOT}/${TASK}/${NAME}"

for TRIAL in {1..5}; do
  CFG_DIR="${BASE_CFG}/${TRIAL}"
  MM_JSON="${CFG_DIR}/mm_timestamp.json"
  OUT_CSV="${CFG_DIR}/sync_table_all.csv"

  if [[ ! -f "${MM_JSON}" ]]; then
    echo "⚠️  Skip TRIAL=${TRIAL}: missing ${MM_JSON}"
    continue
  fi

  echo "▶ Running sync_adaptive.py | TASK=${TASK} NAME=${NAME} TRIAL=${TRIAL}"

  python "${CODE_ROOT}/sync_adaptive.py" \
    --config "${MM_JSON}" \
    --out_csv "${OUT_CSV}"

  echo "✅ Generated ${OUT_CSV}"
done
