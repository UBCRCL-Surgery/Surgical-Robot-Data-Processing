#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 [DATASET_ROOT] [TASK] [CODE_ROOT]"
  echo
  echo "Example:"
  echo "  $0 /home/zijianwu/Codes/surg_gui/data \\"
  echo "     NeedlePassing \\"
  echo "     /home/zijianwu/Codes/surg_gui"
  exit 1
fi

DATASET_ROOT="$1"   # e.g. /home/.../data
TASK="$2"           # e.g. NeedlePassing
CODE_ROOT="$3"      # where validate_formatting.py lives

TASK_ROOT="${DATASET_ROOT}/${TASK}"

if [[ ! -d "${TASK_ROOT}" ]]; then
  echo "❌ TASK directory not found: ${TASK_ROOT}"
  exit 1
fi

echo "▶ Validating datasets under TASK=${TASK}"
echo "▶ Root: ${TASK_ROOT}"
echo

FAILED=0

for DATASET_DIR in "${TASK_ROOT}"/*; do
  [[ -d "${DATASET_DIR}" ]] || continue

  DATASET_ID=$(basename "${DATASET_DIR}")

  echo "======================================"
  echo "▶ Validating dataset: ${DATASET_ID}"
  echo "======================================"

  if python "${CODE_ROOT}/validate_formatting.py" "${DATASET_DIR}"; then
    echo "✅ VALID: ${DATASET_ID}"
  else
    echo "❌ INVALID: ${DATASET_ID}"
    FAILED=1
  fi

  echo
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "❌ One or more datasets failed validation for TASK=${TASK}."
  exit 1
fi

echo "🎉 All datasets passed validation for TASK=${TASK}."
