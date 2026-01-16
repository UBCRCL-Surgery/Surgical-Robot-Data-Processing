#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 [TASK] [NAME] [RAW_ROOT] [OUT_ROOT] [CODE_ROOT]"
  echo
  echo "Example:"
  echo "  $0 NeedlePassing PeterBlack \\"
  echo "     /media/zijianwu/My\\ Book/SurgMani \\"
  echo "     /home/zijianwu/Codes/surg_gui/configs \\"
  echo "     /home/zijianwu/Codes/surg_gui"
  exit 1
fi

TASK="$1"
NAME="$2"
RAW_ROOT="$3"     # /media/.../SurgMani
OUT_ROOT="$4"     # where sync_table_all.csv lives
CODE_ROOT="$5"    # where trim_left_video.py lives

RAW_BASE="${RAW_ROOT}/${TASK}/${NAME}"
OUT_BASE="${OUT_ROOT}/${TASK}/${NAME}"

# ---------- safe glob for paths with spaces ----------
find_one () {
  local pattern="$1"
  local files=()

  while IFS= read -r f; do
    files+=("$f")
  done < <(compgen -G "$pattern")

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "❌ Missing: $pattern" >&2
    exit 1
  fi
  if [[ ${#files[@]} -gt 1 ]]; then
    echo "❌ Multiple matches for pattern: $pattern" >&2
    printf '  %s\n' "${files[@]}" >&2
    exit 1
  fi
  echo "${files[0]}"
}

for TRIAL in {1..5}; do
  RAW_TRIAL="${RAW_BASE}/${TRIAL}"
  OUT_TRIAL="${OUT_BASE}/${TRIAL}"

  SYNC_ALL="${OUT_TRIAL}/sync_table_all.csv"

  if [[ ! -f "${SYNC_ALL}" ]]; then
    echo "⚠️  Skip TRIAL=${TRIAL}: missing sync_table_all.csv"
    continue
  fi

  if [[ ! -d "${RAW_TRIAL}" ]]; then
    echo "⚠️  Skip TRIAL=${TRIAL}: raw dir missing"
    continue
  fi

  LEFT_VIDEO=$(find_one "${RAW_TRIAL}/endo/video_stream_2_*.mp4")

  OUT_VIDEO="${OUT_TRIAL}/proxy_left.mp4"
  OUT_MAP="${OUT_TRIAL}/proxy_left_index_map.csv"

  echo "▶ Trimming left video | TASK=${TASK} NAME=${NAME} TRIAL=${TRIAL}"

  python "${CODE_ROOT}/trim_left_video.py" \
    --sync_csv "${SYNC_ALL}" \
    --video "${LEFT_VIDEO}" \
    --out_video "${OUT_VIDEO}" \
    --out_map "${OUT_MAP}" \
    --idx_col left_idx \
    --ts_col t_ref_s

  echo "✅ Generated:"
  echo "   ${OUT_VIDEO}"
  echo "   ${OUT_MAP}"
done
