#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 [TASK] [NAME] [RAW_ROOT] [OUT_ROOT] [CODE_ROOT]"
  echo
  echo "Example:"
  echo "  $0 NeedlePassing PeterBlack \\"
  echo "     \"/media/zijianwu/My Book/SurgMani\" \\"
  echo "     /home/zijianwu/Codes/surg_gui/data \\"
  echo "     /home/zijianwu/Codes/surg_gui"
  exit 1
fi

TASK="$1"
NAME="$2"
RAW_ROOT="$3"    # /media/.../SurgMani
OUT_ROOT="$4"    # where sync / proxy files live
CODE_ROOT="$5"   # where export_lerobot_adaptive.py lives

RAW_BASE="${RAW_ROOT}/${TASK}/${NAME}"
OUT_BASE="${OUT_ROOT}/${TASK}/${NAME}"

# ---------- safe glob (supports spaces) ----------
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

  # ---------- find sync_table_<project_id>.csv (exclude sync_table_all.csv) ----------
  SYNC_EP_FILES=()

  while IFS= read -r f; do
    if [[ "$(basename "$f")" == "sync_table_all.csv" ]]; then
      continue
    fi
    SYNC_EP_FILES+=("$f")
  done < <(compgen -G "${OUT_TRIAL}/sync_table_*.csv")

  if [[ ${#SYNC_EP_FILES[@]} -ne 1 ]]; then
    echo "⚠️  Skip TRIAL=${TRIAL}: expected exactly one sync_table_<project_id>.csv"
    printf '  %s\n' "${SYNC_EP_FILES[@]}"
    continue
  fi

  SYNC_EP="${SYNC_EP_FILES[0]}"

  # ---------- parse project_id ----------
  BASENAME=$(basename "${SYNC_EP}")
  PROJECT_ID="${BASENAME#sync_table_}"
  PROJECT_ID="${PROJECT_ID%.csv}"

  echo "▶ Export LeRobot | TASK=${TASK} NAME=${NAME} TRIAL=${TRIAL} PROJECT_ID=${PROJECT_ID}"

  LEFT_VIDEO=$(find_one "${RAW_TRIAL}/endo/video_stream_2_*.mp4")
  RIGHT_VIDEO=$(find_one "${RAW_TRIAL}/endo/video_stream_1_*.mp4")
  SIDE_VIDEO=$(find_one "${RAW_TRIAL}/side_camera/output.mp4")
  GAZE_VIDEO=$(find_one "${RAW_TRIAL}/gaze/eyeVideo_*.avi")

  python "${CODE_ROOT}/export_lerobot_adaptive.py" \
    --sync-all "${SYNC_ALL}" \
    --sync-ep  "${SYNC_EP}" \
    --left-video  "${LEFT_VIDEO}" \
    --right-video "${RIGHT_VIDEO}" \
    --side-video  "${SIDE_VIDEO}" \
    --gaze-video  "${GAZE_VIDEO}" \
    --out "${OUT_ROOT}" \
    --dataset-name "${PROJECT_ID}"

  echo "✅ Exported dataset: ${PROJECT_ID}"
done
