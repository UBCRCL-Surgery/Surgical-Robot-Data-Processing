#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 [TASK] [NAME] [OUT_ROOT]"
  echo "Example:"
  echo "  $0 NeedlePassing ChristopherNguan /home/zijianwu/Codes/surg_gui/configs"
  exit 1
fi

TASK="$1"
NAME="$2"
OUT_ROOT="$3"

RAW_BASE="/media/zijianwu/My Book/SurgMani/${TASK}/${NAME}"
OUT_BASE="${OUT_ROOT}/${TASK}/${NAME}"

find_one () {
  local pattern="$1"
  local files=()

  # compgen handles glob safely with spaces
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

  if [[ ! -d "${RAW_TRIAL}" ]]; then
    echo "⚠️  Skip missing trial: ${RAW_TRIAL}"
    continue
  fi

  mkdir -p "${OUT_TRIAL}"
  OUT_JSON="${OUT_TRIAL}/mm_timestamp.json"

  echo "▶ TASK=${TASK} NAME=${NAME} TRIAL=${TRIAL}"

  LEFT_TS=$(find_one "${RAW_TRIAL}/endo/video_stream_2_*_timestamps.txt")
  RIGHT_TS=$(find_one "${RAW_TRIAL}/endo/video_stream_1_*_timestamps.txt")
  SIDE_TS=$(find_one "${RAW_TRIAL}/side_camera/output_timestamps.txt")
  GAZE_LOG=$(find_one "${RAW_TRIAL}/gaze/gazelog_*.txt")
  DVAPI_CSV=$(find_one "${RAW_TRIAL}/data_local_part_1_*.csv")

  cat > "${OUT_JSON}" <<EOF
{
  "timezone": "America/Vancouver",
  "left_ts":  "${LEFT_TS}",
  "right_ts": "${RIGHT_TS}",
  "side_ts":  "${SIDE_TS}",
  "gaze_log": "${GAZE_LOG}",
  "dvapi_csv": "${DVAPI_CSV}"
}
EOF

  echo "✅ Generated ${OUT_JSON}"
done
