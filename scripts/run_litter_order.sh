#!/usr/bin/env bash
set -u

# 逐支呼叫 production pipeline；不修改 scripts/main.py 的單影片 CLI 契約。
INPUT_DIR="${INPUT_DIR:-/mnt/8tb_hdd/under115a/litter_vidshort/litter_order}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/litter_order}"
CONDA_ENV="${CONDA_ENV:-rtdetr}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

found=0
success=0
failed=0

while IFS= read -r -d '' video; do
    found=$((found + 1))
    echo "[$found] Processing: $video"

    if OUTPUT_ROOT="$OUTPUT_ROOT" \
        conda run -n "$CONDA_ENV" python scripts/main.py "$video"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
        echo "[$found] FAILED: $video" >&2
    fi
done < <(
    find "$INPUT_DIR" -type f \
        \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \
           -o -iname '*.wmv' -o -iname '*.m4v' \) \
        -print0 | sort -z
)

echo "Finished: found=$found success=$success failed=$failed"

if [[ "$found" -eq 0 ]]; then
    echo "No supported video files found in: $INPUT_DIR" >&2
    exit 1
fi

[[ "$failed" -eq 0 ]]
