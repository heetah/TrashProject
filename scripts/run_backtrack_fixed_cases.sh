#!/usr/bin/env bash
set -u

# Reproducible production replay for a reviewed/fixed case manifest.
MANIFEST="${MANIFEST:-artifacts/backtrack_study/multi_actor_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/backtrack_multi_actor_v1_current}"
CONDA_ENV="${CONDA_ENV:-rtdetr}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
STATUS_FILE="${STATUS_FILE:-$OUTPUT_ROOT/worker_${WORKER_INDEX}_status.tsv}"
SOURCE_DIRECTORY="${SOURCE_DIRECTORY:-}"
SKIP_EXISTING_SUCCESS="${SKIP_EXISTING_SUCCESS:-0}"
CASE_IDS="${CASE_IDS:-}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest does not exist: $MANIFEST" >&2
    exit 1
fi
if (( WORKER_COUNT < 1 || WORKER_INDEX < 0 || WORKER_INDEX >= WORKER_COUNT )); then
    echo "Invalid worker shard: index=$WORKER_INDEX count=$WORKER_COUNT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
printf 'case_id\tvideo\texit_code\n' > "$STATUS_FILE"

mapfile -t CASE_ROWS < <(
    conda run -n "$CONDA_ENV" python -c \
      'import json,sys; d=json.load(open(sys.argv[1])); root=sys.argv[2] or d["source_directory"]; selected={int(v) for v in sys.argv[3].replace(","," ").split() if v}; [print(f"{i}\t{root}/litter_case_{i}.mp4") for i in d["case_ids"] if not selected or i in selected]' \
      "$MANIFEST" "$SOURCE_DIRECTORY" "$CASE_IDS"
)

found=0
success=0
failed=0
row_index=0
for row in "${CASE_ROWS[@]}"; do
    [[ -z "$row" ]] && continue
    case_id="${row%%$'\t'*}"
    video="${row#*$'\t'}"
    if (( row_index % WORKER_COUNT != WORKER_INDEX )); then
        row_index=$((row_index + 1))
        continue
    fi
    row_index=$((row_index + 1))
    found=$((found + 1))
    output_stem="$OUTPUT_ROOT/litter_case_${case_id}_annotated"
    if [[ "$SKIP_EXISTING_SUCCESS" != "0" \
        && -s "${output_stem}_analysis.json" \
        && -s "${output_stem}_litter_candidates.jsonl" ]]; then
        echo "[worker $WORKER_INDEX][$found] litter_case_$case_id (skip existing success)"
        success=$((success + 1))
        printf '%s\t%s\t0\n' "$case_id" "$video" >> "$STATUS_FILE"
        continue
    fi
    echo "[worker $WORKER_INDEX][$found] litter_case_$case_id"
    if OUTPUT_ROOT="$OUTPUT_ROOT" SMART_BACKTRACK=1 \
        SMART_BACKTRACK_SIDECAR=1 \
        conda run -n "$CONDA_ENV" python scripts/main.py "$video"; then
        exit_code=0
        success=$((success + 1))
    else
        exit_code=$?
        failed=$((failed + 1))
    fi
    printf '%s\t%s\t%s\n' "$case_id" "$video" "$exit_code" >> "$STATUS_FILE"
done

echo "Finished worker=$WORKER_INDEX found=$found success=$success failed=$failed"
(( found > 0 && failed == 0 ))
