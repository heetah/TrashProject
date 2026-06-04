#!/bin/bash
# Re-run the 51 previously-missed urinate videos end-to-end with the fixes
# (action.py person-bbox normalize + detect.py track-id TTL buffer) and the
# newly deployed checkpoint, tallying has_urinate per case.
OUT=/tmp/urinate_missed_recheck
mkdir -p "$OUT"
PY=/home/se_copilot/miniconda3/envs/rtdetr/bin/python
MAIN=/home/se_copilot/trashProject/scripts-old-test-long/main.py
cd /home/se_copilot/trashProject/scripts-old-test-long || exit 1
LIST=/tmp/urinate_missed_list.txt
total=$(grep -c . "$LIST")
i=0; flipped=0
echo "=== RECHECK START $(date) | $total cases ==="
while read -r v; do
  [ -z "$v" ] && continue
  i=$((i+1))
  name=$(basename "$v" .mp4)
  sj="$OUT/${name}.json"
  "$PY" "$MAIN" "$v" --output-root "$OUT" --summary-json "$sj" > "$OUT/${name}.log" 2>&1
  hu=$("$PY" -c "import json;print(json.load(open('$sj')).get('has_urinate'))" 2>/dev/null || echo "ERR")
  [ "$hu" = "True" ] && flipped=$((flipped+1))
  echo "[$i/$total] $name has_urinate=$hu  (recovered so far: $flipped)"
done < "$LIST"
echo "=== RECHECK DONE $(date) | recovered $flipped / $total previously-missed ==="
