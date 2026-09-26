#!/bin/bash
# Chunked grid runner: processes pending combos within a time budget, appends JSONL.
BUDGET=${1:-95}
OUT=results/sepsis_distill_law_runs.jsonl
mkdir -p results; touch "$OUT"
START=$(date +%s)
run_one() {
  local arm=$1 width=$2 epochs=$3 seed=$4
  local key="\"arm\": \"$arm\", \"width\": $width, \"epochs\": $epochs, \"seed\": $seed"
  grep -qF "$key" "$OUT" && return 0
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python3 src/sepsis_distill_law.py --table-dir /tmp/icu-sepsis-csv-tables \
    --worker --arm "$arm" --width "$width" --epochs "$epochs" --seed "$seed" >> "$OUT"
}
export -f run_one; export OUT
# generate pending combos
for arm in plain margin_weighted; do
 for width in 16 64 128 256; do
  for epochs in 50 150 500; do
   for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "$arm $width $epochs $seed"
   done
  done
 done
done > /tmp/grid_combos.txt
TOTAL=$(wc -l < /tmp/grid_combos.txt)
DONE=$(wc -l < "$OUT")
echo "grid: $DONE/$TOTAL done"
# run two workers in parallel until budget exhausted
cat /tmp/grid_combos.txt | xargs -P2 -L1 bash -c '
  START=$0
  run_one $@
' "$START" &
XPID=$!
while kill -0 $XPID 2>/dev/null; do
  NOW=$(date +%s)
  if [ $((NOW-START)) -gt $BUDGET ]; then kill $XPID 2>/dev/null; pkill -P $XPID 2>/dev/null; break; fi
  sleep 2
done
wait $XPID 2>/dev/null
echo "chunk done: $(wc -l < "$OUT")/$TOTAL"
