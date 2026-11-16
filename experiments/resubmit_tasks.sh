#!/bin/bash
# Resubmits array tasks of one batch of an evaluation root with the settings it was submitted with (run_config.tsv,
# batches.tsv), e.g. after a fix. TREMOR_BIN_OVERRIDE: a TREMOR binary to use instead of the root's bin/tremor_main;
# it is copied into <root>/bin. Every resubmission is recorded in <root>/resubmissions.tsv.
# Usage: [TREMOR_BIN_OVERRIDE=path] bash experiments/resubmit_tasks.sh <root> <batch_id> <tasks, e.g. 6,7> "<reason>"
set -euo pipefail
cd /path/to/TREMOR
ROOT=$(realpath "$1") BATCH=$2 TASKS=$3 REASON=${4:?give a reason}
config() { awk -F'\t' -v k="$1" '$1 == k { print $2 }' "$ROOT/run_config.tsv"; }
read -r NODES RUNS_PER_TASK PLAN_FILE < <(awk -F'\t' -v b="$BATCH" '$1 == b { print $5, $8, $9 }' "$ROOT/batches.tsv")
[[ -n ${PLAN_FILE:-} ]] || { echo "unknown batch $BATCH" >&2; exit 1; }

TREMOR_BIN=$ROOT/bin/tremor_main
if [[ -n ${TREMOR_BIN_OVERRIDE:-} ]]; then
  TREMOR_BIN=$ROOT/bin/tremor_main.$(sha256sum "$TREMOR_BIN_OVERRIDE" | cut -c1-8)
  cp "$TREMOR_BIN_OVERRIDE" "$TREMOR_BIN"
fi
ROW_TIMEOUT=$(config row_timeout_seconds) GRACE=$(config row_timeout_grace_seconds)
for entry in $(config row_timeout_by_nodes); do [[ ${entry%%=*} = "$NODES" ]] && ROW_TIMEOUT=${entry#*=}; done
minutes=$(( (RUNS_PER_TASK * (ROW_TIMEOUT + GRACE) + 59) / 60 + 15 ))

export LAUNCH_WRAPPER="$(config launch_wrapper)" PAA_BY_WINDOW="$(config paa_by_window)"
export TREMOR_REFINEMENT="$(config tremor_refinement)" TREMOR_SCAN_FALLBACK_FRACTION="$(config tremor_scan_fallback_fraction)"
export TREMOR_METHOD="$(config tremor_method)"
JOB=$(sbatch --parsable --chdir="$PWD" --job-name="$BATCH" \
  --output="$ROOT/logs/$BATCH.%A_%a.out" --error="$ROOT/logs/$BATCH.%A_%a.err" \
  --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --exclusive --mem=0 --cpus-per-task="$(config cpus_per_task)" \
  --time="$(printf "%d:%02d:00" $((minutes / 60)) $((minutes % 60)))" --array="$TASKS" \
  --export="ALL,PLAN_FILE=$PLAN_FILE,TREMOR_BIN=$TREMOR_BIN,DMASS_BIN=$ROOT/bin/dmass_main,CPUS_PER_TASK=$(config cpus_per_task),INDEX_THREADS=$(config index_threads),SEARCH_WORKERS=$(config search_workers),LEAF_SIZE=2000,PAA_SEGMENTS=$(config paa_segments),THREADS=$(config dmass_threads),THRESHOLD_RESULT_SLOTS=1,PQ_THRESHOLD_DIVISOR=16,TREMOR_MAX_LOCAL_SAMPLES=$(config tremor_max_local_samples),DMASS_MAX_LOCAL_SAMPLES=$(config dmass_max_local_samples),MAX_THRESHOLD_SHARD_BYTES=$(config max_threshold_shard_bytes),THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW=$(config threshold_output_avg_bytes_per_row),THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD=$(config threshold_output_guard_max_threshold),ROW_TIMEOUT_SECONDS=$ROW_TIMEOUT,ROW_TIMEOUT_GRACE_SECONDS=$GRACE,ROW_MAX_ATTEMPTS=$(config row_max_attempts)" \
  experiments/full_eval_array_job.slurm)
[[ -f $ROOT/resubmissions.tsv ]] || printf "date\tbatch_id\ttasks\tjob_id\ttremor_bin\treason\n" > "$ROOT/resubmissions.tsv"
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$BATCH" "$TASKS" "$JOB" "$(basename "$TREMOR_BIN")" "$REASON" >> "$ROOT/resubmissions.tsv"
echo "$BATCH tasks $TASKS -> job $JOB (TREMOR binary $(basename "$TREMOR_BIN"))"
