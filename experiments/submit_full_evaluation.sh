#!/bin/bash
#
# Submit the full TREMOR vs MASS evaluation without waiting for completion.
#
# Defaults:
#   Maule threshold search: size-stratified channel subset, thresholds 0.5 0.8 0.9 0.95
#   SeiFR k-NN search:      all channels, k values 10 50 100 200
#   Node counts:            1 2 4 6 8 10
#   Replication groups:     0 (one data partition per MPI rank)
#   Queries:                at most QUERY_LIMIT per channel, same sample for every run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EVAL_LABEL="${EVAL_LABEL:-full_eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-$SCRIPT_DIR/evals/$EVAL_LABEL}"

NODE_COUNTS="${NODE_COUNTS:-1 2 4 6 8 10}"
MAULE_THRESHOLDS="${MAULE_THRESHOLDS:-0.5 0.8 0.9 0.95}"
MAULE_CASE_COUNT="${MAULE_CASE_COUNT:-24}"
MAULE_MIN_SAMPLES="${MAULE_MIN_SAMPLES:-0}"
MAULE_MAX_SAMPLES="${MAULE_MAX_SAMPLES:-0}"
SEIFR_K_VALUES="${SEIFR_K_VALUES:-10 50 100 200}"
MAULE_K_VALUES="${MAULE_K_VALUES:-1 10 100}"
WORKLOADS="${WORKLOADS:-maule_threshold seifr_knn}"  # also: maule_knn
REPLICATION_GROUPS="${REPLICATION_GROUPS:-0}"

CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
INDEX_THREADS="${INDEX_THREADS:-64}"
SEARCH_WORKERS="${SEARCH_WORKERS:-64}"
LEAF_SIZE="${LEAF_SIZE:-2000}"
PAA_SEGMENTS="${PAA_SEGMENTS:-8}"
THREADS="${THREADS:-64}"
THRESHOLD_RESULT_SLOTS="${THRESHOLD_RESULT_SLOTS:-1}"
PQ_THRESHOLD_DIVISOR="${PQ_THRESHOLD_DIVISOR:-16}"
TREMOR_MAX_LOCAL_SAMPLES="${TREMOR_MAX_LOCAL_SAMPLES:-600000000}"
DMASS_MAX_LOCAL_SAMPLES="${DMASS_MAX_LOCAL_SAMPLES:-350000000}"
MAX_THRESHOLD_SHARD_BYTES="${MAX_THRESHOLD_SHARD_BYTES:-1073741824}"
THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW="${THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW:-64}"
THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD="${THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD:-0.1}"
MERGE_OFFSET_DIVISOR="${MERGE_OFFSET_DIVISOR:-4}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-900}"
RUNS_PER_ARRAY_TASK="${RUNS_PER_ARRAY_TASK:-1}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
ROW_TIMEOUT_SECONDS="${ROW_TIMEOUT_SECONDS:-10800}"
ROW_TIMEOUT_GRACE_SECONDS="${ROW_TIMEOUT_GRACE_SECONDS:-120}"
# Optional per-node-count row timeouts, e.g. "1=14400 2=9000 4=5400 8=3600".
# A batch then asks for runs_per_task * (timeout + grace) + 15 minutes of
# walltime instead of SBATCH_TIME, so short multi-node jobs can backfill.
ROW_TIMEOUT_BY_NODES="${ROW_TIMEOUT_BY_NODES:-}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SUBMIT_DEPENDENCY="${SUBMIT_DEPENDENCY:-}"
# Retries absorb the transient OOM-kills observed under production-scale
# concurrency that isolated/small-scale reproduction could not pin down.
ROW_MAX_ATTEMPTS="${ROW_MAX_ATTEMPTS:-3}"
# Limit each array batch's concurrency by default. The full matrix is still
# submitted; this just keeps scheduler pressure and output bursts manageable.
ARRAY_THROTTLE="${ARRAY_THROTTLE:-2}"
DRY_RUN="${DRY_RUN:-0}"
QUERY_LIMIT="${QUERY_LIMIT:-5000}"
QUERY_SEED="${QUERY_SEED:-20260917}"

MASS_CC="${MASS_CC:-mpicc}"
MASS_CFLAGS="${MASS_CFLAGS:--O3 -std=gnu11 -fopenmp}"
MASS_LDFLAGS="${MASS_LDFLAGS:--lfftw3 -lfftw3_threads -lpthread -lm}"

TREMOR_BIN="$REPO_ROOT/bin/tremor_main"
MASS_SRC="${MASS_SRC:-$REPO_ROOT/dmass/DMASS_V1.c}"
MASS_BIN="${MASS_BIN:-$REPO_ROOT/bin/dmass_main}"  # D-MASS-V1; DMASS_V3.c -> bin/dmass_v3_main
PLAN_SCRIPT="$SCRIPT_DIR/prepare_full_eval_plan.py"
ARRAY_JOB="$SCRIPT_DIR/full_eval_array_job.slurm"

if [[ -f "$EVAL_ROOT/submissions.tsv" ]]; then
    echo "Refusing to overwrite an evaluation with existing submissions: $EVAL_ROOT" >&2
    exit 1
fi
mkdir -p "$EVAL_ROOT"

echo "[+] Full evaluation root: $EVAL_ROOT"
echo "[+] Nodes:      $NODE_COUNTS"
echo "[+] Thresholds: $MAULE_THRESHOLDS"
echo "[+] k values:   $SEIFR_K_VALUES"
echo "[+] Reps:       $REPLICATION_GROUPS"
echo "[+] Merge offset: window/$MERGE_OFFSET_DIVISOR"
echo "[+] Bundle:     $RUNS_PER_ARRAY_TASK run(s) per array task"
echo "[+] Wall time:  $SBATCH_TIME per array task"
echo "[+] Row timeout: ${ROW_TIMEOUT_SECONDS}s (+${ROW_TIMEOUT_GRACE_SECONDS}s grace)"
echo "[+] Row timeout by nodes: ${ROW_TIMEOUT_BY_NODES:-none}"
echo "[+] Dependency: ${SUBMIT_DEPENDENCY:-none}"
echo "[+] Row attempts: $ROW_MAX_ATTEMPTS"
echo "[+] TREMOR skip: local chunk > $TREMOR_MAX_LOCAL_SAMPLES samples"
echo "[+] DMASS skip: local chunk > $DMASS_MAX_LOCAL_SAMPLES samples"
echo "[+] Threshold output guard: threshold <= $THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD and worst-case shard > $MAX_THRESHOLD_SHARD_BYTES bytes"

echo "[+] Preparing dataset audit and Slurm plans"
if [[ -f "$REPO_ROOT/dataprep/regeneration_pending.json" ]]; then
    echo "Dataset regeneration/audit is still pending." >&2
    exit 1
fi
python3 "$PLAN_SCRIPT" \
    --output-root "$EVAL_ROOT" \
    --nodes "$NODE_COUNTS" \
    --thresholds "$MAULE_THRESHOLDS" \
    --k-values "$SEIFR_K_VALUES" \
    --maule-k-values "$MAULE_K_VALUES" \
    --workloads "$WORKLOADS" \
    --replication-groups "$REPLICATION_GROUPS" \
    --paa-segments "$PAA_SEGMENTS" \
    --merge-offset-divisor "$MERGE_OFFSET_DIVISOR" \
    --max-array-tasks "$MAX_ARRAY_TASKS" \
    --runs-per-array-task "$RUNS_PER_ARRAY_TASK" \
    --query-limit "$QUERY_LIMIT" --query-seed "$QUERY_SEED" \
    --maule-case-count "$MAULE_CASE_COUNT" --maule-min-samples "$MAULE_MIN_SAMPLES" --maule-max-samples "$MAULE_MAX_SAMPLES" \
    --fail-on-audit-issues

NODE_COUNTS="$NODE_COUNTS" ROW_TIMEOUT_BY_NODES="$ROW_TIMEOUT_BY_NODES" TREMOR_MAX_LOCAL_SAMPLES="$TREMOR_MAX_LOCAL_SAMPLES" DMASS_MAX_LOCAL_SAMPLES="$DMASS_MAX_LOCAL_SAMPLES" ROW_TIMEOUT_SECONDS="$ROW_TIMEOUT_SECONDS" \
    python3 "$SCRIPT_DIR/write_run_limits.py" "$EVAL_ROOT"

if [[ "$DRY_RUN" = "1" ]]; then
    echo "[+] DRY_RUN=1, not compiling or submitting jobs."
    exit 0
fi

if command -v module >/dev/null 2>&1; then
    module load intel/21U2/suite
    module load fftw/3.3.9
fi

# SKIP_BUILD=1 submits the existing (already validated) binaries unchanged.
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
echo "[+] Compiling TREMOR"
( cd "$REPO_ROOT" && make all )
fi

if [[ ! -x "$TREMOR_BIN" ]]; then
    echo "[!] TREMOR binary not found at $TREMOR_BIN" >&2
    exit 1
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
echo "[+] Compiling MASS/DMASS"
mkdir -p "$(dirname "$MASS_BIN")"
echo "    $MASS_CC ${CFLAGS:-} ${CPPFLAGS:-} $MASS_CFLAGS -o $MASS_BIN $MASS_SRC ${LDFLAGS:-} $MASS_LDFLAGS"
$MASS_CC ${CFLAGS:-} ${CPPFLAGS:-} $MASS_CFLAGS -o "$MASS_BIN" "$MASS_SRC" ${LDFLAGS:-} $MASS_LDFLAGS
fi

if [[ ! -x "$MASS_BIN" ]]; then
    echo "[!] MASS/DMASS binary not found at $MASS_BIN" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "[!] sbatch not found in PATH. Run this on a Slurm login node." >&2
    exit 1
fi

mkdir -p "$EVAL_ROOT/bin"
cp "$TREMOR_BIN" "$EVAL_ROOT/bin/tremor_main"
cp "$MASS_BIN" "$EVAL_ROOT/bin/dmass_main"
TREMOR_BIN="$EVAL_ROOT/bin/tremor_main"
MASS_BIN="$EVAL_ROOT/bin/dmass_main"
sha256sum "$TREMOR_BIN" "$MASS_BIN" > "$EVAL_ROOT/bin/SHA256SUMS"
RUN_CONFIG="$EVAL_ROOT/run_config.tsv"
printf "key\tvalue\n" > "$RUN_CONFIG"
printf "query_limit\t%s\nquery_seed\t%s\n" "$QUERY_LIMIT" "$QUERY_SEED" >> "$RUN_CONFIG"
printf "node_counts\t%s\n" "$NODE_COUNTS" >> "$RUN_CONFIG"
printf "maule_thresholds\t%s\n" "$MAULE_THRESHOLDS" >> "$RUN_CONFIG"
printf "maule_case_count\t%s\nmaule_min_samples\t%s\n" "$MAULE_CASE_COUNT" "$MAULE_MIN_SAMPLES" >> "$RUN_CONFIG"
printf "seifr_k_values\t%s\n" "$SEIFR_K_VALUES" >> "$RUN_CONFIG"
printf "maule_k_values\t%s\n" "$MAULE_K_VALUES" >> "$RUN_CONFIG"
printf "workloads\t%s\n" "$WORKLOADS" >> "$RUN_CONFIG"
printf "replication_groups\t%s\n" "$REPLICATION_GROUPS" >> "$RUN_CONFIG"
printf "cpus_per_task\t%s\n" "$CPUS_PER_TASK" >> "$RUN_CONFIG"
printf "index_threads\t%s\n" "$INDEX_THREADS" >> "$RUN_CONFIG"
printf "search_workers\t%s\n" "$SEARCH_WORKERS" >> "$RUN_CONFIG"
printf "dmass_threads\t%s\n" "$THREADS" >> "$RUN_CONFIG"
printf "runs_per_array_task\t%s\n" "$RUNS_PER_ARRAY_TASK" >> "$RUN_CONFIG"
printf "array_throttle\t%s\n" "$ARRAY_THROTTLE" >> "$RUN_CONFIG"
printf "sbatch_time\t%s\n" "$SBATCH_TIME" >> "$RUN_CONFIG"
printf "launch_wrapper\t%s\npaa_segments\t%s\npaa_by_window\t%s\n" "${LAUNCH_WRAPPER:-}" "$PAA_SEGMENTS" "${PAA_BY_WINDOW:-}" >> "$RUN_CONFIG"
printf "tremor_refinement\t%s\n" "${TREMOR_REFINEMENT:-}" >> "$RUN_CONFIG"
printf "tremor_scan_fallback_fraction\t%s\n" "${TREMOR_SCAN_FALLBACK_FRACTION:-}" >> "$RUN_CONFIG"
printf "tremor_method\t%s\n" "${TREMOR_METHOD:-}" >> "$RUN_CONFIG"
printf "row_timeout_seconds\t%s\n" "$ROW_TIMEOUT_SECONDS" >> "$RUN_CONFIG"
printf "row_max_attempts\t%s\n" "$ROW_MAX_ATTEMPTS" >> "$RUN_CONFIG"
printf "row_timeout_grace_seconds\t%s\n" "$ROW_TIMEOUT_GRACE_SECONDS" >> "$RUN_CONFIG"
printf "row_timeout_by_nodes\t%s\n" "$ROW_TIMEOUT_BY_NODES" >> "$RUN_CONFIG"
printf "merge_offset_divisor\t%s\n" "$MERGE_OFFSET_DIVISOR" >> "$RUN_CONFIG"
printf "tremor_max_local_samples\t%s\n" "$TREMOR_MAX_LOCAL_SAMPLES" >> "$RUN_CONFIG"
printf "dmass_max_local_samples\t%s\n" "$DMASS_MAX_LOCAL_SAMPLES" >> "$RUN_CONFIG"
printf "max_threshold_shard_bytes\t%s\n" "$MAX_THRESHOLD_SHARD_BYTES" >> "$RUN_CONFIG"
printf "threshold_output_avg_bytes_per_row\t%s\n" "$THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW" >> "$RUN_CONFIG"
printf "threshold_output_guard_max_threshold\t%s\n" "$THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD" >> "$RUN_CONFIG"

SUBMISSIONS="$EVAL_ROOT/submissions.tsv"
printf "batch_id\tdataset\tworkload\talgorithm\tnodes\ttask_count\trun_count\truns_per_array_task\tjob_id\tplan_file\tresult_dir\n" > "$SUBMISSIONS"

echo "[+] Submitting Slurm array batches"
tail -n +2 "$EVAL_ROOT/batches.tsv" | while IFS=$'\t' read -r BATCH_ID DATASET WORKLOAD ALGORITHM NODES TASK_COUNT RUN_COUNT RUNS_PER_TASK PLAN_FILE RESULT_DIR LOG_GLOB; do
    if [[ "$TASK_COUNT" -le 0 ]]; then
        continue
    fi
    # ALGORITHMS selects the engines to submit (default: both).
    [[ " ${ALGORITHMS:-tremor dmass} " == *" $ALGORITHM "* ]] || continue

    BATCH_ROW_TIMEOUT="$ROW_TIMEOUT_SECONDS"
    BATCH_TIME="$SBATCH_TIME"
    for entry in $ROW_TIMEOUT_BY_NODES; do
        if [[ "${entry%%=*}" = "$NODES" ]]; then
            BATCH_ROW_TIMEOUT="${entry#*=}"
            batch_minutes=$(( (RUNS_PER_TASK * (BATCH_ROW_TIMEOUT + ROW_TIMEOUT_GRACE_SECONDS) + 59) / 60 + 15 ))
            BATCH_TIME="$(printf "%d:%02d:00" $((batch_minutes / 60)) $((batch_minutes % 60)))"
        fi
    done
    EXTRA_SBATCH=()
    [[ -n "$SBATCH_PARTITION" ]] && EXTRA_SBATCH+=(--partition="$SBATCH_PARTITION")
    [[ -n "$SUBMIT_DEPENDENCY" ]] && EXTRA_SBATCH+=(--dependency="$SUBMIT_DEPENDENCY")

    ARRAY_SPEC="0-$((TASK_COUNT - 1))"
    if [[ -n "$ARRAY_THROTTLE" ]]; then
        ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_THROTTLE}"
    fi

    JOB_ID="$(
        sbatch --parsable --chdir="$REPO_ROOT" \
            --job-name="$BATCH_ID" \
            --output="$EVAL_ROOT/logs/$BATCH_ID.%A_%a.out" \
            --error="$EVAL_ROOT/logs/$BATCH_ID.%A_%a.err" \
            --nodes="$NODES" \
            --ntasks="$NODES" \
            --ntasks-per-node=1 \
            --exclusive --mem=0 \
            --cpus-per-task="$CPUS_PER_TASK" \
            --time="$BATCH_TIME" \
            "${EXTRA_SBATCH[@]}" \
            --array="$ARRAY_SPEC" \
            --export="ALL,PLAN_FILE=$PLAN_FILE,TREMOR_BIN=$TREMOR_BIN,DMASS_BIN=$MASS_BIN,CPUS_PER_TASK=$CPUS_PER_TASK,INDEX_THREADS=$INDEX_THREADS,SEARCH_WORKERS=$SEARCH_WORKERS,LEAF_SIZE=$LEAF_SIZE,PAA_SEGMENTS=$PAA_SEGMENTS,THREADS=$THREADS,THRESHOLD_RESULT_SLOTS=$THRESHOLD_RESULT_SLOTS,PQ_THRESHOLD_DIVISOR=$PQ_THRESHOLD_DIVISOR,TREMOR_MAX_LOCAL_SAMPLES=$TREMOR_MAX_LOCAL_SAMPLES,DMASS_MAX_LOCAL_SAMPLES=$DMASS_MAX_LOCAL_SAMPLES,MAX_THRESHOLD_SHARD_BYTES=$MAX_THRESHOLD_SHARD_BYTES,THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW=$THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW,THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD=$THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD,ROW_TIMEOUT_SECONDS=$BATCH_ROW_TIMEOUT,ROW_TIMEOUT_GRACE_SECONDS=$ROW_TIMEOUT_GRACE_SECONDS,ROW_MAX_ATTEMPTS=$ROW_MAX_ATTEMPTS" \
            "$ARRAY_JOB"
    )"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$BATCH_ID" "$DATASET" "$WORKLOAD" "$ALGORITHM" "$NODES" "$TASK_COUNT" "$RUN_COUNT" "$RUNS_PER_TASK" "$JOB_ID" "$PLAN_FILE" "$RESULT_DIR" \
        >> "$SUBMISSIONS"
    echo "    submitted $BATCH_ID tasks=$TASK_COUNT runs=$RUN_COUNT nodes=$NODES time=$BATCH_TIME row_timeout=$BATCH_ROW_TIMEOUT job=$JOB_ID"
done

JOB_IDS="$(tail -n +2 "$SUBMISSIONS" | cut -f9 | paste -sd: -)"
if [[ -n "$JOB_IDS" ]]; then
    ANALYSIS_JOB="$(sbatch --parsable --chdir="$REPO_ROOT" --dependency="afterany:$JOB_IDS" \
        --export="ALL,EVAL_ROOT=$EVAL_ROOT" "$SCRIPT_DIR/analyze_evaluation.slurm")"
    echo "[+] Analysis job $ANALYSIS_JOB runs after all batches"
fi

echo "[+] Submitted full evaluation. Not waiting for completion."
echo "    Eval root:    $EVAL_ROOT"
echo "    Manifest:     $EVAL_ROOT/manifest.tsv"
echo "    Submissions:  $SUBMISSIONS"
echo "    Run config:   $RUN_CONFIG"
echo "    Logs:         $EVAL_ROOT/logs"
echo "    Results:      $EVAL_ROOT/results"
