#!/bin/bash
#
# Slurm array task for one full-evaluation plan row.
# The submitter sets PLAN_FILE and binary/knob environment variables.

#SBATCH --job-name=full-eval

set -e

cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${PLAN_FILE:?PLAN_FILE not set}"
: "${TREMOR_BIN:?TREMOR_BIN not set}"
: "${DMASS_BIN:?DMASS_BIN not set}"

: "${CPUS_PER_TASK:=64}"
: "${INDEX_THREADS:=64}"
: "${SEARCH_WORKERS:=64}"
: "${LEAF_SIZE:=2000}"
: "${PAA_SEGMENTS:=8}"
: "${THREADS:=64}"
: "${THRESHOLD_RESULT_SLOTS:=1}"
: "${PQ_THRESHOLD_DIVISOR:=16}"
: "${TREMOR_MAX_LOCAL_SAMPLES:=600000000}"
: "${DMASS_MAX_LOCAL_SAMPLES:=350000000}"
: "${MAX_THRESHOLD_SHARD_BYTES:=1073741824}"
: "${THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW:=64}"
: "${THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD:=0.1}"
: "${ROW_TIMEOUT_SECONDS:=10800}"
: "${ROW_TIMEOUT_GRACE_SECONDS:=120}"
: "${ROW_MAX_ATTEMPTS:=3}"
: "${FULL_EVAL_DRY_RUN_TASK:=0}"
: "${SRUN_MPI_TYPE:=pmi2}"
# Optional command placed before either engine's binary inside srun, e.g.
# "numactl --interleave=all" (applied identically to TREMOR and DMASS).
: "${LAUNCH_WRAPPER:=}"
# Optional TREMOR PAA segments per window length, e.g. "1000=10 992=16";
# windows not listed use PAA_SEGMENTS.
: "${PAA_BY_WINDOW:=}"
# Optional TREMOR --refinement (leaf|adaptive); unset keeps the binary default (leaf).
: "${TREMOR_REFINEMENT:=}"
# Optional TREMOR --scan-fallback-fraction (adaptive refinement cutoff).
: "${TREMOR_SCAN_FALLBACK_FRACTION:=}"
# Optional TREMOR --method (tremor|skip-sequential|sequential); skip-sequential is the scan baseline (SSS).
: "${TREMOR_METHOD:=}"
read -r -a LAUNCH_CMD <<< "$LAUNCH_WRAPPER"

paa_for_window() {
    local entry
    for entry in $PAA_BY_WINDOW; do
        if [[ "${entry%%=*}" = "$1" ]]; then
            printf "%s" "${entry#*=}"
            return
        fi
    done
    printf "%s" "$PAA_SEGMENTS"
}

if command -v module >/dev/null 2>&1; then
    module load intel/21U2/suite
fi

if [[ -z "${I_MPI_PMI_LIBRARY:-}" ]]; then
    for pmi_library in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so /usr/lib/x86_64-linux-gnu/libpmi2.so /usr/lib/slurm/libpmi2.so; do
        if [[ -r "$pmi_library" ]]; then
            export I_MPI_PMI_LIBRARY="$pmi_library"
            break
        fi
    done
fi

SRUN_MPI_ARGS=(--kill-on-bad-exit=1)
if [[ -n "$SRUN_MPI_TYPE" ]]; then
    SRUN_MPI_ARGS+=(--mpi="$SRUN_MPI_TYPE")
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"

estimated_max_local_samples() {
    local total_samples="$1"
    local window_length="$2"
    local nodes="$3"
    local requested_groups="$4"

    local groups="$requested_groups"
    if (( groups == 0 )); then
        groups="$nodes"
    fi
    if (( groups <= 0 || groups > nodes )); then
        return 2
    fi

    local base=$(( total_samples / groups ))
    local last=$(( total_samples - base * (groups - 1) ))
    local overlap=$(( window_length - 1 ))
    local max_with_overlap="$base"
    if (( groups > 1 )); then
        max_with_overlap=$(( base + overlap ))
    fi

    if (( last > max_with_overlap )); then
        echo "$last"
    else
        echo "$max_with_overlap"
    fi
}

estimated_max_local_owned_subsequences() {
    local total_samples="$1"
    local window_length="$2"
    local nodes="$3"
    local requested_groups="$4"

    local groups="$requested_groups"
    if (( groups == 0 )); then
        groups="$nodes"
    fi
    if (( groups <= 0 || groups > nodes )); then
        return 2
    fi
    if (( total_samples < window_length )); then
        echo 0
        return 0
    fi

    local total_subsequences=$(( total_samples - window_length + 1 ))
    local base=$(( total_subsequences / groups ))
    local last=$(( total_subsequences - base * (groups - 1) ))

    if (( last > base )); then
        echo "$last"
    else
        echo "$base"
    fi
}

threshold_guard_applies() {
    awk -v threshold="$1" -v max_threshold="$THRESHOLD_OUTPUT_GUARD_MAX_THRESHOLD" \
        'BEGIN { exit !((threshold + 0.0) <= (max_threshold + 0.0)) }'
}

write_skip_record() {
    local output_prefix="$1"
    local run_id="$2"
    local reason="$3"
    local estimated_local_samples="$4"
    local max_local_samples="$5"
    local effective_nodes="$6"
    local replication_groups="$7"
    local window_length="$8"
    local threshold="${9:-NA}"
    local estimated_threshold_rows="${10:-NA}"
    local estimated_threshold_bytes="${11:-NA}"

    local skip_file="${output_prefix}.skipped.tsv"
    mkdir -p "$(dirname "$skip_file")"
    printf "run_id\treason\testimated_local_samples\tmax_local_samples\tnodes\treplication_groups\twindow_length\tthreshold\testimated_threshold_rows\testimated_threshold_bytes\tmax_threshold_shard_bytes\n" > "$skip_file"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$run_id" "$reason" "$estimated_local_samples" "$max_local_samples" \
        "$effective_nodes" "$replication_groups" "$window_length" "$threshold" \
        "$estimated_threshold_rows" "$estimated_threshold_bytes" "$MAX_THRESHOLD_SHARD_BYTES" >> "$skip_file"
    rm -f "${output_prefix}.running.tsv"
    echo "[skip] $reason"
    echo "[skip] wrote $skip_file"
}

write_running_record() {
    local output_prefix="$1"
    local run_id="$2"
    local algorithm="$3"
    local workload="$4"
    local effective_nodes="$5"

    local running_file="${output_prefix}.running.tsv"
    mkdir -p "$(dirname "$running_file")"
    printf "run_id\talgorithm\tworkload\tnodes\tplan_file\tarray_task\tjob_id\tstarted_at\n" > "$running_file"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$run_id" "$algorithm" "$workload" "$effective_nodes" "$PLAN_FILE" \
        "${SLURM_ARRAY_TASK_ID}" "${SLURM_JOB_ID:-NA}" "$(date -Iseconds)" >> "$running_file"
}

clear_running_record() {
    local output_prefix="$1"
    rm -f "${output_prefix}.running.tsv"
}

cleanup_run_outputs() {
    local output_prefix="$1"
    shopt -s nullglob
    local stale=(
        "${output_prefix}.csv"
        "${output_prefix}.timings.csv"
        "${output_prefix}.failed.tsv"
        "${output_prefix}.skipped.tsv"
        "${output_prefix}.running.tsv"
        "${output_prefix}"_*.csv
    )
    if (( ${#stale[@]} > 0 )); then
        rm -f "${stale[@]}"
    fi
    shopt -u nullglob
}

write_failure_record() {
    local output_prefix="$1"
    local run_id="$2"
    local reason="$3"
    local exit_code="$4"
    local algorithm="$5"
    local workload="$6"
    local effective_nodes="$7"

    local failure_file="${output_prefix}.failed.tsv"
    mkdir -p "$(dirname "$failure_file")"
    printf "run_id\treason\texit_code\talgorithm\tworkload\tnodes\tplan_file\tarray_task\n" > "$failure_file"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$run_id" "$reason" "$exit_code" "$algorithm" "$workload" "$effective_nodes" "$PLAN_FILE" "${SLURM_ARRAY_TASK_ID}" >> "$failure_file"
    rm -f "${output_prefix}.timings.csv"
    clear_running_record "$output_prefix"
    echo "[fail] $reason exit_code=$exit_code"
    echo "[fail] wrote $failure_file"
}

run_srun_with_optional_timeout() (
    # Bound each result file while it is written, including correlations >0.1.
    ulimit -f "$(( (MAX_THRESHOLD_SHARD_BYTES + 1023) / 1024 ))"
    if (( ROW_TIMEOUT_SECONDS > 0 )) && command -v timeout >/dev/null 2>&1; then
        timeout --kill-after="${ROW_TIMEOUT_GRACE_SECONDS}s" "${ROW_TIMEOUT_SECONDS}s" "$@" < /dev/null
    else
        "$@" < /dev/null
    fi
)

failure_reason_for_status() {
    local algorithm="$1"
    local status="$2"

    if (( ROW_TIMEOUT_SECONDS > 0 && status == 124 )); then
        printf "%s_row_timeout" "$algorithm"
    else
        printf "%s_srun_failed" "$algorithm"
    fi
}

maybe_skip_threshold_output() {
    local output_prefix="$1"
    local run_id="$2"
    local workload="$3"
    local threshold="$4"
    local query_count="$5"
    local window_length="$6"
    local merge_offset="$7"
    local waveform_samples="$8"
    local effective_nodes="$9"
    local replication_groups="${10}"
    local estimated_local_samples="${11}"
    local max_local_samples="${12}"

    if [[ "$workload" != "threshold" || "$threshold" = "NA" ]]; then
        return 1
    fi
    if ! threshold_guard_applies "$threshold"; then
        return 1
    fi

    local estimated_owned_subsequences
    if ! estimated_owned_subsequences="$(estimated_max_local_owned_subsequences "$waveform_samples" "$window_length" "$effective_nodes" "$replication_groups")"; then
        write_skip_record "$output_prefix" "$run_id" "invalid_replication_groups" "$estimated_local_samples" "$max_local_samples" "$effective_nodes" "$replication_groups" "$window_length" "$threshold" "NA" "NA"
        return 0
    fi

    local stride=$(( merge_offset + 1 ))
    if (( stride <= 0 )); then
        stride=1
    fi
    local estimated_rows=$(( query_count * ((estimated_owned_subsequences + stride - 1) / stride) ))
    local estimated_bytes=$(( estimated_rows * THRESHOLD_OUTPUT_AVG_BYTES_PER_ROW ))

    if (( estimated_bytes > MAX_THRESHOLD_SHARD_BYTES )); then
        write_skip_record "$output_prefix" "$run_id" "threshold_output_estimate_exceeds_safe_limit" "$estimated_local_samples" "$max_local_samples" "$effective_nodes" "$replication_groups" "$window_length" "$threshold" "$estimated_rows" "$estimated_bytes"
        return 0
    fi

    return 1
}

run_plan_line() {
    local line="$1"
    local ARRAY_INDEX RUN_ID DATASET WORKLOAD ALGORITHM MODE NODES REPLICATION_GROUPS CASE_ID
    local QUERY_COUNT WINDOW_LENGTH MERGE_OFFSET THRESHOLD KNN_K WAVEFORM_SAMPLES QUERY_FLOATS
    local WAVEFORM QUERIES OUTPUT_PREFIX RESULT_KIND
    local old_ifs="$IFS"

    IFS=$'\t'
    read -r \
        ARRAY_INDEX RUN_ID DATASET WORKLOAD ALGORITHM MODE NODES REPLICATION_GROUPS CASE_ID \
        QUERY_COUNT WINDOW_LENGTH MERGE_OFFSET THRESHOLD KNN_K WAVEFORM_SAMPLES QUERY_FLOATS \
        WAVEFORM QUERIES OUTPUT_PREFIX RESULT_KIND <<< "$line"
    IFS="$old_ifs"

    if [[ -z "$RUN_ID" || -z "$OUTPUT_PREFIX" || "$ARRAY_INDEX" = "array_index" ]]; then
        echo "[!] Failed to parse plan row for array task ${SLURM_ARRAY_TASK_ID}" >&2
        echo "    PLAN_FILE=$PLAN_FILE" >&2
        echo "    line=$line" >&2
        return 1
    fi

    mkdir -p "$(dirname "$OUTPUT_PREFIX")"
    cleanup_run_outputs "$OUTPUT_PREFIX"

    echo "[job] run_id=$RUN_ID"
    echo "[job] dataset=$DATASET workload=$WORKLOAD algorithm=$ALGORITHM mode=$MODE"
    echo "[job] nodes=${SLURM_NNODES:-$NODES} tasks=${SLURM_NTASKS:-$NODES} cpus/task=${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"
    echo "[job] case=$CASE_ID query_count=$QUERY_COUNT window=$WINDOW_LENGTH merge_offset=$MERGE_OFFSET"
    echo "[job] waveform=$WAVEFORM samples=$WAVEFORM_SAMPLES"
    echo "[job] queries=$QUERIES query_floats=$QUERY_FLOATS"
    echo "[job] output_prefix=$OUTPUT_PREFIX"
    write_running_record "$OUTPUT_PREFIX" "$RUN_ID" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"

    if [[ "$ALGORITHM" = "tremor" ]]; then
        local effective_nodes="${SLURM_NTASKS:-$NODES}"
        local estimated_local_samples
        if ! estimated_local_samples="$(estimated_max_local_samples "$WAVEFORM_SAMPLES" "$WINDOW_LENGTH" "$effective_nodes" "$REPLICATION_GROUPS")"; then
            write_skip_record "$OUTPUT_PREFIX" "$RUN_ID" "invalid_replication_groups" "NA" "$TREMOR_MAX_LOCAL_SAMPLES" "$effective_nodes" "$REPLICATION_GROUPS" "$WINDOW_LENGTH" "$THRESHOLD" "NA" "NA"
            return 0
        fi
        if (( estimated_local_samples > TREMOR_MAX_LOCAL_SAMPLES )); then
            write_skip_record "$OUTPUT_PREFIX" "$RUN_ID" "tremor_local_chunk_exceeds_safe_limit" "$estimated_local_samples" "$TREMOR_MAX_LOCAL_SAMPLES" "$effective_nodes" "$REPLICATION_GROUPS" "$WINDOW_LENGTH" "$THRESHOLD" "NA" "NA"
            return 0
        fi
        if maybe_skip_threshold_output "$OUTPUT_PREFIX" "$RUN_ID" "$WORKLOAD" "$THRESHOLD" "$QUERY_COUNT" "$WINDOW_LENGTH" "$MERGE_OFFSET" "$WAVEFORM_SAMPLES" "$effective_nodes" "$REPLICATION_GROUPS" "$estimated_local_samples" "$TREMOR_MAX_LOCAL_SAMPLES"; then
            return 0
        fi

        local -a ARGS
        ARGS=(
            --dataset "$WAVEFORM"
            --dataset-size "$WAVEFORM_SAMPLES"
            --queries "$QUERIES"
            --query-count "$QUERY_COUNT"
            --window-length "$WINDOW_LENGTH"
            --mode "$MODE"
            --replication-groups "$REPLICATION_GROUPS"
            --index-threads "$INDEX_THREADS"
            --search-workers "$SEARCH_WORKERS"
            --leaf-size "$LEAF_SIZE"
            --paa-segments "$(paa_for_window "$WINDOW_LENGTH")"
            --pq-threshold-divisor "$PQ_THRESHOLD_DIVISOR"
            --merge-offset "$MERGE_OFFSET"
            --benchmark-output "${OUTPUT_PREFIX}.timings.csv"
        )
        if [[ -n "$TREMOR_REFINEMENT" ]]; then
            ARGS+=(--refinement "$TREMOR_REFINEMENT")
        fi
        if [[ -n "$TREMOR_SCAN_FALLBACK_FRACTION" ]]; then
            ARGS+=(--scan-fallback-fraction "$TREMOR_SCAN_FALLBACK_FRACTION")
        fi
        if [[ -n "$TREMOR_METHOD" ]]; then
            ARGS+=(--method "$TREMOR_METHOD")
        fi

        if [[ "$WORKLOAD" = "threshold" ]]; then
            ARGS+=(--threshold "$THRESHOLD" --threshold-result-slots "$THRESHOLD_RESULT_SLOTS" --output "$OUTPUT_PREFIX")
        else
            ARGS+=(--k "$KNN_K" --output "${OUTPUT_PREFIX}.csv")
        fi

        printf "[job] cmd:"
        printf " %q" srun "${SRUN_MPI_ARGS[@]}" -n "${SLURM_NTASKS:-$NODES}" "${LAUNCH_CMD[@]}" "$TREMOR_BIN" "${ARGS[@]}"
        printf "\n"
        if [[ "$FULL_EVAL_DRY_RUN_TASK" = "1" ]]; then
            echo "[job] FULL_EVAL_DRY_RUN_TASK=1, command not executed"
            clear_running_record "$OUTPUT_PREFIX"
            return 0
        fi
        run_srun_with_optional_timeout srun "${SRUN_MPI_ARGS[@]}" -n "${SLURM_NTASKS:-$NODES}" "${LAUNCH_CMD[@]}" "$TREMOR_BIN" "${ARGS[@]}"
        local command_status=$?
        if (( command_status != 0 )); then
            write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "$(failure_reason_for_status tremor "$command_status")" "$command_status" "$ALGORITHM" "$WORKLOAD" "$effective_nodes"
            return "$command_status"
        fi
    elif [[ "$ALGORITHM" = "dmass" ]]; then
        if command -v module >/dev/null 2>&1; then
            module load fftw/3.3.9
        fi

        local effective_nodes="${SLURM_NTASKS:-$NODES}"
        local estimated_local_samples
        if ! estimated_local_samples="$(estimated_max_local_samples "$WAVEFORM_SAMPLES" "$WINDOW_LENGTH" "$effective_nodes" "$REPLICATION_GROUPS")"; then
            write_skip_record "$OUTPUT_PREFIX" "$RUN_ID" "invalid_replication_groups" "NA" "$DMASS_MAX_LOCAL_SAMPLES" "$effective_nodes" "$REPLICATION_GROUPS" "$WINDOW_LENGTH" "$THRESHOLD" "NA" "NA"
            return 0
        fi
        if (( estimated_local_samples > DMASS_MAX_LOCAL_SAMPLES )); then
            write_skip_record "$OUTPUT_PREFIX" "$RUN_ID" "dmass_local_chunk_exceeds_safe_limit" "$estimated_local_samples" "$DMASS_MAX_LOCAL_SAMPLES" "$effective_nodes" "$REPLICATION_GROUPS" "$WINDOW_LENGTH" "$THRESHOLD" "NA" "NA"
            return 0
        fi
        if maybe_skip_threshold_output "$OUTPUT_PREFIX" "$RUN_ID" "$WORKLOAD" "$THRESHOLD" "$QUERY_COUNT" "$WINDOW_LENGTH" "$MERGE_OFFSET" "$WAVEFORM_SAMPLES" "$effective_nodes" "$REPLICATION_GROUPS" "$estimated_local_samples" "$DMASS_MAX_LOCAL_SAMPLES"; then
            return 0
        fi

        local -a ARGS
        ARGS=(
            --dataset "$WAVEFORM"
            --dataset-size "$WAVEFORM_SAMPLES"
            --queries "$QUERIES"
            --queries-size "$QUERY_COUNT"
            --window-length "$WINDOW_LENGTH"
            --mode "$MODE"
            --replication-groups "$REPLICATION_GROUPS"
            --threads "$THREADS"
            --merge-offset "$MERGE_OFFSET"
            --benchmark-output "${OUTPUT_PREFIX}.timings.csv"
        )

        if [[ "$WORKLOAD" = "threshold" ]]; then
            ARGS+=(--threshold "$THRESHOLD" --output "$OUTPUT_PREFIX")
        else
            ARGS+=(--k "$KNN_K" --output "${OUTPUT_PREFIX}.csv")
        fi

        printf "[job] cmd:"
        printf " %q" srun "${SRUN_MPI_ARGS[@]}" -n "${SLURM_NTASKS:-$NODES}" "${LAUNCH_CMD[@]}" "$DMASS_BIN" "${ARGS[@]}"
        printf "\n"
        if [[ "$FULL_EVAL_DRY_RUN_TASK" = "1" ]]; then
            echo "[job] FULL_EVAL_DRY_RUN_TASK=1, command not executed"
            clear_running_record "$OUTPUT_PREFIX"
            return 0
        fi
        run_srun_with_optional_timeout srun "${SRUN_MPI_ARGS[@]}" -n "${SLURM_NTASKS:-$NODES}" "${LAUNCH_CMD[@]}" "$DMASS_BIN" "${ARGS[@]}"
        local command_status=$?
        if (( command_status != 0 )); then
            write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "$(failure_reason_for_status dmass "$command_status")" "$command_status" "$ALGORITHM" "$WORKLOAD" "$effective_nodes"
            return "$command_status"
        fi
    else
        echo "[!] Unknown algorithm: $ALGORITHM" >&2
        return 1
    fi

    if [[ "$WORKLOAD" = "threshold" ]]; then
        shopt -s nullglob
        outputs=("${OUTPUT_PREFIX}"_*.csv)
        shopt -u nullglob
        if [[ "${#outputs[@]}" -ne "${SLURM_NTASKS:-$NODES}" ]]; then
            echo "[!] Missing threshold shard outputs for ${OUTPUT_PREFIX}_<rank>.csv" >&2
            write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "missing_threshold_outputs" "1" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"
            return 1
        fi
        for (( rank=0; rank<${SLURM_NTASKS:-$NODES}; rank++ )); do
            [[ -f "${OUTPUT_PREFIX}_${rank}.csv" ]] || return 1
        done
        for output in "${outputs[@]}"; do
            output_size="$(stat -c '%s' "$output")"
            if (( output_size > MAX_THRESHOLD_SHARD_BYTES )); then
                echo "[!] Threshold shard exceeds safe size: $output (${output_size} bytes > ${MAX_THRESHOLD_SHARD_BYTES})" >&2
                write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "threshold_shard_exceeds_safe_limit" "1" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"
                return 1
            fi
            first_line=""
            IFS= read -r first_line < "$output" || true
            if [[ "$first_line" != "Query, Counter, Position, Cross Correlation" ]]; then
                echo "[!] Malformed threshold shard header: $output" >&2
                write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "malformed_threshold_output" "1" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"
                return 1
            fi
        done
        echo "[job] produced ${#outputs[@]} threshold shard(s)"
    else
        if [[ ! -s "${OUTPUT_PREFIX}.csv" ]]; then
            echo "[!] Missing KNN output CSV: ${OUTPUT_PREFIX}.csv" >&2
            write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "missing_knn_output" "1" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"
            return 1
        fi
        echo "[job] produced KNN CSV: ${OUTPUT_PREFIX}.csv"
    fi

    if [[ ! -s "${OUTPUT_PREFIX}.timings.csv" ]]; then
        echo "[!] Missing benchmark timings CSV: ${OUTPUT_PREFIX}.timings.csv" >&2
        write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "missing_timings_output" "1" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"
        return 1
    fi
    python3 "${SLURM_SUBMIT_DIR:-$PWD}/experiments/validate_run_outputs.py" "$OUTPUT_PREFIX" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}" "$QUERY_COUNT" "$KNN_K" || {
        write_failure_record "$OUTPUT_PREFIX" "$RUN_ID" "invalid_csv_outputs" "1" "$ALGORITHM" "$WORKLOAD" "${SLURM_NTASKS:-$NODES}"
        return 1
    }
    echo "[job] produced timings CSV: ${OUTPUT_PREFIX}.timings.csv"
    clear_running_record "$OUTPUT_PREFIX"
}

rows="$(
    awk -v idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}" 'BEGIN { FS="\t" } NR > 1 && $1 == idx { print; found=1 } END { if (!found) exit 2 }' "$PLAN_FILE"
)"

echo "[bundle] plan_file=$PLAN_FILE array_task=${SLURM_ARRAY_TASK_ID}"
run_count=0
failure_count=0
while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    run_count=$((run_count + 1))
    echo "[bundle] starting run $run_count"
    for (( attempt=1; attempt<=ROW_MAX_ATTEMPTS; attempt++ )); do
        set +e
        run_plan_line "$line"
        run_status=$?
        set -e
        if (( run_status == 0 )); then
            break
        fi
        echo "[bundle] run $run_count attempt $attempt/$ROW_MAX_ATTEMPTS failed with status $run_status"
        # A row timeout is deterministic; retrying only burns the allocation.
        if (( ROW_TIMEOUT_SECONDS > 0 && run_status == 124 )); then
            echo "[bundle] run $run_count hit the row timeout; not retrying"
            break
        fi
    done
    if (( run_status != 0 )); then
        failure_count=$((failure_count + 1))
        echo "[bundle] run $run_count failed with status $run_status after $ROW_MAX_ATTEMPTS attempt(s)"
    fi
done <<< "$rows"

echo "[bundle] completed $run_count run(s), failures=$failure_count"
if (( failure_count > 0 )); then
    exit 1
fi
