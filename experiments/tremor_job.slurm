#!/bin/bash
#
# Slurm job that runs a single TREMOR configuration.
# All parameters come from environment variables set by
# run_maule_tremor.sh via `sbatch --export`.
#
# Do not set --nodes / --ntasks / --cpus-per-task / --time here: the
# submitting script provides them on the sbatch command line so they can
# vary per configuration.

#SBATCH --job-name=tremor-run

set -e

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Runtime environment.
if command -v module >/dev/null 2>&1; then
    module load intel/21U2/suite
fi

# Required inputs (fail early with a clear message if unset).
: "${TREMOR_BIN:?TREMOR_BIN not set}"
: "${WAVEFORM:?WAVEFORM not set}"
: "${QUERIES:?QUERIES not set}"
: "${WINDOW_LENGTH:?WINDOW_LENGTH not set}"
: "${OUTPUT_PREFIX:?OUTPUT_PREFIX not set}"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

# Optional inputs (defaults).
: "${MODE:=threshold}"
: "${THRESHOLD:=0.10}"
: "${KNN_K:=10}"
: "${REPLICATION_GROUPS:=0}"
: "${INDEX_THREADS:=4}"
: "${SEARCH_WORKERS:=4}"
: "${LEAF_SIZE:=2000}"
: "${PAA_SEGMENTS:=16}"
: "${MERGE_OFFSET:=}"
: "${SRUN_MPI_TYPE:=pmi2}"

# A zero merge offset can produce enormous threshold-result CSVs for highly
# similar neighboring subsequences. Default to a quarter query/window length.
if [[ -z "$MERGE_OFFSET" || "$MERGE_OFFSET" = "0" ]]; then
    MERGE_OFFSET=$(( WINDOW_LENGTH / 4 ))
fi
if (( MERGE_OFFSET <= 0 )); then
    echo "[!] Invalid MERGE_OFFSET=$MERGE_OFFSET for WINDOW_LENGTH=$WINDOW_LENGTH" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -z "${I_MPI_PMI_LIBRARY:-}" ]]; then
    for pmi_library in /usr/lib64/libpmi2.so /usr/lib64/slurm/libpmi2.so /usr/lib/x86_64-linux-gnu/libpmi2.so /usr/lib/slurm/libpmi2.so; do
        if [[ -r "$pmi_library" ]]; then
            export I_MPI_PMI_LIBRARY="$pmi_library"
            break
        fi
    done
fi

SRUN_MPI_ARGS=()
if [[ -n "$SRUN_MPI_TYPE" ]]; then
    SRUN_MPI_ARGS=(--mpi="$SRUN_MPI_TYPE")
fi

echo "[job] nodes=${SLURM_NNODES:-?} tasks=${SLURM_NTASKS:-?} cpus/task=${SLURM_CPUS_PER_TASK:-?}"
echo "[job] mode=$MODE threshold=$THRESHOLD k=$KNN_K reps=$REPLICATION_GROUPS window=$WINDOW_LENGTH merge_offset=$MERGE_OFFSET"
echo "[job] waveform=$WAVEFORM"
echo "[job] queries=$QUERIES"
echo "[job] output prefix=$OUTPUT_PREFIX"

# Build the argument list step by step so it is easy to read/edit.
ARGS="--dataset $WAVEFORM"
ARGS="$ARGS --queries $QUERIES"
ARGS="$ARGS --window-length $WINDOW_LENGTH"
ARGS="$ARGS --mode $MODE"
ARGS="$ARGS --replication-groups $REPLICATION_GROUPS"
ARGS="$ARGS --index-threads $INDEX_THREADS"
ARGS="$ARGS --search-workers $SEARCH_WORKERS"
ARGS="$ARGS --leaf-size $LEAF_SIZE"
ARGS="$ARGS --paa-segments $PAA_SEGMENTS"
ARGS="$ARGS --merge-offset $MERGE_OFFSET"
ARGS="$ARGS --benchmark-output ${OUTPUT_PREFIX}.timings.csv"

if [ "$MODE" = "threshold" ]; then
    # In threshold mode --output is treated as a prefix and TREMOR writes
    # one file per rank as <prefix>_<rank>.csv.
    ARGS="$ARGS --threshold $THRESHOLD --output $OUTPUT_PREFIX"
else
    # In KNN mode --output is the full CSV path.
    ARGS="$ARGS --k $KNN_K --output ${OUTPUT_PREFIX}.csv"
fi

echo "[job] cmd: srun ${SRUN_MPI_ARGS[*]} -n ${SLURM_NTASKS:-1} $TREMOR_BIN $ARGS"
srun "${SRUN_MPI_ARGS[@]}" -n "${SLURM_NTASKS:-1}" "$TREMOR_BIN" $ARGS

if [ "$MODE" = "threshold" ]; then
    shopt -s nullglob
    threshold_outputs=("${OUTPUT_PREFIX}"_*.csv)
    shopt -u nullglob
    if [ "${#threshold_outputs[@]}" -eq 0 ]; then
        echo "[!] No threshold CSV outputs were produced for prefix ${OUTPUT_PREFIX}_<rank>.csv" >&2
        exit 1
    fi
    echo "[job] produced ${#threshold_outputs[@]} threshold shard(s)."
else
    if [ ! -s "${OUTPUT_PREFIX}.csv" ]; then
        echo "[!] Missing KNN output CSV: ${OUTPUT_PREFIX}.csv" >&2
        exit 1
    fi
    echo "[job] produced KNN CSV: ${OUTPUT_PREFIX}.csv"
fi

if [ ! -s "${OUTPUT_PREFIX}.timings.csv" ]; then
    echo "[!] Missing benchmark timings CSV: ${OUTPUT_PREFIX}.timings.csv" >&2
    exit 1
fi
echo "[job] produced benchmark timings: ${OUTPUT_PREFIX}.timings.csv"
