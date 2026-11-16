#!/bin/bash
#
# Submit one Slurm job per combination of:
#   waveform/template case  x  k  x  node count  x  replication groups
#
# Unlike Maule, seiFR is run as a top-k workload, so this driver only submits
# KNN-mode DMASS experiments.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DMASS_SRC="$REPO_ROOT/dmass/DMASS_V1.c"
DMASS_BIN="$REPO_ROOT/bin/dmass_main"
JOB_SCRIPT="$SCRIPT_DIR/dmass_job.slurm"

RESULTS_DIR="$SCRIPT_DIR/results_dmass_seifr"
LOGS_DIR="$SCRIPT_DIR/logs_dmass_seifr"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

K_VALUES="${SEIFR_K_VALUES:-10 50 100}"
NODE_COUNTS="${SEIFR_NODE_COUNTS:-4}"
REPLICATION_GROUPS="${SEIFR_REPLICATION_GROUPS:-0}"

MODE="knn-search"
CPUS_PER_TASK=64
THREADS=64
MERGE_OFFSET_RATIO=25

DMASS_CC="${DMASS_CC:-mpicc}"
DMASS_CFLAGS="${DMASS_CFLAGS:--O3 -std=gnu11 -fopenmp}"
DMASS_LDFLAGS="${DMASS_LDFLAGS:--lfftw3 -lfftw3_threads -lpthread -lm}"

source "$SCRIPT_DIR/cases_seifr.sh"

if command -v module >/dev/null 2>&1; then
    module load intel/21U2/suite
    if ! module load fftw/3.3.9; then
        echo "[!] Failed to load module fftw/3.3.9 required by DMASS." >&2
        exit 1
    fi
else
    echo "[!] module command not found; cannot load fftw/3.3.9 for DMASS build." >&2
    exit 1
fi

echo "[+] Compiling DMASS: $DMASS_CC ${CFLAGS:-} ${CPPFLAGS:-} $DMASS_CFLAGS -o $DMASS_BIN $DMASS_SRC ${LDFLAGS:-} $DMASS_LDFLAGS"
mkdir -p "$(dirname "$DMASS_BIN")"
$DMASS_CC ${CFLAGS:-} ${CPPFLAGS:-} $DMASS_CFLAGS -o "$DMASS_BIN" "$DMASS_SRC" ${LDFLAGS:-} $DMASS_LDFLAGS

if [[ ! -x "$DMASS_BIN" ]]; then
    echo "[!] DMASS binary not found at $DMASS_BIN" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "[!] sbatch not found in PATH. Run this on a Slurm login node." >&2
    exit 1
fi

validate_case_inputs() {
    local waveform="$1"
    local queries="$2"
    local window="$3"
    local merge_offset="$4"

    if [[ ! "$window" =~ ^[0-9]+$ ]] || (( window <= 0 )); then
        echo "[!] Invalid window length '$window'." >&2
        return 1
    fi
    if (( window % 8 != 0 )); then
        echo "[!] Invalid window length '$window' (must be divisible by 8 for SIMD)." >&2
        return 1
    fi
    local expected_merge_offset=$(( window * MERGE_OFFSET_RATIO / 100 ))
    if (( merge_offset != expected_merge_offset || merge_offset <= 0 )); then
        echo "[!] Invalid merge offset '$merge_offset' for window '$window'." >&2
        return 1
    fi
    if [[ ! -f "$waveform" ]]; then
        echo "[!] Waveform file not found: $waveform" >&2
        return 1
    fi
    if [[ ! -f "$queries" ]]; then
        echo "[!] Query/template file not found: $queries" >&2
        return 1
    fi
}

job_count=0
for case_line in "${CASES[@]}"; do
    set -- $case_line
    waveform=$1
    queries=$2
    window=$3
    merge_offset=$(( window * MERGE_OFFSET_RATIO / 100 ))

    validate_case_inputs "$waveform" "$queries" "$window" "$merge_offset"

    waveform_base=$(basename "$waveform" .bin)

    for k in $K_VALUES; do
        for nodes in $NODE_COUNTS; do
            for reps in $REPLICATION_GROUPS; do
                run_id="${waveform_base}.k${k}.n${nodes}.r${reps}"
                out_prefix="$RESULTS_DIR/$run_id"

                EXPORTS="ALL"
                EXPORTS="$EXPORTS,DMASS_BIN=$DMASS_BIN"
                EXPORTS="$EXPORTS,WAVEFORM=$waveform"
                EXPORTS="$EXPORTS,QUERIES=$queries"
                EXPORTS="$EXPORTS,WINDOW_LENGTH=$window"
                EXPORTS="$EXPORTS,MODE=$MODE"
                EXPORTS="$EXPORTS,KNN_K=$k"
                EXPORTS="$EXPORTS,REPLICATION_GROUPS=$reps"
                EXPORTS="$EXPORTS,THREADS=$THREADS"
                EXPORTS="$EXPORTS,MERGE_OFFSET=$merge_offset"
                EXPORTS="$EXPORTS,OUTPUT_PREFIX=$out_prefix"

                echo "[+] sbatch $run_id  (nodes=$nodes k=$k reps=$reps)"

                sbatch \
                    --job-name="dmass.$run_id" \
                    --output="$LOGS_DIR/$run_id.%j.out" \
                    --error="$LOGS_DIR/$run_id.%j.err" \
                    --nodes="$nodes" \
                    --ntasks="$nodes" \
                    --ntasks-per-node=1 \
                    --exclusive \
                    --cpus-per-task="$CPUS_PER_TASK" \
                    --export="$EXPORTS" \
                    "$JOB_SCRIPT"

                job_count=$((job_count + 1))
            done
        done
    done
done

echo "[+] Submitted $job_count seiFR DMASS jobs."
echo "    Results:    $RESULTS_DIR"
echo "    Slurm logs: $LOGS_DIR"
