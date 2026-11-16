#!/bin/bash
#
# Submit one Slurm job per combination of:
#   waveform/template case  x  threshold  x  node count  x  replication groups
#
# Same structure as run_maule_tremor.sh, but runs the DMASS reference
# implementation (dmass/DMASS_V1.c) instead of TREMOR. The list of cases is
# read from experiments/cases_maule.sh so both drivers use identical inputs.
#
# Usage:
#   ./run_maule_dmass.sh
#
# The script compiles dmass/DMASS_V1.c before submitting any job.

set -e

# --- Paths --------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DMASS_SRC="$REPO_ROOT/dmass/DMASS_V1.c"
DMASS_BIN="$REPO_ROOT/bin/dmass_main"
JOB_SCRIPT="$SCRIPT_DIR/dmass_job.slurm"

RESULTS_DIR="$SCRIPT_DIR/results_dmass"
LOGS_DIR="$SCRIPT_DIR/logs_dmass"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# --- Sweep parameters ---------------------------------------------------
# Space-separated lists. Add or remove values here.

THRESHOLDS="0.10" #"0.01 0.05 0.10 0.20"
NODE_COUNTS="4" #"1 2 4 8 16"
REPLICATION_GROUPS="${MAULE_REPLICATION_GROUPS:-0}" #"1 2 4 8 16"

# Fixed DMASS knobs (edit if needed).
MODE="threshold-search"          # threshold-search or knn-search
# One dmass instance per server: --exclusive + --ntasks-per-node=1 (below)
# guarantee it. CPUS_PER_TASK is the number of cores Slurm reserves for the
# single MPI rank on that node -- DMASS uses this many pthreads.
CPUS_PER_TASK=64
THREADS=64
MERGE_OFFSET_RATIO=25
KNN_K=10                         # only used if MODE=knn-search

# --- Compiler / libraries -----------------------------------------------
# DMASS needs MPI + pthreads + FFTW3 (double). Adjust these if the cluster
# uses different module / library names.
DMASS_CC="${DMASS_CC:-mpicc}"
# The FFTW module exports CFLAGS/CPPFLAGS/LDFLAGS; fold them into the build
# so the compiler sees both headers and the library search path.
DMASS_CFLAGS="${DMASS_CFLAGS:--O3 -std=gnu11 -fopenmp}"
DMASS_LDFLAGS="${DMASS_LDFLAGS:--lfftw3 -lfftw3_threads -lpthread -lm}"

# --- Cases --------------------------------------------------------------
# Shared with the tremor driver.
source "$SCRIPT_DIR/cases_maule.sh"

# --- Build DMASS --------------------------------------------------------
if command -v module >/dev/null 2>&1; then
    module load intel/21U2/suite
    if ! module load fftw/3.3.9; then
        echo "[!] Failed to load module fftw/3.3.9 required by DMASS." >&2
        echo "    Load it manually or adjust the module name in experiments/run_maule_dmass.sh." >&2
        exit 1
    fi
else
    echo "[!] module command not found; cannot load fftw/3.3.9 for DMASS build." >&2
    echo "    Ensure FFTW libraries are available in compiler/linker paths." >&2
    exit 1
fi

echo "[+] Compiling DMASS: $DMASS_CC ${CFLAGS:-} ${CPPFLAGS:-} $DMASS_CFLAGS -o $DMASS_BIN $DMASS_SRC ${LDFLAGS:-} $DMASS_LDFLAGS"
mkdir -p "$(dirname "$DMASS_BIN")"
$DMASS_CC ${CFLAGS:-} ${CPPFLAGS:-} $DMASS_CFLAGS -o "$DMASS_BIN" "$DMASS_SRC" ${LDFLAGS:-} $DMASS_LDFLAGS

if [ ! -x "$DMASS_BIN" ]; then
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

    if [[ "$waveform" == *"<"* || "$queries" == *"<"* || "$window" == *"<"* ]]; then
        echo "[!] Invalid case contains unresolved placeholders:" >&2
        echo "    waveform=$waveform" >&2
        echo "    queries=$queries" >&2
        echo "    window=$window" >&2
        return 1
    fi

    if [[ ! "$window" =~ ^[0-9]+$ ]]; then
        echo "[!] Invalid window length '$window' (must be a positive integer)." >&2
        return 1
    fi
    if (( window <= 0 )); then
        echo "[!] Invalid window length '$window' (must be > 0)." >&2
        return 1
    fi
    if (( window % 8 != 0 )); then
        echo "[!] Invalid window length '$window' (must be divisible by 8 for SIMD)." >&2
        return 1
    fi
    local expected_merge_offset=$(( window * MERGE_OFFSET_RATIO / 100 ))
    if (( merge_offset != expected_merge_offset )); then
        echo "[!] Invalid merge offset '$merge_offset' for window length '$window'." >&2
        echo "    Expected merge_offset=$expected_merge_offset (${MERGE_OFFSET_RATIO}% of window length, integer arithmetic)." >&2
        return 1
    fi
    if (( merge_offset <= 0 )); then
        echo "[!] Invalid merge offset '$merge_offset' (must be greater than 0)." >&2
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

    if [[ "$queries" =~ \.len([0-9]+)\. ]]; then
        local query_len="${BASH_REMATCH[1]}"
        if [[ "$query_len" != "$window" ]]; then
            echo "[!] Case mismatch: window_length=$window but template filename encodes len$query_len" >&2
            echo "    queries=$queries" >&2
            return 1
        fi
    fi
}

# --- Sweep --------------------------------------------------------------
job_count=0
for case_line in "${CASES[@]}"; do
    # Split the case line into fields:
    #   $1 = waveform.bin, $2 = queries.bin, $3 = window_length
    set -- $case_line
    waveform=$1
    queries=$2
    window=$3
    merge_offset=$(( window * MERGE_OFFSET_RATIO / 100 ))

    validate_case_inputs "$waveform" "$queries" "$window" "$merge_offset"

    # Derive a run id from the waveform file name so the benchmark output
    # files inherit the same naming convention as the tremor / dmass input.
    waveform_base=$(basename "$waveform" .bin)

    for th in $THRESHOLDS; do
        for nodes in $NODE_COUNTS; do
            for reps in $REPLICATION_GROUPS; do

                run_id="${waveform_base}.th${th}.n${nodes}.r${reps}"
                out_prefix="$RESULTS_DIR/$run_id"

                EXPORTS="ALL"
                EXPORTS="$EXPORTS,DMASS_BIN=$DMASS_BIN"
                EXPORTS="$EXPORTS,WAVEFORM=$waveform"
                EXPORTS="$EXPORTS,QUERIES=$queries"
                EXPORTS="$EXPORTS,WINDOW_LENGTH=$window"
                EXPORTS="$EXPORTS,MODE=$MODE"
                EXPORTS="$EXPORTS,THRESHOLD=$th"
                EXPORTS="$EXPORTS,KNN_K=$KNN_K"
                EXPORTS="$EXPORTS,REPLICATION_GROUPS=$reps"
                EXPORTS="$EXPORTS,THREADS=$THREADS"
                EXPORTS="$EXPORTS,MERGE_OFFSET=$merge_offset"
                EXPORTS="$EXPORTS,OUTPUT_PREFIX=$out_prefix"

                echo "[+] sbatch $run_id  (nodes=$nodes threshold=$th reps=$reps)"

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

echo "[+] Submitted $job_count jobs."
echo "    Results:    $RESULTS_DIR"
echo "    Slurm logs: $LOGS_DIR"
