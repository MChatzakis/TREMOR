#!/bin/bash
#
# Submit one Slurm job per combination of:
#   waveform/template case  x  threshold  x  node count  x  replication groups
#
# Edit the lists in the "Sweep parameters" and "Cases" sections below to
# change what gets submitted. The script keeps things intentionally simple:
# plain lists in strings/arrays and a nested for-loop that calls sbatch.
#
# Usage:
#   ./run_maule_tremor.sh
#
# The script first runs `make all` in the repository root so the TREMOR
# binary is up to date before any job is submitted.

set -e

# --- Paths --------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TREMOR_BIN="$REPO_ROOT/bin/tremor_main"
JOB_SCRIPT="$SCRIPT_DIR/tremor_job.slurm"

RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# --- Sweep parameters ---------------------------------------------------
# Space-separated lists. Add or remove values here.

THRESHOLDS="0.10" #"0.01 0.05 0.10 0.20"
NODE_COUNTS="4" #"1 2 4 8 16"
REPLICATION_GROUPS="${MAULE_REPLICATION_GROUPS:-0}" #"1 2 4 8 16"

# Fixed TREMOR knobs (edit if needed).
MODE="threshold"                 # threshold or knn
# One tremor instance per server: --exclusive + --ntasks-per-node=1 (below)
# guarantee it. CPUS_PER_TASK is the number of cores Slurm reserves for the
# single MPI rank on that node -- set to the number of cores you want the
# rank to use (typically all cores on the node).
CPUS_PER_TASK=64
INDEX_THREADS=64
SEARCH_WORKERS=64
LEAF_SIZE=2000
# The current Maule windows are all divisible by 8, but not all by 16.
PAA_SEGMENTS=8
MERGE_OFFSET_RATIO=25
KNN_K=10                         # only used if MODE=knn

# --- Cases --------------------------------------------------------------
# The waveform / template / window-length list is shared with the DMASS
# driver. Edit experiments/cases_maule.sh to change it.
source "$SCRIPT_DIR/cases_maule.sh"

# --- Build TREMOR -------------------------------------------------------
# Make needs the Intel MPI compiler on this cluster; try to load it, but
# don't fail if the module system is not available (make will report a
# clear error if the compiler is missing).
if command -v module >/dev/null 2>&1; then
    module load intel/21U2/suite
fi

echo "[+] Compiling TREMOR in $REPO_ROOT"
( cd "$REPO_ROOT" && make all )

if [ ! -x "$TREMOR_BIN" ]; then
    echo "[!] TREMOR binary not found at $TREMOR_BIN" >&2
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
    if (( window % PAA_SEGMENTS != 0 )); then
        echo "[!] Invalid window length '$window' (must be divisible by PAA_SEGMENTS=$PAA_SEGMENTS)." >&2
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
    # files inherit the same naming convention as the tremor input.
    waveform_base=$(basename "$waveform" .bin)

    for th in $THRESHOLDS; do
        for nodes in $NODE_COUNTS; do
            for reps in $REPLICATION_GROUPS; do

                run_id="${waveform_base}.th${th}.n${nodes}.r${reps}"
                out_prefix="$RESULTS_DIR/$run_id"

                # Build the --export list. ALL keeps PATH / modules; the
                # rest overrides the tremor job parameters.
                EXPORTS="ALL"
                EXPORTS="$EXPORTS,TREMOR_BIN=$TREMOR_BIN"
                EXPORTS="$EXPORTS,WAVEFORM=$waveform"
                EXPORTS="$EXPORTS,QUERIES=$queries"
                EXPORTS="$EXPORTS,WINDOW_LENGTH=$window"
                EXPORTS="$EXPORTS,MODE=$MODE"
                EXPORTS="$EXPORTS,THRESHOLD=$th"
                EXPORTS="$EXPORTS,KNN_K=$KNN_K"
                EXPORTS="$EXPORTS,REPLICATION_GROUPS=$reps"
                EXPORTS="$EXPORTS,INDEX_THREADS=$INDEX_THREADS"
                EXPORTS="$EXPORTS,SEARCH_WORKERS=$SEARCH_WORKERS"
                EXPORTS="$EXPORTS,LEAF_SIZE=$LEAF_SIZE"
                EXPORTS="$EXPORTS,PAA_SEGMENTS=$PAA_SEGMENTS"
                EXPORTS="$EXPORTS,MERGE_OFFSET=$merge_offset"
                EXPORTS="$EXPORTS,OUTPUT_PREFIX=$out_prefix"

                echo "[+] sbatch $run_id  (nodes=$nodes threshold=$th reps=$reps)"

                sbatch \
                    --job-name="tremor.$run_id" \
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
echo "    Results:   $RESULTS_DIR"
echo "    Slurm logs: $LOGS_DIR"
