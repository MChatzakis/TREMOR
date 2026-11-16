#!/bin/bash
#
# Run the full Maule benchmark: submit TREMOR and/or DMASS sweeps, wait for
# the Slurm jobs to finish, and then verify that both algorithms produced
# similar results.
#
# Toggles (env vars OR flags, all default to enabled):
#   RUN_TREMOR=1  --tremor-only         run only TREMOR
#   RUN_DMASS=1   --dmass-only          run only DMASS
#   COMPARE=1     --no-compare          skip the compare step
#   PERF_SUMMARY=1 --no-perf-summary    skip performance summary generation
#   WAIT=1        --no-wait             submit and exit (implies --no-compare)
#
# Compare-only mode (no submission):
#   ./run_maule_all.sh --compare-only
#
# Examples:
#   ./run_maule_all.sh                  # both + wait + compare
#   ./run_maule_all.sh --tremor-only    # tremor only, no compare
#   ./run_maule_all.sh --dmass-only     # dmass only, no compare
#   ./run_maule_all.sh --no-wait        # submit both and exit immediately

set -e

# --- Defaults -----------------------------------------------------------
RUN_TREMOR="${RUN_TREMOR:-1}"
RUN_DMASS="${RUN_DMASS:-1}"
WAIT="${WAIT:-1}"
COMPARE="${COMPARE:-1}"
PERF_SUMMARY="${PERF_SUMMARY:-1}"

CONDA_ENV="${CONDA_ENV:-localPyLibs}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"          # seconds between squeue polls
CORR_TOL="${CORR_TOL:-1e-3}"
POS_TOL="${POS_TOL:-0}"

# --- Parse flags --------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --tremor-only)   RUN_TREMOR=1; RUN_DMASS=0; COMPARE=0 ;;
        --dmass-only)    RUN_TREMOR=0; RUN_DMASS=1; COMPARE=0 ;;
        --compare-only)  RUN_TREMOR=0; RUN_DMASS=0; WAIT=0; COMPARE=1 ;;
        --no-tremor)     RUN_TREMOR=0 ;;
        --no-dmass)      RUN_DMASS=0 ;;
        --no-wait)       WAIT=0; COMPARE=0; PERF_SUMMARY=0 ;;
        --no-compare)    COMPARE=0 ;;
        --no-perf-summary) PERF_SUMMARY=0 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "[!] Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

# --- Paths --------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TREMOR_DRIVER="$SCRIPT_DIR/run_maule_tremor.sh"
DMASS_DRIVER="$SCRIPT_DIR/run_maule_dmass.sh"
TREMOR_RESULTS="$SCRIPT_DIR/results"
DMASS_RESULTS="$SCRIPT_DIR/results_dmass"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_results.py"
COMPARE_REPORT="$SCRIPT_DIR/compare_report.csv"
PERF_SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_performance.py"
PERF_LONG_REPORT="$SCRIPT_DIR/perf_summary_long.csv"
PERF_WIDE_REPORT="$SCRIPT_DIR/perf_summary_wide.csv"

echo "[+] Configuration:"
echo "    RUN_TREMOR=$RUN_TREMOR   RUN_DMASS=$RUN_DMASS   WAIT=$WAIT   COMPARE=$COMPARE   PERF_SUMMARY=$PERF_SUMMARY"
echo "    tremor results: $TREMOR_RESULTS"
echo "    dmass  results: $DMASS_RESULTS"

# --- Submit -------------------------------------------------------------
if [ "$RUN_TREMOR" = "1" ]; then
    echo "[+] Submitting TREMOR sweep..."
    "$TREMOR_DRIVER"
fi

if [ "$RUN_DMASS" = "1" ]; then
    echo "[+] Submitting DMASS sweep..."
    "$DMASS_DRIVER"
fi

# --- Wait for jobs to finish --------------------------------------------
if [ "$WAIT" = "1" ] && { [ "$RUN_TREMOR" = "1" ] || [ "$RUN_DMASS" = "1" ]; }; then
    if ! command -v squeue >/dev/null 2>&1; then
        echo "[!] squeue not found; cannot wait for jobs. Skipping wait+compare." >&2
        exit 0
    fi

    # Build a regex of job-name prefixes we care about.
    patterns=()
    [ "$RUN_TREMOR" = "1" ] && patterns+=("tremor\\.")
    [ "$RUN_DMASS"  = "1" ] && patterns+=("dmass\\.")
    pattern="$(IFS='|'; echo "${patterns[*]}")"

    echo "[+] Waiting for jobs matching /^($pattern)/ (poll every ${POLL_INTERVAL}s)..."
    while :; do
        n=$(squeue -h -u "$USER" -o "%j" | grep -cE "^($pattern)" || true)
        if [ "$n" -eq 0 ]; then
            echo "[+] All matching jobs finished."
            break
        fi
        echo "    ... $n job(s) still queued/running"
        sleep "$POLL_INTERVAL"
    done
fi

# --- Compare ------------------------------------------------------------
if [ "$COMPARE" = "1" ]; then
    if [ ! -f "$COMPARE_SCRIPT" ]; then
        echo "[!] Comparator not found at $COMPARE_SCRIPT" >&2
        exit 1
    fi

    # Activate the conda env that has pandas/numpy, if available.
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        echo "[!] conda not found; running with the current python3." >&2
    fi

    echo "[+] Comparing TREMOR vs DMASS results (corr_tol=$CORR_TOL, pos_tol=$POS_TOL)"
    python3 "$COMPARE_SCRIPT" \
        --tremor-dir "$TREMOR_RESULTS" \
        --dmass-dir  "$DMASS_RESULTS" \
        --corr-tol   "$CORR_TOL" \
        --pos-tol    "$POS_TOL" \
        --report     "$COMPARE_REPORT"
fi

# --- Performance summaries ----------------------------------------------
if [ "$PERF_SUMMARY" = "1" ]; then
    if [ ! -f "$PERF_SUMMARY_SCRIPT" ]; then
        echo "[!] Performance summary script not found at $PERF_SUMMARY_SCRIPT" >&2
        exit 1
    fi

    echo "[+] Building performance summaries from *.timings.csv"
    python3 "$PERF_SUMMARY_SCRIPT" \
        --tremor-dir  "$TREMOR_RESULTS" \
        --dmass-dir   "$DMASS_RESULTS" \
        --long-report "$PERF_LONG_REPORT" \
        --wide-report "$PERF_WIDE_REPORT"
fi

echo "[+] Done."
