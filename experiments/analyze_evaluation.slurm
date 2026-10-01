#!/bin/bash
#SBATCH --job-name=evaluation-plots
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/evaluation-plots.%j.out
#SBATCH --error=experiments/logs/evaluation-plots.%j.err
set -euo pipefail
cd /path/to/TREMOR
export OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="/tmp/matplotlib-$SLURM_JOB_ID"
python3 experiments/analyze_evaluation.py "$EVAL_ROOT"
# Both engines are exact, so every paired run must return the same hits.
for d in "$EVAL_ROOT"/results/*/; do
    [[ -d "$d/tremor" && -d "$d/dmass" ]] || continue
    python3 experiments/compare_results.py \
        --tremor-dir "$d/tremor" --dmass-dir "$d/dmass" --corr-tol 1e-3 \
        --report "$EVAL_ROOT/reports/output_agreement.$(basename "$d").csv" | tail -3 || true
done
