#!/bin/bash
#SBATCH --job-name=search-validation
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/search-validation.%j.out
#SBATCH --error=experiments/logs/search-validation.%j.err
set -euo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python3 experiments/validate_search.py \
    --output "experiments/evals/search_validation_${SLURM_JOB_ID}" --dataset "${VALIDATION_DATASET:-synthetic}" \
    ${VALIDATION_ARGS:-}
