#!/bin/bash
#SBATCH --job-name=smoke-128t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/smoke-128t.%j.out
#SBATCH --error=experiments/logs/smoke-128t.%j.err
#
# Gate for the full evaluation: runs TREMOR and DMASS with all 128 cores on
# real Maule (threshold) and SeiFR (kNN) data, measures per-query speed at a
# realistic query count, and requires both engines to return the same results.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=128
PY=python3

OUT="experiments/evals/smoke128_${SLURM_JOB_ID}"
mkdir -p "$OUT/tremor" "$OUT/dmass"
cp bin/tremor_main bin/dmass_main "$OUT/"
MW=/path/to/data/maule/waveforms
MT=/path/to/data/maule/templates
SW=/path/to/data/seifr/waveforms
ST=/path/to/data/seifr/templates

status=0
run() {
    echo "=== $1"; shift
    srun --mpi=pmi2 -n 1 "$@" || status=1
}

# Maule threshold: ZE_GB0S__HHE (356M samples), first 100 of 1005 templates.
W=$MW/ZE_GB0S__HHE.len356327321.bin; N=356327321
Q=$MT/tem1005.ZE_GB0S__HHE.len1000.fmin1.0.fmax10.0.bin
ID=maule_threshold.GB0S.th0p8
run "$ID tremor" "$OUT/tremor_main" --dataset $W --dataset-size $N --queries $Q --query-count 100 \
    --window-length 1000 --mode threshold --threshold 0.8 --threshold-result-slots 1 --merge-offset 250 \
    --replication-groups 0 --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments 8 \
    --pq-threshold-divisor 16 --output "$OUT/tremor/$ID" --benchmark-output "$OUT/tremor/$ID.timings.csv"
run "$ID dmass" "$OUT/dmass_main" --dataset $W --dataset-size $N --queries $Q --queries-size 100 \
    --window-length 1000 --mode threshold-search --threshold 0.8 --merge-offset 250 \
    --replication-groups 0 --threads 128 --output "$OUT/dmass/$ID" --benchmark-output "$OUT/dmass/$ID.timings.csv"

# SeiFR kNN: RD_EPF__BHZ (1.05G samples), first 20 of 829 templates.
W=$SW/RD_EPF__BHZ.len1046593001.bin; N=1046593001
Q=$ST/tem829.RD_EPF__BHZ.len3000.fmin1.0.fmax10.0.bin
ID=seifr_knn.EPF.k10
run "$ID tremor" "$OUT/tremor_main" --dataset $W --dataset-size $N --queries $Q --query-count 20 \
    --window-length 3000 --mode knn --k 10 --merge-offset 0 \
    --replication-groups 0 --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments 8 \
    --pq-threshold-divisor 16 --output "$OUT/tremor/$ID.csv" --benchmark-output "$OUT/tremor/$ID.timings.csv"
run "$ID dmass" "$OUT/dmass_main" --dataset $W --dataset-size $N --queries $Q --queries-size 20 \
    --window-length 3000 --mode knn-search --k 10 --merge-offset 0 \
    --replication-groups 0 --threads 128 --output "$OUT/dmass/$ID.csv" --benchmark-output "$OUT/dmass/$ID.timings.csv"

for f in "$OUT"/*/*.timings.csv; do echo "$f: $(tail -n +2 "$f")"; done
$PY experiments/compare_results.py --tremor-dir "$OUT/tremor" --dmass-dir "$OUT/dmass" \
    --corr-tol 1e-3 --report "$OUT/compare_report.csv" || status=1
sleep 30 # let accounting flush MaxRSS
sacct -j "$SLURM_JOB_ID" --format=JobID%20,JobName%14,State,Elapsed,MaxRSS,ExitCode -P
echo "smoke status=$status"
exit $status
