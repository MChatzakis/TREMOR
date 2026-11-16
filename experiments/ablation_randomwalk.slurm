#!/bin/bash
#SBATCH --job-name=tremor-ablation-rw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=experiments/logs/tremor-ablation-rw.%j.out
#SBATCH --error=experiments/logs/tremor-ablation-rw.%j.err
#
# Ablation of TREMOR-adaptive's scan fallback fraction on a random walk, where
# the index prunes well and the candidate fraction spans 0.1% to 40% across
# thresholds. "never" is the classic leaf-order search (TREMOR-simple) and
# "sss" the skip-sequential scan baseline. Every run logs its per-template times.
# PARAM=b varies the lower-bound cut-off b (TREMOR_SCAN_LB_FRACTION) instead, with f = 0.01; the column "fallback"
# then holds b.
# Output: $OUT/ablation.<jobid>.csv (OUT defaults to ablation_randomwalk_20260927).
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

OUT=${OUT:-experiments/evals/ablation_randomwalk_20260927}
mkdir -p $OUT/runs
cp -n bin/tremor_main $OUT/tremor_main
CSV=$OUT/ablation.$SLURM_JOB_ID.csv
echo "threshold,fallback,rep,query_time,queries,scanned" > $CSV
DATA=/path/to/data/synthetic   # 100M-point random walk, 20 templates of 256 points

PARAM=${PARAM:-f}
VALUES="0 0.001 0.01 0.05 0.1 0.25 0.5 never sss"
[[ $PARAM = b ]] && VALUES="0 0.01 0.05 0.2 0.5 1"
for rep in 1 2; do
  for th in 0.9 0.95 0.99 0.999; do
    for v in $VALUES; do
      f=$v b=0.05
      [[ $PARAM = b ]] && f=0.01 b=$v
      name=th$th.$PARAM$v.rep$rep
      mode=(--refinement adaptive --scan-fallback-fraction $f)
      [[ $f = never ]] && mode=(--refinement leaf)
      [[ $f = sss ]] && mode=(--method skip-sequential)
      TREMOR_SCAN_LB_FRACTION=$b srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all $OUT/tremor_main --dataset $DATA/randomwalk.len100000000.bin \
        --dataset-size 100000000 --queries $DATA/randomwalk.q20.w256.bin --query-count 20 --window-length 256 \
        --mode threshold --threshold $th --threshold-result-slots 1 --merge-offset 64 --replication-groups 0 \
        --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments 16 --pq-threshold-divisor 16 "${mode[@]}" \
        --output $OUT/runs/$name --benchmark-output $OUT/runs/$name.timings.csv > $OUT/runs/$name.log 2>&1
      query_time=$(awk -F, 'NR > 1 && $3 > t { t = $3 } END { print t }' $OUT/runs/$name.timings.csv)
      scanned=$(grep -c 'sampled candidate fraction .* -> scan' $OUT/runs/$name.log)
      echo "$th,$v,$rep,$query_time,20,$scanned" >> $CSV
      rm -f $OUT/runs/${name}_*.csv
    done
  done
done
