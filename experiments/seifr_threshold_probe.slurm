#!/bin/bash
#SBATCH --job-name=seifr-th-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/seifr-th-probe.%j.out
# Exploratory: threshold search on SeiFR (RD.EPF, 20 templates, one node) with TREMOR, SS, SSS and DMASS-V3,
# t = 0.5 .. 0.9: query time and whether the answers (template, position) agree with TREMOR's.
cd /path/to/TREMOR
module load intel/21U2/suite fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
OUT=${OUT:-/path/to/scratch/bins/seifr_th_probe}; mkdir -p $OUT
Q=experiments/evals/final_sweep_n4_20260925/inputs/seifr/RD_EPF__BHZ.q50.bin
DATA=(--dataset /path/to/data/seifr/waveforms/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001)
TREMOR_ARGS=(--queries $Q --query-count 20 --window-length 3000 --paa-segments 8 --merge-offset 750 --replication-groups 0
             --index-threads 128 --search-workers 128 --leaf-size 2000 --pq-threshold-divisor 16 --threshold-result-slots 1)
DMASS_ARGS=(--queries $Q --queries-size 20 --window-length 3000 --merge-offset 750 --replication-groups 0 --threads 128)
digest() { tail -n +2 "$1" 2>/dev/null | cut -d, -f1,3 | tr -d ' ' | sort | md5sum | cut -c1-12; }
for t in 0.5 0.6 0.7 0.8 0.9; do
  reference=""
  for method in tremor ss sss dmassv3; do
    name=$OUT/th$t.$method
    case $method in
      tremor) srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all bin/tremor_main "${DATA[@]}" "${TREMOR_ARGS[@]}" \
                --mode threshold --threshold $t --refinement adaptive --output $name --benchmark-output $name.timings.csv > $name.log 2>&1 ;;
      ss|sss) srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all bin/tremor_main "${DATA[@]}" "${TREMOR_ARGS[@]}" \
                --mode threshold --threshold $t --method $([[ $method = ss ]] && echo sequential || echo skip-sequential) \
                --output $name --benchmark-output $name.timings.csv > $name.log 2>&1 ;;
      dmassv3) srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all bin/dmass_v3_main "${DATA[@]}" "${DMASS_ARGS[@]}" \
                --mode threshold-search --threshold $t --output $name --benchmark-output $name.timings.csv > $name.log 2>&1 ;;
    esac
    d=$(digest ${name}_0.csv); [[ -z $reference ]] && reference=$d
    echo "t=$t $method: build/query (s) $(tail -1 $name.timings.csv | cut -d, -f2,3), detections $(($(wc -l < ${name}_0.csv)-1)), answers $([[ $d = $reference ]] && echo same || echo differ)"
  done
done
echo finished
