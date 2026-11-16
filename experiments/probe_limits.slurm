#!/bin/bash
#SBATCH --job-name=probe-limits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/logs/probe-limits.%j.out
#SBATCH --error=experiments/logs/probe-limits.%j.err
#
# Measures single-node peak memory (sacct MaxRSS per step) and speed for
# TREMOR and the cached-FFT DMASS, to set the full-evaluation skip guards
# from measurements instead of estimates. Each srun step is one probe.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=64

OUT="experiments/evals/probe_limits_${SLURM_JOB_ID}"
mkdir -p "$OUT"
cp bin/tremor_main bin/dmass_main "$OUT/"
MW=/path/to/data/maule/waveforms
MT=/path/to/data/maule/templates
SW=/path/to/data/seifr/waveforms
ST=/path/to/data/seifr/templates
G04_WAVE=$MW/ZE_G04S__HHZ.len2566388916.bin
G04_Q=experiments/evals/audited_20260923/inputs/maule/ZE_G04S__HHZ.q5000.bin

STEPS="$OUT/steps.tsv"
printf "step\tname\tstatus\n" > "$STEPS"
step=0
probe() {
    local name="$1"; shift
    echo "=== step $step: $name"
    srun --mpi=pmi2 -n 1 "$@"
    printf "%s\t%s\t%s\n" "$step" "$name" "$?" >> "$STEPS"
    step=$((step + 1))
}

dmass() {
    local name="$1" wave="$2" size="$3" queries="$4" qn="$5" w="$6"
    probe "$name" "$OUT/dmass_main" --dataset "$wave" --dataset-size "$size" \
        --queries "$queries" --queries-size "$qn" --window-length "$w" --threads 64 \
        --replication-groups 0 --merge-offset $((w / 4)) --mode threshold-search --threshold 0.8 \
        --output "$OUT/$name" --benchmark-output "$OUT/$name.timings.csv"
}

tremor() {
    local name="$1" wave="$2" size="$3" queries="$4" qn="$5" w="$6" mode="$7"
    local -a extra
    if [[ "$mode" = knn ]]; then
        extra=(--k 10 --merge-offset 0 --output "$OUT/$name.csv")
    else
        extra=(--threshold 0.8 --threshold-result-slots 1 --merge-offset $((w / 4)) --output "$OUT/$name")
    fi
    probe "$name" "$OUT/tremor_main" --dataset "$wave" --dataset-size "$size" \
        --queries "$queries" --query-count "$qn" --window-length "$w" --mode "$mode" \
        --replication-groups 0 --index-threads 64 --search-workers 64 --leaf-size 2000 \
        --paa-segments 8 --pq-threshold-divisor 16 "${extra[@]}" \
        --benchmark-output "$OUT/$name.timings.csv"
}

# PROBE_SET=tremor skips the DMASS probes.
if [[ "${PROBE_SET:-all}" != tremor ]]; then
    # Speed of the cached-FFT DMASS on cases with old-binary timings
    # (old query_time: 1313s, 1270s, 346s).
    dmass dmass_speed_GC1S_HHN $MW/ZE_GC1S__HHN.len8640001.bin 8640001 $MT/tem110.ZE_GC1S__HHN.len1000.fmin1.0.fmax10.0.bin 110 1000
    dmass dmass_speed_GB2S_HHZ $MW/ZE_GB2S__HHZ.len26124343.bin 26124343 $MT/tem100.ZE_GB2S__HHZ.len1000.fmin1.0.fmax10.0.bin 100 1000
    dmass dmass_speed_GC1S_HHE $MW/ZE_GC1S__HHE.len17381829.bin 17381829 $MT/tem110.ZE_GC1S__HHE.len1000.fmin1.0.fmax10.0.bin 110 1000

    # DMASS peak memory at its hard 32-bit FFT limit (INT_MAX / 2 local samples).
    dmass dmass_mem_1073M $G04_WAVE 1073741823 $G04_Q 3 992
fi

# TREMOR peak memory versus local chunk size.
tremor tremor_mem_600M $G04_WAVE 600000000 $G04_Q 3 992 threshold
tremor tremor_mem_1200M $G04_WAVE 1200000000 $G04_Q 3 992 threshold
tremor tremor_mem_2566M $G04_WAVE 2566388916 $G04_Q 3 992 threshold
tremor tremor_mem_seifr_1577M $SW/RD_FLAF__BHZ.len1576800001.bin 1576800001 $ST/tem436.RD_FLAF__BHZ.len3000.fmin1.0.fmax10.0.bin 3 3000 knn

sleep 30 # let accounting flush MaxRSS
sacct -j "$SLURM_JOB_ID" --format=JobID%20,JobName%14,State,Elapsed,MaxRSS,ExitCode -P > "$OUT/sacct.tsv"
cat "$STEPS" "$OUT/sacct.tsv"
