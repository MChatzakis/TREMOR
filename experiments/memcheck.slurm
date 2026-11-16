#!/bin/bash
#SBATCH --job-name=memcheck
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --output=experiments/logs/memcheck.%j.out
#
# Which NUMA nodes (cpuset mems) do the tasks of an srun step get on the node that runs the batch script?
cd /path/to/TREMOR
show='echo "$(hostname) task $SLURM_PROCID: $(grep -h Mems_allowed_list /proc/self/status | tr -s "\t" " ") | $(grep -h Cpus_allowed_list /proc/self/status | tr -s "\t" " ")"'
echo "batch script: $(hostname) $(grep Mems_allowed_list /proc/self/status | tr -s '\t' ' ')"
for opts in "" "--mem-bind=none" "--cpu-bind=none" "--mem-bind=none --cpu-bind=none" "--exact" "--overlap" "-c 128" "--mem=0"; do
  echo "== srun $opts"
  srun $opts -n 4 bash -c "$show" 2>&1 | sort
done
