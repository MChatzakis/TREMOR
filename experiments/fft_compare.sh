#!/bin/bash
# Compares the answers (template, position) of the fft and nofft variants of experiments/fft_check.slurm.
OUT=${OUT:-/path/to/scratch/bins/fft_check}
digest() { tail -n +2 "$1" 2>/dev/null | cut -d, -f1,3 | tr -d ' ' | sort | md5sum | cut -c1-12; }
for a in $OUT/*.fft.csv $OUT/*.fft_0.csv; do
  [[ -f $a ]] || continue
  b=${a/.fft/.nofft}; [[ -f $b ]] || continue
  echo "$(basename ${a/.fft/}): $([[ $(digest $a) = $(digest $b) ]] && echo identical || echo "DIFFERENT ($(diff <(tail -n +2 $a | cut -d, -f1,3 | sort) <(tail -n +2 $b | cut -d, -f1,3 | sort) | grep -c '^[<>]') lines)")"
done
