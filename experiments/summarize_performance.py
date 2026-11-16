#!/usr/bin/env python3
"""
Aggregate per-run benchmark timing CSVs from TREMOR and DMASS sweeps.

Expected input files:
  <results_dir>/<run_id>.timings.csv
with columns:
  node_rank,index_time,query_time

Outputs:
  1) long report: one row per (run_id, algorithm)
  2) wide report: one row per run_id with tremor/dmass metrics side-by-side
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Iterable


TIMING_COLUMNS = ("node_rank", "index_time", "query_time")


@dataclass
class PerfRow:
    run_id: str
    algorithm: str
    timings_file: str
    ranks_reported: int
    index_time_sum: float
    index_time_mean: float
    index_time_max: float
    query_time_sum: float
    query_time_mean: float
    query_time_max: float
    total_time_sum: float
    total_time_mean: float
    total_time_max: float


def _parse_float(value: str, file_path: str, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{file_path}: invalid {field_name} value '{value}'") from exc


def _load_timing_file(file_path: str, algorithm: str) -> PerfRow:
    index_times: list[float] = []
    query_times: list[float] = []

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{file_path}: missing CSV header")

        missing = [col for col in TIMING_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{file_path}: missing required columns {missing}; got {reader.fieldnames}"
            )

        for row in reader:
            index_times.append(_parse_float(row["index_time"], file_path, "index_time"))
            query_times.append(_parse_float(row["query_time"], file_path, "query_time"))

    if not index_times:
        raise ValueError(f"{file_path}: no timing rows found")

    run_id = os.path.basename(file_path)
    if not run_id.endswith(".timings.csv"):
        raise ValueError(f"{file_path}: unexpected timing filename")
    run_id = run_id[: -len(".timings.csv")]

    ranks = len(index_times)
    total_times = [i + q for i, q in zip(index_times, query_times)]

    return PerfRow(
        run_id=run_id,
        algorithm=algorithm,
        timings_file=file_path,
        ranks_reported=ranks,
        index_time_sum=sum(index_times),
        index_time_mean=sum(index_times) / ranks,
        index_time_max=max(index_times),
        query_time_sum=sum(query_times),
        query_time_mean=sum(query_times) / ranks,
        query_time_max=max(query_times),
        total_time_sum=sum(total_times),
        total_time_mean=sum(total_times) / ranks,
        total_time_max=max(index_times) + max(query_times),
    )


def _discover_rows(result_dir: str, algorithm: str) -> list[PerfRow]:
    if not os.path.isdir(result_dir):
        return []

    rows: list[PerfRow] = []
    for file_path in sorted(glob.glob(os.path.join(result_dir, "*.timings.csv"))):
        prefix = file_path[:-len(".timings.csv")]
        if any(os.path.exists(prefix + suffix) for suffix in [".failed.tsv", ".running.tsv", ".skipped.tsv"]):
            continue
        rows.append(_load_timing_file(file_path, algorithm))
    return rows


def _write_long_report(path: str, rows: Iterable[PerfRow]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "run_id",
        "algorithm",
        "timings_file",
        "ranks_reported",
        "index_time_sum",
        "index_time_mean",
        "index_time_max",
        "query_time_sum",
        "query_time_mean",
        "query_time_max",
        "total_time_sum",
        "total_time_mean",
        "total_time_max",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "run_id": r.run_id,
                    "algorithm": r.algorithm,
                    "timings_file": r.timings_file,
                    "ranks_reported": r.ranks_reported,
                    "index_time_sum": f"{r.index_time_sum:.6f}",
                    "index_time_mean": f"{r.index_time_mean:.6f}",
                    "index_time_max": f"{r.index_time_max:.6f}",
                    "query_time_sum": f"{r.query_time_sum:.6f}",
                    "query_time_mean": f"{r.query_time_mean:.6f}",
                    "query_time_max": f"{r.query_time_max:.6f}",
                    "total_time_sum": f"{r.total_time_sum:.6f}",
                    "total_time_mean": f"{r.total_time_mean:.6f}",
                    "total_time_max": f"{r.total_time_max:.6f}",
                }
            )


def _write_wide_report(path: str, rows: Iterable[PerfRow]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    by_run: dict[str, dict[str, PerfRow]] = {}
    for row in rows:
        by_run.setdefault(row.run_id, {})[row.algorithm] = row

    fieldnames = [
        "run_id",
        "tremor_ranks_reported",
        "tremor_index_time_max",
        "tremor_query_time_max",
        "tremor_total_time_max",
        "dmass_ranks_reported",
        "dmass_index_time_max",
        "dmass_query_time_max",
        "dmass_total_time_max",
        "speedup_total_dmass_over_tremor",
        "speedup_query_dmass_over_tremor",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_id in sorted(by_run):
            t = by_run[run_id].get("tremor")
            d = by_run[run_id].get("dmass")

            speedup_total = ""
            speedup_query = ""
            if t and d and t.total_time_max > 0 and t.query_time_max > 0:
                speedup_total = f"{(d.total_time_max / t.total_time_max):.6f}"
                speedup_query = f"{(d.query_time_max / t.query_time_max):.6f}"

            writer.writerow(
                {
                    "run_id": run_id,
                    "tremor_ranks_reported": "" if not t else t.ranks_reported,
                    "tremor_index_time_max": "" if not t else f"{t.index_time_max:.6f}",
                    "tremor_query_time_max": "" if not t else f"{t.query_time_max:.6f}",
                    "tremor_total_time_max": "" if not t else f"{t.total_time_max:.6f}",
                    "dmass_ranks_reported": "" if not d else d.ranks_reported,
                    "dmass_index_time_max": "" if not d else f"{d.index_time_max:.6f}",
                    "dmass_query_time_max": "" if not d else f"{d.query_time_max:.6f}",
                    "dmass_total_time_max": "" if not d else f"{d.total_time_max:.6f}",
                    "speedup_total_dmass_over_tremor": speedup_total,
                    "speedup_query_dmass_over_tremor": speedup_query,
                }
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tremor-dir", required=True, help="Directory with TREMOR result files")
    p.add_argument("--dmass-dir", required=True, help="Directory with DMASS result files")
    p.add_argument("--long-report", required=True, help="Output CSV for long-form summary")
    p.add_argument("--wide-report", required=True, help="Output CSV for side-by-side summary")
    args = p.parse_args()

    tremor_rows = _discover_rows(args.tremor_dir, "tremor")
    dmass_rows = _discover_rows(args.dmass_dir, "dmass")
    rows = sorted(tremor_rows + dmass_rows, key=lambda r: (r.run_id, r.algorithm))

    if not rows:
        print("No .timings.csv files found in either results directory.")
        return 0

    _write_long_report(args.long_report, rows)
    _write_wide_report(args.wide_report, rows)

    print(f"Wrote long performance summary: {args.long_report}")
    print(f"Wrote wide performance summary: {args.wide_report}")
    print(
        "Runs summarized: "
        f"{len(set(r.run_id for r in rows))} "
        f"(tremor={len(tremor_rows)}, dmass={len(dmass_rows)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
