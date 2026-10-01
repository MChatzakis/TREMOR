#!/usr/bin/env python3
"""
Compare TREMOR and DMASS result CSVs for matching run ids.

Both algorithms write the same CSV formats:

  Threshold mode  ->  <prefix>_<rank>.csv   (one shard per MPI rank)
                       header: "Query, Counter, Position, Cross Correlation"

  KNN mode        ->  <prefix>.csv          (single file, rank 0 only)
                       header: "tID, k, pos, corr"

For every run id that exists on BOTH sides this script:
  * concatenates all threshold shards (or loads the single KNN file),
  * pairs hits by (Query, Position) with an optional +/- pos-tol tolerance,
  * checks that hit counts per query match, and
  * checks that cross-correlations agree within --corr-tol.

Usage examples:
    python compare_results.py \\
        --tremor-dir experiments/results \\
        --dmass-dir  experiments/results_dmass

    python compare_results.py --tremor-dir r_t --dmass-dir r_d \\
        --corr-tol 1e-3 --pos-tol 0 --report compare_report.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

THRESHOLD_COLS = ["Query", "Counter", "Position", "Cross Correlation"]
KNN_COLS = ["tID", "k", "pos", "corr"]


def _read_csv(path: str) -> pd.DataFrame:
    """Read a TREMOR/DMASS CSV. Both use ', ' as separator between values."""
    return pd.read_csv(path, skipinitialspace=True)


def load_threshold_run(result_dir: str, run_id: str) -> Optional[pd.DataFrame]:
    """Return concatenated per-rank threshold shards for a run id, or None."""
    shards = sorted(glob.glob(os.path.join(result_dir, f"{run_id}_*.csv")))
    if not shards:
        return None
    frames = []
    for shard in shards:
        try:
            df = _read_csv(shard)
        except pd.errors.EmptyDataError:
            continue
        # Some shards are header-only (no hits on that rank); skip them.
        if df.empty:
            continue
        missing = [c for c in THRESHOLD_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{shard}: missing columns {missing}; got {list(df.columns)}"
            )
        frames.append(df[THRESHOLD_COLS])
    if not frames:
        return pd.DataFrame(columns=THRESHOLD_COLS)
    return pd.concat(frames, ignore_index=True)


def load_knn_run(result_dir: str, run_id: str) -> Optional[pd.DataFrame]:
    """Return the single KNN CSV for a run id, or None."""
    path = os.path.join(result_dir, f"{run_id}.csv")
    if not os.path.isfile(path):
        return None
    df = _read_csv(path)
    missing = [c for c in KNN_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; got {list(df.columns)}"
        )
    return df[KNN_COLS]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_run_ids(result_dir: str) -> dict[str, str]:
    """
    Return {run_id: "threshold"|"knn"} for every run id found in a dir.

    We ignore benchmark files (*.timings.csv) and stderr/stdout logs.
    """
    if not os.path.isdir(result_dir):
        return {}

    runs: dict[str, str] = {}
    for name in os.listdir(result_dir):
        if not name.endswith(".csv"):
            continue
        if name.endswith(".timings.csv"):
            continue

        base = name[:-4]  # strip ".csv"

        # Threshold shards look like "<run_id>_<rank>". Trailing token must be
        # a pure integer; anything else is treated as a KNN single-file run.
        if "_" in base:
            head, tail = base.rsplit("_", 1)
            if tail.isdigit():
                if any(os.path.exists(os.path.join(result_dir, head + suffix)) for suffix in [".failed.tsv", ".running.tsv", ".skipped.tsv"]):
                    continue
                runs.setdefault(head, "threshold")
                continue
        if any(os.path.exists(os.path.join(result_dir, base + suffix)) for suffix in [".failed.tsv", ".running.tsv", ".skipped.tsv"]):
            continue
        runs.setdefault(base, "knn")
    return runs


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class RunReport:
    run_id: str
    mode: str
    status: str                   # "match" | "mismatch" | "missing" | "error"
    tremor_rows: int = 0
    dmass_rows: int = 0
    common_rows: int = 0
    only_tremor: int = 0
    only_dmass: int = 0
    max_abs_corr_diff: float = 0.0
    mean_abs_corr_diff: float = 0.0
    notes: str = ""


def _sort_threshold(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["Query", "Position"]).reset_index(drop=True)


def _sort_knn(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["tID", "pos"]).reset_index(drop=True)


def compare_threshold(
    tremor_df: pd.DataFrame,
    dmass_df: pd.DataFrame,
    corr_tol: float,
    pos_tol: int,
) -> tuple[str, int, int, int, float, float, str]:
    """
    Threshold-mode comparison.

    Match rows by (Query, Position) exactly when pos_tol == 0. Otherwise, for
    every Query, greedily match tremor hits to dmass hits whose Position is
    within +/- pos_tol samples.
    """
    t = _sort_threshold(tremor_df)
    d = _sort_threshold(dmass_df)

    if pos_tol == 0:
        merged = t.merge(
            d,
            on=["Query", "Position"],
            how="outer",
            suffixes=("_tremor", "_dmass"),
            indicator=True,
        )
        common = merged[merged["_merge"] == "both"]
        only_t = int((merged["_merge"] == "left_only").sum())
        only_d = int((merged["_merge"] == "right_only").sum())

        if common.empty:
            diffs = np.array([], dtype=float)
        else:
            diffs = np.abs(
                common["Cross Correlation_tremor"].to_numpy()
                - common["Cross Correlation_dmass"].to_numpy()
            )
    else:
        common_pairs = []
        only_t = 0
        only_d = 0
        all_queries = sorted(set(t["Query"].unique()).union(d["Query"].unique()))
        for query in all_queries:
            t_hits = t[t["Query"] == query]
            d_hits = d[d["Query"] == query]

            if t_hits.empty:
                only_d += int(len(d_hits))
                continue
            if d_hits.empty:
                only_t += int(len(t_hits))
                continue

            t_used = np.zeros(len(t_hits), dtype=bool)
            d_used = np.zeros(len(d_hits), dtype=bool)

            t_pos = t_hits["Position"].to_numpy()
            d_pos = d_hits["Position"].to_numpy()
            t_cc = t_hits["Cross Correlation"].to_numpy()
            d_cc = d_hits["Cross Correlation"].to_numpy()

            for i in range(len(t_pos)):
                # Find nearest unused dmass hit within tolerance.
                diffs_pos = np.abs(d_pos - t_pos[i])
                diffs_pos[d_used] = np.iinfo(np.int64).max
                j = int(np.argmin(diffs_pos))
                if diffs_pos[j] <= pos_tol:
                    t_used[i] = True
                    d_used[j] = True
                    common_pairs.append((t_cc[i], d_cc[j]))

            only_t += int((~t_used).sum())
            only_d += int((~d_used).sum())

        if common_pairs:
            arr = np.asarray(common_pairs, dtype=float)
            diffs = np.abs(arr[:, 0] - arr[:, 1])
        else:
            diffs = np.array([], dtype=float)

    common_rows = int(len(diffs))
    max_diff = float(diffs.max()) if diffs.size else 0.0
    mean_diff = float(diffs.mean()) if diffs.size else 0.0

    status = "match"
    notes = []
    if only_t or only_d:
        status = "mismatch"
        notes.append(f"only_tremor={only_t}, only_dmass={only_d}")
    if diffs.size and max_diff > corr_tol:
        status = "mismatch"
        notes.append(f"max|dCC|={max_diff:.3e} > {corr_tol:.1e}")

    return status, len(t), len(d), common_rows, max_diff, mean_diff, "; ".join(notes)


def compare_knn(
    tremor_df: pd.DataFrame,
    dmass_df: pd.DataFrame,
    corr_tol: float,
    pos_tol: int,
) -> tuple[str, int, int, int, float, float, str]:
    """
    KNN-mode comparison: sort by (tID, pos) on both sides and align row-wise
    within each query. The set of positions must agree (within pos_tol), and
    correlations must agree within corr_tol.
    """
    t = _sort_knn(tremor_df)
    d = _sort_knn(dmass_df)

    only_t = 0
    only_d = 0
    diffs_list = []

    all_queries = sorted(set(t["tID"].unique()).union(d["tID"].unique()))
    for query in all_queries:
        t_hits = t[t["tID"] == query]
        d_hits = d[d["tID"] == query]

        if len(t_hits) != len(d_hits):
            only_t += max(0, len(t_hits) - len(d_hits))
            only_d += max(0, len(d_hits) - len(t_hits))
            continue

        t_pos = t_hits["pos"].to_numpy()
        d_pos = d_hits["pos"].to_numpy()
        t_cc = t_hits["corr"].to_numpy()
        d_cc = d_hits["corr"].to_numpy()

        pos_bad = int(np.sum(np.abs(t_pos - d_pos) > pos_tol))
        if pos_bad:
            only_t += pos_bad
            only_d += pos_bad
        diffs_list.append(np.abs(t_cc - d_cc))

    diffs = np.concatenate(diffs_list) if diffs_list else np.array([], dtype=float)
    common_rows = int(diffs.size)
    max_diff = float(diffs.max()) if diffs.size else 0.0
    mean_diff = float(diffs.mean()) if diffs.size else 0.0

    status = "match"
    notes = []
    if only_t or only_d:
        status = "mismatch"
        notes.append(f"only_tremor={only_t}, only_dmass={only_d}")
    if diffs.size and max_diff > corr_tol:
        status = "mismatch"
        notes.append(f"max|dCC|={max_diff:.3e} > {corr_tol:.1e}")

    return status, len(t), len(d), common_rows, max_diff, mean_diff, "; ".join(notes)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compare_all(
    tremor_dir: str,
    dmass_dir: str,
    corr_tol: float,
    pos_tol: int,
) -> list[RunReport]:
    tremor_runs = discover_run_ids(tremor_dir)
    dmass_runs = discover_run_ids(dmass_dir)

    all_ids = sorted(set(tremor_runs) | set(dmass_runs))
    reports: list[RunReport] = []

    for run_id in all_ids:
        t_mode = tremor_runs.get(run_id)
        d_mode = dmass_runs.get(run_id)

        if t_mode is None or d_mode is None:
            reports.append(
                RunReport(
                    run_id=run_id,
                    mode=t_mode or d_mode or "?",
                    status="missing",
                    notes=(
                        "no tremor results"
                        if t_mode is None
                        else "no dmass results"
                    ),
                )
            )
            continue

        if t_mode != d_mode:
            reports.append(
                RunReport(
                    run_id=run_id,
                    mode=f"{t_mode}/{d_mode}",
                    status="mismatch",
                    notes=f"tremor mode={t_mode} vs dmass mode={d_mode}",
                )
            )
            continue

        try:
            if t_mode == "threshold":
                t_df = load_threshold_run(tremor_dir, run_id)
                d_df = load_threshold_run(dmass_dir, run_id)
                status, tr, dr, cr, mx, mn, notes = compare_threshold(
                    t_df, d_df, corr_tol, pos_tol
                )
            else:
                t_df = load_knn_run(tremor_dir, run_id)
                d_df = load_knn_run(dmass_dir, run_id)
                status, tr, dr, cr, mx, mn, notes = compare_knn(
                    t_df, d_df, corr_tol, pos_tol
                )
        except Exception as e:  # pragma: no cover - defensive
            reports.append(
                RunReport(
                    run_id=run_id,
                    mode=t_mode,
                    status="error",
                    notes=f"{type(e).__name__}: {e}",
                )
            )
            continue

        reports.append(
            RunReport(
                run_id=run_id,
                mode=t_mode,
                status=status,
                tremor_rows=tr,
                dmass_rows=dr,
                common_rows=cr,
                only_tremor=tr - cr,
                only_dmass=dr - cr,
                max_abs_corr_diff=mx,
                mean_abs_corr_diff=mn,
                notes=notes,
            )
        )

    return reports


def _print_reports(reports: list[RunReport]) -> None:
    if not reports:
        print("No result files found in either directory.")
        return

    header = (
        f"{'STATUS':10s} {'MODE':10s} {'RUN_ID':60s} "
        f"{'#T':>8s} {'#D':>8s} {'#COMMON':>8s} "
        f"{'MAX|dCC|':>12s} {'MEAN|dCC|':>12s}  NOTES"
    )
    print(header)
    print("-" * len(header))

    for r in reports:
        print(
            f"{r.status:10s} {r.mode:10s} {r.run_id:60s} "
            f"{r.tremor_rows:8d} {r.dmass_rows:8d} {r.common_rows:8d} "
            f"{r.max_abs_corr_diff:12.3e} {r.mean_abs_corr_diff:12.3e}  "
            f"{r.notes}"
        )


def _write_report(reports: list[RunReport], path: str) -> None:
    df = pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "mode": r.mode,
                "status": r.status,
                "tremor_rows": r.tremor_rows,
                "dmass_rows": r.dmass_rows,
                "common_rows": r.common_rows,
                "only_tremor": r.only_tremor,
                "only_dmass": r.only_dmass,
                "max_abs_corr_diff": r.max_abs_corr_diff,
                "mean_abs_corr_diff": r.mean_abs_corr_diff,
                "notes": r.notes,
            }
            for r in reports
        ]
    )
    df.to_csv(path, index=False)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tremor-dir", required=True, help="Directory with TREMOR CSV outputs")
    p.add_argument("--dmass-dir", required=True, help="Directory with DMASS CSV outputs")
    p.add_argument("--corr-tol", type=float, default=1e-3, help="Max abs diff in cross-correlation (default: 1e-3)")
    p.add_argument("--pos-tol", type=int, default=0, help="Allowed position offset in samples (default: 0)")
    p.add_argument("--report", default=None, help="Optional CSV report path")
    args = p.parse_args(argv)

    reports = compare_all(args.tremor_dir, args.dmass_dir, args.corr_tol, args.pos_tol)
    _print_reports(reports)

    if args.report:
        _write_report(reports, args.report)
        print(f"\nReport written to {args.report}")

    # Exit non-zero if any run failed to match.
    bad = sum(1 for r in reports if r.status in ("mismatch", "missing", "error"))
    total = len(reports)
    print(f"\nSummary: {total - bad}/{total} runs match (corr_tol={args.corr_tol}, pos_tol={args.pos_tol})")
    return 0 if bad == 0 and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
