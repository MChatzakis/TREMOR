"""Analysis helpers for the full TREMOR vs DMASS evaluation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except ModuleNotFoundError:
    plt = None
    TwoSlopeNorm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT: Path | None = None
TIMING_SUFFIX = ".timings.csv"
TIMING_COLUMNS = {"node_rank", "index_time", "query_time"}


INT_COLUMNS = [
    "manifest_index",
    "array_index",
    "nodes",
    "replication_groups",
    "query_count",
    "window_length",
    "merge_offset",
    "waveform_samples",
    "query_floats",
]

FLOAT_COLUMNS = ["threshold", "k"]

PAIR_INDEX = [
    "dataset",
    "workload",
    "case_id",
    "nodes",
    "replication_groups",
    "window_length",
    "query_count",
    "waveform_samples",
    "merge_offset",
    "param_name",
    "param_value",
]


def _require_pandas():
    if pd is None:
        raise ModuleNotFoundError("pandas is required for analysis table cells")
    return pd


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for analysis and plotting cells")
    return np


def resolve_eval_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an evaluation root from an explicit value, EVAL_ROOT, or the latest known run."""
    raw = value or os.environ.get("EVAL_ROOT")
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            candidates = [Path.cwd() / path, REPO_ROOT / path]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
        return path.resolve()

    if DEFAULT_EVAL_ROOT is not None and DEFAULT_EVAL_ROOT.exists():
        return DEFAULT_EVAL_ROOT.resolve()

    roots = list_eval_roots()
    if not roots:
        raise FileNotFoundError("No evaluation roots found under experiments/evals")
    return roots[-1]


def list_eval_roots(repo_root: Path | None = None) -> list[Path]:
    """Return eval roots sorted by modification time."""
    base = (repo_root or REPO_ROOT) / "experiments" / "evals"
    if not base.exists():
        return []
    roots = [p for p in base.iterdir() if (p / "manifest.tsv").exists() and (p / "submissions.tsv").exists()]
    return sorted(roots, key=lambda p: p.stat().st_mtime)


def _read_tsv(path: Path) -> pd.DataFrame:
    _require_pandas()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _coerce_manifest_types(df: pd.DataFrame) -> pd.DataFrame:
    _require_pandas()
    _require_numpy()
    out = df.copy()
    for col in INT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace({"NA": np.nan}), errors="coerce")
    return out


def load_eval_tables(eval_root: str | os.PathLike[str] | None = None) -> dict[str, pd.DataFrame | Path]:
    """Load the plan/audit tables for one evaluation root."""
    root = resolve_eval_root(eval_root)
    manifest = _coerce_manifest_types(_read_tsv(root / "manifest.tsv"))
    submissions = _read_tsv(root / "submissions.tsv")
    run_config = _read_tsv(root / "run_config.tsv")
    summary = _read_tsv(root / "summary.tsv")
    audit_issues = _read_tsv(root / "audit_issues.tsv")
    batches = _read_tsv(root / "batches.tsv")
    return {
        "eval_root": root,
        "manifest": manifest,
        "submissions": submissions,
        "run_config": run_config,
        "summary": summary,
        "audit_issues": audit_issues,
        "batches": batches,
    }


def _timing_metrics(path: Path) -> dict[str, float | int | str]:
    _require_pandas()
    df = pd.read_csv(path)
    missing = TIMING_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing timing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path}: no timing rows")

    index_time = pd.to_numeric(df["index_time"], errors="raise")
    query_time = pd.to_numeric(df["query_time"], errors="raise")
    ranks = pd.to_numeric(df["node_rank"], errors="raise")
    if sorted(ranks.tolist()) != list(range(len(df))):
        raise ValueError(f"{path}: duplicated or missing timing ranks")
    if not np.isfinite(index_time).all() or not np.isfinite(query_time).all() or (index_time < 0).any() or (query_time < 0).any():
        raise ValueError(f"{path}: nonfinite or negative timings")
    total_time = index_time + query_time
    return {
        "timings_file": str(path),
        "ranks_reported": int(len(df)),
        "index_time_sum": float(index_time.sum()),
        "index_time_mean": float(index_time.mean()),
        "index_time_max": float(index_time.max()),
        "query_time_sum": float(query_time.sum()),
        "query_time_mean": float(query_time.mean()),
        "query_time_max": float(query_time.max()),
        "total_time_sum": float(total_time.sum()),
        "total_time_mean": float(total_time.mean()),
        # Separate phases have an MPI barrier: use each phase's slowest rank.
        "total_time_max": float(index_time.max() + query_time.max()),
    }


def load_timings(
    eval_root: str | os.PathLike[str] | None = None,
    manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover and aggregate timing CSVs.

    Returns (timings, parse_errors). Missing runs are reported by completion_status().
    """
    _require_pandas()
    root = resolve_eval_root(eval_root)
    manifest_df = _coerce_manifest_types(manifest) if manifest is not None else load_eval_tables(root)["manifest"]
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for path in sorted((root / "results").rglob(f"*{TIMING_SUFFIX}")):
        prefix = str(path)[: -len(TIMING_SUFFIX)]
        if any(Path(prefix + suffix).exists() for suffix in [".failed.tsv", ".running.tsv", ".skipped.tsv"]):
            continue
        try:
            row = {"output_prefix": prefix, **_timing_metrics(path)}
        except Exception as exc:  # noqa: BLE001 - keep parse diagnostics in the notebook.
            errors.append({"timings_file": str(path), "error": str(exc)})
            continue
        rows.append(row)

    timings_raw = pd.DataFrame(rows)
    if timings_raw.empty:
        timings = manifest_df.iloc[0:0].copy()
        metric_cols = [
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
            "timing_ok",
        ]
        for col in metric_cols:
            timings[col] = pd.Series(dtype="float64")
        return timings, pd.DataFrame(errors)

    timings = manifest_df.merge(timings_raw, on="output_prefix", how="inner")
    timings["timing_ok"] = timings["ranks_reported"].eq(timings["nodes"].astype(int))
    return timings, pd.DataFrame(errors)


def load_skips(
    eval_root: str | os.PathLike[str] | None = None,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load explicit skipped-run records written by the full-eval Slurm runner."""
    _require_pandas()
    root = resolve_eval_root(eval_root)
    rows: list[pd.DataFrame] = []
    for path in sorted((root / "results").rglob("*.skipped.tsv")):
        df = pd.read_csv(path, sep="\t")
        df["skip_file"] = str(path)
        df["output_prefix"] = str(path)[: -len(".skipped.tsv")]
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    skips = pd.concat(rows, ignore_index=True)
    if manifest is None:
        return skips

    meta_cols = [
        "output_prefix",
        "dataset",
        "workload",
        "algorithm",
        "nodes",
        "replication_groups",
        "case_id",
        "query_count",
        "window_length",
        "merge_offset",
        "threshold",
        "k",
    ]
    meta = _coerce_manifest_types(manifest)[[c for c in meta_cols if c in manifest.columns]]
    return meta.merge(skips, on="output_prefix", how="inner")


def _load_marker_tsvs(
    eval_root: str | os.PathLike[str] | None,
    suffix: str,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_pandas()
    root = resolve_eval_root(eval_root)
    rows: list[pd.DataFrame] = []
    for path in sorted((root / "results").rglob(f"*{suffix}")):
        df = pd.read_csv(path, sep="\t")
        df["marker_file"] = str(path)
        df["output_prefix"] = str(path)[: -len(suffix)]
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    markers = pd.concat(rows, ignore_index=True)
    if manifest is None:
        return markers

    meta_cols = [
        "output_prefix",
        "dataset",
        "workload",
        "algorithm",
        "nodes",
        "replication_groups",
        "case_id",
        "query_count",
        "window_length",
        "merge_offset",
        "threshold",
        "k",
    ]
    meta = _coerce_manifest_types(manifest)[[c for c in meta_cols if c in manifest.columns]]
    return meta.merge(markers, on="output_prefix", how="inner")


def load_failures(
    eval_root: str | os.PathLike[str] | None = None,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load explicit failed-run records written by the full-eval Slurm runner."""
    return _load_marker_tsvs(eval_root, ".failed.tsv", manifest)


def load_running(
    eval_root: str | os.PathLike[str] | None = None,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load rows that were marked running and did not finish cleanly."""
    return _load_marker_tsvs(eval_root, ".running.tsv", manifest)


def completion_status(
    manifest: pd.DataFrame,
    timings: pd.DataFrame,
    skips: pd.DataFrame | None = None,
    failures: pd.DataFrame | None = None,
    running: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize timing completion by dataset/workload/algorithm/node count."""
    _require_pandas()
    base = _coerce_manifest_types(manifest)
    found = set(timings.get("output_prefix", pd.Series(dtype=str)).astype(str))
    ok = set(timings.loc[timings.get("timing_ok", False).astype(bool), "output_prefix"].astype(str)) if not timings.empty else set()
    skipped = set(skips.get("output_prefix", pd.Series(dtype=str)).astype(str)) if skips is not None and not skips.empty else set()
    failed = set(failures.get("output_prefix", pd.Series(dtype=str)).astype(str)) if failures is not None and not failures.empty else set()
    active = set(running.get("output_prefix", pd.Series(dtype=str)).astype(str)) if running is not None and not running.empty else set()
    base["timing_found"] = base["output_prefix"].astype(str).isin(found)
    base["timing_ok"] = base["output_prefix"].astype(str).isin(ok)
    base["skipped"] = base["output_prefix"].astype(str).isin(skipped)
    base["failed"] = base["output_prefix"].astype(str).isin(failed)
    base["running"] = base["output_prefix"].astype(str).isin(active)

    status = (
        base.groupby(["dataset", "workload", "algorithm", "nodes"], dropna=False)
        .agg(
            planned=("run_id", "count"),
            timings_found=("timing_found", "sum"),
            timings_ok=("timing_ok", "sum"),
            skipped=("skipped", "sum"),
            failed=("failed", "sum"),
            running=("running", "sum"),
        )
        .reset_index()
    )
    status["missing"] = status["planned"] - status["timings_ok"] - status["skipped"] - status["failed"] - status["running"]
    status["completed_pct"] = 100.0 * status["timings_ok"] / status["planned"]
    status["accounted_pct"] = 100.0 * (
        status["timings_ok"] + status["skipped"] + status["failed"] + status["running"]
    ) / status["planned"]
    return status.sort_values(["dataset", "workload", "algorithm", "nodes"]).reset_index(drop=True)


def collect_log_errors(
    eval_root: str | os.PathLike[str] | None = None,
    limit: int | None = 50,
) -> pd.DataFrame:
    """Collect non-empty stderr logs with a short sample."""
    _require_pandas()
    root = resolve_eval_root(eval_root)
    rows: list[dict[str, object]] = []
    for path in sorted((root / "logs").glob("*.err")):
        size = path.stat().st_size
        if size == 0:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rows.append(
            {
                "log_file": str(path),
                "bytes": size,
                "first_line": lines[0] if lines else "",
                "sample": "\n".join(lines[:5]),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def output_inventory(
    eval_root: str | os.PathLike[str] | None = None,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize result CSV files by output prefix without reading large outputs."""
    _require_pandas()
    root = resolve_eval_root(eval_root)
    by_prefix: dict[str, dict[str, object]] = {}

    for path in sorted((root / "results").rglob("*.csv")):
        if path.name.endswith(TIMING_SUFFIX):
            continue
        full = str(path)
        shard_match = re.match(r"^(?P<prefix>.+)_\d+\.csv$", full)
        if shard_match:
            prefix = shard_match.group("prefix")
            kind = "threshold_shards"
        else:
            prefix = full[: -len(".csv")]
            kind = "single_csv"

        entry = by_prefix.setdefault(
            prefix,
            {
                "output_prefix": prefix,
                "output_kind": kind,
                "output_file_count": 0,
                "output_bytes": 0,
                "sample_output_file": full,
            },
        )
        entry["output_file_count"] = int(entry["output_file_count"]) + 1
        entry["output_bytes"] = int(entry["output_bytes"]) + path.stat().st_size

    inv = pd.DataFrame(by_prefix.values())
    if inv.empty:
        return inv
    inv["output_mb"] = inv["output_bytes"] / (1024 * 1024)

    if manifest is None:
        return inv.sort_values("output_prefix").reset_index(drop=True)

    meta_cols = [
        "output_prefix",
        "dataset",
        "workload",
        "algorithm",
        "nodes",
        "replication_groups",
        "case_id",
        "query_count",
        "window_length",
        "merge_offset",
        "threshold",
        "k",
    ]
    meta = _coerce_manifest_types(manifest)[[c for c in meta_cols if c in manifest.columns]]
    return meta.merge(inv, on="output_prefix", how="inner").sort_values(
        ["dataset", "workload", "algorithm", "nodes", "case_id"]
    )


def add_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add param_name/param_value for threshold and k workloads."""
    _require_numpy()
    out = df.copy()
    out["param_name"] = np.where(out["workload"].eq("threshold"), "threshold", "k")
    out["param_value"] = np.where(out["workload"].eq("threshold"), out["threshold"], out["k"])
    return out


def matched_timings(timings: pd.DataFrame) -> pd.DataFrame:
    """Select the same completed cases for both algorithms at each configuration."""
    if timings.empty:
        return timings
    df = add_parameter_columns(timings[timings["timing_ok"].astype(bool)])
    shared = df.groupby(PAIR_INDEX, dropna=False)["algorithm"].transform(
        lambda values: {"tremor", "dmass"}.issubset(set(values)))
    return df[shared.astype(bool)]


def paired_speedups(timings: pd.DataFrame) -> pd.DataFrame:
    """Pair TREMOR and DMASS rows for the same case and compute speedups."""
    _require_pandas()
    if timings.empty:
        return pd.DataFrame()

    df = add_parameter_columns(timings[timings["timing_ok"].astype(bool)])
    metrics = ["index_time_max", "query_time_max", "total_time_max"]
    wide = df.pivot_table(index=PAIR_INDEX, columns="algorithm", values=metrics, aggfunc="median")
    if wide.empty:
        return pd.DataFrame()

    wide.columns = [f"{algorithm}_{metric}" for metric, algorithm in wide.columns]
    paired = wide.reset_index()
    needed = {"dmass_total_time_max", "tremor_total_time_max"}
    if not needed.issubset(paired.columns):
        return pd.DataFrame()

    paired = paired.dropna(subset=list(needed))
    if paired.empty:
        return paired

    paired["speedup_total_dmass_over_tremor"] = paired["dmass_total_time_max"] / paired["tremor_total_time_max"]
    paired["speedup_query_dmass_over_tremor"] = paired["dmass_query_time_max"] / paired["tremor_query_time_max"]
    paired["speedup_index_dmass_over_tremor"] = paired["dmass_index_time_max"] / paired["tremor_index_time_max"]
    return paired.sort_values(["dataset", "workload", "nodes", "param_value", "case_id"]).reset_index(drop=True)


def save_analysis_tables(
    eval_root: str | os.PathLike[str],
    timings: pd.DataFrame,
    status: pd.DataFrame,
    paired: pd.DataFrame,
    outputs: pd.DataFrame | None = None,
    skips: pd.DataFrame | None = None,
    failures: pd.DataFrame | None = None,
    running: pd.DataFrame | None = None,
) -> Path:
    """Write derived tables next to the evaluation results for reproducible plotting."""
    root = resolve_eval_root(eval_root)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings.to_csv(out_dir / "timings_long.csv", index=False)
    status.to_csv(out_dir / "completion_status.csv", index=False)
    paired.to_csv(out_dir / "paired_speedups.csv", index=False)
    if outputs is not None:
        outputs.to_csv(out_dir / "output_inventory.csv", index=False)
    if skips is not None:
        skips.to_csv(out_dir / "skipped_runs.csv", index=False)
    if failures is not None:
        failures.to_csv(out_dir / "failed_runs.csv", index=False)
    if running is not None:
        running.to_csv(out_dir / "running_runs.csv", index=False)
    return out_dir


def _filtered(df: pd.DataFrame, dataset: str | None, workload: str | None) -> pd.DataFrame:
    out = df
    if dataset is not None:
        out = out[out["dataset"].eq(dataset)]
    if workload is not None:
        out = out[out["workload"].eq(workload)]
    return out.copy()


def _require_matplotlib():
    _require_numpy()
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting cells")
    return plt


def _empty_plot(message: str):
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig, ax


def plot_completion(status: pd.DataFrame):
    _require_matplotlib()
    if status.empty:
        return _empty_plot("No planned runs found.")

    df = status.copy()
    df["label"] = (
        df["dataset"].astype(str)
        + " / "
        + df["workload"].astype(str)
        + " / "
        + df["algorithm"].astype(str)
        + " / n="
        + df["nodes"].astype(str)
    )
    colors = df["algorithm"].map({"tremor": "#2f6fbb", "dmass": "#8f4f24"}).fillna("#666666")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.28 * len(df))))
    bars = ax.barh(df["label"], df["completed_pct"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Timing files complete (%)")
    ax.set_title("Full Evaluation Completion")
    ax.grid(axis="x", alpha=0.25)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            min(99, bar.get_width() + 1),
            bar.get_y() + bar.get_height() / 2,
            (
                f"{int(row['timings_ok'])}/{int(row['planned'])}, "
                f"skip={int(row.get('skipped', 0))}, "
                f"fail={int(row.get('failed', 0))}, "
                f"run={int(row.get('running', 0))}"
            ),
            va="center",
            fontsize=8,
        )
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def plot_median_runtime_by_nodes(
    timings: pd.DataFrame,
    dataset: str | None = None,
    workload: str | None = None,
    metric: str = "total_time_max",
    log_y: bool = False,
):
    _require_matplotlib()
    sub = _filtered(matched_timings(timings), dataset, workload)
    if sub.empty:
        return _empty_plot("No completed timing rows match this selection.")

    summary = (
        sub.groupby(["algorithm", "nodes"], dropna=False)[metric]
        .median()
        .reset_index()
        .sort_values(["algorithm", "nodes"])
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algorithm, group in summary.groupby("algorithm"):
        ax.plot(group["nodes"], group[metric], marker="o", linewidth=2, label=algorithm)
    ax.set_xlabel("Nodes")
    ax.set_ylabel(metric.replace("_", " "))
    title_bits = [bit for bit in [dataset, workload] if bit]
    ax.set_title("Median Runtime by Nodes (paired cases per node count)" + (f" ({' / '.join(title_bits)})" if title_bits else ""))
    ax.grid(alpha=0.25)
    if log_y:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_runtime_by_parameter(
    timings: pd.DataFrame,
    dataset: str,
    workload: str,
    metric: str = "total_time_max",
    nodes: Iterable[int] | None = None,
    log_y: bool = False,
):
    _require_matplotlib()
    sub = _filtered(matched_timings(timings), dataset, workload)
    if sub.empty:
        return _empty_plot("No completed timing rows match this selection.")

    param_col = "threshold" if workload == "threshold" else "k"
    selected_nodes = sorted(set(nodes or sub["nodes"].dropna().astype(int).tolist()))
    ncols = min(3, len(selected_nodes))
    nrows = int(np.ceil(len(selected_nodes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.7 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, node in zip(axes_flat, selected_nodes):
        node_df = sub[sub["nodes"].astype(int).eq(node)]
        summary = (
            node_df.groupby(["algorithm", param_col], dropna=False)[metric]
            .median()
            .reset_index()
            .sort_values(["algorithm", param_col])
        )
        for algorithm, group in summary.groupby("algorithm"):
            ax.plot(group[param_col], group[metric], marker="o", linewidth=2, label=algorithm)
        ax.set_title(f"n={node}")
        ax.set_xlabel(param_col)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
        if param_col == "threshold":
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend(frameon=False)

    for ax in axes_flat[len(selected_nodes) :]:
        ax.axis("off")
    fig.suptitle(f"{dataset} {workload}: Median Runtime by {param_col}", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_speedup_heatmap(
    paired: pd.DataFrame,
    dataset: str,
    workload: str | None = None,
    value_col: str = "speedup_total_dmass_over_tremor",
):
    _require_matplotlib()
    sub = _filtered(paired, dataset, workload)
    if sub.empty:
        return _empty_plot("No paired TREMOR/DMASS runs are available yet.")

    pivot = sub.groupby(["param_value", "nodes"], dropna=False)[value_col].median().unstack("nodes")
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return _empty_plot("No finite speedup values are available.")

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax) if vmin < 1.0 < vmax else None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Threshold" if dataset == "maule" else "k")
    ax.set_title(f"{dataset}: median {value_col.replace('_', ' ')}")
    fig.colorbar(im, ax=ax, label="DMASS / TREMOR")

    if values.size <= 80:
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if np.isfinite(values[y, x]):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_output_volume_heatmap(
    outputs: pd.DataFrame,
    dataset: str = "maule",
    workload: str = "threshold",
):
    _require_matplotlib()
    sub = _filtered(add_parameter_columns(outputs), dataset, workload)
    if sub.empty:
        return _empty_plot("No result CSV outputs are available yet.")

    algorithms = sorted(sub["algorithm"].dropna().unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5.5 * len(algorithms), 4.2), squeeze=False)
    for ax, algorithm in zip(axes.ravel(), algorithms):
        part = sub[sub["algorithm"].eq(algorithm)]
        pivot = part.groupby(["param_value", "nodes"], dropna=False)["output_mb"].median().unstack("nodes")
        values = pivot.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap="Blues")
        ax.set_title(algorithm)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Threshold" if workload == "threshold" else "k")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        fig.colorbar(im, ax=ax, label="Median output MB")
    fig.suptitle(f"{dataset} {workload}: result volume", y=1.03)
    fig.tight_layout()
    return fig, axes
