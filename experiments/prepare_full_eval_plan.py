#!/usr/bin/env python3
"""Prepare dataset audits and Slurm-array plans for the full TREMOR vs MASS evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


MAULE_WAVEFORM_RE = re.compile(r"^(?P<key>.+)\.len(?P<samples>\d+)\.bin$")
SEIFR_WAVEFORM_RE = re.compile(r"^(?P<key>.+)\.len(?P<samples>\d+)(?:\..*)?\.bin$")
MAULE_TEMPLATE_RE = re.compile(
    r"^tem(?P<count>\d+)\.(?P<key>.+)\.len(?P<window>\d+)"
    r"\.fmin(?P<fmin>[0-9]+(?:\.[0-9]+)?)"
    r"\.fmax(?P<fmax>[0-9]+(?:\.[0-9]+)?)\.bin$"
)
SEIFR_TEMPLATE_RE = MAULE_TEMPLATE_RE
SEIFR_LEGACY_TEMPLATE_RE = re.compile(
    r"^templ(?P<count>\d+)\.(?P<key>.+)\.len(?P<window>\d+)"
    r"(?:\.sr_[0-9]+(?:\.[0-9]+)?)?\.bin$"
)


CASE_FIELDS = [
    "dataset",
    "case_id",
    "query_count",
    "window_length",
    "waveform_samples",
    "query_floats",
    "waveform",
    "queries",
]

PLAN_FIELDS = [
    "array_index",
    "run_id",
    "dataset",
    "workload",
    "algorithm",
    "mode",
    "nodes",
    "replication_groups",
    "case_id",
    "query_count",
    "window_length",
    "merge_offset",
    "threshold",
    "k",
    "waveform_samples",
    "query_floats",
    "waveform",
    "queries",
    "output_prefix",
    "result_kind",
]

MANIFEST_FIELDS = ["manifest_index", "batch_id", "plan_file"] + PLAN_FIELDS
BATCH_FIELDS = [
    "batch_id",
    "dataset",
    "workload",
    "algorithm",
    "nodes",
    "task_count",
    "run_count",
    "runs_per_array_task",
    "plan_file",
    "result_dir",
    "log_glob",
]


@dataclass(frozen=True)
class Case:
    dataset: str
    case_id: str
    query_count: int
    window_length: int
    waveform_samples: int
    query_floats: int
    waveform: Path
    queries: Path


def parse_space_list(value: str, item_name: str) -> list[str]:
    items = [item.strip() for item in value.split() if item.strip()]
    if not items:
        raise ValueError(f"{item_name} list is empty")
    return items


def file_float_count(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    return size // 4, size % 4


def slug(value: str) -> str:
    return value.replace("-", "m").replace(".", "p")


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_+.-]+", "_", value)
    return safe.strip("._") or "case"


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_waveforms(
    directory: Path,
    regex: re.Pattern[str],
    issues: list[dict[str, str]],
    dataset: str,
) -> dict[str, tuple[Path, int]]:
    waveforms: dict[str, tuple[Path, int]] = {}
    for path in sorted(directory.glob("*.bin")):
        if path.name.startswith("tem") or path.name.endswith(".tmp.bin"):
            continue
        match = regex.match(path.name)
        if not match:
            issues.append({"dataset": dataset, "kind": "waveform", "path": str(path), "issue": "bad_name"})
            continue
        expected = int(match.group("samples"))
        floats, rem = file_float_count(path)
        if rem != 0 or floats != expected:
            issues.append(
                {
                    "dataset": dataset,
                    "kind": "waveform",
                    "path": str(path),
                    "issue": f"size_mismatch:floats={floats},byte_rem={rem},expected={expected}",
                }
            )
            continue
        key = match.group("key")
        if key in waveforms:
            issues.append({"dataset": dataset, "kind": "waveform", "path": str(path), "issue": "duplicate_channel"})
            continue
        waveforms[key] = (path, floats)
    return waveforms


def validate_template_case(
    *,
    dataset: str,
    path: Path,
    match: re.Match[str],
    waveforms: dict[str, tuple[Path, int]],
    paa_segments: int,
    merge_offset_divisor: int,
    issues: list[dict[str, str]],
) -> Optional[Case]:
    key = match.group("key")
    query_count = int(match.group("count"))
    window = int(match.group("window"))
    query_floats, rem = file_float_count(path)
    expected_query_floats = query_count * window

    reasons: list[str] = []
    if query_count <= 0 or window <= 0:
        reasons.append("nonpositive_query_count_or_length")
    if rem != 0:
        reasons.append(f"byte_rem={rem}")
    if query_floats != expected_query_floats:
        reasons.append(f"query_floats={query_floats},expected={expected_query_floats}")
    if window % 8 != 0:
        reasons.append("window_not_divisible_by_8")
    if window % paa_segments != 0:
        reasons.append(f"window_not_divisible_by_paa_segments_{paa_segments}")
    if window // merge_offset_divisor <= 0:
        reasons.append(f"merge_offset_divisor_{merge_offset_divisor}_too_large")
    if key not in waveforms:
        reasons.append("missing_matching_waveform")
    elif waveforms[key][1] < window:
        reasons.append("waveform_shorter_than_query")

    metadata_dir = Path(__file__).resolve().parents[1] / 'dataprep' / ('seiFR' if dataset == 'seifr' else dataset) / 'metadata'
    try:
        query_metadata = json.loads((metadata_dir / path.with_suffix('.json').name).read_text())
        wave_path = waveforms[key][0]
        wave_metadata = json.loads((metadata_dir / wave_path.with_suffix('.json').name).read_text())
        if any(r.get('pipeline_version') != 2 for r in [query_metadata, wave_metadata]):
            reasons.append('obsolete_preprocessing')
        if query_metadata['frequency']['sampling_rate_hz'] != wave_metadata['frequency']['sampling_rate_hz']:
            reasons.append('sampling_rate_mismatch')
        for record in [query_metadata, wave_metadata]:
            if record['amplitude_stats']['nan_points'] or record['amplitude_stats']['inf_points']:
                reasons.append('nonfinite_sidecar')
    except (OSError, ValueError, KeyError):
        reasons.append('missing_or_invalid_metadata')

    if reasons:
        issues.append({"dataset": dataset, "kind": "template", "path": str(path), "issue": ";".join(reasons)})
        return None

    waveform, waveform_samples = waveforms[key]
    return Case(
        dataset=dataset,
        case_id=safe_id(key),
        query_count=query_count,
        window_length=window,
        waveform_samples=waveform_samples,
        query_floats=query_floats,
        waveform=waveform,
        queries=path,
    )


def discover_maule_cases(args: argparse.Namespace, issues: list[dict[str, str]]) -> list[Case]:
    waveforms = discover_waveforms(args.maule_waveform_dir, MAULE_WAVEFORM_RE, issues, "maule")
    cases: list[Case] = []
    for path in sorted(args.maule_template_dir.glob("tem*.bin")):
        match = MAULE_TEMPLATE_RE.match(path.name)
        if not match:
            issues.append({"dataset": "maule", "kind": "template", "path": str(path), "issue": "bad_name"})
            continue
        case = validate_template_case(
            dataset="maule",
            path=path,
            match=match,
            waveforms=waveforms,
            paa_segments=args.paa_segments,
            merge_offset_divisor=args.merge_offset_divisor,
            issues=issues,
        )
        if case is not None:
            cases.append(case)
    return cases


def discover_seifr_cases(args: argparse.Namespace, issues: list[dict[str, str]]) -> tuple[list[Case], Path]:
    waveforms = discover_waveforms(args.seifr_waveform_dir, SEIFR_WAVEFORM_RE, issues, "seifr")

    template_dir = args.seifr_template_dir
    templates = sorted(template_dir.glob("tem*.bin"))
    template_regex = SEIFR_TEMPLATE_RE
    if not templates:
        template_dir = args.seifr_legacy_template_dir
        templates = sorted(template_dir.glob("templ*.bin"))
        template_regex = SEIFR_LEGACY_TEMPLATE_RE

    cases: list[Case] = []
    for path in templates:
        match = template_regex.match(path.name)
        if not match:
            issues.append({"dataset": "seifr", "kind": "template", "path": str(path), "issue": "bad_name"})
            continue
        case = validate_template_case(
            dataset="seifr",
            path=path,
            match=match,
            waveforms=waveforms,
            paa_segments=args.paa_segments,
            merge_offset_divisor=args.merge_offset_divisor,
            issues=issues,
        )
        if case is not None:
            cases.append(case)
    return cases, template_dir


def case_rows(cases: list[Case]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.append(
            {
                "dataset": case.dataset,
                "case_id": case.case_id,
                "query_count": case.query_count,
                "window_length": case.window_length,
                "waveform_samples": case.waveform_samples,
                "query_floats": case.query_floats,
                "waveform": str(case.waveform),
                "queries": str(case.queries),
            }
        )
    return rows


def select_cases_by_size(cases: list[Case], count: int, min_samples: int) -> list[Case]:
    """Pick `count` cases at evenly spaced ranks of waveform length.

    Deterministic and size-stratified, so the subset spans small to large
    channels instead of following filename order.
    """
    eligible = sorted((c for c in cases if c.waveform_samples >= min_samples),
                      key=lambda c: (c.waveform_samples, c.case_id))
    if count <= 0 or count >= len(eligible):
        return eligible
    if count == 1:
        return [eligible[len(eligible) // 2]]
    ranks = sorted({round(i * (len(eligible) - 1) / (count - 1)) for i in range(count)})
    return [eligible[r] for r in ranks]


def sample_queries(cases: list[Case], output_root: Path, limit: int, seed: int) -> list[Case]:
    """Materialize one reproducible query sample shared by every algorithm/rank count."""
    sampled = []
    for case in cases:
        if limit <= 0 or case.query_count <= limit:
            sampled.append(case)
            continue
        case_seed = int.from_bytes(hashlib.sha256(f'{seed}:{case.dataset}:{case.case_id}'.encode()).digest()[:8], 'little')
        indices = sorted(random.Random(case_seed).sample(range(case.query_count), limit))
        target = output_root / 'inputs' / case.dataset / f'{case.case_id}.q{limit}.bin'
        target.parent.mkdir(parents=True, exist_ok=True)
        with case.queries.open('rb') as src, target.open('wb') as dst:
            for index in indices:
                src.seek(index * case.window_length * 4)
                block = src.read(case.window_length * 4)
                if len(block) != case.window_length * 4:
                    raise ValueError(f'Truncated query input: {case.queries}')
                dst.write(block)
        target.with_suffix('.json').write_text(json.dumps({
            'source': str(case.queries), 'source_count': case.query_count,
            'seed': seed, 'source_indices': indices, 'window_length': case.window_length,
        }, indent=2) + '\n')
        sampled.append(replace(case, query_count=limit, query_floats=limit*case.window_length, queries=target))
    return sampled


def make_run_id(
    *,
    dataset: str,
    workload: str,
    case: Case,
    nodes: str,
    replication_groups: str,
    threshold: Optional[str] = None,
    k_value: Optional[str] = None,
) -> str:
    base = f"{dataset}_{workload}.{case.case_id}.w{case.window_length}.q{case.query_count}"
    if threshold is not None:
        base += f".th{slug(threshold)}"
    if k_value is not None:
        base += f".k{k_value}"
    return f"{base}.n{nodes}.r{replication_groups}"


def result_kind(workload: str) -> str:
    if workload == "threshold":
        return "threshold_shards"
    if workload == "knn":
        return "knn_csv"
    raise ValueError(f"unknown workload {workload}")


def plan_rows_for_algorithm(
    *,
    output_root: Path,
    cases: list[Case],
    dataset: str,
    workload: str,
    algorithm: str,
    mode: str,
    nodes_values: list[str],
    replication_values: list[str],
    thresholds: list[str] | None,
    k_values: list[str] | None,
    merge_offset_divisor: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    result_dir = output_root / "results" / f"{dataset}_{workload}" / algorithm
    for case in cases:
        # Detections (threshold) and neighbors (kNN, exclusion zone) closer
        # than the merge offset are merged; both engines use the same rules.
        merge_offset = max(1, case.window_length // merge_offset_divisor)
        for nodes in nodes_values:
            for reps in replication_values:
                values = thresholds if workload == "threshold" else k_values
                assert values is not None
                for value in values:
                    threshold = value if workload == "threshold" else None
                    k_value = value if workload == "knn" else None
                    run_id = make_run_id(
                        dataset=dataset,
                        workload=workload,
                        case=case,
                        nodes=nodes,
                        replication_groups=reps,
                        threshold=threshold,
                        k_value=k_value,
                    )
                    rows.append(
                        {
                            "array_index": "NA",
                            "run_id": run_id,
                            "dataset": dataset,
                            "workload": workload,
                            "algorithm": algorithm,
                            "mode": mode,
                            "nodes": nodes,
                            "replication_groups": reps,
                            "case_id": case.case_id,
                            "query_count": case.query_count,
                            "window_length": case.window_length,
                            "merge_offset": merge_offset,
                            "threshold": threshold or "NA",
                            "k": k_value or "NA",
                            "waveform_samples": case.waveform_samples,
                            "query_floats": case.query_floats,
                            "waveform": str(case.waveform),
                            "queries": str(case.queries),
                            "output_prefix": str(result_dir / run_id),
                            "result_kind": result_kind(workload),
                        }
                    )
    return rows


def build_all_plan_rows(
    output_root: Path,
    maule_cases: list[Case],
    seifr_cases: list[Case],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    nodes = parse_space_list(args.nodes, "nodes")
    reps = parse_space_list(args.replication_groups, "replication groups")
    workloads = parse_space_list(args.workloads, "workloads")
    # workload name -> (dataset, cases, workload, parameter list option, parameter name)
    table = {
        "maule_threshold": ("maule", maule_cases, "threshold", args.thresholds, "thresholds"),
        "maule_knn": ("maule", maule_cases, "knn", args.maule_k_values, "maule k values"),
        "seifr_knn": ("seifr", seifr_cases, "knn", args.k_values, "k values"),
        "seifr_threshold": ("seifr", seifr_cases, "threshold", args.thresholds, "thresholds"),
    }

    rows: list[dict[str, object]] = []
    for name in workloads:
        dataset, cases, workload, values, item_name = table[name]
        values = parse_space_list(values, item_name)
        for algorithm, mode in [("tremor", workload), ("dmass", f"{workload}-search")]:
            rows.extend(
                plan_rows_for_algorithm(
                    output_root=output_root,
                    cases=cases,
                    dataset=dataset,
                    workload=workload,
                    algorithm=algorithm,
                    mode=mode,
                    nodes_values=nodes,
                    replication_values=reps,
                    thresholds=values if workload == "threshold" else None,
                    k_values=values if workload == "knn" else None,
                    merge_offset_divisor=args.merge_offset_divisor,
                )
            )
    return rows


def write_plan_batches(
    output_root: Path,
    rows: list[dict[str, object]],
    max_array_tasks: int,
    runs_per_array_task: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if max_array_tasks <= 0:
        raise ValueError("--max-array-tasks must be positive")
    if runs_per_array_task <= 0:
        raise ValueError("--runs-per-array-task must be positive")

    plans_dir = output_root / "plans"
    logs_dir = output_root / "logs"
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["workload"]), str(row["algorithm"]), str(row["nodes"]))].append(row)

    def group_sort_key(item):
        dataset, workload, algorithm, nodes = item[0]
        node_key = int(nodes) if nodes.isdigit() else nodes
        return dataset, workload, algorithm, node_key

    manifest_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    manifest_index = 0

    for (dataset, workload, algorithm, nodes), group_rows in sorted(grouped.items(), key=group_sort_key):
        max_rows_per_plan = max_array_tasks * runs_per_array_task
        parts = math.ceil(len(group_rows) / max_rows_per_plan)
        for part in range(parts):
            chunk = group_rows[part * max_rows_per_plan : (part + 1) * max_rows_per_plan]
            batch_id = f"{dataset}_{workload}.{algorithm}.n{nodes}.part{part + 1:03d}"
            plan_file = plans_dir / f"{batch_id}.tsv"
            result_dir = output_root / "results" / f"{dataset}_{workload}" / algorithm
            log_glob = logs_dir / f"{batch_id}.%A_%a.*"
            task_count = math.ceil(len(chunk) / runs_per_array_task)

            plan_rows: list[dict[str, object]] = []
            for row_index, row in enumerate(chunk):
                row = dict(row)
                row["array_index"] = row_index // runs_per_array_task
                plan_rows.append(row)

                manifest_row = {"manifest_index": manifest_index, "batch_id": batch_id, "plan_file": str(plan_file)}
                manifest_row.update(row)
                manifest_rows.append(manifest_row)
                manifest_index += 1

            write_tsv(plan_file, PLAN_FIELDS, plan_rows)
            batch_rows.append(
                {
                    "batch_id": batch_id,
                    "dataset": dataset,
                    "workload": workload,
                    "algorithm": algorithm,
                    "nodes": nodes,
                    "task_count": task_count,
                    "run_count": len(plan_rows),
                    "runs_per_array_task": runs_per_array_task,
                    "plan_file": str(plan_file),
                    "result_dir": str(result_dir),
                    "log_glob": str(log_glob),
                }
            )

    write_tsv(output_root / "manifest.tsv", MANIFEST_FIELDS, manifest_rows)
    write_tsv(output_root / "batches.tsv", BATCH_FIELDS, batch_rows)
    return manifest_rows, batch_rows


def write_summary(
    *,
    output_root: Path,
    args: argparse.Namespace,
    maule_cases: list[Case],
    seifr_cases: list[Case],
    seifr_template_dir: Path,
    issues: list[dict[str, str]],
    manifest_rows: list[dict[str, object]],
    batch_rows: list[dict[str, object]],
) -> None:
    write_tsv(output_root / "cases" / "maule_cases.tsv", CASE_FIELDS, case_rows(maule_cases))
    write_tsv(output_root / "cases" / "seifr_cases.tsv", CASE_FIELDS, case_rows(seifr_cases))
    write_tsv(output_root / "audit_issues.tsv", ["dataset", "kind", "path", "issue"], issues)

    summary = {
        "output_root": str(output_root),
        "parameters": {
            "nodes": parse_space_list(args.nodes, "nodes"),
            "thresholds": parse_space_list(args.thresholds, "thresholds"),
            "k_values": parse_space_list(args.k_values, "k values"),
            "replication_groups": parse_space_list(args.replication_groups, "replication groups"),
            "paa_segments": args.paa_segments,
            "merge_offset_divisor": args.merge_offset_divisor,
            "max_array_tasks": args.max_array_tasks,
            "runs_per_array_task": args.runs_per_array_task,
            "query_limit": args.query_limit,
            "query_seed": args.query_seed,
            "maule_case_count": args.maule_case_count,
            "maule_min_samples": args.maule_min_samples,
        },
        "datasets": {
            "maule": {
                "waveform_dir": str(args.maule_waveform_dir),
                "template_dir": str(args.maule_template_dir),
                "valid_cases": len(maule_cases),
            },
            "seifr": {
                "waveform_dir": str(args.seifr_waveform_dir),
                "template_dir": str(seifr_template_dir),
                "valid_cases": len(seifr_cases),
            },
        },
        "runs": {
            "total_plan_rows": len(manifest_rows),
            "total_batches": len(batch_rows),
        },
        "audit": {
            "issue_count": len(issues),
            "issues_file": str(output_root / "audit_issues.tsv"),
        },
        "analysis_inputs": {
            "manifest": str(output_root / "manifest.tsv"),
            "batches": str(output_root / "batches.tsv"),
            "maule_tremor_results": str(output_root / "results" / "maule_threshold" / "tremor"),
            "maule_dmass_results": str(output_root / "results" / "maule_threshold" / "dmass"),
            "seifr_tremor_results": str(output_root / "results" / "seifr_knn" / "tremor"),
            "seifr_dmass_results": str(output_root / "results" / "seifr_knn" / "dmass"),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    summary_rows = [
        {"metric": "maule_valid_cases", "value": len(maule_cases)},
        {"metric": "seifr_valid_cases", "value": len(seifr_cases)},
        {"metric": "total_plan_rows", "value": len(manifest_rows)},
        {"metric": "total_batches", "value": len(batch_rows)},
        {"metric": "audit_issue_count", "value": len(issues)},
    ]
    write_tsv(output_root / "summary.tsv", ["metric", "value"], summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maule-waveform-dir", type=Path, default=Path("/path/to/data/maule/waveforms"))
    parser.add_argument("--maule-template-dir", type=Path, default=Path("/path/to/data/maule/templates"))
    parser.add_argument("--seifr-waveform-dir", type=Path, default=Path("/path/to/data/seifr/waveforms"))
    parser.add_argument("--seifr-template-dir", type=Path, default=Path("/path/to/data/seifr/templates"))
    parser.add_argument("--seifr-legacy-template-dir", type=Path, default=Path("/path/to/data/seifr/waveforms"))
    parser.add_argument("--nodes", default="1 2 4 6 8 10")
    parser.add_argument("--thresholds", default="0.3 0.5 0.8 0.9 0.95 0.99")
    parser.add_argument("--maule-k-values", default="1 10 100", help="k values of the maule_knn workload")
    parser.add_argument("--workloads", default="maule_threshold seifr_knn",
                        help="subset of: maule_threshold maule_knn seifr_knn seifr_threshold")
    parser.add_argument("--k-values", default="10 50 100 200")
    parser.add_argument("--replication-groups", default="0")
    parser.add_argument("--paa-segments", type=int, default=8)
    parser.add_argument("--merge-offset-divisor", type=int, default=4)
    parser.add_argument("--max-array-tasks", type=int, default=900)
    parser.add_argument("--runs-per-array-task", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=5000, help="Maximum queries per channel; 0 uses all queries")
    parser.add_argument("--query-seed", type=int, default=20260917)
    parser.add_argument("--maule-case-count", type=int, default=0,
                        help="Size-stratified subset of Maule channels; 0 keeps all")
    parser.add_argument("--maule-min-samples", type=int, default=0,
                        help="Drop Maule channels shorter than this before selection")
    parser.add_argument("--maule-max-samples", type=int, default=0,
                        help="Drop Maule channels longer than this before selection (0: no limit)")
    parser.add_argument("--fail-on-audit-issues", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()

    for directory in [
        args.maule_waveform_dir,
        args.maule_template_dir,
        args.seifr_waveform_dir,
    ]:
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    issues: list[dict[str, str]] = []
    maule_cases = discover_maule_cases(args, issues)
    seifr_cases, seifr_template_dir = discover_seifr_cases(args, issues)

    if not maule_cases:
        raise RuntimeError("No valid Maule cases discovered")
    if not seifr_cases:
        raise RuntimeError("No valid SeiFR cases discovered")
    if issues and args.fail_on_audit_issues:
        write_tsv(output_root / "audit_issues.tsv", ["dataset", "kind", "path", "issue"], issues)
        raise RuntimeError(f"Dataset audit found {len(issues)} issue(s)")

    for cases in [maule_cases, seifr_cases]:
        if len({c.case_id for c in cases}) != len(cases):
            raise ValueError('Multiple query files for a channel; remove stale generation outputs')
    if args.maule_max_samples > 0:
        maule_cases = [c for c in maule_cases if c.waveform_samples <= args.maule_max_samples]
    maule_cases = select_cases_by_size(maule_cases, args.maule_case_count, args.maule_min_samples)
    if not maule_cases:
        raise RuntimeError("No Maule cases left after size selection")
    maule_cases = sample_queries(maule_cases, output_root, args.query_limit, args.query_seed)
    seifr_cases = sample_queries(seifr_cases, output_root, args.query_limit, args.query_seed)

    all_rows = build_all_plan_rows(output_root, maule_cases, seifr_cases, args)
    for row in all_rows:
        Path(str(row["output_prefix"])).parent.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest_rows, batch_rows = write_plan_batches(
        output_root,
        all_rows,
        args.max_array_tasks,
        args.runs_per_array_task,
    )
    write_summary(
        output_root=output_root,
        args=args,
        maule_cases=maule_cases,
        seifr_cases=seifr_cases,
        seifr_template_dir=seifr_template_dir,
        issues=issues,
        manifest_rows=manifest_rows,
        batch_rows=batch_rows,
    )

    print(f"Prepared full evaluation under {output_root}")
    print(f"  Maule valid cases: {len(maule_cases)}")
    print(f"  SeiFR valid cases: {len(seifr_cases)} from {seifr_template_dir}")
    print(f"  Plan rows: {len(manifest_rows)}")
    print(f"  Slurm array batches: {len(batch_rows)}")
    print(f"  Audit issues: {len(issues)}")
    print(f"  Manifest: {output_root / 'manifest.tsv'}")
    print(f"  Batches:  {output_root / 'batches.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
