"""Fail-closed static checks for the public research repository.

This gate inspects files tracked by Git rather than every local file. Internal
notes, raw data, models, and outputs may remain on the user's machine, but they
must not enter the public commit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PUBLIC_FILES = {
    ".github/workflows/public-smoke.yml",
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    ".env.example",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-mining.txt",
    "requirements-tree.txt",
    "docs/README.md",
    "docs/DATA.md",
    "docs/METHODOLOGY.md",
    "docs/RESULTS.md",
    "docs/LIMITATIONS.md",
    "docs/REPRODUCIBILITY.md",
    "results/public/us300_release_v1/README.md",
    "scripts/build_public_figures.py",
    "scripts/check_public_repository.py",
    "scripts/export_public_evidence.py",
    "scripts/run_canonical_us300.ps1",
    "scripts/run_canonical_us300.sh",
    "scripts/run_canonical_us300_backtest.ps1",
    "scripts/run_canonical_us300_backtest.sh",
    "scripts/run_public_smoke.ps1",
    "scripts/run_public_smoke.sh",
    "tests/test_temporal_boundaries.py",
}
FORBIDDEN_TRACKED_PREFIXES = (
    "data/",
    "models/",
    "outputs/",
    ".cache/",
    ".venv/",
    "backtest_from_outputs/",
    "desktop_tauri/",
    "web_app/",
    "docs/learning/",
    "docs/interview/",
    "docs/presentation/",
    "docs/overview/",
    "results/public/us300_v013/",
)
FORBIDDEN_TRACKED_PATHS = {
    "factor_mining_workspace/AUTO_FACTOR_MINING_SUMMARY.md",
    "factor_mining_workspace/CURRENT_FACTOR_DIRECTION.md",
    "factor_mining_workspace/GENERATIVE_FACTOR_MINING_SUMMARY.md",
    "factor_mining_workspace/NEW_AI_PROMPT.md",
    "factor_mining_workspace/RECENT_FACTOR_MINING_PAPERS.md",
    "factor_mining_workspace/RL_GENERATIVE_FACTOR_RESULTS_SUMMARY.md",
}
FORBIDDEN_BINARY_SUFFIXES = {".joblib", ".pkl", ".pt", ".parquet", ".npy", ".npz"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".sh", ".ps1", ".csv"}
TEXT_FILENAMES = {"README", "LICENSE", ".env.example", ".gitignore", ".gitattributes"}
REQUIRED_RELEASE_EVIDENCE_FILES = (
    "run_manifest.json",
    "source_run_manifest.json",
    "data_quality_summary.json",
    "corporate_action_audit.csv",
    "universe_coverage_audit.csv",
    "walk_forward_fold_metrics.csv",
    "walk_forward_model_summary.csv",
    "stage_timing.csv",
    "final_model_timing.csv",
    "selected_features.csv",
    "selected_feature_scores.csv",
    "model_weights.csv",
    "feature_family_summary.csv",
    "feature_selection_funnel.csv",
    "oos_metrics.json",
    "data_summary.json",
    "runtime_summary.json",
    "portfolio_grid_summary.csv",
    "portfolio_cost_summary.csv",
    "portfolio_anomaly_summary.csv",
    "backtest_run_manifest.json",
    "figures/walk_forward_ic.png",
    "figures/feature_family_inventory.png",
    "figures/portfolio_diagnostic_20bps.png",
    "figures/portfolio_cost_sensitivity.png",
)
REQUIRED_PUBLIC_PORTFOLIO_DETAIL_FILES = (
    "daily_returns.csv",
    "portfolio_weights.csv",
    "turnover_cost.csv",
    "skipped_trades.csv",
    "sector_exposure.csv",
    "extreme_return_days.csv",
    "position_daily_contributions.csv",
    "instrument_return_attribution.csv",
    "portfolio_metrics.json",
    "portfolio_report.md",
)
MACHINE_PATH_PATTERN_ALLOWLIST = {"scripts/check_public_repository.py"}
MACHINE_PATH_PATTERN = re.compile(r"/Users/[^/\s]+|/opt/anaconda3|[A-Za-z]:\\Users\\[^\\\s]+")
SECRET_PATTERNS = [
    re.compile(r"(?i)(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]([A-Za-z0-9_-]{20,})['\"]"),
    re.compile(r"(?i)FMP_API_KEY\s*=\s*['\"](?!your_api_key_here)([^'$\"]{12,})['\"]"),
]
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check public Git-tracked repository hygiene.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument(
        "--release",
        action="store_true",
        help="Treat pre-release evidence markers or a non-release manifest as blocking errors.",
    )
    return parser.parse_args()


def tracked_files(root: Path) -> list[str]:
    """Read the Git index so ignored private files do not create false alarms."""

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def release_head_history_paths(root: Path) -> set[str]:
    """Return every path reachable from the release candidate's HEAD.

    Deleting a private note in the newest commit does not remove it from Git
    history. The gate therefore scans the complete ancestry that would be pushed
    for the current release branch. It intentionally does not scan unrelated local
    branches: a clean public orphan branch must be able to coexist with private
    local history that will never be pushed.
    """

    completed = subprocess.run(
        ["git", "rev-list", "--objects", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: set[str] = set()
    for line in completed.stdout.splitlines():
        _, separator, path = line.partition(" ")
        if separator and path:
            paths.add(path.strip())
    return paths


def git_commit_is_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    """Verify that the evidence-producing source commit belongs to the release history."""

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def is_forbidden_tracked_path(relative_path: str) -> bool:
    """Keep local notes, generated experiments, and app prototypes out of Git."""

    if relative_path in FORBIDDEN_TRACKED_PATHS:
        return True
    if relative_path.startswith(FORBIDDEN_TRACKED_PREFIXES):
        return True
    path = Path(relative_path)
    if len(path.parts) >= 2 and path.parts[0] == "factor_mining_workspace":
        # The mining source code is public, while any run directory whose first
        # component contains "outputs" remains local and can include large model
        # binaries, exploratory OOS rankings, or stale result narratives.
        return "outputs" in path.parts[1].lower()
    return False


def check_markdown_links(root: Path, relative_path: str, text: str) -> list[str]:
    """Validate repository-relative file links; URLs and document anchors are skipped."""

    problems: list[str] = []
    source_path = root / relative_path
    for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target_without_anchor = target.split("#", maxsplit=1)[0]
        if not target_without_anchor:
            continue
        # Percent-encoded spaces and other URL escapes are uncommon in the public
        # docs. Keep validation conservative and avoid accepting absolute paths.
        if target_without_anchor.startswith(("/", "file:")):
            problems.append(f"{relative_path}: absolute Markdown link: {target}")
            continue
        resolved = (source_path.parent / target_without_anchor).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            problems.append(f"{relative_path}: link escapes repository: {target}")
            continue
        if not resolved.exists():
            problems.append(f"{relative_path}: missing linked file: {target}")
    return problems


def run_checks(root: Path, *, release: bool = False) -> tuple[list[str], list[str]]:
    tracked = tracked_files(root)
    tracked_set = set(tracked)
    errors: list[str] = []
    warnings: list[str] = []

    for required_file in sorted(REQUIRED_PUBLIC_FILES - tracked_set):
        errors.append(f"required public file is not tracked: {required_file}")

    for relative_path in tracked:
        path = root / relative_path
        if not path.exists():
            errors.append(f"tracked path is missing from worktree: {relative_path}")
            continue
        if is_forbidden_tracked_path(relative_path):
            errors.append(f"private/generated path is tracked: {relative_path}")
        if path.suffix.lower() in FORBIDDEN_BINARY_SUFFIXES:
            errors.append(f"binary research artifact is tracked: {relative_path}")
        if path.is_file() and path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f"tracked file exceeds 25 MiB: {relative_path}")
        if not path.is_file() or not is_text_candidate(path):
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if relative_path not in MACHINE_PATH_PATTERN_ALLOWLIST and MACHINE_PATH_PATTERN.search(text):
            errors.append(f"machine-specific path found: {relative_path}")
        for secret_pattern in SECRET_PATTERNS:
            if secret_pattern.search(text):
                errors.append(f"possible plaintext credential found: {relative_path}")
                break
        if path.suffix.lower() == ".md":
            errors.extend(check_markdown_links(root, relative_path, text))

    evidence_readme = root / "results/public/us300_release_v1/README.md"
    evidence_text = evidence_readme.read_text(encoding="utf-8").lower() if evidence_readme.exists() else ""
    if "pre-release" in evidence_text or "pending clean rerun" in evidence_text:
        message = "public evidence package is still pending a clean rerun"
        (errors if release else warnings).append(message)
    if release:
        historical_private_paths = sorted(
            path
            for path in release_head_history_paths(root)
            if is_forbidden_tracked_path(path)
        )
        if historical_private_paths:
            preview = ", ".join(historical_private_paths[:5])
            remaining = len(historical_private_paths) - min(5, len(historical_private_paths))
            suffix = f" (+{remaining} more)" if remaining else ""
            errors.append(
                "private/generated paths remain reachable from release HEAD: "
                f"{preview}{suffix}. Create a clean public history or explicitly sanitize history."
            )

        root_readme = root / "README.md"
        if root_readme.exists() and "pre-release evidence boundary" in root_readme.read_text(
            encoding="utf-8"
        ).lower():
            errors.append("root README still presents pre-release evidence")
        results_doc = root / "docs/RESULTS.md"
        if results_doc.exists() and "status: pre-release" in results_doc.read_text(encoding="utf-8").lower():
            errors.append("docs/RESULTS.md is still marked pre-release")
        evidence_dir = root / "results/public/us300_release_v1"
        for relative_evidence_path in REQUIRED_RELEASE_EVIDENCE_FILES:
            evidence_path = evidence_dir / relative_evidence_path
            tracked_evidence_path = (
                Path("results/public/us300_release_v1") / relative_evidence_path
            ).as_posix()
            if not evidence_path.is_file():
                errors.append(f"release evidence file is missing: {relative_evidence_path}")
            elif tracked_evidence_path not in tracked_set:
                errors.append(f"release evidence file is not tracked: {relative_evidence_path}")
        manifest_path = evidence_dir / "experiment_manifest.json"
        if not manifest_path.is_file():
            errors.append("release evidence manifest is missing")
        elif "results/public/us300_release_v1/experiment_manifest.json" not in tracked_set:
            errors.append("release evidence manifest is not tracked")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"release evidence manifest is invalid JSON: {exc}")
            else:
                if manifest.get("public_status") != "release_candidate_requires_review":
                    errors.append(
                        "release evidence manifest has non-release status: "
                        f"{manifest.get('public_status')!r}"
                    )
                source_git = manifest.get("source_git", {}) or {}
                source_commit = str(source_git.get("commit") or "").strip()
                if not source_commit:
                    errors.append("release evidence manifest has no source commit")
                elif not git_commit_is_ancestor(root, source_commit):
                    errors.append(
                        "release evidence source commit is not reachable from release HEAD: "
                        f"{source_commit}"
                    )
                if source_git.get("dirty_tracked_worktree") is not False:
                    errors.append("release evidence was generated from a dirty worktree")
                public_run_names = manifest.get("public_portfolio_run_names", [])
                if not isinstance(public_run_names, list) or not public_run_names:
                    errors.append("release evidence manifest has no public portfolio runs")
                else:
                    for run_name in public_run_names:
                        if not isinstance(run_name, str) or not run_name.strip():
                            errors.append(f"invalid public portfolio run name: {run_name!r}")
                            continue
                        run_dir = evidence_dir / "portfolio_runs" / run_name
                        for filename in REQUIRED_PUBLIC_PORTFOLIO_DETAIL_FILES:
                            detail_path = run_dir / filename
                            tracked_detail_path = (
                                Path("results/public/us300_release_v1")
                                / "portfolio_runs"
                                / run_name
                                / filename
                            ).as_posix()
                            if not detail_path.is_file():
                                errors.append(
                                    "release portfolio detail is missing: "
                                    f"portfolio_runs/{run_name}/{filename}"
                                )
                            elif tracked_detail_path not in tracked_set:
                                errors.append(
                                    "release portfolio detail is not tracked: "
                                    f"portfolio_runs/{run_name}/{filename}"
                                )
    return errors, warnings


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    errors, warnings = run_checks(root, release=bool(args.release))
    for warning in warnings:
        print(f"[Warning] {warning}")
    for error in errors:
        print(f"[Error] {error}")
    if errors:
        print(f"[Failed] public repository gate found {len(errors)} error(s).")
        return 1
    print("[Passed] public repository static gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
