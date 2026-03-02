#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: haskell/scripts/generate_roi_scorecard.sh [options]

Options:
  --week YYYY-WW      ISO week to report (default: current ISO week in UTC)
  --output PATH       Output markdown path
  --repo OWNER/REPO   GitHub repo override for CI metrics (default: infer from git remote)
  --no-gh             Skip GitHub Actions metrics (git-only scorecard)
  -h, --help          Show this help

Examples:
  ./haskell/scripts/generate_roi_scorecard.sh
  ./haskell/scripts/generate_roi_scorecard.sh --week 2026-09
  ./haskell/scripts/generate_roi_scorecard.sh --week 2026-09 --output tmp/roi-scorecard-2026-09.md
EOF
}

WEEK=""
OUTPUT=""
REPO_OVERRIDE=""
USE_GH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --week)
      [[ $# -ge 2 ]] || {
        echo "error: --week requires a value" >&2
        exit 1
      }
      WEEK="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || {
        echo "error: --output requires a value" >&2
        exit 1
      }
      OUTPUT="$2"
      shift 2
      ;;
    --repo)
      [[ $# -ge 2 ]] || {
        echo "error: --repo requires a value" >&2
        exit 1
      }
      REPO_OVERRIDE="$2"
      shift 2
      ;;
    --no-gh)
      USE_GH=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

python3 - "$ROOT_DIR" "$WEEK" "$OUTPUT" "$REPO_OVERRIDE" "$USE_GH" <<'PY'
import datetime as dt
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


ROOT = Path(sys.argv[1])
WEEK_ARG = sys.argv[2].strip()
OUTPUT_ARG = sys.argv[3].strip()
REPO_OVERRIDE = sys.argv[4].strip()
USE_GH = sys.argv[5] == "1"


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def run_cmd(args: Sequence[str], *, cwd: Path = ROOT, check: bool = True) -> CmdResult:
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result = CmdResult(proc.returncode, proc.stdout, proc.stderr)
    if check and proc.returncode != 0:
        joined = " ".join(args)
        raise RuntimeError(f"command failed ({proc.returncode}): {joined}\n{proc.stderr.strip()}")
    return result


def parse_iso_week(week_text: str) -> Tuple[int, int]:
    if not week_text:
        today = dt.datetime.now(dt.timezone.utc).date()
        iso = today.isocalendar()
        return iso.year, iso.week
    match = re.match(r"^(\d{4})-(\d{2})$", week_text)
    if not match:
        raise ValueError("week must match YYYY-WW")
    year = int(match.group(1))
    week = int(match.group(2))
    dt.date.fromisocalendar(year, week, 1)  # validate
    return year, week


def parse_dt(text: str) -> dt.datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def parse_github_repo_slug(remote_url: str) -> Optional[str]:
    candidates = [
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$",
        r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"^ssh://git@github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
    ]
    for pattern in candidates:
        match = re.match(pattern, remote_url.strip())
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    return None


def collect_commit_rows(start: dt.datetime, end: dt.datetime) -> List[Tuple[str, dt.datetime, str]]:
    fmt = "%H%x09%cI%x09%s"
    res = run_cmd(
        [
            "git",
            "-C",
            str(ROOT),
            "log",
            "--no-merges",
            f"--since={start.isoformat()}",
            f"--until={end.isoformat()}",
            f"--pretty=format:{fmt}",
        ],
        check=True,
    )
    rows: List[Tuple[str, dt.datetime, str]] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        commit_sha, commit_iso, subject = parts
        rows.append((commit_sha, parse_dt(commit_iso), subject.strip()))
    rows.sort(key=lambda x: x[1])
    return rows


def commit_files(commit_sha: str) -> Set[str]:
    res = run_cmd(
        ["git", "-C", str(ROOT), "show", "--pretty=format:", "--name-only", commit_sha],
        check=True,
    )
    return {line.strip() for line in res.stdout.splitlines() if line.strip()}


def collect_git_metrics(start: dt.datetime, end: dt.datetime) -> Dict[str, object]:
    commits = collect_commit_rows(start, end)
    commit_count = len(commits)

    cycle_hours_median: Optional[float] = None
    if commit_count >= 2:
        gaps = []
        for idx in range(1, commit_count):
            delta = commits[idx][1] - commits[idx - 1][1]
            gaps.append(delta.total_seconds() / 3600.0)
        cycle_hours_median = float(statistics.median(gaps))

    commit_file_map: Dict[str, Set[str]] = {}
    for sha, _, _ in commits:
        commit_file_map[sha] = commit_files(sha)

    reworked = 0
    rework_window = dt.timedelta(hours=24)
    for idx, (sha_i, ts_i, _) in enumerate(commits):
        files_i = commit_file_map.get(sha_i, set())
        if not files_i:
            continue
        found_followup = False
        for jdx in range(idx + 1, commit_count):
            sha_j, ts_j, _ = commits[jdx]
            if ts_j - ts_i > rework_window:
                break
            files_j = commit_file_map.get(sha_j, set())
            if files_i.intersection(files_j):
                found_followup = True
                break
        if found_followup:
            reworked += 1

    first_pass_pct: Optional[float] = None
    rework_pct: Optional[float] = None
    if commit_count > 0:
        first_pass_pct = 100.0 * (commit_count - reworked) / commit_count
        rework_pct = 100.0 * reworked / commit_count

    escaped_defects = 0
    defect_re = re.compile(r"\b(fix|bug|hotfix|regress(?:ion)?)\b", re.IGNORECASE)
    for _, _, subject in commits:
        if defect_re.search(subject):
            escaped_defects += 1

    return {
        "commit_count": commit_count,
        "cycle_hours_median": cycle_hours_median,
        "first_pass_pct": first_pass_pct,
        "rework_pct": rework_pct,
        "escaped_defects": escaped_defects,
    }


def collect_ci_metrics(start: dt.datetime, end: dt.datetime) -> Dict[str, object]:
    unavailable = {
        "available": False,
        "reason": "",
        "haskell_rate": None,
        "web_rate": None,
        "workflow_success_first_pass_rate": None,
    }

    if not USE_GH:
        unavailable["reason"] = "disabled via --no-gh"
        return unavailable

    gh_path = shutil.which("gh")
    if not gh_path:
        unavailable["reason"] = "gh CLI not found"
        return unavailable

    auth = run_cmd(["gh", "auth", "status", "-h", "github.com"], check=False)
    if auth.returncode != 0:
        unavailable["reason"] = "gh auth invalid or missing"
        return unavailable

    repo_slug = REPO_OVERRIDE
    if not repo_slug:
        remote = run_cmd(["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"], check=False)
        if remote.returncode == 0 and remote.stdout.strip():
            repo_slug = parse_github_repo_slug(remote.stdout.strip()) or ""
    if not repo_slug:
        unavailable["reason"] = "could not infer GitHub repo slug"
        return unavailable

    runs_res = run_cmd(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo_slug,
            "--workflow",
            "CI",
            "--limit",
            "200",
            "--json",
            "databaseId,createdAt,conclusion,attempt",
        ],
        check=False,
    )
    if runs_res.returncode != 0:
        unavailable["reason"] = "failed to fetch workflow runs via gh"
        return unavailable

    try:
        runs = json.loads(runs_res.stdout or "[]")
    except json.JSONDecodeError:
        unavailable["reason"] = "invalid JSON from gh run list"
        return unavailable

    in_week = []
    for run in runs:
        created_at = run.get("createdAt")
        if not created_at:
            continue
        try:
            created_dt = parse_dt(created_at)
        except Exception:
            continue
        if start <= created_dt < end:
            in_week.append(run)

    workflow_total = 0
    workflow_first_pass_success = 0

    haskell_total = 0
    haskell_first_pass_success = 0
    web_total = 0
    web_first_pass_success = 0

    for run in in_week:
        attempt = int(run.get("attempt") or 1)
        conclusion = (run.get("conclusion") or "").lower()
        if conclusion:
            workflow_total += 1
            if conclusion == "success" and attempt == 1:
                workflow_first_pass_success += 1

        run_id = run.get("databaseId")
        if run_id is None:
            continue
        view_res = run_cmd(
            [
                "gh",
                "run",
                "view",
                str(run_id),
                "--repo",
                repo_slug,
                "--json",
                "jobs",
            ],
            check=False,
        )
        if view_res.returncode != 0:
            continue
        try:
            payload = json.loads(view_res.stdout or "{}")
        except json.JSONDecodeError:
            continue
        jobs = payload.get("jobs") or []
        for job in jobs:
            name = (job.get("name") or "").strip().lower()
            job_conclusion = (job.get("conclusion") or "").strip().lower()
            if not job_conclusion:
                continue
            if name.startswith("haskell"):
                haskell_total += 1
                if job_conclusion == "success" and attempt == 1:
                    haskell_first_pass_success += 1
            elif name.startswith("web"):
                web_total += 1
                if job_conclusion == "success" and attempt == 1:
                    web_first_pass_success += 1

    haskell_rate = None
    web_rate = None
    workflow_rate = None
    if haskell_total > 0:
        haskell_rate = 100.0 * haskell_first_pass_success / haskell_total
    if web_total > 0:
        web_rate = 100.0 * web_first_pass_success / web_total
    if workflow_total > 0:
        workflow_rate = 100.0 * workflow_first_pass_success / workflow_total

    return {
        "available": True,
        "reason": "",
        "repo_slug": repo_slug,
        "runs_seen": len(in_week),
        "workflow_success_first_pass_rate": workflow_rate,
        "haskell_rate": haskell_rate,
        "web_rate": web_rate,
        "haskell_total": haskell_total,
        "web_total": web_total,
    }


def build_markdown(
    week_label: str,
    start: dt.datetime,
    end: dt.datetime,
    git_metrics: Dict[str, object],
    ci_metrics: Dict[str, object],
) -> str:
    period_end = (end - dt.timedelta(days=1)).date().isoformat()
    lines: List[str] = []
    lines.append(f"# ROI Scorecard - {week_label}")
    lines.append("")
    lines.append(f"Period (UTC): {start.date().isoformat()} to {period_end}")
    lines.append("")
    lines.append("## Delivery")
    lines.append(f"- Tasks completed: {git_metrics['commit_count']} (non-merge commits)")
    lines.append(
        "- Median cycle time (hours): "
        + f"{fmt_num(git_metrics['cycle_hours_median'])} "
        + "(median gap between consecutive commits)"
    )
    lines.append(
        "- First-pass acceptance (%): "
        + f"{fmt_num(git_metrics['first_pass_pct'])} "
        + "(heuristic: no same-file follow-up commit within 24h)"
    )
    lines.append(
        "- Rework rate (%): "
        + f"{fmt_num(git_metrics['rework_pct'])} "
        + "(heuristic complement of first-pass acceptance)"
    )
    lines.append("")
    lines.append("## Quality")
    lines.append(
        f"- Escaped defects (count): {git_metrics['escaped_defects']} "
        + "(heuristic: commit subjects containing fix/bug/hotfix/regression)"
    )

    if ci_metrics.get("available"):
        haskell_rate = fmt_num(ci_metrics.get("haskell_rate"))
        web_rate = fmt_num(ci_metrics.get("web_rate"))
        workflow_rate = fmt_num(ci_metrics.get("workflow_success_first_pass_rate"))
        lines.append(
            f"- Haskell gate first-pass rate (%): {haskell_rate} "
            + f"(GitHub CI jobs, n={ci_metrics.get('haskell_total', 0)})"
        )
        lines.append(
            f"- Web gate first-pass rate (%): {web_rate} "
            + f"(GitHub CI jobs, n={ci_metrics.get('web_total', 0)})"
        )
        lines.append(
            f"- Workflow first-pass success rate (%): {workflow_rate} "
            + f"(workflow runs, n={ci_metrics.get('runs_seen', 0)})"
        )
    else:
        reason = ci_metrics.get("reason") or "unavailable"
        lines.append(f"- Haskell gate first-pass rate (%): N/A ({reason})")
        lines.append(f"- Web gate first-pass rate (%): N/A ({reason})")
        lines.append(f"- Workflow first-pass success rate (%): N/A ({reason})")

    lines.append("")
    lines.append("## Efficiency")
    lines.append("- Mean agent turns per task: N/A (manual capture)")
    lines.append("- Mean human intervention points per task: N/A (manual capture)")
    lines.append("- Estimated hours saved: N/A (enter baseline/actual manually)")
    lines.append("  - Formula: (baseline_hours_per_task - actual_hours_per_task) * tasks_completed")
    lines.append("")
    lines.append("## Top ROI lanes this week")
    lines.append("- Lane:")
    lines.append("- Why it paid off:")
    lines.append("")
    lines.append("## Low ROI patterns this week")
    lines.append("- Pattern:")
    lines.append("- Change for next week:")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is auto-generated from local git history and optional GitHub Actions data.")
    lines.append("- Heuristic fields should be reviewed before sharing externally.")
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        year, week = parse_iso_week(WEEK_ARG)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    start_date = dt.date.fromisocalendar(year, week, 1)
    start_dt = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)
    end_dt = start_dt + dt.timedelta(days=7)
    week_label = f"{year}-{week:02d}"

    output_path = Path(OUTPUT_ARG) if OUTPUT_ARG else ROOT / "tmp" / f"roi-scorecard-{week_label}.md"
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        git_metrics = collect_git_metrics(start_dt, end_dt)
    except Exception as exc:
        print(f"error: failed to collect git metrics: {exc}", file=sys.stderr)
        return 1

    ci_metrics = collect_ci_metrics(start_dt, end_dt)
    report = build_markdown(week_label, start_dt, end_dt, git_metrics, ci_metrics)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote ROI scorecard: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
