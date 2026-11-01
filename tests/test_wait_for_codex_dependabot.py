"""Guard for wait-for-codex.yml's Dependabot pass-through (PRO-18).

The required check "Codex review clean (gate before expensive downstream jobs)"
exists only when a job writes it through the Checks API. A Dependabot update
must get a completed, successful one from the producer job. pull_request_target
runs main's copy of the workflow, so a PR cannot exercise its own change; this
test is the pre-merge check. It runs create-check's real script with a stub
`gh` on PATH.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "wait-for-codex.yml"
CHECK = "Codex review clean (gate before expensive downstream jobs)"
DEPENDABOT_EXPR = (
    "${{ github.actor == 'dependabot[bot]' && "
    "github.event.pull_request.user.login == 'dependabot[bot]' && "
    "github.event.pull_request.head.repo.full_name == github.repository }}"
)

# Logs each call as one JSON array, then answers the two calls upsert_check_run makes:
# the paginated lookup (no check-run yet) and the POST (`--jq .id`).
STUB_GH = """#!/usr/bin/env bash
"{python}" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@" >> "$GH_LOG"
case "$*" in
  *"--jq"*) echo 4242 ;;
  *"--paginate"*) echo '{{"check_runs": []}}' ;;
esac
"""


def _producer() -> dict:
    # PyYAML reads the bare `on:` key as True; only jobs matter here.
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["codex-review-request"]


def _step(step_id: str) -> dict:
    return next(s for s in _producer()["steps"] if s.get("id") == step_id)


def _run_create_check(tmp_path: Path, is_dependabot: str) -> tuple[list[list[str]], dict]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(STUB_GH.format(python=sys.executable))
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    log, out = tmp_path / "gh.log", tmp_path / "github_output"
    log.touch()
    out.touch()
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "GH_LOG": str(log),
        "GITHUB_OUTPUT": str(out),
        "GH_TOKEN": "app-token",
        "REPO": "HagaleTechnologies/propagation",
        "HEAD_SHA": "76cb40eac1a38773a255ed2904c87e59774b0a51",
        "RUN_ID": "39243326855",
        "ALREADY_CLEAN": "false",
        "BOOTSTRAP_STATUS": "ready",
        "IS_DEPENDABOT": is_dependabot,
    }
    # GitHub Actions runs `run:` blocks with `bash -e {0}`.
    subprocess.run(["bash", "-e", "-c", _step("create-check")["run"]], env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    return calls, outputs


def _posts(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[:3] == ["api", "-X", "POST"]]


def test_producer_runs_for_dependabot():
    cond = _producer()["if"]
    assert "dependabot" not in cond
    assert "github.event_name == 'pull_request_target'" in cond
    assert "github.event.pull_request.draft == false" in cond


def test_app_token_minted_for_every_pull_request_target_run():
    assert _step("app-token")["if"] == "github.event_name == 'pull_request_target'"


def test_create_check_classifies_dependabot_by_actor_author_and_repo():
    assert _step("create-check")["env"]["IS_DEPENDABOT"] == DEPENDABOT_EXPR


def test_dependabot_update_gets_completed_success(tmp_path):
    calls, outputs = _run_create_check(tmp_path, is_dependabot="true")
    [post] = _posts(calls)
    assert f"name={CHECK}" in post
    assert "status=completed" in post and "conclusion=success" in post
    assert "external_id=39243326855" in post
    assert outputs["conclusion"] == "success"
    assert outputs["check_run_id"] == "4242"


def test_other_prs_still_wait_for_codex(tmp_path):
    calls, outputs = _run_create_check(tmp_path, is_dependabot="false")
    [post] = _posts(calls)
    assert "status=in_progress" in post
    assert not any(a.startswith("conclusion=") for a in post)
    assert outputs["conclusion"] == ""


def test_request_a_review_skipped_on_pass_through():
    cond = next(s for s in _producer()["steps"] if s["name"] == "Request a review")["if"]
    assert "steps.create-check.outputs.conclusion != 'success'" in cond
