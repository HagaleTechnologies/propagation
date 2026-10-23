"""Guard for .serena/project.yml (PRO-13).

Serena rewrites project.yml in place, dropping its comments, when a field is
missing or carries a pre-2.0 name. Pinning the exact key set makes that drift
fail here instead of silently on someone's machine.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SERENA = ROOT / ".serena"

# dataclasses.fields(ProjectConfig) on Serena 2.0.0.dev0
# (oraios/serena@1de556f71569f3acfc0743e526dd60aca40a545e), public fields only.
EXPECTED_KEYS = {
    "project_name", "language_servers", "encoding", "activation_command",
    "activation_command_timeout", "line_ending", "language_backend", "agent_interface",
    "ignore_all_files_in_gitignore", "ls_specific_settings", "ls_workspace_folders",
    "ls_additional_workspace_folders", "ignored_paths", "read_only", "excluded_tools",
    "included_optional_tools", "fixed_tools", "excluded_apis", "included_apis",
    "default_modes", "added_modes", "initial_prompt", "symbol_info_budget",
    "read_only_memory_patterns", "ignored_memory_patterns",
}


def _config() -> dict:
    return yaml.safe_load((SERENA / "project.yml").read_text())


def test_key_set_matches_pinned_serena():
    assert set(_config()) == EXPECTED_KEYS


def test_values():
    cfg = _config()
    assert cfg["project_name"] == "propagation"
    assert cfg["language_servers"] == ["python"]
    assert cfg["read_only"] is True
    assert cfg["ignore_all_files_in_gitignore"] is True
    assert not cfg["activation_command"]  # Serena runs this as a shell command
    assert {".serena/cache", "baselines/p533/upstream", "thoughts"} <= set(cfg["ignored_paths"])


def test_initial_prompt_names_claude_md_canon():
    claude = (ROOT / "CLAUDE.md").read_text()
    section = claude.split("## Documents (read in this order)", 1)[1].split("\n## ", 1)[0]
    docs = set(re.findall(r"`([\w./-]+\.(?:md|json))`", section))
    assert len(docs) >= 5
    prompt = _config()["initial_prompt"]
    assert sorted(d for d in docs if d not in prompt) == []


def test_cache_ignored_and_memory_present():
    ignored = (SERENA / ".gitignore").read_text().split()
    assert "/cache" in ignored and "/project.local.yml" in ignored
    assert (SERENA / "memories" / "codebase_map.md").read_text().strip()
