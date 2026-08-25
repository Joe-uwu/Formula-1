"""Maintains REPORT.md as an append-only experiment log. The summary table at
the top is regenerated from the log every run (it's an index, not a past
entry); everything under '## Log' is appended to and never edited or removed.
"""
import re
from datetime import date
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parents[1] / "REPORT.md"
SUMMARY_START = "<!-- SUMMARY_START -->"
SUMMARY_END = "<!-- SUMMARY_END -->"


def _read() -> str:
    if REPORT_PATH.exists():
        return REPORT_PATH.read_text(encoding="utf-8")
    return (
        "# F1 Predictor — Experiment Log\n\n"
        f"{SUMMARY_START}\n(no entries yet)\n{SUMMARY_END}\n\n"
        "## Log\n"
    )


def _entries_from_log(text: str) -> list[dict]:
    entries = []
    for block in re.findall(r"### (.+?)\n(.*?)(?=\n### |\Z)", text.split("## Log", 1)[-1], re.S):
        title, body = block
        headline = re.search(r"\*\*Headline:\*\* (.+)", body)
        verdict = re.search(r"\*\*Verdict:\*\* (.+)", body)
        entries.append(dict(
            title=title.strip(),
            headline=headline.group(1).strip() if headline else "",
            verdict=verdict.group(1).strip() if verdict else "",
        ))
    return entries


def _render_summary(entries: list[dict]) -> str:
    if not entries:
        return "(no entries yet)"
    lines = ["| Entry | Headline metric | Verdict |", "|---|---|---|"]
    for e in entries:
        lines.append(f"| {e['title']} | {e['headline']} | {e['verdict']} |")
    return "\n".join(lines)


def next_entry_number() -> int:
    return len(_entries_from_log(_read())) + 1


def append_entry(
    title: str,
    commit_sha: str,
    what_changed: str,
    hypothesis: str,
    config_diff: str,
    metrics_table_md: str,
    headline: str,
    verdict: str,
    next_steps: str,
) -> None:
    text = _read()
    entry = (
        f"\n### {title}\n\n"
        f"**Date:** {date.today().isoformat()}  **Commit:** {commit_sha}\n\n"
        f"**What changed / hypothesis:** {what_changed} {hypothesis}\n\n"
        f"**Config diff from previous run:**\n```\n{config_diff}\n```\n\n"
        f"**Metrics:**\n\n{metrics_table_md}\n\n"
        f"**Headline:** {headline}\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"**Next:** {next_steps}\n"
    )

    if "## Log" not in text:
        text += "\n## Log\n"
    text = text.rstrip() + "\n" + entry

    entries = _entries_from_log(text)
    new_summary = f"{SUMMARY_START}\n{_render_summary(entries)}\n{SUMMARY_END}"
    text = re.sub(f"{SUMMARY_START}.*?{SUMMARY_END}", new_summary, text, flags=re.S)

    REPORT_PATH.write_text(text, encoding="utf-8")
