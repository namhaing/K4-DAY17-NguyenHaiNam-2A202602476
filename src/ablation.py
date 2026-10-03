"""Reproducible ablations for REPORT.md (offline, deterministic).

1. Compact memory: threshold / keep_messages sweep on the long-context stress dataset.
2. Guardrails: turn off the confidence threshold and/or clause-level negation and
   measure recall on every dataset.

Run from the repo root: `python src/ablation.py`
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, run_agent_benchmark
from config import load_config
from memory_store import DEFAULT_CONFIDENCE_THRESHOLD

NO_COMPACTION = 10**9


def _isolated(config, **overrides):
    """Config copy whose state lives in a fresh temp dir, so runs never share User.md."""

    return replace(config, state_dir=Path(tempfile.mkdtemp(prefix="day17-ablation-")), **overrides)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        return "\n".join(lines)


def compaction_sweep(config) -> str:
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    settings = [
        ("no compaction", NO_COMPACTION, config.compact_keep_messages),
        ("threshold 2000", 2000, config.compact_keep_messages),
        ("threshold 1200", 1200, config.compact_keep_messages),
        ("threshold 800 (default)", 800, config.compact_keep_messages),
        ("threshold 500", 500, config.compact_keep_messages),
        ("threshold 800, keep 8", 800, 8),
    ]
    rows = []
    for label, threshold, keep in settings:
        cfg = _isolated(config, compact_threshold_tokens=threshold, compact_keep_messages=keep)
        row = run_agent_benchmark("Advanced", AdvancedAgent(cfg, force_offline=True), stress, cfg)
        rows.append([f"Advanced – {label}", row.prompt_tokens_processed, row.compactions, f"{row.recall_score:.3f}"])
    cfg = _isolated(config)
    base = run_agent_benchmark("Baseline", BaselineAgent(cfg, force_offline=True), stress, cfg)
    rows.append(["Baseline", base.prompt_tokens_processed, base.compactions, f"{base.recall_score:.3f}"])
    return _table(["Configuration", "Prompt tokens processed", "Compactions", "Cross-session recall"], rows)


def guardrail_ablation(config) -> str:
    variants = [
        ("full system", DEFAULT_CONFIDENCE_THRESHOLD, True),
        ("no confidence threshold", 0.0, True),
        ("no negation handling", DEFAULT_CONFIDENCE_THRESHOLD, False),
        ("neither", 0.0, False),
    ]
    datasets = [
        ("Standard", "conversations.json"),
        ("Stress", "advanced_long_context.json"),
        ("Guardrail", "guardrail_cases.json"),
    ]
    rows = []
    for label, min_confidence, use_negation in variants:
        cells: list[object] = [label]
        stale_total = 0
        for _, filename in datasets:
            conversations = load_conversations(config.data_dir / filename)
            cfg = _isolated(config)
            agent = AdvancedAgent(cfg, force_offline=True, min_confidence=min_confidence, use_negation=use_negation)
            row = run_agent_benchmark("Advanced", agent, conversations, cfg)
            stale_total += row.stale_answers
            cells.append(f"{row.recall_score:.3f}")
        cells.append(stale_total)
        rows.append(cells)
    headers = ["Advanced variant", *[f"Recall – {name}" for name, _ in datasets], "Answers with stale fact"]
    return _table(headers, rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config(Path(__file__).resolve().parent.parent)

    print("# Day 17 Ablations (offline, deterministic)\n")
    print("## 1. Compact memory sweep – Long-Context Stress dataset\n")
    print(compaction_sweep(config))
    print("\n## 2. Guardrail ablation – Advanced agent\n")
    print(guardrail_ablation(config))


if __name__ == "__main__":
    main()
