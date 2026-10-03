from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, normalize_text

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]

# Phrases that mean "I don't know" — they earn no quality credit.
_UNKNOWN_MARKERS = ("chưa có thông tin", "không biết", "không nhớ")


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hits(answer: str, expected: list[str]) -> int:
    normalized = normalize_text(answer).lower()
    return sum(1 for item in expected if normalize_text(item).lower() in normalized)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 when every expected fact appears, 0.5 when some do, 0 when none do."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1].

    quality = coverage * (0.7 + 0.15 * concise + 0.15 * structured)
    - coverage: share of expected facts present, halved if the answer admits missing info
    - concise: 1 up to 60 words, then decays linearly to 0 at 160 words
    - structured: 1 for bullet lines, 0.5 otherwise
    """

    if not expected:
        return 0.0
    coverage = _hits(answer, expected) / len(expected)
    if any(marker in answer.lower() for marker in _UNKNOWN_MARKERS):
        coverage *= 0.5
    words = len(answer.split())
    concise = 1.0 if words <= 60 else max(0.0, 1 - (words - 60) / 100)
    structured = 1.0 if re.search(r"^\s*[-•*]\s", answer, re.MULTILINE) else 0.5
    return round(coverage * (0.7 + 0.15 * concise + 0.15 * structured), 4)


def reset_user_memory(config, conversations: list[dict[str, Any]]) -> None:
    """Delete persisted `User.md` files of the benchmark users so every run starts clean."""

    store = UserProfileStore(config.state_dir / "profiles")
    for user_id in {conv["user_id"] for conv in conversations}:
        profile_dir = store.path_for(user_id).parent
        if profile_dir.exists():
            shutil.rmtree(profile_dir)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, verbose: bool = False
) -> BenchmarkRow:
    """Feed every conversation, then ask its recall questions in a fresh thread.

    Token columns count the conversation threads only; recall threads are measured
    for recall / quality but not added to the token totals (same rule for both agents).
    """

    user_ids = {conv["user_id"] for conv in conversations}
    size_before = sum(agent.memory_file_size(uid) for uid in user_ids)
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        thread_id = conv["id"]
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for index, question in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            answer = agent.reply(conv["user_id"], recall_thread, question["question"])["response"]
            expected = question["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            if verbose:
                print(f"[{agent_name}] {recall_thread}: {question['question']}\n{answer}\n")

    size_after = sum(agent.memory_file_size(uid) for uid in user_ids)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=size_after - size_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(title: str, dataset: Path, config, live: bool, verbose: bool) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    reset_user_memory(config, conversations)
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=not live), conversations, config, verbose),
        run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=not live), conversations, config, verbose),
    ]
    print(f"## {title}\n")
    print(f"Dataset: `{dataset.relative_to(config.base_dir).as_posix()}` "
          f"({len(conversations)} conversation(s), "
          f"{sum(len(c['turns']) for c in conversations)} turns, "
          f"{sum(len(c.get('recall_questions', [])) for c in conversations)} recall questions)\n")
    print(format_rows(rows))
    print()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced")
    parser.add_argument("--live", action="store_true", help="use the configured LLM provider instead of offline mode")
    parser.add_argument("--verbose", action="store_true", help="print every recall question and answer")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    mode = f"live ({config.model.provider}/{config.model.model_name})" if args.live else "offline (deterministic)"
    print(f"# Day 17 Memory Benchmark — mode: {mode}")
    print(f"compact_threshold_tokens={config.compact_threshold_tokens}, "
          f"compact_keep_messages={config.compact_keep_messages}\n")

    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, args.live, args.verbose)
    run_suite(
        "Long-Context Stress Benchmark",
        config.data_dir / "advanced_long_context.json",
        config,
        args.live,
        args.verbose,
    )


if __name__ == "__main__":
    main()
