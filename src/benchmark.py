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
from model_provider import build_chat_model, message_text

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]

SUITES = (
    ("Standard Benchmark", "conversations.json"),
    ("Long-Context Stress Benchmark", "advanced_long_context.json"),
    ("Guardrail Benchmark (bonus: correction / negation / confidence)", "guardrail_cases.json"),
)

# Phrases that mean "I don't know" — they earn no quality credit.
_UNKNOWN_MARKERS = ("chưa có thông tin", "không biết", "không nhớ")

JUDGE_PROMPT = """Bạn là giám khảo chấm câu trả lời của một AI agent có bộ nhớ về người dùng.

Câu hỏi của người dùng: {question}
Các fact bắt buộc phải có: {expected}
Các fact cũ/sai KHÔNG được xuất hiện như thông tin hiện tại: {forbidden}

Câu trả lời của agent:
<answer>
{answer}
</answer>

Chấm trên thang 0-10:
- 6 điểm: nêu đúng và đủ các fact bắt buộc
- 2 điểm: không khẳng định fact cũ/sai
- 2 điểm: ngắn gọn, rõ ràng, đúng trọng tâm câu hỏi
Chỉ trả về MỘT số nguyên từ 0 đến 10, không giải thích."""


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    stale_answers: int = 0  # recall answers that still state an `expected_not_contains` fact


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hits(answer: str, expected: list[str]) -> int:
    normalized = normalize_text(answer).lower()
    return sum(1 for item in expected if normalize_text(item).lower() in normalized)


def recall_points(answer: str, expected: list[str], forbidden: list[str] = ()) -> float:
    """1 when every expected fact appears, 0.5 when some do, 0 when none do.

    An answer that still states a stale / wrong fact from `forbidden` scores 0.
    """

    if forbidden and _hits(answer, forbidden):
        return 0.0
    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str], forbidden: list[str] = ()) -> float:
    """Offline quality score in [0, 1].

    quality = coverage * (0.7 + 0.15 * concise + 0.15 * structured)
    - coverage: share of expected facts present, halved if the answer admits missing info
      and halved again if it states a stale / wrong fact
    - concise: 1 up to 60 words, then decays linearly to 0 at 160 words
    - structured: 1 for bullet lines, 0.5 otherwise
    """

    if not expected:
        return 0.0
    coverage = _hits(answer, expected) / len(expected)
    if any(marker in answer.lower() for marker in _UNKNOWN_MARKERS):
        coverage *= 0.5
    if forbidden and _hits(answer, forbidden):
        coverage *= 0.5
    words = len(answer.split())
    concise = 1.0 if words <= 60 else max(0.0, 1 - (words - 60) / 100)
    structured = 1.0 if re.search(r"^\s*[-•*]\s", answer, re.MULTILINE) else 0.5
    return round(coverage * (0.7 + 0.15 * concise + 0.15 * structured), 4)


class LLMJudge:
    """LLM-as-judge for `Response quality` in live mode (uses `config.judge_model`).

    Falls back to `heuristic_quality` when the judge call fails or returns no number.
    """

    def __init__(self, judge_config) -> None:
        self.name = f"{judge_config.provider}/{judge_config.model_name}"
        self.model = build_chat_model(judge_config)
        self.fallbacks = 0

    def __call__(self, question: str, answer: str, expected: list[str], forbidden: list[str]) -> float:
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected=", ".join(expected),
            forbidden=", ".join(forbidden) or "(không có)",
            answer=answer,
        )
        try:
            match = re.search(r"\d+", message_text(self.model.invoke(prompt)))
        except Exception:
            match = None
        if not match:
            self.fallbacks += 1
            return heuristic_quality(answer, expected, forbidden)
        return min(int(match.group()), 10) / 10


def reset_user_memory(config, conversations: list[dict[str, Any]]) -> None:
    """Delete persisted `User.md` files of the benchmark users so every run starts clean."""

    store = UserProfileStore(config.state_dir / "profiles")
    for user_id in {conv["user_id"] for conv in conversations}:
        profile_dir = store.path_for(user_id).parent
        if profile_dir.exists():
            shutil.rmtree(profile_dir)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
    verbose: bool = False,
    judge: LLMJudge | None = None,
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
    stale_answers = 0

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
            forbidden = question.get("expected_not_contains", [])
            recall_scores.append(recall_points(answer, expected, forbidden))
            stale_answers += bool(forbidden and _hits(answer, forbidden))
            if judge is not None:
                quality_scores.append(judge(question["question"], answer, expected, forbidden))
            else:
                quality_scores.append(heuristic_quality(answer, expected, forbidden))
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
        stale_answers=stale_answers,
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


def _require_live(agent_name: str, agent) -> None:
    if agent.langchain_agent is None:
        raise SystemExit(f"--live was requested but {agent_name} could not start live mode: {agent.live_error}")


def run_suite(
    title: str, dataset: Path, config, live: bool, verbose: bool, judge: LLMJudge | None = None
) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    reset_user_memory(config, conversations)
    baseline = BaselineAgent(config, force_offline=not live)
    advanced = AdvancedAgent(config, force_offline=not live)
    if live:
        _require_live("Baseline", baseline)
        _require_live("Advanced", advanced)
    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config, verbose, judge),
        run_agent_benchmark("Advanced", advanced, conversations, config, verbose, judge),
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
    judge = LLMJudge(config.judge_model) if args.live else None
    quality = f"LLM judge ({judge.name})" if judge else "heuristic"
    print(f"# Day 17 Memory Benchmark — mode: {mode}")
    print(f"compact_threshold_tokens={config.compact_threshold_tokens}, "
          f"compact_keep_messages={config.compact_keep_messages}, response quality: {quality}\n")

    for title, filename in SUITES:
        dataset = config.data_dir / filename
        if dataset.exists():
            run_suite(title, dataset, config, args.live, args.verbose, judge)
    if judge and judge.fallbacks:
        print(f"(judge fell back to the heuristic score {judge.fallbacks} time(s))")


if __name__ == "__main__":
    main()
