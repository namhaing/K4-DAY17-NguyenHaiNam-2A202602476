from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import (
    MAX_LIST_ITEMS,
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_facts,
    extract_profile_updates,
)
from model_provider import message_text, turn_usage

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path):
    """Isolated config: state lives in tmp_path and compaction triggers quickly."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        load_config(REPO_ROOT),
        state_dir=state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=4,
    )


def make_agents(tmp_path: Path) -> tuple[BaselineAgent, AdvancedAgent]:
    config = make_config(tmp_path)
    return BaselineAgent(config, force_offline=True), AdvancedAgent(config, force_offline=True)


def stress_turns() -> list[str]:
    data = json.loads((REPO_ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))
    return data[0]["turns"]


# --------------------------------------------------------------------------- #
# Required tests
# --------------------------------------------------------------------------- #


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.file_size("dungct") == 0
    assert store.read_text("dungct").startswith("# User Profile: dungct")

    path = store.write_text("dungct", "# User Profile: dungct\n\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path == store.path_for("dungct") and path.exists()
    assert "- location: Đà Nẵng" in store.read_text("dungct")
    assert store.file_size("dungct") == len(store.read_text("dungct").encode("utf-8"))

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert store.facts("dungct")["location"] == "Huế"
    assert store.edit_text("dungct", "Hà Nội", "Sài Gòn") is False


def test_compact_trigger(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)
    for turn in stress_turns()[:6]:
        advanced.reply("dungct_stress", "long-thread", turn)

    ctx = advanced.compact_memory.context("long-thread")
    assert advanced.compaction_count("long-thread") > 0
    assert ctx["summary"]
    assert len(ctx["messages"]) <= advanced.config.compact_keep_messages + 1


def test_cross_session_recall(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
    ]
    for turn in turns:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    question = "Mình tên gì, đang ở đâu và làm nghề gì?"
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]

    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer and "Đà Nẵng" not in advanced_answer
    assert "MLOps engineer" in advanced_answer and "backend" not in advanced_answer
    assert "DũngCT" not in baseline_answer
    assert recall_points(baseline_answer, ["DũngCT", "Huế"]) == 0.0

    # A brand-new agent instance still remembers: User.md is persisted on disk.
    fresh = AdvancedAgent(advanced.config, force_offline=True)
    assert "DũngCT" in fresh.reply("dungct", "session-3", "Mình tên gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    baseline, advanced = make_agents(tmp_path)
    for turn in stress_turns():
        baseline.reply("dungct_stress", "stress", turn)
        advanced.reply("dungct_stress", "stress", turn)

    assert advanced.compaction_count("stress") >= 2
    assert advanced.prompt_token_usage("stress") < baseline.prompt_token_usage("stress")


# --------------------------------------------------------------------------- #
# Beyond the happy path
# --------------------------------------------------------------------------- #


def test_baseline_remembers_within_same_thread(tmp_path: Path) -> None:
    baseline, _ = make_agents(tmp_path)
    baseline.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT.")
    assert "DũngCT" in baseline.reply("dungct", "t1", "Mình tên gì?")["response"]
    assert baseline.compaction_count("t1") == 0
    assert baseline.memory_file_size("dungct") == 0


def test_questions_are_not_stored_as_facts() -> None:
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}


def test_noise_does_not_override_profile() -> None:
    assert "location" not in extract_profile_updates(
        "Tương tự, Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại."
    )
    assert "location" not in extract_profile_updates(
        "Nếu sau này mình có nhắc lại Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại nhé."
    )
    joke = extract_profile_updates(
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa."
    )
    assert "profession" not in joke


def test_correction_overwrites_stale_fact(tmp_path: Path) -> None:
    _, advanced = make_agents(tmp_path)
    advanced.reply("dungct", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    advanced.reply("dungct", "t2", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    profile = advanced.profile_store.read_text("dungct")
    assert "- profession: MLOps engineer" in profile
    assert "backend engineer" not in profile
    assert profile.count("- profession:") == 1


def test_confidence_threshold_skips_hedged_facts() -> None:
    hedged = "Có lẽ mình sẽ chuyển ra Hà Nội, nhưng chưa chắc."
    locations = [fact for fact in extract_profile_facts(hedged) if fact.key == "location"]
    assert locations and all(fact.confidence < 0.6 for fact in locations)
    assert "location" not in extract_profile_updates(hedged)


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.upsert_fact("dungct", "favorite_drink", "cà phê sữa đá") is True
    size = store.file_size("dungct")
    assert store.upsert_fact("dungct", "favorite_drink", "cà phê sữa đá") is False
    assert store.file_size("dungct") == size


def test_compact_manager_bounds_summary() -> None:
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2, max_summary_lines=3)
    for index in range(30):
        manager.append("t", "user", f"tin nhắn số {index} " + "nội dung dài " * 10)
    ctx = manager.context("t")
    assert manager.compaction_count("t") > 1
    assert len(ctx["messages"]) <= 3
    assert len(str(ctx["summary"]).splitlines()) <= 3


def test_recall_and_quality_scoring() -> None:
    expected = ["DũngCT", "Huế"]
    assert recall_points("- Tên: DũngCT\n- Nơi ở hiện tại: Huế", expected) == 1.0
    assert recall_points("- Tên: dũngct", expected) == 0.5
    assert recall_points("Mình chưa có thông tin.", expected) == 0.0
    assert heuristic_quality("- Tên: DũngCT\n- Nơi ở: Huế", expected) > heuristic_quality("DũngCT", expected)
    assert heuristic_quality("Mình chưa có thông tin.", expected) == 0.0


def test_correction_value_stops_at_contrast_word() -> None:
    updates = extract_profile_updates("À đính chính: đồ uống yêu thích của mình là matcha latte chứ không phải trà đào.")
    assert updates["favorite_drink"] == "matcha latte"


def test_guardrails_are_needed_on_guardrail_dataset(tmp_path: Path) -> None:
    conversations = load_conversations(REPO_ROOT / "data" / "guardrail_cases.json")

    def recall(**kwargs) -> float:
        # Each variant gets its own state dir so User.md files never leak between runs.
        config = replace(make_config(tmp_path / ("_".join(kwargs) or "full")), compact_threshold_tokens=800)
        agent = AdvancedAgent(config, force_offline=True, **kwargs)
        return run_agent_benchmark("Advanced", agent, conversations, config).recall_score

    assert recall() == 1.0
    assert recall(min_confidence=0.0) < 1.0
    assert recall(use_negation=False) < 1.0


def test_stale_fact_in_answer_scores_zero_recall() -> None:
    assert recall_points("- Nơi ở hiện tại: Huế", ["Huế"], ["Đà Nẵng"]) == 1.0
    assert recall_points("- Nơi ở: Huế (trước đây Đà Nẵng)", ["Huế"], ["Đà Nẵng"]) == 0.0


def test_live_mode_reports_why_it_is_unavailable(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config = replace(config, model=replace(config.model, provider="openai", api_key=None))
    for agent in (BaselineAgent(config), AdvancedAgent(config)):
        assert agent.langchain_agent is None
        assert "API key" in agent.live_error


def test_turn_usage_sums_every_model_call_in_the_turn() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    usage = lambda i, o: {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}  # noqa: E731
    messages = [
        HumanMessage("lượt cũ"),
        AIMessage("cũ", usage_metadata=usage(999, 999)),
        HumanMessage("lượt mới"),
        AIMessage("", usage_metadata=usage(100, 10)),
        ToolMessage("updated", tool_call_id="1"),
        AIMessage([{"type": "text", "text": "Xong"}], usage_metadata=usage(130, 5)),
    ]
    assert turn_usage(messages) == (230, 15)
    assert message_text(messages[-1]) == "Xong"


def test_parallel_tool_writes_do_not_corrupt_user_md(tmp_path: Path) -> None:
    # Live agents run tool calls in parallel; reproduces the UnicodeDecodeError seen in the first live run.
    from concurrent.futures import ThreadPoolExecutor

    store = UserProfileStore(tmp_path / "profiles")
    values = [f"sở thích số {index} – Đà Nẵng" for index in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: store.upsert_fact("dungct", "interests", value), values))
        list(pool.map(lambda index: store.upsert_fact("dungct", "location", ["Huế", "Đà Nẵng"][index % 2]), range(40)))

    facts = store.facts("dungct")  # raises UnicodeDecodeError if a write was torn
    assert facts["location"] in {"Huế", "Đà Nẵng"}
    assert len(facts["interests"].split(", ")) == MAX_LIST_ITEMS


def test_list_fields_decay_by_recency_and_reject_long_values(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("dungct", "interests", "Python, AI")
    for index in range(MAX_LIST_ITEMS - 1):
        store.upsert_fact("dungct", "interests", f"chủ đề {index}")
    store.upsert_fact("dungct", "interests", "Python")  # re-mentioned -> refreshed
    store.upsert_fact("dungct", "interests", "chủ đề mới")

    interests = store.facts("dungct")["interests"].split(", ")
    assert len(interests) == MAX_LIST_ITEMS
    assert "Python" in interests and "AI" not in interests  # AI was the least recently mentioned
    assert store.upsert_fact("dungct", "pet", "corgi tên Bơ, " + "rất hay phá khi mình họp online " * 3) is False
