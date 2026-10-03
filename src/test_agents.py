from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import heuristic_quality, recall_points
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_facts, extract_profile_updates

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
