from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    PROFILE_FIELDS,
    CompactMemoryManager,
    UserProfileStore,
    compose_profile_answer,
    estimate_tokens,
    extract_profile_updates,
    is_recall_request,
)
from model_provider import build_chat_model

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn trong file User.md.\n"
    "- Dùng User.md để cá nhân hoá câu trả lời và trả lời câu hỏi về người dùng.\n"
    "- Khi người dùng cung cấp fact ổn định mới hoặc đính chính, gọi tool `save_user_fact` để cập nhật.\n"
    "- Không lưu câu hỏi, câu đùa, hay địa điểm chỉ ghé qua.\n"
    "- Nếu có mâu thuẫn, luôn ưu tiên fact mới nhất trong User.md."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent `User.md` + compact memory.

    1. within-session memory  -> `CompactMemoryManager` keeps the recent messages
    2. persistent `User.md`   -> `UserProfileStore`, survives new threads
    3. compact memory         -> older messages folded into a bounded summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # Live mode: the tools and dynamic prompt read the user of the turn being processed.
        self.active_context: AgentContext | None = None
        self.langchain_agent = None
        if not force_offline and self.config.model.is_live_ready():
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _persist_profile_updates(self, user_id: str, message: str) -> dict[str, str]:
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        return updates

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens

        response = self._offline_response(user_id, thread_id, message, updates)

        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = (
            self.token_usage(thread_id) + estimate_tokens(message) + estimate_tokens(response)
        )
        return {
            "response": response,
            "thread_id": thread_id,
            "agent_tokens": self.token_usage(thread_id),
            "prompt_tokens": prompt_tokens,
            "profile_updates": updates,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: `User.md` + compact summary + recent kept messages."""

        ctx = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(ctx["summary"]))
            + sum(estimate_tokens(m["content"]) for m in ctx["messages"])
        )

    def _offline_response(
        self, user_id: str, thread_id: str, message: str, updates: dict[str, str] | None = None
    ) -> str:
        """Deterministic answer built from persisted memory (never echoes the question)."""

        if is_recall_request(message):
            answer = compose_profile_answer(message, self.profile_store.facts(user_id))
            if answer:
                return answer
            return "Mình chưa có thông tin này trong bộ nhớ."
        if updates:
            saved = ", ".join(f"{PROFILE_FIELDS[key]} = {value}" for key, value in updates.items())
            return f"Đã ghi nhớ vào User.md: {saved}."
        return "Mình đã ghi nhận."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # The deterministic extractor still runs so User.md stays correct even if the model forgets a tool call.
        updates = self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        self.active_context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        last = result["messages"][-1]
        response = last.content if isinstance(last.content, str) else str(last.content)
        usage = getattr(last, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or self._estimate_prompt_context_tokens(user_id, thread_id)
        output_tokens = usage.get("output_tokens") or estimate_tokens(response)

        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + estimate_tokens(message) + output_tokens
        return {
            "response": response,
            "thread_id": thread_id,
            "agent_tokens": self.token_usage(thread_id),
            "prompt_tokens": prompt_tokens,
            "profile_updates": updates,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + dynamic prompt + summarization."""

        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
        from langchain.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver

        store = self.profile_store
        model = build_chat_model(self.config.model)

        def current_user() -> str:
            return self.active_context.user_id if self.active_context else "anonymous"

        @tool
        def read_user_profile() -> str:
            """Đọc toàn bộ User.md của người dùng hiện tại."""

            return store.read_text(current_user())

        @tool
        def save_user_fact(key: str, value: str) -> str:
            """Lưu hoặc cập nhật một fact ổn định vào User.md.

            key là một trong: name, location, profession, response_style, interests,
            favorite_drink, favorite_food, pet.
            """

            if key not in PROFILE_FIELDS:
                return f"Key không hợp lệ: {key}"
            changed = store.upsert_fact(current_user(), key, value)
            return "updated" if changed else "unchanged"

        @tool
        def edit_user_profile(search_text: str, replacement: str) -> str:
            """Thay một đoạn text trong User.md (dùng khi cần sửa fact sai)."""

            return "edited" if store.edit_text(current_user(), search_text, replacement) else "not found"

        @dynamic_prompt
        def profile_prompt(request) -> str:
            profile = store.read_text(current_user())
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n<user_md>\n{profile}\n</user_md>"

        summarization = SummarizationMiddleware(
            model=model,
            trigger=("tokens", self.config.compact_threshold_tokens),
            keep=("messages", self.config.compact_keep_messages),
        )

        return create_agent(
            model=model,
            tools=[read_user_profile, save_user_fact, edit_user_profile],
            middleware=[profile_prompt, summarization],
            checkpointer=InMemorySaver(),
        )
