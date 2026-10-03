from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import compose_profile_answer, estimate_tokens, extract_profile_updates, is_recall_request
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Bạn chỉ nhớ nội dung trong cuộc hội thoại hiện tại. "
    "Nếu không có thông tin trong cuộc hội thoại này, hãy nói rõ là bạn không biết."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full message list per `thread_id` and re-reads all of it every turn.
    - No persistent `User.md`, no compaction.
    - A new thread starts from zero, so long-term facts are forgotten.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if not force_offline and self.config.model.is_live_ready():
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        # The baseline carries the whole thread history into every turn.
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        response = self._offline_response(session, message)

        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)
        return {
            "response": response,
            "thread_id": thread_id,
            "agent_tokens": session.token_usage,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _offline_response(self, session: SessionState, message: str) -> str:
        if not is_recall_request(message):
            return "Mình đã ghi nhận."
        # Only facts mentioned earlier in *this* thread are available.
        facts: dict[str, str] = {}
        for item in session.messages[:-1]:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        answer = compose_profile_answer(message, facts)
        return answer or "Mình chưa có thông tin này trong phiên hiện tại."

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        last = result["messages"][-1]
        response = last.content if isinstance(last.content, str) else str(last.content)
        usage = getattr(last, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or sum(estimate_tokens(m["content"]) for m in session.messages)
        output_tokens = usage.get("output_tokens") or estimate_tokens(response)

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + output_tokens
        return {
            "response": response,
            "thread_id": thread_id,
            "agent_tokens": session.token_usage,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Live baseline: `create_agent` + `InMemorySaver`, no tools, no long-term memory."""

        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
