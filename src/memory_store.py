from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def normalize_text(text: str) -> str:
    """NFC-normalize Vietnamese text so composed / decomposed accents compare equal."""

    return unicodedata.normalize("NFC", text or "")


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token, stable across runs."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


# --------------------------------------------------------------------------- #
# Persistent memory: User.md
# --------------------------------------------------------------------------- #

# Structured fields stored in User.md, in display order.
PROFILE_FIELDS: dict[str, str] = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "interests": "Mối quan tâm kỹ thuật",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}

# List-like preferences are merged (union); every other field is overwritten by the newest value.
MERGE_FIELDS = {"response_style", "interests"}

_FACT_LINE = re.compile(r"^- (\w+): (.*)$")


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user id."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (user_id or "").strip()) or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def default_profile(self, user_id: str) -> str:
        return f"# User Profile: {user_id}\n\n"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return self.default_profile(user_id)
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE.match(line.strip())
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one `- key: value` line. Returns True when the file changed.

        Replacing (instead of appending) is the conflict-handling rule: a correction
        overwrites the stale fact, so User.md never holds two values for one field.
        """

        value = " ".join(normalize_text(value).split())
        if not value:
            return False
        current = self.facts(user_id)
        if key in MERGE_FIELDS and key in current:
            value = merge_list_values(current[key], value)
        if current.get(key) == value:
            return False

        content = self.read_text(user_id)
        line = f"- {key}: {value}"
        if key in current:
            pattern = re.compile(rf"^- {re.escape(key)}: .*$", re.MULTILINE)
            content = pattern.sub(lambda _: line, content, count=1)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += line + "\n"
        self.write_text(user_id, content)
        return True


def merge_list_values(old: str, new: str) -> str:
    items: list[str] = []
    for part in [*old.split(","), *new.split(",")]:
        part = part.strip()
        if part and part.lower() not in {item.lower() for item in items}:
            items.append(part)
    return ", ".join(items)


# --------------------------------------------------------------------------- #
# Fact extraction (structured fields + confidence threshold)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class ProfileFact:
    key: str
    value: str
    confidence: float
    evidence: str


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"[,;:]|\s(?:chứ|nhưng)\s")

# Sentences that ask for / announce a question are never treated as facts.
_QUESTION_MARKERS = ("là gì", "con gì", "nhớ lại xem", "hỏi lại", "hỏi tiếp", "sẽ hỏi")
# Negation / staleness markers that cancel an entity mentioned after them in the same clause.
_NEGATION_MARKERS = (
    "không còn", "không phải", "đừng", "lúc đầu", "trước đó", "trước đây",
    "ví dụ cũ", "đùa", "nghề cũ", "hay là",
)
# Hedges lower confidence; conditionals lower it a bit less.
_HEDGE_MARKERS = ("có lẽ", "hình như", "chắc là", "không chắc", "giả sử", "đùa")
_CONDITIONAL_MARKERS = ("nếu",)

KNOWN_LOCATIONS = (
    "Đà Nẵng", "Huế", "Hà Nội", "Hồ Chí Minh", "Sài Gòn", "Hải Phòng",
    "Cần Thơ", "Nha Trang", "Đà Lạt", "Quy Nhơn", "Vũng Tàu",
)
_LOCATION_PATTERN = re.compile("|".join(re.escape(loc) for loc in KNOWN_LOCATIONS))
_LOCATION_TRIGGER = re.compile(
    r"(?:đang ở|hiện ở|vẫn ở|mình ở|tôi ở|sống ở|làm việc ở|chuyển (?:về|đến|tới|ra|vào)|sang"
    r"|nơi ở hiện tại là|nơi ở là)\s+$",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"\b((?:backend|frontend|fullstack|full-stack|mlops|ml|ai|data|devops|software|platform|qa) engineer"
    r"|product manager|project manager|data scientist|data analyst)\b",
    re.IGNORECASE,
)
_ROLE_TRIGGER = re.compile(r"(?:làm|là|sang|nghề)\s+$", re.IGNORECASE)
_CANONICAL_ROLES = {"mlops engineer": "MLOps engineer", "ml engineer": "ML engineer", "ai engineer": "AI engineer"}

_NAME_PATTERN = re.compile(r"\btên (?:mình |tôi |em )?là ([^\W\d_]\w*(?:\s+[^\W\d_]\w*)*)", re.IGNORECASE)
_DRINK_PATTERNS = (
    (re.compile(r"đồ uống yêu thích(?: của mình)? là ([^,.;!?]+)", re.IGNORECASE), 0.95),
    (re.compile(r"\buống ((?:cà phê|trà|nước)[\w ]*?)(?= như| nhưng| mỗi|[,.;!?]|$)", re.IGNORECASE), 0.75),
)
_FOOD_PATTERN = re.compile(r"món ăn yêu thích(?: của mình)? là ([^,.;!?]+)", re.IGNORECASE)
_PET_PATTERN = re.compile(
    r"\bnuôi (?:một |1 )?(?:bé |con |chú |em )?([^\W\d_]+)(?: tên ([^\W\d_]+))?", re.IGNORECASE
)

_STYLE_CONTEXT = ("trả lời", "giải thích", "style")
_STYLE_TAGS = (
    ("3 bullet", "3 bullet"),
    ("ngắn gọn", "ngắn gọn"),
    ("ngắn", "ngắn gọn"),
    ("gọn", "ngắn gọn"),
    ("bullet", "bullet"),
    ("cấu trúc", "có cấu trúc"),
    ("ví dụ thực chiến", "có ví dụ thực chiến"),
    ("ví dụ thực tế", "có ví dụ thực tế"),
    ("trade-off", "ưu tiên trade-off"),
)

_INTEREST_CONTEXT = ("thích", "quan tâm", "đam mê")
_INTEREST_KEYWORDS = ("Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "LLM")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(normalize_text(text)) if s.strip()]


def is_question_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return sentence.rstrip().endswith("?") or any(marker in lowered for marker in _QUESTION_MARKERS)


def _is_negated(prefix: str, suffix: str = "") -> bool:
    prefix = prefix.lower()
    return any(marker in prefix for marker in _NEGATION_MARKERS) or suffix.lower().lstrip().startswith("chỉ là")


def _confidence(base: float, sentence: str, conditional_penalty: float = 0.3) -> float:
    lowered = sentence.lower()
    score = base
    if any(marker in lowered for marker in _HEDGE_MARKERS):
        score -= 0.4
    if any(re.search(rf"\b{marker}\b", lowered) for marker in _CONDITIONAL_MARKERS):
        score -= conditional_penalty
    return round(max(score, 0.0), 2)


def _affirmed_entities(sentence: str, pattern: re.Pattern, trigger: re.Pattern) -> list[str]:
    """Entities introduced by an affirmative trigger and not negated earlier in their clause."""

    found: list[str] = []
    for clause in _CLAUSE_SPLIT.split(sentence):
        for match in pattern.finditer(clause):
            prefix, suffix = clause[: match.start()], clause[match.end():]
            if trigger.search(prefix) and not _is_negated(prefix, suffix):
                found.append(match.group(0))
    return found


def _extract_name(sentence: str) -> str | None:
    match = _NAME_PATTERN.search(sentence)
    if not match:
        return None
    tokens: list[str] = []
    for token in match.group(1).split():
        if not token[0].isupper():
            break
        tokens.append(token)
    return " ".join(tokens) or None


def _extract_style(sentence: str) -> str | None:
    lowered = sentence.lower()
    if not any(word in lowered for word in _STYLE_CONTEXT):
        return None
    tags: list[str] = []
    for needle, tag in _STYLE_TAGS:
        if needle in lowered and tag not in tags:
            if tag == "bullet" and "3 bullet" in tags:
                continue
            tags.append(tag)
    return ", ".join(tags) or None


def _extract_interests(sentence: str) -> str | None:
    lowered = sentence.lower()
    if not any(word in lowered for word in _INTEREST_CONTEXT):
        return None
    found = [kw for kw in _INTEREST_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", sentence)]
    return ", ".join(found) or None


def extract_profile_facts(message: str) -> list[ProfileFact]:
    """Return every candidate fact in `message` together with a confidence score."""

    facts: list[ProfileFact] = []
    for sentence in split_sentences(message):
        if is_question_sentence(sentence):
            continue

        def add(key: str, value: str | None, base: float, conditional_penalty: float = 0.3) -> None:
            if value:
                value = " ".join(value.strip().removesuffix("nhé").split())
                facts.append(ProfileFact(key, value, _confidence(base, sentence, conditional_penalty), sentence))

        add("name", _extract_name(sentence), 0.95)

        for location in _affirmed_entities(sentence, _LOCATION_PATTERN, _LOCATION_TRIGGER):
            add("location", location, 0.85)

        for role in _affirmed_entities(sentence, _ROLE_PATTERN, _ROLE_TRIGGER):
            add("profession", _CANONICAL_ROLES.get(role.lower(), role.lower()), 0.85)

        for pattern, base in _DRINK_PATTERNS:
            match = pattern.search(sentence)
            if match:
                add("favorite_drink", match.group(1), base)
                break

        food = _FOOD_PATTERN.search(sentence)
        if food:
            add("favorite_food", food.group(1), 0.95)

        pet = _PET_PATTERN.search(sentence)
        if pet:
            species, pet_name = pet.group(1), pet.group(2)
            add("pet", f"{species} tên {pet_name}" if pet_name else species, 0.9)

        # Style instructions are often phrased conditionally ("nếu bạn giải thích, hãy...").
        add("response_style", _extract_style(sentence), 0.9, conditional_penalty=0.0)
        add("interests", _extract_interests(sentence), 0.8)
    return facts


def extract_profile_updates(message: str, min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Convert raw user text into stable profile facts that pass the confidence threshold.

    Within one message the latest single-valued fact wins; list-like fields are merged.
    """

    updates: dict[str, str] = {}
    for fact in extract_profile_facts(message):
        if fact.confidence < min_confidence:
            continue
        if fact.key in MERGE_FIELDS and fact.key in updates:
            updates[fact.key] = merge_list_values(updates[fact.key], fact.value)
        else:
            updates[fact.key] = fact.value
    return updates


# --------------------------------------------------------------------------- #
# Answering from remembered facts (shared by both agents' offline mode)
# --------------------------------------------------------------------------- #

_RECALL_MARKERS = ("nhắc lại", "tóm tắt", "nhớ lại", "là gì", "là ai", "mô tả")
_QUESTION_ROUTES: dict[str, tuple[str, ...]] = {
    "name": ("tên", "là ai", "mô tả", "tóm tắt"),
    "location": ("ở đâu", "nơi ở", "còn ở", "sống ở"),
    "profession": ("nghề", "làm gì", "công việc", "là ai", "mô tả"),
    "response_style": ("style", "kiểu trả lời", "cách trả lời", "phong cách"),
    "interests": ("quan tâm", "là ai", "mô tả", "sở thích"),
    "favorite_drink": ("đồ uống", "uống gì"),
    "favorite_food": ("món ăn", "ăn gì"),
    "pet": ("nuôi", "con gì", "thú cưng"),
}


def is_recall_request(message: str) -> bool:
    lowered = normalize_text(message).lower()
    return "?" in message or any(marker in lowered for marker in _RECALL_MARKERS)


def requested_fields(question: str) -> list[str]:
    lowered = normalize_text(question).lower()
    return [key for key in PROFILE_FIELDS if any(kw in lowered for kw in _QUESTION_ROUTES[key])]


def compose_profile_answer(question: str, facts: dict[str, str]) -> str | None:
    """Short bullet answer built only from `facts`; None when nothing relevant is known."""

    keys = requested_fields(question) or ["name", "profession", "location"]
    known = [key for key in keys if facts.get(key)]
    if not known:
        return None
    lines = [f"- {PROFILE_FIELDS[key]}: {facts[key]}" for key in known]
    lines += [f"- {PROFILE_FIELDS[key]}: mình chưa có thông tin." for key in keys if key not in known]
    return "Theo những gì mình nhớ:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Compact memory
# --------------------------------------------------------------------------- #


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6, max_chars: int = 120) -> str:
    """Heuristic summary: the last `max_items` user messages, each truncated to `max_chars`."""

    if not messages:
        return ""
    user_messages = [m for m in messages if m.get("role") == "user"] or messages
    selected = user_messages[-max_items:]
    lines: list[str] = []
    skipped = len(user_messages) - len(selected)
    if skipped:
        lines.append(f"- ({skipped} tin nhắn cũ hơn đã lược bỏ)")
    for message in selected:
        content = " ".join(message.get("content", "").split())
        if len(content) > max_chars:
            content = content[: max_chars - 1].rstrip() + "…"
        lines.append(f"- {message.get('role', 'user')}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Short-term thread memory that folds old messages into a bounded summary."""

    threshold_tokens: int
    keep_messages: int
    max_summary_lines: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        ctx = self.context(thread_id)
        ctx["messages"].append({"role": role, "content": content})
        if self.token_count(thread_id) > self.threshold_tokens and len(ctx["messages"]) > self.keep_messages:
            self._compact(ctx)

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def token_count(self, thread_id: str) -> int:
        ctx = self.context(thread_id)
        return estimate_tokens(ctx["summary"]) + sum(estimate_tokens(m["content"]) for m in ctx["messages"])

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])

    def _compact(self, ctx: dict[str, object]) -> None:
        messages: list[dict[str, str]] = ctx["messages"]
        keep = max(self.keep_messages, 0)
        older, recent = (messages[:-keep], messages[-keep:]) if keep else (messages, [])
        merged = f"{ctx['summary']}\n{summarize_messages(older)}"
        lines = [line for line in merged.splitlines() if line.strip()]
        ctx["summary"] = "\n".join(lines[-self.max_summary_lines:])
        ctx["messages"] = recent
        ctx["compactions"] = int(ctx["compactions"]) + 1
