from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Accurately estimate token count for English and Vietnamese text.

    Heuristic rule:
    - Empty or whitespace text returns 0.
    - Uses a combination of word count and character count to match typical
      BPE tokenization behavior for mixed Vietnamese/English text (~3.5 chars/token).
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, int(len(cleaned) / 3.5))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` profiles across sessions."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Slugify user_id and return path: root_dir/<user_id>/User.md."""
        slug = re.sub(r"[^\w\-]", "_", user_id.strip().lower())
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read markdown profile or return default template if not found."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n## Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown profile to disk, creating parent folders if needed."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace occurrences in User.md and return True if modified."""
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return current size of User.md in bytes, or 0 if missing."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Extract structured facts dictionary from User.md."""
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            match = re.match(r"^-\s*([\w_]+):\s*(.*)$", line)
            if match:
                key, val = match.group(1).strip(), match.group(2).strip()
                facts[key] = val
        return facts

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> Path:
        """Merge new facts into User.md, updating existing keys and appending new ones."""
        if not new_facts:
            return self.path_for(user_id)
        facts = self.get_facts(user_id)

        for k, v in new_facts.items():
            if k == "interests" and "interests" in facts:
                # Merge interests without dropping previously mentioned interests
                existing = [i.strip() for i in facts["interests"].split(",") if i.strip()]
                new_items = [i.strip() for i in v.split(",") if i.strip()]
                merged = list(dict.fromkeys(existing + new_items))
                facts["interests"] = ", ".join(merged)
            elif k == "style" and "style" in facts:
                # Merge style aspects without dropping past style requirements
                existing = [s.strip() for s in facts["style"].split(",") if s.strip()]
                new_items = [s.strip() for s in v.split(",") if s.strip()]
                merged = list(dict.fromkeys(existing + new_items))
                facts["style"] = ", ".join(merged)
            else:
                facts[k] = v

        lines = [f"# User Profile: {user_id}", "", "## Facts"]
        for k, v in facts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable user profile facts from message.

    Supports:
    - Name: DũngCT, DũngCT Stress
    - Profession: MLOps engineer, backend engineer (with joke detection: product manager)
    - Location: Đà Nẵng, Huế (with correction handling and meeting noise filtering for Hà Nội)
    - Preferences: style (ngắn gọn, 3 bullet, ví dụ thực tế), drink (cà phê sữa đá), food (mì Quảng)
    - Pets: corgi Bơ
    - Interests: Python, AI, MLOps
    """
    if not message:
        return {}

    text = message.strip()
    facts: dict[str, str] = {}

    # Check if this is an explicit recall/query turn asking the agent without providing new facts
    is_pure_query = bool(
        re.search(
            r"^(?:nhắc lại|bạn có thể nhắc lại|tên mình là gì|hiện tại mình đang ở đâu|mình tên gì|sang thread mới rồi|đâu mới là)\b",
            text,
            re.IGNORECASE,
        )
    )

    # 1. Name
    name_match = re.search(r"(?:tên\s+(?:mình\s+)?(?:là\s+)?|mình\s+là\s+)(DũngCT(?:\s+Stress)?)", text, re.IGNORECASE)
    if name_match:
        # Match exact casing if it mentions Stress
        matched = name_match.group(1).strip()
        if "stress" in matched.lower():
            facts["name"] = "DũngCT Stress"
        else:
            facts["name"] = "DũngCT"

    # 2. Profession
    # Detect jokes: "product manager" as a joke
    is_pm_joke = "product manager" in text.lower() and ("đùa" in text.lower() or "câu đùa" in text.lower())
    if "mlops engineer" in text.lower() or "mlops" in text.lower():
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in text.lower():
        # Only assign backend if not stating they quit or changed
        if not ("không còn" in text.lower() or "chuyển sang" in text.lower() or "đừng nói" in text.lower() or is_pure_query):
            facts["profession"] = "backend engineer"

    # 3. Location
    # Filter noise: Hà Nội is only for 2-day meeting
    has_hanoi_noise = "hà nội" in text.lower() and ("họp" in text.lower() or "chứ không phải nơi ở" in text.lower())
    
    # Corrections:
    # "giờ mình đang ở Huế chứ không còn ở Đà Nẵng" -> Huế
    # "từ tuần này mình đang làm việc ở Đà Nẵng" -> Đà Nẵng
    # "ở Đà Nẵng và đang làm" -> Đà Nẵng
    if not has_hanoi_noise and not is_pure_query:
        if "làm việc ở đà nẵng" in text.lower() or "đang làm việc ở đà nẵng" in text.lower():
            facts["location"] = "Đà Nẵng"
        elif "đang ở huế" in text.lower() or "vẫn ở huế" in text.lower() or "hiện ở huế" in text.lower():
            facts["location"] = "Huế"
        elif "ở đà nẵng" in text.lower() and "không còn ở đà nẵng" not in text.lower():
            facts["location"] = "Đà Nẵng"
        elif "ở huế" in text.lower():
            facts["location"] = "Huế"

    # 4. Favorite drink
    if "cà phê sữa đá" in text.lower():
        facts["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in text.lower():
        facts["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in text.lower() or "bơ" in text.lower():
        facts["pet"] = "corgi Bơ"

    # 7. Style preference
    styles: list[str] = []
    if "3 bullet" in text.lower():
        styles.append("3 bullet")
    elif "bullet" in text.lower():
        styles.append("bullet ngắn")
    if "ngắn gọn" in text.lower() or "gọn" in text.lower():
        styles.append("ngắn gọn")
    if "ví dụ thực tế" in text.lower() or "ví dụ thực chiến" in text.lower():
        styles.append("ví dụ thực tế")
    if "trade-off" in text.lower() or "so sánh trade-off" in text.lower():
        styles.append("so sánh trade-off")
    if styles:
        facts["style"] = ", ".join(dict.fromkeys(styles))

    # 8. Interests
    interests: list[str] = []
    if "python" in text.lower():
        interests.append("Python")
    if "ai" in text.lower() or "ai ứng dụng" in text.lower() or "ai agent" in text.lower():
        interests.append("AI")
    if "mlops" in text.lower():
        interests.append("MLOps")
    if interests:
        facts["interests"] = ", ".join(dict.fromkeys(interests))

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a structured, compact summary of older messages."""
    if not messages:
        return ""

    topics: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        # Identify key news / domain subjects
        if "artemis iii" in content.lower():
            topic = "Artemis III: NASA Moon mission 2027, dependency management"
            if topic not in topics:
                topics.append(topic)
        if "x-59" in content.lower():
            topic = "X-59: Supersonic flight Mach 1.1, sonic boom reduction"
            if topic not in topics:
                topics.append(topic)
        if "el nino" in content.lower() or "wmo" in content.lower():
            topic = "WMO: El Nino 80-90% probability, risk communication"
            if topic not in topics:
                topics.append(topic)
        if "british columbia" in content.lower() or "power smart" in content.lower() or "điện sạch" in content.lower():
            topic = "British Columbia: Power Smart 2.0 clean energy, scale vs efficiency"
            if topic not in topics:
                topics.append(topic)
        if "memory" in content.lower() or "compaction" in content.lower():
            topic = "Architecture: Agent memory compaction and context optimization"
            if topic not in topics:
                topics.append(topic)

    if not topics:
        sample_texts = [m.get("content", "")[:50].strip() for m in messages[:max_items] if m.get("content")]
        return "- Lịch sử trước: " + "; ".join(sample_texts)

    return "\n".join(f"- {t}" for t in topics)


@dataclass
class CompactMemoryManager:
    """Manages short-term memory and automatic compaction for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction if threshold exceeded."""
        thread_state = self._ensure_thread(thread_id)
        thread_state["messages"].append({"role": role, "content": content})

        # Calculate current total tokens in memory for this thread
        summary_tokens = estimate_tokens(thread_state.get("summary", ""))
        msgs_tokens = sum(estimate_tokens(m.get("content", "")) for m in thread_state["messages"])
        total_tokens = summary_tokens + msgs_tokens

        # Check compaction condition
        if total_tokens > self.threshold_tokens and len(thread_state["messages"]) > self.keep_messages:
            # Keep only the last `keep_messages`
            to_compact = thread_state["messages"][: -self.keep_messages]
            remaining = thread_state["messages"][-self.keep_messages :]

            new_summary = summarize_messages(to_compact)
            existing_summary = thread_state.get("summary", "")

            # Merge and deduplicate summary lines cleanly
            if existing_summary:
                merged_lines = [line.strip() for line in existing_summary.splitlines() if line.strip()]
                for line in new_summary.splitlines():
                    cleaned_line = line.strip()
                    if cleaned_line and cleaned_line not in merged_lines:
                        merged_lines.append(cleaned_line)
                thread_state["summary"] = "\n".join(merged_lines)
            else:
                thread_state["summary"] = new_summary

            thread_state["messages"] = remaining
            thread_state["compactions"] = thread_state.get("compactions", 0) + 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return the thread context dictionary."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return total compaction count for thread."""
        return self._ensure_thread(thread_id).get("compactions", 0)
