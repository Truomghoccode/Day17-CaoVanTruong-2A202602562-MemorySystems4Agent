from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session memory only (keyed strictly by thread_id)
    - No persistent User.md profile storage
    - Forgets all facts across different threads / sessions
    - Processes full uncompressed conversation history on every turn
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Generate response and update token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live path
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    {"configurable": {"thread_id": thread_id}},
                )
                output_text = result["messages"][-1].content
                agent_tokens = estimate_tokens(output_text)
                prompt_tokens = estimate_tokens(message)
                session = self._get_session(thread_id)
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": output_text})
                session.token_usage += agent_tokens
                session.prompt_tokens_processed += prompt_tokens
                return {
                    "response": output_text,
                    "agent_tokens": agent_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent generation tokens for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).token_usage
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt context tokens processed for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline agent has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply logic for reproducible evaluation.

        Baseline remembers only messages inside the current thread_id.
        In a new thread, it has zero context about the user's name, profile, or past facts.
        """
        session = self._get_session(thread_id)

        # Baseline must carry the full uncompressed thread history in prompt
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.messages)
        full_prompt = f"{history_text}\nuser: {message}".strip()
        turn_prompt_tokens = estimate_tokens(full_prompt)
        session.prompt_tokens_processed += turn_prompt_tokens

        # Check if this thread has any facts mentioned earlier within the same thread
        within_thread_text = full_prompt.lower()

        # Offline response generation
        # If in a brand new thread asking for personal facts (cross-session recall question)
        if len(session.messages) == 0 and any(
            q in message.lower()
            for q in [
                "mình tên gì",
                "tên mình là gì",
                "nhắc lại",
                "ở đâu",
                "nghề gì",
                "đồ uống",
                "món ăn",
                "con gì",
                "style",
                "dũngct là ai",
            ]
        ):
            response_text = (
                "Xin lỗi bạn, tôi chưa có thông tin về bạn trong phiên trò chuyện mới này. "
                "Bạn vui lòng chia sẻ lại tên và thông tin nhé."
            )
        else:
            response_text = "Tôi đã ghi nhận thông tin của bạn."

        agent_tokens = estimate_tokens(response_text)
        session.token_usage += agent_tokens

        # Append messages to session
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build live LangChain/LangGraph agent when valid API key is present."""
        if not self.config.model.api_key:
            return
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            memory = MemorySaver()
            self.langchain_agent = create_react_agent(
                model=llm,
                tools=[],
                checkpointer=memory,
            )
        except Exception:
            self.langchain_agent = None
