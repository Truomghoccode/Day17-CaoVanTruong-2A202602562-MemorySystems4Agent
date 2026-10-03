from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with low compact threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cfg.state_dir = state_dir
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, edited, and queried for size."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)
    user_id = "test_user_01"

    # Write initial profile
    initial_content = "# User Profile: test_user_01\n\n## Facts\n- location: Da Nang\n"
    file_path = store.write_text(user_id, initial_content)
    assert file_path.exists()
    assert store.file_size(user_id) > 0

    # Read profile
    read_back = store.read_text(user_id)
    assert "Da Nang" in read_back

    # Edit profile (e.g. location correction)
    edited = store.edit_text(user_id, "Da Nang", "Hue")
    assert edited is True

    # Confirm edit
    updated_content = store.read_text(user_id)
    assert "Hue" in updated_content
    assert "Da Nang" not in updated_content


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction and summarize older context."""
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "test_compact_thread"

    # Append multiple long messages to exceed 50 tokens
    for i in range(5):
        manager.append(
            thread_id,
            "user",
            f"Message {i}: Đây là một thông điệp dài nhằm mục đích kích hoạt compaction của bộ nhớ trong thread.",
        )
        manager.append(
            thread_id,
            "assistant",
            f"Phản hồi {i}: Tôi đã nhận được thông điệp dài này từ bạn và đang xử lý dữ liệu tương ứng.",
        )

    assert manager.compaction_count(thread_id) > 0

    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 2
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced agent remembers across sessions and baseline agent does not."""
    cfg = make_config(tmp_path)
    user_id = "dungct_recall_test"

    base_agent = BaselineAgent(config=cfg, force_offline=True)
    adv_agent = AdvancedAgent(config=cfg, force_offline=True)

    # Turn 1 in thread_1: Provide personal facts
    intro_message = "Chào bạn, mình tên là DũngCT. Mình ở Huế và đồ uống yêu thích là cà phê sữa đá."
    base_agent.reply(user_id, "thread_1", intro_message)
    adv_agent.reply(user_id, "thread_1", intro_message)

    # Turn 2 in thread_2 (brand new session): Ask recall questions
    recall_question = "Mình tên gì và đồ uống yêu thích của mình là gì?"
    base_reply = base_agent.reply(user_id, "thread_2", recall_question)["response"]
    adv_reply = adv_agent.reply(user_id, "thread_2", recall_question)["response"]

    # Advanced agent has persistent User.md memory across threads
    assert "DũngCT" in adv_reply
    assert "cà phê sữa đá" in adv_reply

    # Baseline agent has naive within-session memory only, forgets across new threads
    assert "DũngCT" not in base_reply
    assert "cà phê sữa đá" not in base_reply


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    cfg.compact_threshold_tokens = 80
    cfg.compact_keep_messages = 2

    user_id = "dungct_stress_test"
    thread_id = "long_test_thread"

    base_agent = BaselineAgent(config=cfg, force_offline=True)
    adv_agent = AdvancedAgent(config=cfg, force_offline=True)

    long_payload = (
        "Đây là một lượt trao đổi ngữ cảnh rất dài trong chuỗi hội thoại kỹ thuật. "
        "NASA công bố kế hoạch Artemis III năm 2027 và máy bay X-59 bay siêu thanh Mach 1.1. "
        "WMO cảnh báo El Nino với xác suất cao và British Columbia công bố kế hoạch tiết kiệm điện. "
    )

    for i in range(8):
        turn_text = f"Turn {i}: {long_payload}"
        base_agent.reply(user_id, thread_id, turn_text)
        adv_agent.reply(user_id, thread_id, turn_text)

    # Advanced agent should have triggered compaction
    assert adv_agent.compaction_count(thread_id) > 0

    base_prompt_tokens = base_agent.prompt_token_usage(thread_id)
    adv_prompt_tokens = adv_agent.prompt_token_usage(thread_id)

    # Advanced agent should have significantly lower prompt tokens due to compaction
    assert adv_prompt_tokens < base_prompt_tokens
