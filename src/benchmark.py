from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure stdout handles UTF-8 on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


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
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Compute recall score based on fraction of expected strings found in answer."""
    if not expected:
        return 1.0
    if not answer:
        return 0.0
    ans_lower = answer.lower()
    matched = sum(1 for exp in expected if exp.lower() in ans_lower)
    return round(matched / len(expected), 4)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Evaluate response quality for offline mode.

    Takes into account recall accuracy, structured formatting, and concise style.
    """
    rec = recall_points(answer, expected)
    if not answer or rec == 0.0:
        return 0.20  # Minimum score for polite uninformative fallback

    structure_bonus = 0.15 if ("-" in answer or "\n" in answer) else 0.0
    length_bonus = 0.15 if (30 <= len(answer) <= 400) else 0.05
    score = 0.60 * rec + structure_bonus + length_bonus + 0.10
    return round(min(1.0, score), 4)


def run_agent_benchmark(
    agent_name: str,
    agent_or_cls: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent across conversations and cross-session recall questions."""
    # Instantiate agent if a class was provided
    if isinstance(agent_or_cls, type):
        agent = agent_or_cls(config=config, force_offline=True)
    else:
        agent = agent_or_cls

    all_user_ids = list({c["user_id"] for c in conversations})

    # Initial memory size
    initial_memory_size = 0
    if hasattr(agent, "memory_file_size"):
        initial_memory_size = sum(agent.memory_file_size(u) for u in all_user_ids)

    recalls: list[float] = []
    qualities: list[float] = []

    for conv in conversations:
        user_id = conv["user_id"]
        conv_id = conv["id"]
        main_thread = f"{conv_id}-main"

        # 1. Feed all turns to main thread
        for turn in conv["turns"]:
            agent.reply(user_id, main_thread, turn)

        # 2. Ask recall questions in fresh, separate threads (cross-session evaluation)
        recall_questions = conv.get("recall_questions", [])
        for q_idx, q_item in enumerate(recall_questions):
            recall_thread = f"{conv_id}-recall-{q_idx}"
            reply_dict = agent.reply(user_id, recall_thread, q_item["question"])
            answer = reply_dict.get("response", "")

            rec = recall_points(answer, q_item["expected_contains"])
            qual = heuristic_quality(answer, q_item["expected_contains"])

            recalls.append(rec)
            qualities.append(qual)

    # Calculate final metrics
    agent_tokens = agent.token_usage()
    prompt_tokens = agent.prompt_token_usage()
    avg_recall = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    avg_quality = round(sum(qualities) / len(qualities), 4) if qualities else 0.0

    final_memory_size = 0
    if hasattr(agent, "memory_file_size"):
        final_memory_size = sum(agent.memory_file_size(u) for u in all_user_ids)
    memory_growth = max(0, final_memory_size - initial_memory_size)

    compactions = agent.compaction_count()

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows into a clean comparison table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality * 100:.1f}%",
                f"{r.memory_growth_bytes:,}",
                r.compactions,
            ]
        )

    try:
        from tabulate import tabulate

        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback markdown table generator
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table_data]
        return "\n".join([header_line, sep_line] + data_lines)


def safe_clean_dir(target_dir: Path) -> None:
    """Safely clear contents of directory without failing on Windows/OneDrive locks."""
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        return
    for root_path, dirs, files in os.walk(target_dir, topdown=False):
        for f in files:
            file_path = Path(root_path) / f
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
        for d in dirs:
            dir_path = Path(root_path) / d
            try:
                dir_path.rmdir()
            except Exception:
                pass


def main() -> None:
    """Execute both Standard Benchmark and Long-Context Stress Benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("================================================================================")
    print("                    AI AGENT MEMORY SYSTEMS BENCHMARK SUITE                     ")
    print("================================================================================\n")

    # 1. Standard Benchmark
    print("### Standard Benchmark (Dataset: data/conversations.json - 10 conversations)")
    clean_profiles_dir = config.state_dir / "profiles"
    safe_clean_dir(clean_profiles_dir)

    std_conversations = load_conversations(standard_path)
    std_baseline_row = run_agent_benchmark("Baseline", BaselineAgent, std_conversations, config)
    std_advanced_row = run_agent_benchmark("Advanced", AdvancedAgent, std_conversations, config)
    print(format_rows([std_baseline_row, std_advanced_row]))
    print("\n")

    # 2. Long-Context Stress Benchmark
    print("### Long-Context Stress Benchmark (Dataset: data/advanced_long_context.json - 16 long turns)")
    safe_clean_dir(clean_profiles_dir)

    stress_conversations = load_conversations(stress_path)
    stress_baseline_row = run_agent_benchmark("Baseline", BaselineAgent, stress_conversations, config)
    stress_advanced_row = run_agent_benchmark("Advanced", AdvancedAgent, stress_conversations, config)
    print(format_rows([stress_baseline_row, stress_advanced_row]))
    print("\n================================================================================")


if __name__ == "__main__":
    main()
