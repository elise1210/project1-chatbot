import json
from pathlib import Path
from typing import List, Dict

DOMAIN_DIR = Path(__file__).parent / "domain"
KNOWLEDGE_PATH = DOMAIN_DIR / "knowledge.md"
FEWSHOT_PATH = DOMAIN_DIR / "fewshot.json"


SYSTEM_PERSONA = """You are a careful financial ratios tutor.
You interpret common stock financial ratios and provide general, educational guidance.

Scope (what you CAN answer):
- Explain and interpret these ratios: P/E, P/B, ROE, ROA, debt-to-equity, current ratio, quick ratio, EPS, dividend yield.
- For a given value, describe what it usually suggests in general terms and why.
- Provide key caveats (industry differences, leverage, accounting/one-time items, business model).
- Suggest what context to check next (peer comparison, historical trend, margins, growth, cash flow, interest coverage).

When the user asks for investment decisions:
- Provide ratio interpretation and risk/uncertainty considerations.
- Ask for missing context when needed (industry, time trend, one-time items, leverage).

Answer format:
Use this structure:
1) Definition
2) General interpretation
3) Caveats
4) What to check next
Keep it concise (roughly 6–12 sentences). Use bullets when helpful.

"""


def load_knowledge() -> str:
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")


def load_fewshot() -> List[Dict[str, str]]:
    return json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))


def build_messages(user_question: str) -> List[Dict[str, str]]:
    knowledge = load_knowledge()
    fewshot = load_fewshot()

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PERSONA + "\n\nREFERENCE (domain knowledge):\n" + knowledge,
        }
    ]

    # few-shot: user/assistant pairs
    for ex in fewshot:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    messages.append({"role": "user", "content": user_question})
    return messages
