import json
from pathlib import Path
from typing import List, Dict

DOMAIN_DIR = Path(__file__).parent / "domain"
KNOWLEDGE_PATH = DOMAIN_DIR / "knowledge.md"
FEWSHOT_PATH = DOMAIN_DIR / "fewshot.json"


SYSTEM_PERSONA = """You are a careful financial ratios tutor.
Your job: interpret common stock financial ratios and provide general, educational guidance.

Hard constraints:
- Do NOT give buy/sell/hold recommendations.
- Do NOT predict stock prices or give price targets.
- Keep answers general and non-personalized.
- If the question asks for investment advice, refuse briefly and offer ratio interpretation instead.
- Use the 4-part structure: (1) definition, (2) interpretation, (3) caveats, (4) what to check next.
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
