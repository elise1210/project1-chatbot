from pathlib import Path
import yaml

SCOPE_PATH = Path(__file__).parent / "domain" / "scope.yaml"
_SCOPE = yaml.safe_load(SCOPE_PATH.read_text(encoding="utf-8")) or {}
if not _SCOPE:
    raise RuntimeError(f"scope.yaml is empty or invalid: {SCOPE_PATH}")

def route_question(q: str) -> str:
    text = q.lower()

    for k in _SCOPE.get("safety_keywords", []):
        if k in text:
            return "SAFETY_TRIGGER"

    for k in _SCOPE.get("in_scope_keywords", []):
        if k in text:
            return "IN_SCOPE"

    for k in _SCOPE.get("oos_keywords", []):
        if k in text:
            return "OUT_OF_SCOPE"

    return "OUT_OF_SCOPE"

def oos_response() -> str:
    domain = _SCOPE.get("domain_name", "this domain")
    topics = _SCOPE.get("in_scope_topics", [])
    lines = [
        f"I can help with **{domain}** questions.",
        "Examples of what I can help with:",
    ]
    for t in topics[:6]:
        lines.append(f"- {t}")
    lines.append("\nIf you rephrase your question to match one of these, I’ll answer directly.")
    return "\n".join(lines)

def safety_response() -> str:
    return (
        "I’m sorry you’re dealing with this. If you’re in immediate danger, call your local emergency number.\n"
        "If you’re in the U.S., you can call or text **988** (Suicide & Crisis Lifeline).\n"
        "If you want, tell me what’s going on and what kind of help you’re looking for."
    )
