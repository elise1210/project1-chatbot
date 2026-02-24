# app/escape_hatch.py

VAGUE_PATTERNS = [
    "is it good",
    "is it bad",
    "good or bad",
    "should i buy",
    "should i sell",
    "recommend",
    "worth buying",
    "worth selling",
]

def needs_context(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in VAGUE_PATTERNS)

def escape_answer() -> str:
    return (
        "I can provide a general interpretation, but I’m missing important context.\n\n"
        "To properly interpret a financial ratio, it helps to know:\n"
        "- The company’s industry or peer group\n"
        "- Historical trend over several years\n"
        "- Whether earnings or equity include one-time effects\n\n"
        "If you can share that context, I can give a more precise explanation."
    )
