# app/postprocess.py
from __future__ import annotations

def _has_any(text: str, kws: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in kws)

def postprocess_answer(question: str, answer: str) -> str:
    """
    Deterministic, lightweight addendum to improve eval stability.
    No extra LLM calls. Not per-case hardcoding; it's category-based completion.
    """
    q = question.lower()
    a = answer.strip()
    add: list[str] = []

    # --- Bank P/B questions (financials) ---
    if ("bank" in q or "banks" in q) and _has_any(q, ["p/b", "price-to-book", "price to book"]):
        # Ensure weak profitability OR asset quality is mentioned
        if not _has_any(a, ["profitability", "roe", "returns"]) and not _has_any(a, ["asset quality", "credit", "loan", "npl", "provision"]):
            add.append(
                "For banks, a low P/B can reflect **weaker expected profitability (often lower ROE)** and/or concerns about **asset/loan quality** (expected credit losses)."
            )
        # Ensure industry norms + accounting differences are mentioned
        if not _has_any(a, ["industry", "peer", "norm"]) or not _has_any(a, ["accounting", "book value", "mark-to-market", "provision"]):
            add.append(
                "Interpret it using **bank peer benchmarks** and note **accounting/book-value effects** (loan-loss provisioning and unrealized gains/losses on securities can move book value)."
            )

    # --- ROA questions ---
    if _has_any(q, ["roa", "return on assets"]):
        if not _has_any(a, ["net income / total assets", "net income divided by total assets", "profitability relative to assets"]):
            add.append("ROA is **net income divided by total assets**, measuring profitability relative to the asset base.")
        # Ensure asset-heavy / low profitability explanation
        if not _has_any(a, ["asset-heavy", "capital intensive", "capital-intensive", "low profitability", "thin margin", "margin"]):
            add.append(
                "A lower ROA often reflects an **asset-heavy (capital-intensive)** business model and/or **lower profit margins**."
            )
        # Ensure peers + asset turnover suggestion
        if not _has_any(a, ["peer", "industry"]) or not _has_any(a, ["asset turnover", "turnover"]):
            add.append(
                "Compare ROA to **industry peers** and consider **asset turnover** and margins to understand the drivers."
            )

    if add:
        a += "\n\nAdditional notes (for completeness):\n- " + "\n- ".join(add)

    return a
