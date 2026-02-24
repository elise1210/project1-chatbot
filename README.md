# Project 1 – Domain Q&A Chatbot  
**Domain:** Common Stock Financial Ratio Interpretation  

## Overview

This project implements a domain-specific Q&A chatbot that interprets common stock financial ratios and provides general, educational guidance.

The system includes:

- Few-shot prompting (≥ 3 examples)
- Scope guardrails (in-scope / out-of-scope detection)
- Safety trigger handling
- Escape hatch for vague or advisory-style questions
- Structured answer style (4-part format)
- FastAPI backend
- Cloud Run deployment

Live API URL:  
`https://project1-chatbot-slkooky2va-uc.a.run.app/`

---

## Domain

**Mission:**  
Interpret financial ratios (P/E, ROE, D/E, liquidity ratios, etc.) in general terms.

**Hard Constraints:**
- No buy/sell/hold recommendations
- No price targets or predictions
- No personalized financial advice
- Provide structured explanations:
  1. Definition  
  2. Interpretation  
  3. Caveats  
  4. What to check next  

---

## Architecture

## Evaluation

Run the evaluation script (FastAPI must be running):

```bash
uv run python eval/run_eval.py
```

Environment variables used by the eval:

- `GEMINI_API_KEY` (required for judge)
- `BASE_URL` (optional, default `http://127.0.0.1:8000`)
- `JUDGE_MODEL` (optional, default `gemini/gemini-2.5-flash`)
- `JUDGE_MIN_INTERVAL_SECONDS` (optional, default `0.2`; increase if you still see rate limits)
