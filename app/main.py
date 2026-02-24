from .escape_hatch import needs_context, escape_answer
from fastapi import FastAPI
from pydantic import BaseModel

from .guardrails import route_question, oos_response, safety_response
from .prompts import build_messages
from .llm import chat_completion

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    in_scope: bool
    route: str  # IN_SCOPE / OUT_OF_SCOPE / SAFETY_TRIGGER / ESCAPE_HATCH

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    route = route_question(req.question)

    if route == "SAFETY_TRIGGER":
        return ChatResponse(answer=safety_response(), in_scope=False, route=route)

    if route == "OUT_OF_SCOPE":
        return ChatResponse(answer=oos_response(), in_scope=False, route=route)

    if needs_context(req.question):
        return ChatResponse(
            answer=escape_answer(),
            in_scope=True,
            route="ESCAPE_HATCH",
        )

    # IN_SCOPE -> LLM
    try:
        messages = build_messages(req.question)
        answer = chat_completion(messages)
        return ChatResponse(answer=answer, in_scope=True, route="IN_SCOPE")
    except Exception as e:
        # simple fallback for now
        return ChatResponse(
            answer=f"Error calling LLM: {e}",
            in_scope=True,
            route="IN_SCOPE",
        )
