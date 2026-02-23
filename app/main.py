from fastapi import FastAPI
from pydantic import BaseModel

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
    return ChatResponse(
        answer=f"You asked: {req.question}",
        in_scope=True,
        route="IN_SCOPE",
    )
