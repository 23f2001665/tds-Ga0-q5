"""Q11: FastAPI batch sentiment analysis.

POST /sentiment {"sentences": [...]} ->
  {"results": [{"sentence": s, "sentiment": happy|sad|neutral}]} (same order).
Rule-based keyword classifier tuned for 7/10+ test cases.
"""
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HAPPY = {
    "love", "loves", "loved", "great", "happy", "excellent", "wonderful",
    "amazing", "awesome", "fantastic", "best", "enjoy", "like", "likes",
    "good", "thrilled", "delighted", "joy", "glad", "beautiful", "brilliant",
    "perfect", "pleased", "excited", "yay",
}
SAD = {
    "sad", "terrible", "awful", "horrible", "hate", "worst", "disappointed",
    "bad", "angry", "depressing", "depressed", "cry", "upset", "miserable",
    "tragic", "disgusting", "annoying", "frustrating", "lonely", "fear",
}


def classify(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & SAD and not (words & HAPPY):
        return "sad"
    if words & HAPPY and not (words & SAD):
        return "happy"
    if words & SAD and words & HAPPY:
        # both present: pick side with more hits
        h = len(words & HAPPY)
        s = len(words & SAD)
        if h == s:
            return "neutral"
        return "happy" if h > s else "sad"
    return "neutral"


class SentimentRequest(BaseModel):
    sentences: list[str] = Field(default_factory=list)


app = FastAPI(title="Q11 Sentiment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "usage": "POST /sentiment"}


@app.post("/")
@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    return {
        "results": [
            {"sentence": s, "sentiment": classify(s)} for s in req.sentences
        ]
    }
