"""Q10: FastAPI server to serve q-fastapi.csv.

GET /api -> {"students": [{"studentId": int, "class": str}, ...]} in CSV order.
GET /api?class=1A&class=1B -> filtered, preserving CSV order.
CORS enabled for GET from any origin.
"""
import csv
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


def _find_csv() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "q-fastapi.csv",  # tds-Ga0-q5/q-fastapi.csv
        here.parents[1] / "q-fastapi.csv",
        Path.cwd() / "tds-Ga0-q5" / "q-fastapi.csv",
        Path.cwd() / "q-fastapi.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


CSV_PATH = _find_csv()
ROWS: list[dict] = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ROWS.append({"studentId": int(r["studentId"]), "class": r["class"]})

app = FastAPI(title="Q10 FastAPI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "rows": len(ROWS), "usage": "/api?class=1A&class=1B"}


@app.get("/api")
def get_api(request: Request):
    wanted = request.query_params.getlist("class")
    if not wanted:
        return {"students": ROWS}
    allowed = set(wanted)
    return {"students": [r for r in ROWS if r["class"] in allowed]}
