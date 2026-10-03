from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.inference import RecommendationEngine
from src.utils import load_config

engine: RecommendationEngine


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = Field(default=20, ge=1, le=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    cfg = load_config()
    engine = RecommendationEngine(cfg)
    yield
    engine = None


app = FastAPI(title="Two-Stage RecSys", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_id not in engine.known_users:
        raise HTTPException(status_code=404, detail="Unknown user")
    return engine.recommend(req.user_id, req.top_k)
