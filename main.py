import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("⚠  OPENAI_API_KEY 미설정 — .env 파일에 키를 추가하세요")
    else:
        print(f"✓  OpenAI API 키 확인 ({key[:12]}...)")
    print("✓  LangGraph 파이프라인: preprocess → classify → format (gpt-4o-mini)")
    print("✓  서버 시작 완료 — http://localhost:8000")
    yield


app = FastAPI(title="Spectrum Analyzer Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudioSnapshot(BaseModel):
    bass_energy: float = Field(0.0, ge=0.0, le=1.0)
    mid_energy: float = Field(0.0, ge=0.0, le=1.0)
    high_energy: float = Field(0.0, ge=0.0, le=1.0)
    peak_freq: float = Field(0.0, ge=0.0)
    peak_level: float = Field(0.0, ge=0.0, le=1.0)
    rms: float = Field(0.0, ge=0.0, le=1.0)
    spectral_bands: List[float] = Field(default_factory=list)


@app.post("/api/analyze")
async def analyze(snapshot: AudioSnapshot):
    from graph import run_analysis
    try:
        return await run_analysis(snapshot.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "model": "gpt-4o-mini",
        "pipeline": "preprocess → classify → format",
    }


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response

# 정적 파일 마운트 (마지막에 위치해야 함)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
