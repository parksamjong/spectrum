import os
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("  OPENAI_API_KEY not set - add to .env file")
    else:
        print(f"  OpenAI API key ok ({key[:12]}...)")
    print("  LangGraph pipeline: preprocess -> classify -> format (gpt-4o-mini)")
    if _SD_OK:
        devs = sd.query_devices()
        n_in = sum(1 for d in devs if d['max_input_channels'] > 0)
        print(f"  sounddevice ok - {n_in} input device(s) found")
    else:
        print("  sounddevice not installed - system audio streaming unavailable")
    print("  Server ready: http://localhost:8000")
    yield


app = FastAPI(title="Spectrum Analyzer Pro", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST: LangGraph AI analysis ─────────────────────────────────────────────

class AudioSnapshot(BaseModel):
    bass_energy: float = Field(0.0, ge=0.0, le=1.0)
    mid_energy:  float = Field(0.0, ge=0.0, le=1.0)
    high_energy: float = Field(0.0, ge=0.0, le=1.0)
    peak_freq:   float = Field(0.0, ge=0.0)
    peak_level:  float = Field(0.0, ge=0.0, le=1.0)
    rms:         float = Field(0.0, ge=0.0, le=1.0)
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
        "status":      "ok",
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "model":       "gpt-4o-mini",
        "pipeline":    "preprocess -> classify -> format",
        "sounddevice": _SD_OK,
    }


# ── System audio: device list ────────────────────────────────────────────────

@app.get("/api/audio/devices")
async def get_audio_devices():
    if not _SD_OK:
        return {"devices": [], "error": "sounddevice not installed"}
    try:
        devices = sd.query_devices()
        try:
            default_in = int(sd.default.device[0])
        except Exception:
            default_in = -1
        result = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                result.append({
                    "id":         int(i),
                    "name":       str(d['name']),
                    "channels":   int(d['max_input_channels']),
                    "sample_rate": int(d['default_samplerate']),
                    "is_default": (int(i) == int(default_in)) if default_in is not None else False,
                })
        return {"devices": result, "default_id": default_in}
    except Exception as e:
        return {"devices": [], "error": str(e)}


# ── System audio: WebSocket FFT stream ───────────────────────────────────────

@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket, device_id: int = -1):
    await websocket.accept()

    if not _SD_OK:
        await websocket.send_json({"type": "error", "msg": "sounddevice not installed - run: pip install sounddevice numpy"})
        await websocket.close()
        return

    BLOCK = 2048
    SR = 44100

    try:
        dev_info = sd.query_devices(device_id if device_id >= 0 else None)
        SR = int(dev_info['default_samplerate'])
    except Exception:
        pass

    hann = np.hanning(BLOCK).astype(np.float32)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)

    def callback(indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        try:
            loop.call_soon_threadsafe(queue.put_nowait, mono.copy())
        except asyncio.QueueFull:
            pass

    try:
        stream_kwargs: dict = dict(
            samplerate=SR,
            channels=1,
            blocksize=BLOCK,
            callback=callback,
            dtype="float32",
        )
        if device_id >= 0:
            stream_kwargs["device"] = device_id

        with sd.InputStream(**stream_kwargs):
            await websocket.send_json({"type": "ready", "sr": SR, "bins": BLOCK // 2})
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
                    continue

                fft = np.abs(np.fft.rfft(data * hann))[: BLOCK // 2]
                fft_db = 20.0 * np.log10(fft + 1e-9)
                fft_norm = np.clip((fft_db + 90.0) / 90.0, 0.0, 1.0).astype(np.float32)
                rms = float(np.sqrt(np.mean(data ** 2)))

                await websocket.send_json({
                    "type": "fft",
                    "d":   [round(float(v), 3) for v in fft_norm],
                    "rms": round(rms, 5),
                    "sr":  SR,
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass


# ── no-cache middleware ───────────────────────────────────────────────────────

@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory="static", html=True), name="static")
