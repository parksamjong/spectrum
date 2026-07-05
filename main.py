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


# ── 장치 분류 태그 helper ────────────────────────────────────────────────────

_LOOPBACK_KW = ('loopback', 'stereo mix', '스테레오 믹스', 'what u hear', 'wave out')

def _dev_type(name: str) -> str:
    lo = name.lower()
    return 'system' if any(k in lo for k in _LOOPBACK_KW) else 'mic'


# ── 장치 목록 API (type 필드 추가) ──────────────────────────────────────────

@app.get("/api/audio/devices2")
async def get_audio_devices2():
    if not _SD_OK:
        return {"devices": [], "error": "sounddevice not installed"}
    try:
        devs = sd.query_devices()
        try:
            default_in = int(sd.default.device[0])
        except Exception:
            default_in = -1
        result = []
        for i, d in enumerate(devs):
            if d['max_input_channels'] > 0:
                name = str(d['name'])
                result.append({
                    "id":          int(i),
                    "name":        name,
                    "channels":    int(d['max_input_channels']),
                    "sample_rate": int(d['default_samplerate']),
                    "is_default":  (int(i) == default_in),
                    "type":        _dev_type(name),
                })
        return {"devices": result, "default_id": default_in}
    except Exception as e:
        return {"devices": [], "error": str(e)}


# ── 내부 helper: FFT 계산 ────────────────────────────────────────────────────

def _raw_fft(data: np.ndarray, hann: np.ndarray, block: int) -> np.ndarray:
    """Hann 윈도우 적용 후 선형 FFT 크기 반환 (노이즈 처리 전)"""
    mono = data.mean(axis=1) if data.ndim > 1 else data.ravel()
    if len(mono) < block:
        mono = np.pad(mono, (0, block - len(mono)))
    return np.abs(np.fft.rfft(mono[:block] * hann))[: block // 2]

def _normalize_fft(fft: np.ndarray) -> np.ndarray:
    """선형 FFT → dB → 0~1 정규화"""
    fft_db = 20.0 * np.log10(fft + 1e-9)
    return np.clip((fft_db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)

def _make_fft(data: np.ndarray, hann: np.ndarray, block: int,
              noise_floor: 'np.ndarray | None' = None,
              noise_strength: float = 2.0) -> np.ndarray:
    """FFT + 스펙트럴 서브트랙션 + 정규화"""
    fft = _raw_fft(data, hann, block)
    if noise_floor is not None:
        # 과감산(oversubtraction) + 스펙트럴 플로어 0.01 유지
        fft = np.maximum(fft - noise_strength * noise_floor, 0.01 * fft)
    return _normalize_fft(fft)


# ── System audio: WebSocket FFT stream ───────────────────────────────────────

@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket, device_id: int = -1, mix_device_id: int = -1,
                   noise_reduce: float = 2.0):
    """
    device_id     : 주 입력 장치 (마이크 or 시스템 사운드)
    mix_device_id : 추가 혼합 장치 (시스템 사운드 or 마이크)  -1이면 단일 장치
    noise_reduce  : 스펙트럴 서브트랙션 강도 (0=끔, 2=기본, 4=최대)
    두 FFT는 element-wise max 로 합성
    """
    await websocket.accept()

    if not _SD_OK:
        await websocket.send_json({"type": "error", "msg": "sounddevice not installed"})
        await websocket.close()
        return

    BLOCK = 2048
    SR = 44100

    # 주 장치 샘플레이트
    try:
        SR = int(sd.query_devices(device_id if device_id >= 0 else None)['default_samplerate'])
    except Exception:
        pass

    hann = np.hanning(BLOCK).astype(np.float32)
    ev_loop = asyncio.get_event_loop()
    q1: asyncio.Queue = asyncio.Queue(maxsize=4)
    q2: asyncio.Queue = asyncio.Queue(maxsize=4)
    smooth_buf = None

    def _cb(q):
        def cb(indata, frames, time_info, status):
            try:
                ev_loop.call_soon_threadsafe(q.put_nowait, indata.copy())
            except asyncio.QueueFull:
                pass
        return cb

    def _open(dev_id, q):
        n_ch = 1
        sr = SR
        if dev_id >= 0:
            try:
                info = sd.query_devices(dev_id)
                n_ch = min(int(info['max_input_channels']), 2)
                sr = int(info['default_samplerate'])
            except Exception:
                pass
        kw = dict(samplerate=sr, channels=n_ch, blocksize=BLOCK, callback=_cb(q), dtype='float32')
        if dev_id >= 0:
            kw['device'] = dev_id
        return sd.InputStream(**kw)

    try:
        streams = []
        use_primary = device_id >= 0
        use_mix     = mix_device_id >= 0

        if not use_primary and not use_mix:
            await websocket.send_json({"type": "error", "msg": "장치를 선택하세요"})
            return

        if use_primary:
            streams.append(_open(device_id, q1))
        if use_mix:
            streams.append(_open(mix_device_id, q2))

        import contextlib
        with contextlib.ExitStack() as stack:
            for s in streams:
                stack.enter_context(s)

            await websocket.send_json({"type": "ready", "sr": SR, "bins": BLOCK // 2})

            # 주 큐 결정 (primary or mix-only)
            main_q = q1 if use_primary else q2
            sec_q  = q2 if (use_primary and use_mix) else None
            last_sec = np.zeros(BLOCK // 2, dtype=np.float32)

            # 노이즈 플로어 추정 (초기 N프레임의 선형 FFT를 수집)
            NOISE_EST_FRAMES = 25          # ~1.2 초
            noise_est_buf: list = []
            noise_floor: np.ndarray | None = None

            while True:
                try:
                    raw1 = await asyncio.wait_for(main_q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
                    continue

                # 선형 FFT 크기
                lin1 = _raw_fft(raw1, hann, BLOCK)

                # 노이즈 플로어 추정 단계
                if noise_reduce > 0 and noise_floor is None:
                    noise_est_buf.append(lin1.copy())
                    if len(noise_est_buf) >= NOISE_EST_FRAMES:
                        # 하위 30th percentile → 잡음 플로어 추정
                        noise_floor = np.percentile(
                            np.stack(noise_est_buf), 30, axis=0
                        ).astype(np.float32)
                    # 추정 중에는 노이즈 감소 없이 그냥 정규화
                    fft_main = _normalize_fft(lin1)
                else:
                    fft_main = _make_fft(raw1, hann, BLOCK,
                                         noise_floor=noise_floor if noise_reduce > 0 else None,
                                         noise_strength=noise_reduce)

                # 보조 장치 FFT (가장 최신 프레임 사용)
                if sec_q is not None:
                    while not sec_q.empty():
                        try:
                            raw2 = sec_q.get_nowait()
                            last_sec = _make_fft(raw2, hann, BLOCK)
                        except Exception:
                            break
                    fft_combined = np.maximum(fft_main, last_sec)
                else:
                    fft_combined = fft_main

                # EMA 스무딩
                if smooth_buf is None:
                    smooth_buf = fft_combined.copy()
                else:
                    smooth_buf = 0.65 * smooth_buf + 0.35 * fft_combined
                    fft_combined = smooth_buf

                rms = float(np.sqrt(np.mean((raw1.mean(axis=1) if raw1.ndim > 1 else raw1.ravel()) ** 2)))

                await websocket.send_json({
                    "type": "fft",
                    "d":   [round(float(v), 3) for v in fft_combined],
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
