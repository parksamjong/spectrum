import json
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

MODEL = "gpt-4o-mini"


class AudioState(TypedDict):
    # 입력
    bass_energy: float
    mid_energy: float
    high_energy: float
    peak_freq: float
    peak_level: float
    rms: float
    spectral_bands: list
    # 전처리
    frequency_profile: str
    # 분류
    genre: str
    instruments: list
    description: str
    color_theme: str
    # 이상감지
    anomalies: list       # 룰 기반 이상 목록
    threat_level: str     # "normal" | "warning" | "critical"
    threat_detail: str    # AI 이상 설명


def preprocess_node(state: AudioState) -> dict:
    bass  = state["bass_energy"]
    mid   = state["mid_energy"]
    high  = state["high_energy"]
    total = bass + mid + high

    if total < 0.005:
        profile = f"무음 | peak:{state['peak_freq']:.0f}Hz | RMS:{state['rms']:.1%}"
    else:
        profile = (
            f"bass:{bass:.1%} mid:{mid:.1%} high:{high:.1%} | "
            f"peak:{state['peak_freq']:.0f}Hz @ {state['peak_level']:.1%} | "
            f"RMS:{state['rms']:.1%}"
        )
    return {"frequency_profile": profile}


async def classify_node(state: AudioState) -> dict:
    llm = ChatOpenAI(model=MODEL, max_tokens=300, temperature=0.2)
    prompt = f"""You are an expert audio analyst. Analyze this real-time audio spectrum and identify genre and instruments.

Spectrum: {state['frequency_profile']}

Respond with ONLY valid JSON (no markdown):
{{
  "genre": "<EDM|Jazz|Rock|Classical|Hip-hop|Pop|speech|noise|silence|...>",
  "instruments": ["<instrument1>", "<instrument2>"],
  "description": "<one sentence in Korean, max 35 chars>",
  "color_theme": "<hex color matching the mood>"
}}

For non-music audio, instruments should be [].
For silence or very low signal, use genre "silence"."""

    try:
        resp = await llm.ainvoke(prompt)
        content = resp.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]).lstrip("json").strip()
        result = json.loads(content)
        return {
            "genre":       result.get("genre", "Unknown"),
            "instruments": result.get("instruments", [])[:5],
            "description": result.get("description", ""),
            "color_theme": result.get("color_theme", "#00e5ff"),
        }
    except Exception:
        return {
            "genre":       "Unknown",
            "instruments": [],
            "description": "분석 완료",
            "color_theme": "#00e5ff",
        }


async def anomaly_detect_node(state: AudioState) -> dict:
    """
    룰 기반 이상 감지 + AI 위협 해석.
    - 클리핑 / 과부하 / 주파수 불균형 / 급격한 레벨 변화 등
    - '해킹 탐지' 맥락: 오디오 채널을 통한 초음파 신호, 비정상 패턴 등
    """
    anomalies = []
    bass  = state["bass_energy"]
    mid   = state["mid_energy"]
    high  = state["high_energy"]
    peak  = state["peak_level"]
    rms   = state["rms"]
    pf    = state["peak_freq"]
    total = bass + mid + high

    # ── 룰 기반 이상 감지
    if peak > 0.95:
        anomalies.append("클리핑 — 피크 레벨 한계 초과")
    if rms > 0.85:
        anomalies.append("과부하 — 레벨 낮춤 권장")
    if high > 0.6 and bass < 0.05:
        anomalies.append("저음 결핍 — 고주파 과다")
    if bass > 0.75 and high < 0.05:
        anomalies.append("고음 결핍 — 저주파 과다")
    if rms > 0.001 and (peak / rms) > 25:
        anomalies.append("극단적 다이나믹 레인지")
    # 초음파 대역 이상 (17kHz 이상에 에너지 집중 → 초음파 공격 의심)
    if pf > 17000 and peak > 0.15:
        anomalies.append("⚡ 초음파 대역 신호 감지 (17kHz+) — 해킹 의심")
    # 비정상 협대역 신호 (특정 주파수에 에너지 과집중)
    if total > 0.05 and peak > 0.8 and (high + bass) < 0.05:
        anomalies.append("협대역 신호 이상 — 단일 주파수 과부하")
    # DC 오프셋 의심 (초저음역 과다)
    if pf < 30 and peak > 0.3:
        anomalies.append("DC 오프셋 또는 초저주파 신호 이상")

    # 위협 수준 결정
    if any("해킹" in a or "초음파" in a or "협대역" in a or "DC" in a for a in anomalies):
        threat_level = "critical"
    elif len(anomalies) >= 2:
        threat_level = "warning"
    elif len(anomalies) == 1:
        threat_level = "warning"
    else:
        threat_level = "normal"

    # AI 위협 해석 (이상이 있을 때만 호출)
    threat_detail = ""
    if anomalies:
        llm = ChatOpenAI(model=MODEL, max_tokens=150, temperature=0.1)
        prompt = f"""오디오 이상 감지 시스템입니다. 아래 이상 목록을 바탕으로 짧고 명확한 한국어 설명을 작성하세요.

감지된 이상: {', '.join(anomalies)}
스펙트럼 데이터: {state['frequency_profile']}

20자 이내로 핵심만 설명하세요. 예: "클리핑 발생, 게인 조정 필요"
설명만 출력하고 다른 텍스트는 없이:"""
        try:
            resp = await llm.ainvoke(prompt)
            threat_detail = resp.content.strip()
        except Exception:
            threat_detail = "이상 신호 감지됨"

    return {
        "anomalies":    anomalies,
        "threat_level": threat_level,
        "threat_detail": threat_detail,
    }


def format_node(state: AudioState) -> dict:
    return {
        "genre":       (state.get("genre") or "Unknown").strip(),
        "instruments": [i.strip() for i in (state.get("instruments") or []) if i.strip()],
    }


# ── 그래프 빌드: preprocess → classify → anomaly_detect → format
builder = StateGraph(AudioState)
builder.add_node("preprocess",     preprocess_node)
builder.add_node("classify",       classify_node)
builder.add_node("anomaly_detect", anomaly_detect_node)
builder.add_node("format",         format_node)

builder.add_edge(START,            "preprocess")
builder.add_edge("preprocess",     "classify")
builder.add_edge("classify",       "anomaly_detect")
builder.add_edge("anomaly_detect", "format")
builder.add_edge("format",         END)

audio_graph = builder.compile()


async def run_analysis(snapshot: dict) -> dict:
    state = AudioState(
        bass_energy=snapshot.get("bass_energy", 0.0),
        mid_energy=snapshot.get("mid_energy",   0.0),
        high_energy=snapshot.get("high_energy", 0.0),
        peak_freq=snapshot.get("peak_freq",     0.0),
        peak_level=snapshot.get("peak_level",   0.0),
        rms=snapshot.get("rms",                 0.0),
        spectral_bands=snapshot.get("spectral_bands", []),
        frequency_profile="",
        genre="",
        instruments=[],
        description="",
        color_theme="#00e5ff",
        anomalies=[],
        threat_level="normal",
        threat_detail="",
    )
    result = await audio_graph.ainvoke(state)
    return {
        "genre":             result.get("genre",         "Unknown"),
        "instruments":       result.get("instruments",   []),
        "description":       result.get("description",   ""),
        "color_theme":       result.get("color_theme",   "#00e5ff"),
        "frequency_profile": result.get("frequency_profile", ""),
        "anomalies":         result.get("anomalies",     []),
        "threat_level":      result.get("threat_level",  "normal"),
        "threat_detail":     result.get("threat_detail", ""),
    }
