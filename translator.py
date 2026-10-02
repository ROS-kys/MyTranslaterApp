import json
import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-6-astra"
MAX_CHARS = 5000
# 응답 대기 시간(초)과 SDK 자동 재시도 횟수. 기본값(600초, 2회)이면 장애 시 너무 오래 멈춘다.
REQUEST_TIMEOUT = 60
MAX_RETRIES = 1
# 번역에는 깊은 추론이 필요 없으므로 추론 토큰을 줄여 응답 속도를 높인다. (gpt-6-astra 지원값: low 이상)
REASONING_EFFORT = "low"

# 번역 프롬프트에 쓰는 언어 이름
LANGUAGES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Simplified Chinese",
    "th": "Thai",
}

# 화면과 다운로드 파일에 표시하는 언어 이름
LANGUAGE_LABELS = {
    "en": "English",
    "ja": "日本語",
    "zh": "简体中文",
    "th": "ภาษาไทย",
}


class TranslationError(Exception):
    """모델 응답을 번역 결과로 해석할 수 없을 때 발생한다."""


def get_setting(name: str, default: str | None = None) -> str | None:
    """st.secrets를 먼저 조회하고, 없으면 환경변수(.env)를 조회한다."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def secrets_problem() -> str | None:
    """Secrets가 있는데도 API 키를 읽지 못하는 원인을 반환한다. ("parse", "section" 또는 None)"""
    try:
        items = list(st.secrets.items())
    except Exception as e:
        # 파일이 없는 것은 정상(로컬 .env 사용), 형식 오류는 st.secrets 전체를 못 읽게 만든다.
        return None if "No secrets found" in str(e) else "parse"
    for _, value in items:
        if hasattr(value, "keys") and any(k.upper() == "OPENAI_API_KEY" for k in value.keys()):
            return "section"
    return None


def get_api_key() -> str | None:
    return get_setting("OPENAI_API_KEY") or None


def get_model() -> str:
    return get_setting("OPENAI_MODEL") or DEFAULT_MODEL


def build_system_prompt(targets: list[str]) -> str:
    names = ", ".join(f"{LANGUAGES[code]} ({code})" for code in targets)
    keys = ", ".join(f'"{code}"' for code in targets)
    return (
        "You are a professional translator. Detect the source language automatically "
        f"and translate the user's text into {names}. Preserve meaning, tone, line breaks, "
        f"and formatting. Return only a JSON object with keys {keys}."
    )


def _request(client: OpenAI, model: str, text: str, targets: list[str]) -> dict:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": build_system_prompt(targets)},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise TranslationError("응답이 올바른 JSON이 아닙니다.") from e
    if not isinstance(data, dict):
        raise TranslationError("응답이 JSON 객체가 아닙니다.")
    missing = [code for code in targets if not isinstance(data.get(code), str)]
    if missing:
        raise TranslationError(f"응답에 누락된 언어가 있습니다: {', '.join(missing)}")
    return {code: data[code] for code in targets}


def _translate_one(client: OpenAI, model: str, text: str, code: str) -> str:
    try:
        return _request(client, model, text, [code])[code]
    except TranslationError:
        # JSON 파싱 실패 또는 언어 누락 시 1회 재시도
        return _request(client, model, text, [code])[code]


@st.cache_data(show_spinner=False, max_entries=500, ttl=86400)
def _cached_translate(text: str, targets: tuple[str, ...], model: str) -> dict:
    # API 키는 캐시 키에 포함되지 않도록 함수 안에서 조회한다.
    client = OpenAI(api_key=get_api_key(), timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
    # 한 번에 모든 언어를 생성하면 출력이 길어져 느리므로 언어별로 동시에 요청한다.
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {code: pool.submit(_translate_one, client, model, text, code) for code in targets}
        return {code: future.result() for code, future in futures.items()}


def translate(text: str, targets: list[str]) -> dict:
    """선택한 언어로 번역한 결과를 {언어코드: 번역문} 형태로 반환한다."""
    ordered = tuple(code for code in LANGUAGES if code in targets)
    return _cached_translate(text, ordered, get_model())
