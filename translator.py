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


TRANSLATE_PROMPT = (
    "You are a professional translator. Detect the source language automatically "
    "and translate the user's text into {language}. Preserve meaning, tone, line breaks, "
    'and formatting. Return only a JSON object with the key "translation".'
)

PRONOUNCE_PROMPT = (
    "The user's text is written in {language}. Write how it is pronounced using only Korean Hangul, "
    "the way a Korean speaker would read it aloud. Keep the same line breaks. "
    'Return only a JSON object with the key "pronunciation".'
)


def _new_client() -> OpenAI:
    # API 키는 캐시 키에 포함되지 않도록 캐시 함수 안에서 조회한다.
    return OpenAI(api_key=get_api_key(), timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)


def _request(client: OpenAI, model: str, system_prompt: str, text: str, key: str) -> str:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": system_prompt},
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
    if not isinstance(data.get(key), str):
        raise TranslationError(f"응답에 '{key}' 항목이 없습니다.")
    return data[key]


def _request_with_retry(client: OpenAI, model: str, system_prompt: str, text: str, key: str) -> str:
    try:
        return _request(client, model, system_prompt, text, key)
    except TranslationError:
        # JSON 파싱 실패 또는 항목 누락 시 1회 재시도
        return _request(client, model, system_prompt, text, key)


def _run_parallel(jobs: dict, model: str, key: str) -> dict:
    """{언어코드: (시스템 프롬프트, 입력 텍스트)}를 언어별로 동시에 요청한다."""
    client = _new_client()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            code: pool.submit(_request_with_retry, client, model, prompt, text, key)
            for code, (prompt, text) in jobs.items()
        }
        return {code: future.result() for code, future in futures.items()}


@st.cache_data(show_spinner=False, max_entries=500, ttl=86400)
def _cached_translate(text: str, targets: tuple[str, ...], model: str) -> dict:
    # 한 번에 모든 언어를 생성하면 출력이 길어져 느리므로 언어별로 동시에 요청한다.
    jobs = {code: (TRANSLATE_PROMPT.format(language=LANGUAGES[code]), text) for code in targets}
    return _run_parallel(jobs, model, "translation")


@st.cache_data(show_spinner=False, max_entries=500, ttl=86400)
def _cached_pronounce(items: tuple[tuple[str, str], ...], model: str) -> dict:
    jobs = {code: (PRONOUNCE_PROMPT.format(language=LANGUAGES[code]), text) for code, text in items}
    return _run_parallel(jobs, model, "pronunciation")


def translate(text: str, targets: list[str]) -> dict:
    """선택한 언어로 번역한 결과를 {언어코드: 번역문} 형태로 반환한다."""
    ordered = tuple(code for code in LANGUAGES if code in targets)
    return _cached_translate(text, ordered, get_model())


def pronounce(translations: dict) -> dict:
    """번역문의 한글 발음을 {언어코드: 발음} 형태로 반환한다.

    번역과 따로 요청해 번역 결과를 먼저 보여줄 수 있게 하고, 완성된 번역문을 읽게 해 번역과 발음이 어긋나지 않게 한다.
    """
    return _cached_pronounce(tuple(translations.items()), get_model())
