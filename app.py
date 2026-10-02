import html

import openai
import streamlit as st

from translator import (
    LANGUAGE_LABELS,
    MAX_CHARS,
    get_api_key,
    get_model,
    pronounce,
    secrets_problem,
    translate,
)

# PRD 11장 오류 메시지
MESSAGES = {
    "empty_input": "번역할 내용을 입력해 주세요.",
    "too_long": f"최대 {MAX_CHARS:,}자까지 입력할 수 있습니다.",
    "no_targets": "번역할 언어를 하나 이상 선택해 주세요.",
    "no_api_key": "OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 Secrets를 확인해 주세요.",
    "secrets_parse": 'Secrets 형식이 올바르지 않아 읽지 못했습니다. 값을 큰따옴표로 감싸 주세요. 예: OPENAI_API_KEY = "sk-..."',
    "secrets_section": "Secrets에서 OPENAI_API_KEY가 [섹션] 아래에 들어 있습니다. [ ] 제목 줄 없이 맨 위에 입력해 주세요.",
    "auth": "API 키가 올바르지 않습니다.",
    "rate_limit": "요청이 많아 잠시 후 다시 시도해 주세요.",
    "not_found": "모델({model})을 사용할 수 없습니다. OPENAI_MODEL 값을 확인해 주세요.",
    "timeout": "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
    "connection": "네트워크 연결에 실패했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.",
    "unknown": "번역 중 오류가 발생했습니다: {error}",
    "pron_loading": "발음을 불러오는 중입니다...",
    "pron_failed": "발음을 불러오지 못했습니다.",
}

# PRD 9장 화면 구성의 탭 라벨
TAB_LABELS = {
    "en": "🇺🇸 English",
    "ja": "🇯🇵 日本語",
    "zh": "🇨🇳 中文",
    "th": "🇹🇭 ไทย",
}

# 색상은 지정하지 않고 Streamlit 테마를 따르게 해 라이트/다크 모드 모두 대응한다.
CUSTOM_CSS = """
<style>
/* Windows는 국기 이모지를 지원하지 않으므로 국기 문자에만 쓰는 웹 글꼴을 불러온다. */
@font-face {
    font-family: "Twemoji Country Flags";
    unicode-range: U+1F1E6-1F1FF, U+1F3F4, U+E0062-E0063, U+E0065, U+E0067, U+E006C, U+E006E, U+E0073-E0074, U+E0077, U+E007F;
    src: url("https://cdn.jsdelivr.net/npm/country-flag-emoji-polyfill@0.1/dist/TwemojiCountryFlags.woff2") format("woff2");
    font-display: swap;
}
[role="tab"] p, [data-testid="stSidebar"] label p {
    font-family: "Twemoji Country Flags", "Source Sans Pro", sans-serif;
}
h1 { word-break: keep-all; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
.app-subtitle { margin-top: -0.75rem; margin-bottom: 1.5rem; opacity: 0.7; }
.stTextArea textarea { font-size: 1rem; line-height: 1.6; }

/* 번역 결과: 다국어 글자가 잘 보이도록 일반 글꼴, 넉넉한 줄 간격, 자동 줄바꿈 */
[data-testid="stCode"] { border-radius: 0.75rem; }
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    font-family: "Source Sans Pro", "Noto Sans", "Noto Sans JP", "Noto Sans SC",
                 "Noto Sans Thai", "Malgun Gothic", sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
    white-space: pre-wrap !important;
    word-break: break-word;
}
[data-testid="stCode"] pre { padding: 1rem 3rem 1rem 1.1rem !important; }

/* 번역문 아래 한글 발음: 작고 흐리게, 줄바꿈 유지 */
.pronunciation { display: flex; gap: 0.5rem; font-size: 0.9rem; line-height: 1.7; opacity: 0.7; margin: -0.5rem 0 0.5rem 0.25rem; }
.pronunciation-label { font-weight: 600; flex-shrink: 0; }
.pronunciation-text { white-space: pre-wrap; }
</style>
"""


def error_message(e: Exception) -> str:
    if isinstance(e, openai.AuthenticationError):
        return MESSAGES["auth"]
    if isinstance(e, openai.RateLimitError):
        return MESSAGES["rate_limit"]
    if isinstance(e, openai.NotFoundError):
        return MESSAGES["not_found"].format(model=get_model())
    if isinstance(e, openai.APITimeoutError):
        return MESSAGES["timeout"]
    if isinstance(e, openai.APIConnectionError):
        return MESSAGES["connection"]
    return MESSAGES["unknown"].format(error=e)


def build_download_text(source: str, result: dict, pronunciation: dict | None = None) -> str:
    sections = [f"[원문]\n{source}"]
    for code, text in result.items():
        section = f"[{LANGUAGE_LABELS[code]}]\n{text}"
        if pronunciation:
            section += f"\n(발음) {pronunciation[code]}"
        sections.append(section)
    return "\n\n".join(sections) + "\n"


st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("번역 언어")
    targets = [
        code for code in LANGUAGE_LABELS
        if st.checkbox(TAB_LABELS[code], value=True, key=f"target_{code}")
    ]
    st.divider()
    st.caption(f"사용 모델: `{get_model()}`")

st.title("🌐 다국어 번역기")
st.markdown(
    '<p class="app-subtitle">입력한 글을 영어·일본어·중국어·태국어로 번역합니다</p>',
    unsafe_allow_html=True,
)

api_key = get_api_key()
if not api_key:
    st.error(MESSAGES["no_api_key"])
    problem = secrets_problem()
    if problem:
        st.info(MESSAGES[f"secrets_{problem}"])

text = st.text_area(
    "번역할 내용",
    placeholder="번역할 내용을 입력하세요...",
    height=200,
    label_visibility="collapsed",
)
count = f"{len(text):,} / {MAX_CHARS:,}자"
if len(text) > MAX_CHARS:
    st.markdown(f":red[**{count} · {MESSAGES['too_long']}**]", text_alignment="right")
else:
    st.markdown(f":gray[:small[{count}]]", text_alignment="right")

if st.button("🔄 번역하기", type="primary", width="stretch", disabled=not api_key):
    if not text.strip():
        st.warning(MESSAGES["empty_input"])
    elif len(text) > MAX_CHARS:
        st.warning(MESSAGES["too_long"])
    elif not targets:
        st.warning(MESSAGES["no_targets"])
    else:
        try:
            with st.spinner("번역 중입니다..."):
                result = translate(text, targets)
            st.session_state["translation"] = {"source": text, "result": result}
        except Exception as e:
            st.error(error_message(e))

st.subheader("번역 결과")
translation = st.session_state.get("translation")
if translation:
    pronunciation = translation.get("pronunciation")
    codes = list(translation["result"])
    for tab, code in zip(st.tabs([TAB_LABELS[c] for c in codes]), codes):
        with tab:
            st.code(translation["result"][code], language=None, wrap_lines=True)
            if pronunciation:
                body = html.escape(pronunciation[code])
            else:
                body = MESSAGES["pron_failed" if translation.get("pronunciation_failed") else "pron_loading"]
            st.markdown(
                '<div class="pronunciation"><span class="pronunciation-label">발음</span>'
                f'<span class="pronunciation-text">{body}</span></div>',
                unsafe_allow_html=True,
            )
    st.download_button(
        "⬇ 전체 결과 다운로드",
        data=build_download_text(translation["source"], translation["result"], pronunciation),
        file_name="translation.txt",
        mime="text/plain",
        width="stretch",
    )

    # 번역 결과를 먼저 보여준 뒤 발음을 불러오고, 다 받으면 다시 그려 발음과 다운로드 파일에 반영한다.
    if pronunciation is None and not translation.get("pronunciation_failed"):
        try:
            translation["pronunciation"] = pronounce(translation["result"])
        except Exception:
            translation["pronunciation_failed"] = True
        st.rerun()
else:
    st.info("번역 결과가 여기에 표시됩니다")
