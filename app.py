# -*- coding: utf-8 -*-
import calendar
import json
import os
import re
import time
from datetime import date
import numpy as np
import anthropic
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import yfinance as yf
except ImportError:
    yf = None

load_dotenv(encoding="utf-8-sig")

# region agent log
DEBUG_LOG_PATH = "debug-a0c597.log"
DEBUG_SESSION_ID = "a0c597"


def _agent_debug_log(run_id, hypothesis_id, location, message, data):
    try:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion

# ============================================================
# ETF 포트폴리오 시뮬레이션 - app.py
# ============================================================

# ============================================================
# 섹션 1. ETF 데이터 정의
# ============================================================

# 보수율: 발행사 공시 기준 순운용보수(연율, 소수). 2026년 6월 기준.
ETF_DATA = {
    "VOO":  {"이름": "Vanguard S&P 500 ETF",              "카테고리": "지수추적", "보수율": 0.0003, "배당": True,  "레버리지": False},
    "QQQ":  {"이름": "Invesco QQQ Trust",                   "카테고리": "지수추적", "보수율": 0.0020, "배당": True,  "레버리지": False},
    "VTI":  {"이름": "Vanguard Total Stock Market ETF",     "카테고리": "지수추적", "보수율": 0.0003, "배당": True,  "레버리지": False},
    "SCHD": {"이름": "Schwab U.S. Dividend Equity ETF",     "카테고리": "배당성장",   "보수율": 0.0006, "배당": True,  "레버리지": False},
    "DGRO": {"이름": "iShares Core Dividend Growth ETF",    "카테고리": "배당성장",   "보수율": 0.0008, "배당": True,  "레버리지": False},
    "VYM":  {"이름": "Vanguard High Dividend Yield ETF",    "카테고리": "배당성장",   "보수율": 0.0004, "배당": True,  "레버리지": False},
    "JEPI": {"이름": "JPMorgan Equity Premium Income ETF",  "카테고리": "배당집중", "보수율": 0.0035, "배당": True,  "레버리지": False},
    "JEPQ": {"이름": "JPMorgan Nasdaq Equity Premium Income ETF", "카테고리": "배당집중", "보수율": 0.0035, "배당": True,  "레버리지": False},
    "QYLD": {"이름": "Global X Nasdaq 100 Covered Call ETF", "카테고리": "배당집중", "보수율": 0.0060, "배당": True,  "레버리지": False},
    "QLD":  {"이름": "ProShares Ultra QQQ",                 "카테고리": "레버리지", "보수율": 0.0095, "배당": False, "레버리지": True},
    "TQQQ": {"이름": "ProShares UltraPro QQQ",              "카테고리": "레버리지", "보수율": 0.0082, "배당": False, "레버리지": True},
    "SSO":  {"이름": "ProShares Ultra S&P500",              "카테고리": "레버리지", "보수율": 0.0087, "배당": False, "레버리지": True},
    "UPRO": {"이름": "ProShares UltraPro S&P500",           "카테고리": "레버리지", "보수율": 0.0089, "배당": False, "레버리지": True},
    "SOXL": {"이름": "Direxion Daily Semiconductor Bull 3X", "카테고리": "레버리지", "보수율": 0.0075, "배당": False, "레버리지": True},
}


@st.cache_data
def load_etf_cagr():
    import yfinance as yf
    for ticker in ETF_DATA:
        try:
            hist = yf.Ticker(ticker).history(period="max")
            if len(hist) > 0:
                start = hist['Close'].iloc[0]
                end = hist['Close'].iloc[-1]
                years = len(hist) / 252
                cagr = (end / start) ** (1 / years) - 1
                ETF_DATA[ticker]['cagr'] = round(cagr, 4)
        except:
            ETF_DATA[ticker]['cagr'] = 0.10
    return ETF_DATA

# ============================================================
# 섹션 2. AI 성향 분석 함수
# ============================================================

AI_PROFILE_CATEGORIES = ["지수추적", "배당성장", "배당집중", "레버리지"]

# 포트폴리오 내 레버리지 ETF 비중 상한 (0~1)
LEVERAGE_MAX_PORTFOLIO_WEIGHT = 0.30

COVERED_CALL_EXPLANATION = """
**배당집중**은 쉽게 말해, 보유한 자산에서 나오는 일부 상승 기회를 대신해
매달 비교적 규칙적인 현금흐름(프리미엄)을 기대하는 방식입니다.

- **어떻게 수익이 나나요?** → 월수익처럼 들어오는 추가 현금흐름을 노립니다.
- **장점** → 시장이 크게 오르지 않는 구간에서 꾸준한 흐름을 기대하기 좋습니다.
- **단점** → 시장이 급등하면 상승 수익 일부를 놓칠 수 있고, 하락 시 손실 위험은 남습니다.

즉, “크게 한 번 벌기”보다 “꾸준히 받기”에 가까운 성향입니다.
"""

LEVERAGE_EXPLANATION = """
### 🎯 레버리지 ETF가 뭔가요?

**간단히 말해:**
- 일반 지수(예: 나스닥)를 **2배 또는 3배** 빠르게 움직이도록 한 상품
- 수익도 2-3배 크지만, **손실도 2-3배 크다**

---

### ⚠️ 가장 중요한 주의사항 - "일일 변동성"

**왜 중요한가?** 레버리지 ETF는 **매일매일** 정산되기 때문입니다.

**쉬운 예시:**
```
지수가 상승 +10% → 하락 -10% → 다시 +10% 반복된다면?

📊 일반 ETF(예: QQQ):
   전체 손실 없음 (원점 복귀) ✓

📊 레버리지 ETF(TQQQ 3배):
   매번 정산되면서 조금씩 손실 발생...
   결과: 원점에서 약간 손실 ✗
```

**일반인이 이해할 포인트:**
- 🔄 시장이 오르내릴 때마다 자산이 천천히 깎임
- 📉 특히 "횡보(요동치는 구간)"에서 손실 많음

---

### ✅ 레버리지 ETF 사용 가이드

| 좋은 경우 | 안 좋은 경우 |
|---|---|
| 📈 **지속적 상승** 추세 | 🔀 **횡보/요동** 구간 |
| ⏱️ **1개월~1년** 단기 | 📅 **5년 이상** 장기 보유 |
| 💪 **여유 자금** 소액 투자 | 💰 **전체 자산 대부분** |

---

### 💡 결론

레버리지 ETF는 **단기 고수익을 노리는 고급 투자자용** 상품입니다.

**초보자는:**
- 🟢 **안전 선택:** VOO, QQQ 같은 일반 지수 추적 ETF
- 🟡 **도전 가능:** 작은 비중만 레버리지 시도 (전체의 10-20%)
- 🔴 **피할 것:** 장기 보유 또는 전체 자산 투자
"""

PROFILE_METRIC_HELP = {
    "배당집중": (
        "보유 자산에 콜옵션을 매도해 프리미엄(월배당 등)을 받는 전략. "
        "급등 시 수익은 제한될 수 있습니다."
    ),
    "레버리지": (
        "2배 또는 3배 레버리지를 사용하는 상품. 수익이 크지만 손실도 크고 "
        "일일변동성 영향이 있으므로 주의가 필요합니다."
    ),
}

# 복수 선택 문항: 카테고리당 점수 상한 (무제한 선택 시 성향이 평탄해지는 것 방지)
PROFILE_MULTI_SELECT_MAX = 2
PROFILE_MULTI_SELECT_POINTS_PER_CHOICE = 2

# 질문 구조: 단일 선택 + Q7 복수 선택(최대 2개) + Q5 조건부
AI_PROFILE_QUESTIONS = {
    "Q1": {
        "text": (
            "1단계: 투자 가치관\n\n"
            "**Q1. 투자할 때 가장 중요한 목표는 무엇인가요?**"
        ),
        "choices": [
            "① 원금이 크게 흔들리지 않고, 꾸준한 수익이 나면 좋겠다.",
            "② 어렵게 고르지 않고, 시장 평균 흐름을 따라가고 싶다.",
            "③ 큰 수익보다 매달 들어오는 현금흐름이 더 중요하다.",
            "④ 손실 위험이 커도 높은 수익을 노리고 싶다.",
        ],
        "scores": [
            {"배당성장": 2, "배당집중": 2},
            {"지수추적": 3},
            {"배당집중": 2, "배당성장": 1},
            {"레버리지": 3},
        ],
    },
    "Q2": {
        "text": (
            "2단계: 포트폴리오 스타일\n\n"
            "**Q2. 투자 방법으로 더 마음이 가는 쪽은 어느 쪽인가요?**"
        ),
        "choices": [
            "① 여러 기업에 넓게 나눠 담아 위험을 줄이고 싶다.",
            "② 대표 기업들을 한 번에 담는 단순한 방식이 좋다.",
            "③ 매달 들어오는 수익이 있는 구성이 마음이 편하다.",
            "④ 변동이 커도 수익 기회가 크면 적극적으로 시도하고 싶다.",
        ],
        "scores": [
            {"배당성장": 1, "지수추적": 2},
            {"지수추적": 3},
            {"배당집중": 2, "배당성장": 1},
            {"배당집중": 1, "레버리지": 2, "배당성장": 1},
        ],
    },
    "Q3": {
        "text": (
            "3단계: 수익 매력\n\n"
            "**Q3. 어떤 수익 방식이 더 좋나요?**"
        ),
        "choices": [
            "① 시간이 지나며 자산 가격이 오르는 수익",
            "② 정기적으로 통장에 들어오는 현금 수익",
            "③ 성장 수익이 조금 더 중요하지만, 현금 수익도 원한다.",
            "④ 현금 수익이 조금 더 중요하지만, 성장 수익도 원한다.",
        ],
        "scores": [
            {"지수추적": 2, "레버리지": 1},
            {"배당성장": 2, "배당집중": 2},
            {"지수추적": 2, "배당성장": 1},
            {"배당성장": 2, "배당집중": 1, "지수추적": 1},
        ],
    },
    "Q4": {
        "text": (
            "4단계: 위험 감수 (중요)\n\n"
            "**Q4. 1,000만 원을 투자했는데 한 달 뒤 700만 원이 되었다면?**"
        ),
        "choices": [
            "① 너무 불안하다. 더 안정적인 쪽으로 옮기고 싶다.",
            "② 걱정되지만 당장 팔지 않고 더 지켜본다.",
            "③ 오히려 기회라고 보고 추가 매수도 생각한다.",
        ],
        "scores": [
            {"배당성장": 3, "배당집중": 1},
            {"지수추적": 2, "배당성장": 1},
            {"지수추적": 2, "레버리지": 2},
        ],
        "follow_up": {2: "Q5"},
    },
    "Q5": {
        "text": (
            "**Q5. (심화) 가격이 더 크게 오르내리는 고위험 상품도 감당할 수 있나요?**"
        ),
        "choices": [
            "① 어렵다. 변동이 덜한 상품이 좋다.",
            "② 가능하다. 위험이 커도 높은 수익을 노려볼 수 있다.",
        ],
        "scores": [
            {"지수추적": 1},
            {"레버리지": 3},
        ],
    },
    "Q6": {
        "text": (
            "5단계: 시장 vs 현금흐름\n\n"
            "**Q6. 미국 주식에 투자한다면 어떤 방식이 더 편한가요?**"
        ),
        "choices": [
            "① 시장 전체에 넓게 나눠 담는 방식",
            "② 배당/월수익처럼 현금흐름을 챙기는 방식",
        ],
        "scores": [
            {"지수추적": 3},
            {"배당성장": 2, "배당집중": 1},
        ],
    },
    "Q7": {
        "text": (
            "6단계: 관심 상품 유형\n\n"
            "**Q7. 아래 중 가장 관심 있는 투자 유형은 무엇인가요?**"
        ),
        "choices": [
            "① 대표 기업들을 넓게 담아 시장 흐름 따라가기",
            "② 배당 중심으로 정기 수익 받기",
            "③ 매달 현금흐름 중심(배당집중 포함)",
            "④ 변동이 큰 고위험·고수익 상품",
        ],
        "scores": [
            {"지수추적": 3},
            {"배당성장": 3},
            {"배당집중": 3},
            {"레버리지": 3},
        ],
        "follow_up": {3: "Q7_SUB"},
    },
    "Q7_SUB": {
        "text": (
            "**Q7-추가. (심화) 레버리지 ETF를 이해하고 있나요?**\n\n"
            "고위험 상품 중 하나인 '레버리지 ETF'에 대해 얼마나 알고 계신가요?"
        ),
        "choices": [
            "① 레버리지 ETF가 뭔지 잘 몰라요. 자세히 설명해주세요.",
            "② 대충 2배~3배 움직인다고만 알아요.",
            "③ 일일 변동성과 음의 복리 개념을 알고 있어요.",
            "④ 레버리지 ETF의 위험성을 충분히 이해하고 있어요.",
        ],
        "scores": [
            {"레버리지": 0},
            {"레버리지": 1},
            {"레버리지": 2},
            {"레버리지": 2},
        ],
    },
    "Q8": {
        "text": "7단계: 투자 자금\n\n**Q8. 현재 투자 가능한 자금 상황은?**",
        "choices": [
            "① 목돈이 있고 매달 추가 투자도 가능해요",
            "② 목돈은 있지만 매달 추가 투자는 어려워요",
            "③ 목돈은 없고 매달 일정액만 투자 가능해요",
        ],
        "scores": [{}, {}, {}],
    },
}

AI_PROFILE_BASE_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q6", "Q7", "Q8"]


def get_profile_question_order(responses):
    """응답 상태에 따른 질문 순서(Q5 조건부 포함, Q7_SUB 조건부)."""
    order = list(AI_PROFILE_BASE_QUESTIONS)
    if responses.get("Q4") == 2:
        q4_idx = order.index("Q4")
        order.insert(q4_idx + 1, "Q5")
    # Q7에서 "④ 변동이 큰 고위험·고수익 상품" (인덱스 3)을 선택했으면 Q7_SUB 추가
    if responses.get("Q7") == 3:
        q7_idx = order.index("Q7")
        order.insert(q7_idx + 1, "Q7_SUB")
    return order


def count_total_profile_questions(responses):
    """진행률 표시용 전체 문항 수."""
    return len(get_profile_question_order(responses))


def get_next_question(current_q, responses):
    """현재 질문 다음에 표시할 질문을 결정합니다."""
    order = get_profile_question_order(responses)
    try:
        idx = order.index(current_q)
        if idx + 1 < len(order):
            return order[idx + 1]
    except ValueError:
        pass
    return None


def _apply_choice_score_dict(scores, choice_scores):
    for category, points in choice_scores.items():
        if category in scores:
            scores[category] += points


def _is_multi_select_question(q_key):
    return AI_PROFILE_QUESTIONS.get(q_key, {}).get("type") == "multi"


def calculate_profile_scores(responses):
    """사용자 응답을 기반으로 4개 투자 성향 카테고리별 점수(%)를 계산합니다."""
    scores = {cat: 0 for cat in AI_PROFILE_CATEGORIES}

    for q_key, answer in responses.items():
        if q_key not in AI_PROFILE_QUESTIONS:
            continue

        q_data = AI_PROFILE_QUESTIONS[q_key]
        score_table = q_data.get("scores", [])

        if _is_multi_select_question(q_key):
            indices = answer if isinstance(answer, list) else []
            max_select = q_data.get("max_select", PROFILE_MULTI_SELECT_MAX)
            for choice_idx in indices[:max_select]:
                if 0 <= choice_idx < len(score_table):
                    _apply_choice_score_dict(scores, score_table[choice_idx])
        else:
            if answer is None or not isinstance(answer, int):
                continue
            if answer >= len(score_table):
                continue
            _apply_choice_score_dict(scores, score_table[answer])

    if responses.get("Q5") == 0:
        scores["레버리지"] = 0

    total = sum(scores.values())
    if total == 0:
        profile = {cat: round(100 / len(AI_PROFILE_CATEGORIES), 2) for cat in AI_PROFILE_CATEGORIES}
    else:
        profile = {cat: round((scores[cat] / total) * 100, 2) for cat in AI_PROFILE_CATEGORIES}

    leverage_from_q5 = responses.get("Q5") == 1 if "Q5" in responses else None
    leverage_from_q7 = responses.get("Q7") == 3  # Q7에서 레버리지(④) 선택 여부
    if leverage_from_q5 is not None:
        profile["_leverage_allowed"] = leverage_from_q5
    else:
        profile["_leverage_allowed"] = scores["레버리지"] > 0 or leverage_from_q7

    return profile


def get_question_text_and_choices(q_key):
    """질문 키에 해당하는 텍스트와 선택지를 반환합니다."""
    if q_key not in AI_PROFILE_QUESTIONS:
        return None, None

    q_data = AI_PROFILE_QUESTIONS[q_key]
    return q_data["text"], q_data["choices"]


GEMINI_PROFILE_SYSTEM_PROMPT = (
    "투자 성향 질문 생성. 쉬운 말로 다음 질문 1개와 보기 4개를 JSON만 반환. "
    "직접 관련 항목만 질문: 투자목표(자산성장/현금흐름), 위험감내도, 투자기간, 레버리지 허용, "
    "배당/현금흐름 선호, 지수추적/배당성장/배당집중 선호, 투자경험, 자금상황(목돈/매달적립/둘다). "
    "메인 질문은 총 8문항으로 구성하고, 한 번에 한 문항씩 순서대로 생성. "
    "문항별 주제는 1번 투자목표, 2번 위험감내도, 3번 투자기간/경험, 4번 레버리지 허용, "
    "5번 배당/현금흐름 관심, 6번 지수추적 선호, 7번 상품 이해도, 8번 자금상황으로 고정. "
    "다른 주제는 절대 묻지 말 것. "
    "형식: {\"question\": str, \"choices\": [str,str,str,str], \"is_complete\": bool}. "
    "8문항 전 is_complete=false, 8문항 후 is_complete=true."
)

GEMINI_PROFILE_MODEL = "gemini-flash-lite-latest"
GEMINI_MIN_PROFILE_QUESTIONS = 8
GEMINI_MAX_PROFILE_QUESTIONS = 8
GEMINI_MAX_SUBQUESTIONS_PER_MAIN = 2
GEMINI_SUBQUESTION_RULES = {
    4: "레버리지 허용 답변이면 2배 vs 3배 감수 가능 여부, 나스닥 vs S&P500 선호를 최대 2개까지 확인. 레버리지 거부 답변이면 하위 질문 없음.",
    5: "배당 관심 답변이면 배당성장 vs 고배당 중심, 월배당 vs 분기배당 선호를 최대 2개까지 확인. 배당 관심 없는 답변이면 하위 질문 없음.",
    6: "지수추적 선호 답변이면 기술주 집중(QQQ) vs 전체 분산(VTI) vs 균형(VOO)을 최대 2개까지 확인. 지수추적 선호가 아니면 하위 질문 없음.",
    8: "목돈 있음 답변이면 거치식 vs 혼합식(목돈+적립)을 최대 2개까지 확인. 목돈 없다는 답변이면 하위 질문 없음.",
}
PROFILE_KEYWORD_EXPLANATIONS = [
    {
        "key": "leverage",
        "keywords": ["레버리지", "2배", "3배", "TQQQ", "UPRO", "SOXL"],
        "message": (
            "💡 레버리지 ETF란?\n"
            "간단히 말해:\n"
            "- 일반 지수(예: 나스닥)를 2배 또는 3배\n"
            "  빠르게 움직이도록 한 상품\n"
            "- 수익도 2-3배 크지만, 손실도 2-3배 크다\n\n"
            "⚠️ 가장 중요한 주의사항 - \"일일 변동성\"\n"
            "레버리지 ETF는 매일매일 정산되기 때문입니다.\n\n"
            "쉬운 예시:\n"
            "지수가 상승 +10% → 하락 -10% → 다시 +10% 반복된다면?\n\n"
            "일반 ETF (예: QQQ):\n"
            "전체 손실 없음 (원점 복귀) ✓\n\n"
            "레버리지 ETF (TQQQ 3배):\n"
            "매번 정산되면서 조금씩 손실 발생...\n"
            "결과: 원점에서 약간 손실 ✗\n\n"
            "😱 음의 복리란?\n"
            "이게 바로 \"음의 복리\"예요.\n\n"
            "일반 복리는 수익이 쌓이는 것:\n"
            "100만원 → +10% → 110만원\n"
            "110만원 → +10% → 121만원 (이익이 이익을 낳음)\n\n"
            "음의 복리는 손실이 쌓이는 것:\n"
            "100만원 → +30% → 130만원\n"
            "130만원 → -30% → 91만원\n"
            "(원금 100만원인데 91만원으로 줄어듦!)\n\n"
            "레버리지는 이 효과가 3배라\n"
            "횡보장에서 특히 손실이 커요."
        ),
    },
    {
        "key": "covered_call",
        "keywords": ["커버드콜", "배당집중", "JEPI", "JEPQ", "QYLD", "월배당"],
        "message": COVERED_CALL_EXPLANATION,
    },
    {
        "key": "dividend_growth",
        "keywords": ["배당성장", "배당", "SCHD", "DGRO", "VYM"],
        "message": (
            "💡 배당이란?\n"
            "배당이란 기업이 이익의 일부를 주주에게\n"
            "현금으로 나눠주는 것이에요.\n"
            "예를 들어 삼성전자 주식 100주를 가지고 있으면\n"
            "1년에 몇 만원씩 통장으로 입금되는 방식이에요."
        ),
    },
]


def _extract_json_object(text):
    if not text:
        raise ValueError("Gemini 응답이 비어 있습니다.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini JSON 응답을 찾지 못했습니다.")
    return json.loads(cleaned[start:end + 1])


def _get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if genai is None or not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_PROFILE_MODEL)


def _render_profile_keyword_explanations(text_parts, context_key):
    shown = st.session_state.setdefault("profile_keyword_explanations_shown", set())
    contexts = st.session_state.setdefault("profile_keyword_explanation_contexts", {})
    combined_text = " ".join(str(part) for part in text_parts if part)
    matched_keys = set()

    for explanation in PROFILE_KEYWORD_EXPLANATIONS:
        if any(keyword in combined_text for keyword in explanation["keywords"]):
            matched_keys.add(explanation["key"])

    if "covered_call" in matched_keys and "dividend_growth" in matched_keys:
        matched_keys.remove("dividend_growth")

    for explanation in PROFILE_KEYWORD_EXPLANATIONS:
        if explanation["key"] in matched_keys:
            if explanation["key"] not in shown:
                shown.add(explanation["key"])
                contexts[explanation["key"]] = context_key
            elif explanation["key"] not in contexts:
                contexts[explanation["key"]] = context_key

            if contexts.get(explanation["key"]) == context_key:
                st.info(explanation["message"])


def _profile_answers_for_gemini(responses, questions, subquestions=None):
    answers = []
    subquestions = subquestions or {}
    for idx, question_data in enumerate(questions, start=1):
        q_key = f"G{idx}"
        if q_key not in responses:
            continue
        choice_idx = responses[q_key]
        choices = question_data.get("choices", [])
        choice_text = choices[choice_idx] if isinstance(choice_idx, int) and 0 <= choice_idx < len(choices) else ""
        answer_data = {
            "question_number": idx,
            "question": question_data.get("question", ""),
            "answer": choice_text,
        }
        sub_state = subquestions.get(q_key, {})
        sub_answers = []
        for sub_idx, sub_data in enumerate(sub_state.get("items", []), start=1):
            sub_answer_idx = sub_data.get("answer")
            sub_choices = sub_data.get("choices", [])
            if isinstance(sub_answer_idx, int) and 0 <= sub_answer_idx < len(sub_choices):
                sub_answers.append({
                    "question_number": sub_idx,
                    "question": sub_data.get("question", ""),
                    "answer": sub_choices[sub_answer_idx],
                })
        if sub_answers:
            answer_data["sub_answers"] = sub_answers
        answers.append(answer_data)
    return answers


def _generate_gemini_profile_question(answers, next_question_number):
    model = _get_gemini_model()
    if model is None:
        raise RuntimeError("Gemini API를 사용할 수 없습니다.")

    prompt = (
        f"{GEMINI_PROFILE_SYSTEM_PROMPT}\n"
        f"문항번호: {next_question_number}\n"
        f"이전답변: {json.dumps(answers, ensure_ascii=False, separators=(',', ':'))}"
    )
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    data = _extract_json_object(getattr(response, "text", ""))
    question = str(data.get("question", "")).strip()
    choices = data.get("choices", [])
    if not question or not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("Gemini 질문 형식이 올바르지 않습니다.")
    return {
        "question": question,
        "choices": [str(choice).strip() for choice in choices],
        "is_complete": bool(data.get("is_complete", False)) and next_question_number > GEMINI_MIN_PROFILE_QUESTIONS,
    }


def _generate_gemini_profile_subquestion(answers, main_question_number, main_question, main_answer, sub_answers):
    model = _get_gemini_model()
    if model is None:
        raise RuntimeError("Gemini API를 사용할 수 없습니다.")

    prompt = (
        "투자 성향 분석의 같은 페이지 하위 질문 필요 여부를 판단하고 JSON만 반환. "
        "하위 질문은 메인 4번, 5번, 6번, 8번에서만 가능. "
        "이전 답변 맥락과 현재 메인 답변을 보고, 아래 현재 문항 규칙에 맞을 때만 하위 질문 1개와 보기 4개를 생성. "
        "현재 메인 답변이 규칙의 조건에 해당하지 않으면 반드시 needs_sub_question=false. "
        "이미 하위 답변으로 충분히 확인된 내용은 반복 질문하지 말 것. "
        "하위 질문은 현재 메인 문항당 최대 2개까지만 필요하다고 판단. "
        "필요 없으면 needs_sub_question=false. "
        "형식: {\"needs_sub_question\": bool, \"question\": str, \"choices\": [str,str,str,str]}.\n"
        f"현재문항번호: {main_question_number}\n"
        f"현재문항규칙: {GEMINI_SUBQUESTION_RULES.get(main_question_number, '하위 질문 없음')}\n"
        f"이전답변: {json.dumps(answers, ensure_ascii=False, separators=(',', ':'))}\n"
        f"현재메인질문: {main_question}\n"
        f"현재메인답변: {main_answer}\n"
        f"이미받은하위답변: {json.dumps(sub_answers, ensure_ascii=False, separators=(',', ':'))}"
    )
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    data = _extract_json_object(getattr(response, "text", ""))
    if not bool(data.get("needs_sub_question", False)):
        return {"needs_sub_question": False}

    question = str(data.get("question", "")).strip()
    choices = data.get("choices", [])
    if not question or not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("Gemini 하위 질문 형식이 올바르지 않습니다.")
    return {
        "needs_sub_question": True,
        "question": question,
        "choices": [str(choice).strip() for choice in choices],
    }


def _generate_gemini_profile_result(answers):
    model = _get_gemini_model()
    if model is None:
        raise RuntimeError("Gemini API를 사용할 수 없습니다.")

    prompt = (
        "답변으로 최종 투자 성향을 JSON만 반환. "
        "profile_weights는 지수추적/배당성장/배당집중/레버리지 합계 100, "
        "leverage_allowed boolean, funding_situation은 has_lump_sum/can_monthly_invest boolean. "
        "buy_mode는 목돈만 있으면 거치식, 매달 적립만 가능하면 적립식, 둘 다면 혼합식.\n"
        f"답변: {json.dumps(answers, ensure_ascii=False, separators=(',', ':'))}"
    )
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    data = _extract_json_object(getattr(response, "text", ""))
    raw_weights = data.get("profile_weights", {})
    weights = {cat: max(0.0, float(raw_weights.get(cat, 0))) for cat in AI_PROFILE_CATEGORIES}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Gemini 최종 비율이 올바르지 않습니다.")
    weights = {cat: round((weights[cat] / total) * 100, 2) for cat in AI_PROFILE_CATEGORIES}
    weights["_leverage_allowed"] = bool(data.get("leverage_allowed", weights.get("레버리지", 0) > 0))
    funding = data.get("funding_situation", {})
    return weights, {
        "has_lump_sum": bool(funding.get("has_lump_sum", True)),
        "can_monthly_invest": bool(funding.get("can_monthly_invest", True)),
        "buy_mode": str(data.get("buy_mode", "")).strip(),
    }


def _generate_gemini_step4_analysis(profile_weights, selected_etfs, investment_method, target_amount, years):
    model = _get_gemini_model()
    if model is None:
        raise RuntimeError("Gemini API를 사용할 수 없습니다.")

    prompt = f"""당신은 주식 투자 전문가이자 
친절한 재테크 멘토입니다.

주식을 한 번도 해본 적 없는 완전 초보자에게
아래 내용을 설명해주세요:

1. 이 사람의 투자 성향이 어떤 타입인지
   (예: 공격적 성장형, 안정적 배당형 등)

2. 왜 이 매수 방식 (거치식/적립식/혼합식)이
   이 사람에게 맞는지 이유 설명

3. 이 투자 기간 동안 가져야 할 마인드셋
   (예: 단기 하락에 흔들리지 않는 법,
    장기 복리의 힘 등)

4. 초보 투자자가 이 포트폴리오로 
   투자할 때 알아야 할 것들

전문용어는 그대로 쓰되
반드시 괄호로 쉬운 설명 추가.
예: ETF(여러 주식을 한 번에 담은 바구니)
친근하고 따뜻한 말투.
너무 길지 않게 핵심만.

성향 분석 결과: {profile_weights}
추천 ETF: {selected_etfs}
매수 방식: {investment_method}
목표 금액: {target_amount}원
투자 기간: {years}년"""

    response = model.generate_content(prompt)
    result = getattr(response, "text", "").strip()
    if not result:
        raise ValueError("Gemini 응답이 비어 있습니다.")
    return result


def _default_step4_analysis_text(profile_weights, selected_etfs, investment_method, target_amount, years):
    top_profile = max(
        ((cat, profile_weights.get(cat, 0)) for cat in AI_PROFILE_CATEGORIES),
        key=lambda item: item[1],
    )[0]
    return (
        f"현재 성향은 **{top_profile} 중심형**에 가깝습니다. "
        f"추천 ETF(여러 주식을 한 번에 담은 바구니)는 {', '.join(selected_etfs)}이며, "
        f"매수 방식은 **{investment_method}**입니다.\n\n"
        f"목표 금액은 {target_amount:,}원, 투자 기간은 {years:g}년입니다. "
        "초보 투자자는 단기 하락에 너무 흔들리기보다 정해 둔 기간 동안 꾸준히 유지하는 마음가짐이 중요합니다. "
        "ETF도 원금 손실 가능성이 있으므로 비상금은 따로 두고, 투자 금액과 비중을 정기적으로 확인해 주세요."
    )


def _funding_situation_to_q8(funding):
    has_lump_sum = funding.get("has_lump_sum", True)
    can_monthly = funding.get("can_monthly_invest", True)
    if has_lump_sum and can_monthly:
        return 0
    if has_lump_sum and not can_monthly:
        return 1
    return 2


def _activate_fixed_profile_fallback():
    st.session_state["profile_ai_fallback"] = True
    st.session_state["current_question"] = "Q1"
    st.session_state["profile_responses"] = {}
    st.session_state["gemini_profile_questions"] = []
    st.session_state["gemini_profile_subquestions"] = {}
    st.session_state["profile_keyword_explanations_shown"] = set()
    st.session_state["profile_keyword_explanation_contexts"] = {}


def render_profile_metrics(profile_weights):
    """투자 성향 비율과 배당집중 설명을 표시합니다."""
    cols = st.columns(len(AI_PROFILE_CATEGORIES))
    for i, cat in enumerate(AI_PROFILE_CATEGORIES):
        with cols[i]:
            st.metric(
                cat,
                f"{profile_weights.get(cat, 0)}%",
                help=PROFILE_METRIC_HELP.get(cat),
            )
    with st.expander("💡 배당집중이란? (처음이시라면 읽어보세요)"):
        st.markdown(COVERED_CALL_EXPLANATION)
    with st.expander("💡 레버리지 ETF란? (처음이시라면 읽어보세요)"):
        st.markdown(LEVERAGE_EXPLANATION)

# ============================================================
# 섹션 3. ETF 추천 함수
# ============================================================

CATEGORY_TO_ETFS = {
    "지수추적": ["VOO", "QQQ", "VTI"],
    "배당성장": ["SCHD", "DGRO", "VYM"],
    "배당집중": ["JEPI", "JEPQ", "QYLD"],
    "레버리지": ["TQQQ", "UPRO", "SOXL", "QLD", "SSO"],
}


def _normalize_profile_weights(profile_weights):
    """4개 투자 성향 카테고리 비중을 0~1로 정규화합니다."""
    regular_weights = {cat: float(profile_weights.get(cat, 0)) for cat in AI_PROFILE_CATEGORIES}
    if not profile_weights.get("_leverage_allowed", True):
        regular_weights["레버리지"] = 0
    total = sum(regular_weights.values())
    if total <= 0:
        raise ValueError("프로필 가중치 합계는 0보다 커야 합니다.")
    return {cat: regular_weights[cat] / total for cat in AI_PROFILE_CATEGORIES}


def _adjust_category_counts(normalized, top_n):
    """성향 비율에 따라 카테고리별 ETF 개수를 배분합니다."""
    category_counts = {
        cat: max(0, int(round(normalized[cat] * top_n)))
        for cat in AI_PROFILE_CATEGORIES
    }
    total_allocated = sum(category_counts.values())
    if total_allocated < top_n:
        top_category = max(AI_PROFILE_CATEGORIES, key=lambda c: normalized[c])
        category_counts[top_category] += top_n - total_allocated
    elif total_allocated > top_n:
        lowest_category = min(AI_PROFILE_CATEGORIES, key=lambda c: normalized[c])
        category_counts[lowest_category] = max(
            0, category_counts[lowest_category] - (total_allocated - top_n)
        )
    return category_counts


def _cap_leverage_weights(weights, max_share=LEVERAGE_MAX_PORTFOLIO_WEIGHT):
    """레버리지 ETF 합산 비중이 상한을 넘으면 조정합니다."""
    leverage_tickers = [t for t in weights if ETF_DATA.get(t, {}).get("레버리지")]
    if not leverage_tickers:
        return weights

    lev_total = sum(weights[t] for t in leverage_tickers)
    if lev_total <= max_share:
        return weights

    scale = max_share / lev_total
    new_weights = dict(weights)
    freed = 0.0
    for ticker in leverage_tickers:
        old = new_weights[ticker]
        new_weights[ticker] = old * scale
        freed += old - new_weights[ticker]

    non_leverage = [t for t in weights if t not in leverage_tickers]
    if non_leverage and freed > 0:
        bonus = freed / len(non_leverage)
        for ticker in non_leverage:
            new_weights[ticker] += bonus

    total = sum(new_weights.values())
    if total <= 0:
        return weights
    return {ticker: round(w / total, 4) for ticker, w in new_weights.items()}


def recommend_etfs_with_weights(profile_weights, top_n=5):
    """투자 성향 비율(4축)을 받아 ETF와 비중을 함께 추천합니다.

    반환: {ticker: weight} 딕셔너리
    """
    leverage_allowed = profile_weights.get("_leverage_allowed", True)
    normalized = _normalize_profile_weights(profile_weights)
    category_counts = _adjust_category_counts(normalized, top_n)
    sorted_categories = sorted(
        AI_PROFILE_CATEGORIES, key=lambda c: normalized[c], reverse=True
    )

    selected = []
    etf_to_category = {}

    for category in sorted_categories:
        needed = category_counts[category]
        if needed <= 0:
            continue
        if category == "레버리지" and not leverage_allowed:
            continue

        for ticker in CATEGORY_TO_ETFS.get(category, [])[:needed]:
            if ticker not in selected and len(selected) < top_n:
                selected.append(ticker)
                etf_to_category[ticker] = category

    if len(selected) < top_n:
        all_etfs = []
        for cat_etfs in CATEGORY_TO_ETFS.values():
            all_etfs.extend(cat_etfs)
        for ticker in all_etfs:
            if ticker not in selected and len(selected) < top_n:
                selected.append(ticker)
                etf_to_category[ticker] = ETF_DATA[ticker]["카테고리"]

    weights = {}
    for ticker in selected:
        category = etf_to_category[ticker]
        same_category = [t for t in selected if etf_to_category[t] == category]
        weights[ticker] = (
            normalized[category] / len(same_category)
            if same_category
            else 1 / len(selected)
        )

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {ticker: round(w / total_weight, 4) for ticker, w in weights.items()}

    weights = _cap_leverage_weights(weights)
    return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))


def recommend_etfs(profile_weights, top_n=5):
    """투자 성향 비율을 받아 ETF를 추천합니다."""
    weights = recommend_etfs_with_weights(profile_weights, top_n)
    return list(weights.keys())[:top_n]


def _get_etf_annual_return(ticker):
    """ETF 예상 연 수익률을 계산합니다."""
    mu, _ = _get_ticker_return_profile(ticker)
    return max(0, mu * 100)  # 음수 제거


def _get_risk_stars(ticker):
    """ETF 위험도를 별점(0~5)로 반환합니다."""
    _, sigma = _get_ticker_return_profile(ticker)
    # 변동성 범위: 0.05 (1별) ~ 0.25 (5별)
    if sigma < 0.07:
        return 1
    elif sigma < 0.12:
        return 2
    elif sigma < 0.15:
        return 3
    elif sigma < 0.20:
        return 4
    else:
        return 5


ETF_SUMMARY_DESCRIPTIONS = {
    "VOO": "미국 대형주 500개 지수 추종",
    "QQQ": "미국 기술 대형주 100개 지수 추종",
    "VTI": "미국 주식 시장 전체 지수 추종",
    "SCHD": "미국 배당성장주 묶음",
    "DGRO": "배당이 늘어나는 기업 위주",
    "VYM": "배당 수익 위주",
    "JEPI": "대형주 + 배당집중 · 월배당",
    "JEPQ": "기술 대형주 + 배당집중 · 월배당",
    "QYLD": "기술 지수 + 배당집중 · 월배당",
    "QLD": "기술 지수 · 2배 레버리지",
    "TQQQ": "기술 지수 · 3배 레버리지",
    "SSO": "미국 대형주 지수 · 2배 레버리지",
    "UPRO": "미국 대형주 지수 · 3배 레버리지",
    "SOXL": "반도체 · 3배 레버리지",
}

# 카드 UI 등에서 사용 (ETF_SUMMARY_DESCRIPTIONS와 동일)
ETF_CARD_DESCRIPTIONS = ETF_SUMMARY_DESCRIPTIONS

CATEGORY_DISPLAY_LABELS = {
    "지수추적": "지수추종형",
    "배당성장": "배당성장형",
    "배당집중": "배당집중형",
    "레버리지": "레버리지형",
}

ETF_DETAIL_GUIDE = {
    "VOO": {
        "intro": "미국을 대표하는 대형 기업 500개에 한 번에 분산 투자하는 상품입니다. 미국 경제 성장 흐름을 비교적 단순하게 따라가고 싶을 때 자주 선택됩니다.",
        "recommended_for": [
            "ETF를 처음 시작하는 분",
            "종목 선택보다 시장 전체 흐름에 투자하고 싶은 분",
        ],
        "top_holdings": ["Apple", "Microsoft", "NVIDIA"],
        "caution": "시장 전체가 하락하면 함께 떨어질 수 있어 단기보다 장기 관점이 더 적합합니다.",
    },
    "QQQ": {
        "intro": "미국 기술 대형주 중심의 대표 지수를 추종합니다. 성장성이 큰 만큼 변동성도 더 큰 편입니다.",
        "recommended_for": [
            "기술 성장주 비중을 높이고 싶은 분",
            "중장기 성장 수익을 우선하는 분",
        ],
        "top_holdings": ["Microsoft", "Apple", "NVIDIA"],
        "caution": "기술주 조정 구간에서는 하락 폭이 크게 나올 수 있어 분할매수가 유리합니다.",
    },
    "VTI": {
        "intro": "미국 주식시장 거의 전체를 담아 매우 넓게 분산되는 상품입니다. 한 상품으로 시장 전반에 투자하기 쉽습니다.",
        "recommended_for": [
            "최대한 넓게 분산하고 싶은 분",
            "장기 적립식 투자를 선호하는 분",
        ],
        "top_holdings": ["Apple", "Microsoft", "NVIDIA"],
        "caution": "시장 전반이 약세일 때는 방어력이 제한될 수 있습니다.",
    },
    "SCHD": {
        "intro": "재무 건전성과 배당 지속성이 높은 기업 위주로 구성된 배당 성장형 상품입니다.",
        "recommended_for": [
            "배당 + 장기 성장의 균형을 원하는 분",
            "현금흐름도 챙기고 싶은 분",
        ],
        "top_holdings": ["Coca-Cola", "Home Depot", "Cisco"],
        "caution": "고성장 구간에서는 기술 성장주 대비 수익이 낮게 보일 수 있습니다.",
    },
    "DGRO": {
        "intro": "배당이 꾸준히 늘어난 기업을 중심으로 구성되어 장기적으로 배당 성장 흐름을 노리는 상품입니다.",
        "recommended_for": [
            "배당 성장 기업에 투자하고 싶은 분",
            "안정성과 성장의 균형을 중시하는 분",
        ],
        "top_holdings": ["Microsoft", "Apple", "JPMorgan"],
        "caution": "배당 성장 전략 특성상 급등장에서 상대 수익이 낮을 수 있습니다.",
    },
    "VYM": {
        "intro": "배당수익률이 상대적으로 높은 미국 대형주에 분산 투자하는 고배당 성격의 상품입니다.",
        "recommended_for": [
            "정기적인 배당 흐름을 선호하는 분",
            "변동성을 조금 낮추고 싶은 분",
        ],
        "top_holdings": ["Broadcom", "JPMorgan", "Exxon Mobil"],
        "caution": "금리와 경기 흐름에 따라 배당주가 약세를 보일 수 있습니다.",
    },
    "JEPI": {
        "intro": "주식 포지션에 옵션 전략을 결합해 월 단위 현금흐름을 추구하는 배당집중 상품입니다.",
        "recommended_for": [
            "매달 들어오는 수익 흐름이 중요한 분",
            "큰 급등보다 안정적 흐름을 선호하는 분",
        ],
        "top_holdings": ["Trane", "Progressive", "Meta"],
        "caution": "강한 상승장에서는 주가 상승 이익 일부를 놓칠 수 있습니다.",
    },
    "JEPQ": {
        "intro": "기술 대형주 기반에 배당집중 전략을 적용해 현금흐름을 강화한 상품입니다.",
        "recommended_for": [
            "기술주 노출 + 월수익을 같이 원하는 분",
            "배당성장보다 높은 변동성도 감수 가능한 분",
        ],
        "top_holdings": ["Microsoft", "Apple", "NVIDIA"],
        "caution": "기술주 변동성과 옵션 구조 영향으로 가격이 빠르게 출렁일 수 있습니다.",
    },
    "QYLD": {
        "intro": "기술 지수 기반 배당집중 전략으로 월수익 흐름을 우선하는 성격의 상품입니다.",
        "recommended_for": [
            "현금흐름 우선 투자자",
            "가격 상승보다 월배당 성향을 선호하는 분",
        ],
        "top_holdings": ["Microsoft", "Apple", "NVIDIA"],
        "caution": "장기 총수익은 성장주 중심 상품보다 낮아질 수 있습니다.",
    },
    "QLD": {
        "intro": "기술 지수의 일일 움직임을 2배로 추종하는 레버리지 상품입니다.",
        "recommended_for": [
            "중단기 공격적 비중 조절이 가능한 분",
            "높은 변동성을 감수할 수 있는 분",
        ],
        "top_holdings": ["Microsoft", "Apple", "NVIDIA"],
        "caution": "횡보장에서는 복리 효과로 기대보다 수익이 낮아질 수 있습니다.",
    },
    "TQQQ": {
        "intro": "기술 지수의 일일 수익률을 3배로 추종하는 고위험 레버리지 상품입니다.",
        "recommended_for": [
            "공격적 운용 경험이 있는 분",
            "단기 변동성 대응이 가능한 분",
        ],
        "top_holdings": ["Microsoft", "Apple", "NVIDIA"],
        "caution": "하락장에서 손실이 매우 빠르게 커질 수 있어 장기 방치에 특히 주의해야 합니다.",
    },
    "SSO": {
        "intro": "미국 대형주 지수의 일일 움직임을 2배로 추종하는 레버리지 상품입니다.",
        "recommended_for": [
            "시장 상승 관점에서 공격 비중을 일부 두고 싶은 분",
            "리스크 관리 계획이 있는 분",
        ],
        "top_holdings": ["Apple", "Microsoft", "NVIDIA"],
        "caution": "큰 하락 구간에서는 원금 회복에 오래 걸릴 수 있습니다.",
    },
    "UPRO": {
        "intro": "미국 대형주 지수의 일일 움직임을 3배로 추종하는 초고변동 레버리지 상품입니다.",
        "recommended_for": [
            "고위험·고수익 전략을 명확히 이해한 분",
            "짧은 주기로 리밸런싱 가능한 분",
        ],
        "top_holdings": ["Apple", "Microsoft", "NVIDIA"],
        "caution": "단기간 큰 손실 가능성이 높아 포트폴리오 소수 비중으로 제한하는 것이 일반적입니다.",
    },
    "SOXL": {
        "intro": "미국 반도체 지수의 일일 수익률을 3배로 추종하는 초고변동 레버리지 상품입니다.",
        "recommended_for": [
            "반도체 업황에 대한 강한 관점을 가진 분",
            "매우 높은 변동성 대응이 가능한 분",
        ],
        "top_holdings": ["NVIDIA", "Broadcom", "AMD"],
        "caution": "섹터 집중 + 3배 레버리지 구조라 가격 변동이 매우 큽니다.",
    },
}


def _format_risk_label(star_count):
    if star_count <= 2:
        return "낮음"
    if star_count == 3:
        return "중간"
    return "높음"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _get_usdkrw_rate():
    if yf is None:
        return 1400.0
    try:
        fx = yf.Ticker("KRW=X")
        hist = fx.history(period="5d", auto_adjust=True)
        if hist.empty:
            return 1400.0
        return float(hist["Close"].iloc[-1])
    except Exception:
        return 1400.0


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _get_live_etf_snapshot(ticker):
    """Yahoo Finance 기반 실시간 스냅샷(현재가, 배당률, 과거 연환산 수익률)."""
    if yf is None:
        return {}

    try:
        etf = yf.Ticker(ticker)
        info = etf.info or {}
        fast_info = getattr(etf, "fast_info", {}) or {}

        price_usd = (
            fast_info.get("lastPrice")
            or fast_info.get("regularMarketPrice")
            or info.get("regularMarketPrice")
        )
        dividend_yield = info.get("yield")
        if dividend_yield is None:
            dividend_yield = info.get("dividendYield")

        annual_return_pct = None
        hist = etf.history(period="10y", auto_adjust=True)
        if hist is not None and not hist.empty and len(hist) >= 252 * 3:
            first_close = float(hist["Close"].iloc[0])
            last_close = float(hist["Close"].iloc[-1])
            elapsed_years = max((hist.index[-1] - hist.index[0]).days / 365.25, 1e-6)
            if first_close > 0 and last_close > 0 and elapsed_years > 0:
                annual_return_pct = ((last_close / first_close) ** (1 / elapsed_years) - 1) * 100

        return {
            "price_usd": float(price_usd) if price_usd is not None else None,
            "dividend_yield_pct": float(dividend_yield * 100) if dividend_yield is not None else None,
            "annual_return_pct": annual_return_pct,
        }
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _get_etf_dividend_yield(ticker):
    if yf is None:
        return None

    try:
        dividend_yield = (yf.Ticker(ticker).info or {}).get("dividendYield")
        if dividend_yield is None:
            return None
        dividend_yield = float(dividend_yield)
        return dividend_yield / 100 if dividend_yield > 1 else dividend_yield
    except Exception:
        return None


def _etf_card_html(ticker, is_selected=False):
    """ETF 카드 HTML (한 줄, 들여쓰기 없음 — markdown 코드블록 방지)."""
    info = ETF_DATA.get(ticker, {})
    annual_return = _get_etf_annual_return(ticker)
    risk_stars = _get_risk_stars(ticker)
    border_color = "#FF6B6B" if is_selected else "#CCCCCC"
    border_width = "3px" if is_selected else "1px"
    bg_color = "#F0F8FF" if is_selected else "#FFFFFF"
    selected_badge = (
        '<div style="font-size:11px;font-weight:bold;color:#FF6B6B;margin-bottom:6px;">✓ 선택됨</div>'
        if is_selected
        else ""
    )
    return (
        f'<div style="border:{border_width} solid {border_color};border-radius:10px;padding:14px;'
        f'background-color:{bg_color};text-align:center;min-height:260px;display:flex;'
        f'flex-direction:column;justify-content:space-between;box-sizing:border-box;cursor:pointer;">'
        f"{selected_badge}"
        f'<div style="font-size:18px;font-weight:bold;margin-bottom:4px;">{ticker}</div>'
        f'<div style="font-size:12px;color:#666;margin-bottom:8px;">{info.get("이름", "")}</div>'
        f'<div style="text-align:left;font-size:12px;line-height:1.6;margin-bottom:8px;">'
        f'<div>카테고리: {info.get("카테고리", "")}</div>'
        f'<div>보수율: {info.get("보수율", 0) * 100:.2f}%</div>'
        f'<div>배당: {"있음" if info.get("배당") else "없음"}</div>'
        f'<div>위험도: {"⭐" * risk_stars}{"☆" * (5 - risk_stars)}</div>'
        f"</div>"
        f'<div style="font-size:11px;color:#555;margin-bottom:8px;font-style:italic;">'
        f'{ETF_CARD_DESCRIPTIONS.get(ticker, "")}</div>'
        f'<div style="font-size:13px;font-weight:bold;color:#1f77b4;">'
        f"예상 수익률: {annual_return:.1f}%</div></div>"
    )


def _display_etf_card_clickable(ticker, is_selected=False):
    """ETF 카드를 HTML로 렌더링합니다."""
    html = _etf_card_html(ticker, is_selected)
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def get_same_category_replacement_options(ticker, selected_etfs):
    """포트폴리오 종목과 동일 카테고리의 교체 후보(현재 종목 유지 + 미선택 종목)."""
    category = ETF_DATA.get(ticker, {}).get("카테고리")
    if not category:
        return [ticker]
    alternatives = sorted(
        t
        for t, info in ETF_DATA.items()
        if info.get("카테고리") == category and t not in selected_etfs
    )
    return [ticker] + alternatives


def _format_replace_option_label(option_ticker, current_ticker):
    """종목 교체 라디오 한 줄(멀티라인) 라벨."""
    info = ETF_DATA.get(option_ticker, {})
    current_tag = " (현재 보유)" if option_ticker == current_ticker else ""
    summary = ETF_SUMMARY_DESCRIPTIONS.get(option_ticker, info.get("이름", ""))
    return (
        f"{option_ticker}{current_tag}\n"
        f"{summary}\n"
        f"보수율 {info.get('보수율', 0) * 100:.2f}%"
        f"  ·  배당 {'있음' if info.get('배당') else '없음'}"
    )


def _replace_radio_key(slot_idx):
    return f"replace_radio_{slot_idx}"


def _resolve_replace_index(raw_value, replace_options, default_idx=0):
    """라디오 세션 값(int/str 혼재)을 안전한 인덱스로 정규화합니다."""
    if isinstance(raw_value, int):
        if 0 <= raw_value < len(replace_options):
            return raw_value
        return default_idx if 0 <= default_idx < len(replace_options) else 0
    if isinstance(raw_value, str) and raw_value in replace_options:
        return replace_options.index(raw_value)
    return default_idx if 0 <= default_idx < len(replace_options) else 0


def _apply_etf_replace(slot_idx, old_ticker, replace_options):
    """같은 카테고리 ETF로만 포트폴리오 종목을 교체합니다."""
    radio_key = _replace_radio_key(slot_idx)
    picked_idx = _resolve_replace_index(st.session_state.get(radio_key, 0), replace_options, 0)
    new_ticker = replace_options[picked_idx]
    if not new_ticker or new_ticker == old_ticker:
        return

    old_category = ETF_DATA.get(old_ticker, {}).get("카테고리")
    new_category = ETF_DATA.get(new_ticker, {}).get("카테고리")
    if old_category != new_category:
        st.session_state[f"replace_error_{slot_idx}"] = (
            "같은 카테고리 ETF만 선택할 수 있습니다."
        )
        return

    selected = list(st.session_state.get("selected_etfs", []))
    if old_ticker not in selected:
        return
    if slot_idx >= len(selected) or selected[slot_idx] != old_ticker:
        slot_idx = selected.index(old_ticker)

    selected[slot_idx] = new_ticker
    st.session_state["selected_etfs"] = selected

    etf_weights = st.session_state.get("etf_weights", {})
    if old_ticker in etf_weights:
        etf_weights[new_ticker] = etf_weights.pop(old_ticker)
        st.session_state["etf_weights"] = etf_weights

    st.session_state.pop(f"replace_error_{slot_idx}", None)
    st.session_state[f"replace_msg_{slot_idx}"] = f"{old_ticker} → {new_ticker}로 교체됨"


# ============================================================
# 섹션 4. 시뮬레이션 함수
# ============================================================

SIMULATION_BASE_STATS = {
    "지수추적": {"mu": 0.075, "sigma": 0.15},
    "배당성장": {"mu": 0.055, "sigma": 0.12},
    "배당집중": {"mu": 0.050, "sigma": 0.10},
    "레버리지": {"mu": 0.075, "sigma": 0.15},
}

DIVIDEND_TAX_RATE = 0.15
DIVIDEND_YIELD_ESTIMATE = 0.02
MONTE_CARLO_TRIALS = 1000


def _get_ticker_return_profile(ticker):
    info = ETF_DATA[ticker]
    category = info["카테고리"]
    base = SIMULATION_BASE_STATS.get(category, SIMULATION_BASE_STATS["지수추적"])
    mu = base["mu"]
    sigma = base["sigma"]

    if info["레버리지"]:
        leverage = 3 if ticker in {"TQQQ", "UPRO", "SOXL"} else 2
        mu = leverage * mu - 0.5 * leverage * (leverage - 1) * sigma ** 2
        sigma = leverage * sigma

    mu -= info["보수율"]
    if info["배당"]:
        mu -= DIVIDEND_YIELD_ESTIMATE * DIVIDEND_TAX_RATE

    return mu, sigma


def _simulate_portfolio_once(weights, months, initial_capital, monthly_contribution, exchange_rate):
    value = initial_capital
    for month in range(months):
        monthly_return = 0.0
        for ticker, weight in weights.items():
            mu, sigma = _get_ticker_return_profile(ticker)
            monthly_mu = mu / 12
            monthly_sigma = sigma / np.sqrt(12)
            sample = np.random.normal(monthly_mu, monthly_sigma)
            monthly_return += weight * sample
        value *= np.exp(monthly_return)
        value += monthly_contribution
    return value * exchange_rate


def _normalize_weights(weights):
    total = sum(weights.values())
    # If total is zero or negative, fall back to equal weighting for non-empty portfolios
    if total <= 0:
        if not weights:
            raise ValueError("포트폴리오 가중치 합계는 0보다 커야 합니다.")
        n = len(weights)
        return {ticker: 1.0 / n for ticker in weights}
    return {ticker: w / total for ticker, w in weights.items()}


def simulate_portfolio(
    weights,
    years=10,
    mode="accumulation",
    initial_capital=0.0,
    monthly_contribution=100000.0,
    exchange_rate=1300.0,
):
    """Monte Carlo 시뮬레이션을 실행하고 상위/중앙/하위 시나리오를 반환합니다."""
    normalized_weights = _normalize_weights(weights)
    months = int(years * 12)

    if mode == "거치형":
        monthly_contribution = 0.0
    elif mode == "적립형":
        initial_capital = 0.0
    elif mode == "혼합형":
        pass
    else:
        raise ValueError("mode는 '적립형', '거치형', '혼합형' 중 하나여야 합니다")

    final_values = np.zeros(MONTE_CARLO_TRIALS)
    for i in range(MONTE_CARLO_TRIALS):
        final_values[i] = _simulate_portfolio_once(
            normalized_weights,
            months,
            initial_capital,
            monthly_contribution,
            exchange_rate,
        )

    def _scenario_label(pct):
        value = float(np.percentile(final_values, pct))
        total_invested = initial_capital + monthly_contribution * months
        annualized = ((value / max(total_invested, 1e-6)) ** (1 / years) - 1) if years > 0 else 0
        return {
            "최종가": round(value, 2),
            "연환수익률": round(annualized * 100, 2),
            "투자금": round(total_invested, 2),
        }

    return {
        "상위": _scenario_label(90),
        "중간": _scenario_label(50),
        "하위": _scenario_label(10),
    }

# ============================================================
# 섹션 5. 리밸런싱 비교 함수
# ============================================================

REBALANCE_FREQUENCIES = {
    "연간": 12,
    "반기": 6,
    "분기": 3,
}

REALIZATION_TAX_RATE = 0.22


def _simulate_strategy(
    weights,
    years,
    initial_capital,
    monthly_contribution,
    exchange_rate,
    rebalance_months=None,
    invest_monthly=True,
    dca=False,
):
    months = int(years * 12)
    normalized_weights = _normalize_weights(weights)
    holdings = {ticker: 0.0 for ticker in weights}

    dca_monthly_add = 0.0
    if dca and months > 0:
        dca_monthly_add = initial_capital / months
        initial_capital = 0.0

    if initial_capital > 0:
        for ticker, weight in normalized_weights.items():
            holdings[ticker] += initial_capital * weight

    for month in range(1, months + 1):
        for ticker, value in holdings.items():
            mu, sigma = _get_ticker_return_profile(ticker)
            monthly_mu = mu / 12
            monthly_sigma = sigma / np.sqrt(12)
            holdings[ticker] = value * np.exp(np.random.normal(monthly_mu, monthly_sigma))

        if invest_monthly:
            contribution = monthly_contribution + dca_monthly_add
            if contribution > 0:
                for ticker, weight in normalized_weights.items():
                    holdings[ticker] += contribution * weight

        if rebalance_months and month % rebalance_months == 0:
            total_value = sum(holdings.values())
            target_values = {ticker: total_value * normalized_weights[ticker] for ticker in holdings}
            turnover = sum(max(holdings[ticker] - target_values[ticker], 0) for ticker in holdings)
            tax = turnover * REALIZATION_TAX_RATE
            net_value = total_value - tax
            for ticker in holdings:
                holdings[ticker] = net_value * normalized_weights[ticker]

    return sum(holdings.values()) * exchange_rate


def compare_rebalancing_strategies(
    weights,
    years=10,
    initial_capital=1000000.0,
    monthly_contribution=100000.0,
    frequency="연간",
    exchange_rate=1300.0,
    trials=500,
):
    """Buy&Hold, 분기 리밸런싱, DCA 전략 비교합니다."""
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"지원하지 않는 리밸런싱 주기입니다: {frequency}")

    rebalance_months = REBALANCE_FREQUENCIES[frequency]
    results = {
        "Buy&Hold": [],
        "리밸런싱": [],
        "DCA": [],
    }

    for _ in range(trials):
        results["Buy&Hold"].append(
            _simulate_strategy(
                weights,
                years,
                initial_capital,
                monthly_contribution,
                exchange_rate,
                rebalance_months=None,
                invest_monthly=True,
            )
        )
        results["리밸런싱"].append(
            _simulate_strategy(
                weights,
                years,
                initial_capital,
                monthly_contribution,
                exchange_rate,
                rebalance_months=rebalance_months,
                invest_monthly=True,
            )
        )
        results["DCA"].append(
            _simulate_strategy(
                weights,
                years,
                initial_capital,
                monthly_contribution,
                exchange_rate,
                rebalance_months=None,
                invest_monthly=True,
                dca=True,
            )
        )

    def summarize(values):
        arr = np.array(values)
        return {
            "평균최종가": round(float(np.mean(arr)), 2),
            "중간최종가": round(float(np.median(arr)), 2),
            "표준편차": round(float(np.std(arr)), 2),
            "상위(90%)": round(float(np.percentile(arr, 90)), 2),
            "하위(10%)": round(float(np.percentile(arr, 10)), 2),
        }

    return {strategy: summarize(vals) for strategy, vals in results.items()}

# ============================================================
# 섹션 6. Plotly 그래프 함수
# ============================================================

def simulate_portfolio_paths(
    weights,
    years=10,
    mode="거치형",
    initial_capital=0.0,
    monthly_contribution=100000.0,
    exchange_rate=1300.0,
    trials=500,
):
    """Monte Carlo 경로를 생성합니다."""
    months = int(years * 12)
    normalized_weights = _normalize_weights(weights)
    paths = np.zeros((trials, months + 1))

    for i in range(trials):
        value = initial_capital
        paths[i, 0] = value * exchange_rate
        for month in range(1, months + 1):
            monthly_return = 0.0
            for ticker, weight in normalized_weights.items():
                mu, sigma = _get_ticker_return_profile(ticker)
                monthly_mu = mu / 12
                monthly_sigma = sigma / np.sqrt(12)
                sample = np.random.normal(monthly_mu, monthly_sigma)
                monthly_return += weight * sample
            value *= np.exp(monthly_return)
            if mode in {"적립형", "혼합형"}:
                value += monthly_contribution
            paths[i, month] = value * exchange_rate
    return paths


def plot_growth_curves(paths, years=10, title="투자 성장 곡선"):
    """상위/중앙/하위 성장 곡선을 그립니다."""
    months = paths.shape[1] - 1
    x = np.arange(months + 1)
    median = np.percentile(paths, 50, axis=0)
    optimistic = np.percentile(paths, 90, axis=0)
    pessimistic = np.percentile(paths, 10, axis=0)
    lower = np.percentile(paths, 25, axis=0)
    upper = np.percentile(paths, 75, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=median, mode="lines", name="중앙", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=x, y=optimistic, mode="lines", name="상위", line=dict(color="#2ca02c", dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=pessimistic, mode="lines", name="하위", line=dict(color="#d62728", dash="dash")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='중앙 25-75% 구간',
    ))
    fig.update_layout(
        title=title,
        xaxis_title="개월",
        yaxis_title="투자 금액",
        template="plotly_white",
    )
    add_milestone_lines(fig)
    return fig


def plot_mode_comparison(mode_paths, years=10, title="적립형 vs 거치형 vs 혼합형 비교"):
    """투자 모드별 평균 경로를 비교합니다."""
    months = int(years * 12)
    x = np.arange(months + 1)
    fig = go.Figure()
    for mode, paths in mode_paths.items():
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(x=x, y=mean_path, mode="lines", name=mode))
    fig.update_layout(
        title=title,
        xaxis_title="개월",
        yaxis_title="투자 금액",
        template="plotly_white",
    )
    add_milestone_lines(fig)
    return fig


def plot_rebalance_comparison(summary, title="Buy&Hold vs 리밸런싱 vs DCA"):
    """리밸런싱 결과를 막대 그래프로 표시합니다."""
    strategies = list(summary.keys())
    metrics = ["평균최종가", "중간최종가", "표준편차", "상위(90%)", "하위(10%)"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=strategies,
            y=[summary[s][metric] for s in strategies],
        ))
    fig.update_layout(
        title=title,
        xaxis_title="전략",
        yaxis_title="금액",
        barmode='group',
        template="plotly_white",
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def load_voo_daily_return_profile():
    if yf is None:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다.")

    hist = yf.Ticker("VOO").history(start="2008-01-01", auto_adjust=True)
    if hist.empty or "Close" not in hist:
        raise RuntimeError("VOO 가격 데이터를 불러오지 못했습니다.")

    daily_returns = hist["Close"].pct_change().dropna()
    if daily_returns.empty:
        raise RuntimeError("VOO 일별 수익률 데이터를 계산하지 못했습니다.")

    return float(daily_returns.mean()), float(daily_returns.std())


def _recommended_investment_mode_from_profile():
    q8_answer = st.session_state.get("profile_responses", {}).get("Q8")
    if q8_answer == 1:
        return "거치형"
    if q8_answer == 2:
        return "적립형"
    return "혼합형"


def _leverage_multiplier(ticker):
    if ticker in {"QLD", "SSO"}:
        return 2
    if ticker in {"TQQQ", "UPRO", "SOXL"}:
        return 3
    return 1


@st.cache_data(ttl=3600, show_spinner=False)
def _simulate_daily_to_monthly_median_path(
    annual_return,
    annual_volatility,
    month_end_date_keys,
    initial_capital,
    monthly_contribution,
    mode,
    trials=100,
):
    month_end_dates = tuple(pd.Timestamp(date_key) for date_key in month_end_date_keys)
    months = len(month_end_dates) - 1
    paths = np.zeros((trials, months + 1))
    paths[:, 0] = initial_capital if mode != "적립형" else 0.0
    business_days_by_month = [
        max(
            1,
            len(pd.bdate_range(
                month_end_dates[month - 1] + pd.Timedelta(days=1),
                month_end_dates[month],
            )),
        )
        for month in range(1, months + 1)
    ]
    daily_mu = annual_return / 252
    daily_sigma = annual_volatility / np.sqrt(252)

    values = paths[:, 0].copy()
    for month, days in enumerate(business_days_by_month, start=1):
        if mode in {"적립형", "혼합형"}:
            values += monthly_contribution
        portfolio_daily_returns = np.maximum(
            np.random.normal(daily_mu, daily_sigma, size=(trials, days)),
            -0.99,
        )
        values *= np.prod(1 + portfolio_daily_returns, axis=1)
        paths[:, month] = values

    median_final = np.percentile(paths[:, -1], 50)
    median_path_idx = int(np.argmin(np.abs(paths[:, -1] - median_final)))
    return paths[median_path_idx]


def _weighted_portfolio_return_profile(weight_items):
    normalized_weights = _normalize_weights(dict(weight_items))
    annual_return = sum(
        weight * ETF_DATA[ticker].get("cagr", _get_ticker_return_profile(ticker)[0])
        for ticker, weight in normalized_weights.items()
    )
    annual_volatility = sum(
        weight * _get_ticker_return_profile(ticker)[1]
        for ticker, weight in normalized_weights.items()
    )
    return float(annual_return), float(annual_volatility)


def _detect_drawdown_marker_indices(values, threshold=-0.15):
    monthly_changes = np.diff(values) / np.maximum(values[:-1], 1)
    marker_indices = set()
    idx = 1

    while idx < len(values):
        if monthly_changes[idx - 1] > threshold:
            idx += 1
            continue

        start_idx = idx - 1
        start_value = values[start_idx]
        trough_idx = idx
        scan_idx = idx

        while scan_idx < len(values):
            if values[scan_idx] < values[trough_idx]:
                trough_idx = scan_idx
            if values[scan_idx] >= start_value:
                break
            scan_idx += 1

        recovery_idx = min(scan_idx, len(values) - 1)
        marker_indices.update({start_idx, trough_idx, recovery_idx})
        idx = recovery_idx + 1

    return sorted(marker_indices)


def plot_portfolio_vs_sp500_monte_carlo(
    port_weights,
    start_date,
    end_date,
    initial_capital,
    monthly_contribution,
    recommended_mode,
    title="내 포트폴리오 vs S&P500 기준선",
):
    return _plot_portfolio_vs_sp500_monte_carlo_cached(
        _portfolio_weight_items(port_weights),
        _date_cache_key(start_date),
        _date_cache_key(end_date),
        float(initial_capital),
        float(monthly_contribution),
        recommended_mode,
        title,
    )


def _portfolio_weight_items(weights):
    return tuple(sorted((ticker, float(weight)) for ticker, weight in weights.items()))


def _date_cache_key(value):
    return pd.Timestamp(value).date().isoformat()


@st.cache_data(ttl=3600, show_spinner=False)
def _plot_portfolio_vs_sp500_monte_carlo_cached(
    port_weight_items,
    start_date_key,
    end_date_key,
    initial_capital,
    monthly_contribution,
    recommended_mode,
    title,
):
    port_weights = dict(port_weight_items)
    start_date = pd.Timestamp(start_date_key).date()
    end_date = pd.Timestamp(end_date_key).date()
    months = max(1, int(round(period_between_dates_years(start_date, end_date) * 12)))
    monthly_dates = [pd.Timestamp(start_date) + pd.DateOffset(months=i) for i in range(months + 1)]
    monthly_dates[-1] = pd.Timestamp(end_date)
    month_end_date_keys = tuple(_date_cache_key(month_date) for month_date in monthly_dates)
    _, voo_sigma_daily = load_voo_daily_return_profile()
    portfolio_return, portfolio_volatility = _weighted_portfolio_return_profile(port_weight_items)
    voo_return = ETF_DATA["VOO"].get("cagr", _get_ticker_return_profile("VOO")[0])
    voo_volatility = voo_sigma_daily * np.sqrt(252)

    portfolio_median = _simulate_daily_to_monthly_median_path(
        portfolio_return,
        portfolio_volatility,
        month_end_date_keys,
        initial_capital,
        monthly_contribution,
        recommended_mode,
        trials=100,
    )
    sp500_median = _simulate_daily_to_monthly_median_path(
        voo_return,
        voo_volatility,
        month_end_date_keys,
        initial_capital,
        monthly_contribution,
        recommended_mode,
        trials=100,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=portfolio_median,
        mode="lines",
        name="내 포트폴리오",
        line=dict(color="#2563eb", width=3),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}원<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_dates,
        y=sp500_median,
        mode="lines",
        name="S&P500 기준선",
        line=dict(color="#9ca3af", width=3),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}원<extra></extra>",
    ))

    monthly_changes = np.diff(portfolio_median) / np.maximum(portfolio_median[:-1], 1)
    y_midpoint = (float(np.min(portfolio_median)) + float(np.max(portfolio_median))) / 2
    for idx, monthly_change in enumerate(monthly_changes, start=1):
        if monthly_change <= -0.15:
            annotation_ay = 90 if portfolio_median[idx] >= y_midpoint else -90
            fig.add_annotation(
                x=monthly_dates[idx],
                y=portfolio_median[idx],
                text="급락 구간",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#dc2626",
                ax=0,
                ay=annotation_ay,
                font=dict(color="#dc2626", size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#dc2626",
                borderwidth=1,
            )

    fig.update_layout(
        title=f"{title} ({recommended_mode})",
        xaxis_title="투자 기간",
        yaxis_title="자산 금액 (원)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickformat=",")
    return fig


def add_milestone_lines(fig, milestones=None):
    """그래프에 마일스톤 선을 추가합니다."""
    if milestones is None:
        milestones = [100000000, 500000000, 1000000000]
    for value in milestones:
        fig.add_hline(
            y=value,
            line=dict(color="gray", dash="dot"),
            annotation_text=f"{int(value/100000000)}억",
            annotation_position="top left",
            annotation_font_size=10,
            opacity=0.7,
        )
    return fig

# ============================================================
# 섹션 7. Streamlit UI
# ============================================================

INVESTMENT_MODES = ["거치형", "적립형", "혼합형"]
# MODE1(현재 포트 진단) 제거: MODE2=목표 금액 달성, MODE3=목표 기간 확인
ANALYSIS_MODES = ["목표 금액 달성", "목표 기간 확인"]

PERIOD_QUICK_OFFSETS = [
    ("1년 후", 1),
    ("5년 후", 5),
    ("10년 후", 10),
    ("20년 후", 20),
    ("30년 후", 30),
]


def add_calendar_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, month=2, day=28)


def _date_diff_ymd(start: date, end: date):
    if end < start:
        return 0, 0, 0
    years = end.year - start.year
    months = end.month - start.month
    days = end.day - start.day
    if days < 0:
        months -= 1
        prev_month = end.month - 1 if end.month > 1 else 12
        prev_year = end.year if end.month > 1 else end.year - 1
        days += calendar.monthrange(prev_year, prev_month)[1]
    if months < 0:
        years -= 1
        months += 12
    return years, months, days


def period_between_dates_years(start: date, end: date) -> float:
    if end < start:
        return 1.0
    days = (end - start).days
    return max(days / 365.25, 30 / 365.25)


def format_period_label(start: date, end: date) -> str:
    if end < start:
        return "종료일은 시작일 이후로 선택해 주세요."
    y, m, d = _date_diff_ymd(start, end)
    parts = []
    if y:
        parts.append(f"{y}년")
    if m:
        parts.append(f"{m}개월")
    if d or not parts:
        parts.append(f"{d}일")
    approx = period_between_dates_years(start, end)
    return f"{' '.join(parts)} (약 {approx:.1f}년)"


def _set_period_end_callback(key_prefix: str, years_offset: int):
    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"
    start = st.session_state.get(start_key, date.today())
    st.session_state[end_key] = add_calendar_years(start, years_offset)


def init_investment_period_state(key_prefix: str = "invest_period"):
    today = date.today()
    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"
    if start_key not in st.session_state:
        st.session_state[start_key] = today
    if end_key not in st.session_state:
        default_years = int(st.session_state.get("years", st.session_state.get("target_years", 10)))
        st.session_state[end_key] = add_calendar_years(today, default_years)


def render_investment_period_selector(
    key_prefix: str = "invest_period",
    section_title: str = "투자 기간",
    period_label: str = "투자 기간",
) -> float:
    """시작일·종료일 달력과 빠른 선택 버튼으로 기간을 설정하고, 연 단위 기간을 반환합니다."""
    init_investment_period_state(key_prefix)
    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"
    today = date.today()

    st.markdown(f"#### {section_title}")
    date_cols = st.columns(2)
    with date_cols[0]:
        st.date_input(
            "시작일",
            key=start_key,
            min_value=today,
            format="YYYY.MM.DD",
        )
    start_val = st.session_state[start_key]
    with date_cols[1]:
        st.date_input(
            "종료일",
            key=end_key,
            min_value=start_val,
            format="YYYY.MM.DD",
        )

    btn_cols = st.columns(len(PERIOD_QUICK_OFFSETS))
    for col, (label, yrs) in zip(btn_cols, PERIOD_QUICK_OFFSETS):
        col.button(
            label,
            key=f"{key_prefix}_quick_{yrs}",
            on_click=_set_period_end_callback,
            args=(key_prefix, yrs),
            use_container_width=True,
        )

    end_val = st.session_state[end_key]
    if end_val < start_val:
        st.error("종료일은 시작일 이후로 선택해 주세요.")
        years_val = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
    else:
        st.markdown(
            f'<p style="color:#6b7280;font-size:0.875rem;margin:0.25rem 0;">'
            f"· {period_label}: <strong>{format_period_label(start_val, end_val)}</strong><br>"
            f"· 빠른 선택 버튼은 시작일 기준으로 종료일을 자동 설정합니다."
            f"</p>",
            unsafe_allow_html=True,
        )
        years_val = period_between_dates_years(start_val, end_val)

    return years_val


def estimate_profile_locally(responses):
    """사용자 응답을 기반으로 투자 성향을 로컬에서 계산합니다."""
    return calculate_profile_scores(responses)


def build_portfolio_weights(etfs, etf_weights=None):
    """포트폴리오 비중을 계산합니다.
    
    etf_weights가 제공되면 그 값을 사용하고, 없으면 균등 분배합니다.
    """
    if not etfs:
        raise ValueError("추천 ETF 목록이 비어 있습니다.")
    
    # etf_weights가 있으면 사용
    if etf_weights:
        return {ticker: etf_weights.get(ticker, 1/len(etfs)) for ticker in etfs}
    
    # 없으면 균등 분배
    weight = 1.0 / len(etfs)
    return {ticker: weight for ticker in etfs}


def _max_profile_pct(profile_weights):
    return max(profile_weights.get(c, 0) for c in AI_PROFILE_CATEGORIES)


def build_strategy_reason(profile_weights, selected_etfs, analysis_mode):
    reasons = []
    top_pct = _max_profile_pct(profile_weights)
    if profile_weights.get("지수추적", 0) >= top_pct:
        reasons.append("지수추적 성향을 반영해 시장 지수 ETF를 중심으로 구성했습니다.")
    if profile_weights.get("배당성장", 0) >= top_pct:
        reasons.append("배당 성향이 높아 안정적인 배당 ETF를 포함했습니다.")
    if profile_weights.get("배당집중", 0) >= top_pct:
        reasons.append(
            "배당집중 성향을 반영해 옵션 프리미엄·월배당형 ETF를 포함했습니다. "
            "급등 구간에서는 수익이 제한될 수 있습니다."
        )
    if profile_weights.get("레버리지", 0) >= top_pct:
        reasons.append("레버리지 성향이 높아 공격적 수익 추구 ETF를 포함했습니다. 분할매수와 비중 상한을 권장합니다.")
    if analysis_mode == "목표 금액 달성":
        reasons.append("목표 금액 달성을 위해 적절한 월적립 전략을 고려하세요.")
    elif analysis_mode == "목표 기간 확인":
        reasons.append("목표 기간에 맞춘 리밸런싱과 적립 전략을 병행하세요.")
    return "\n".join(reasons) if reasons else "ETF 성향에 따라 균형 있게 포트폴리오를 구성하는 것을 권장합니다."


def estimate_required_contribution(target_amount, years, current_capital, avg_return=0.06):
    months = int(years * 12)
    monthly_r = avg_return / 12
    if months <= 0:
        return float('inf')
    if monthly_r == 0:
        return max(0.0, (target_amount - current_capital) / months)
    factor = (np.power(1 + monthly_r, months) - 1) / monthly_r
    return max(0.0, (target_amount - current_capital * np.power(1 + monthly_r, months)) / factor)


def estimate_years_to_target(target_amount, monthly_contribution, current_capital, avg_return=0.06):
    monthly_r = avg_return / 12
    if monthly_contribution <= 0:
        return float('inf')
    value = current_capital
    months = 0
    while value < target_amount and months < 1200:
        value *= (1 + monthly_r)
        value += monthly_contribution
        months += 1
    return months / 12


def generate_growth_path(initial_capital, monthly_contribution, years, avg_return=0.06, interval_months=12):
    """주어진 간격(interval_months)으로 성장 경로 값을 반환합니다.
    예: interval_months=6이면 6개월 단위로 값을 반환합니다. 반환 길이는 (years*12)//interval_months + 1 입니다.
    """
    monthly_r = avg_return / 12
    total_months = int(years * 12)
    values = [initial_capital]
    v = initial_capital
    for m in range(1, total_months + 1):
        v *= (1 + monthly_r)
        v += monthly_contribution
        if m % interval_months == 0:
            values.append(v)
    # If final point missing due to rounding, pad with last value
    expected_len = total_months // interval_months + 1
    if len(values) < expected_len:
        values += [values[-1]] * (expected_len - len(values))
    return values


def run_streamlit_app():
    st.set_page_config(page_title="ETF 포트폴리오 시뮬레이션", layout="wide")
    st.title("ETF 포트폴리오 시뮬레이션")
    load_etf_cagr()

    # 유틸: 금액 콤마 포맷 및 파서 (천단위 콤마, 소수 .00 제거)
    def fmt_money(n):
        try:
            f = float(n)
        except Exception:
            return "0"
        if f.is_integer():
            return f"{int(f):,}"
        s = f"{f:,}"
        if "." in s:
            s = s.rstrip('0').rstrip('.')
        return s

    def parse_number_input(s):
        if s is None:
            return 0.0
        s2 = str(s).replace(',', '').strip()
        if s2 == '':
            return 0.0
        try:
            return float(s2)
        except Exception:
            return 0.0

    def parse_int(s):
        if s is None:
            return 0
        s2 = str(s).replace(',', '').strip()
        if s2 == '':
            return 0
        try:
            return int(float(s2))
        except Exception:
            return 0

    def format_money_input(key):
        raw_value = st.session_state.get(key, "")
        if raw_value is None:
            raw_value = ""
        raw_text = str(raw_value)
        parsed = parse_int(raw_value)
        formatted = fmt_money(parsed) if raw_text.strip() != "" else ""
        if raw_value != formatted:
            st.session_state[key] = formatted

    def format_won_from_manwon_key(key):
        raw_value = st.session_state.get(key, "")
        parsed = parse_int(raw_value)
        return parsed * 10000

    # 콜백: 만원 단위 입력을 증가시키거나 초기화합니다.
    def change_man_amount(key, inc):
        cur = parse_int(st.session_state.get(key, "0"))
        st.session_state[key] = fmt_money(cur + inc)

    def reset_man_amount(key):
        st.session_state[key] = ""

    def init_man_amount_key(key, default_won):
        if key not in st.session_state:
            st.session_state[key] = fmt_money(int(default_won) // 10000)

    def render_manwon_amount_input(title, key, default_won):
        """직접 입력 + 프리셋 버튼 방식의 금액 입력"""
        init_man_amount_key(key, default_won)
        
        st.markdown(f"#### 💰 {title}")
        
        # 입력창 + 프리셋 버튼 한 줄로
        input_cols = st.columns([2, 1, 1, 1, 1, 1])
        
        with input_cols[0]:
            st.text_input(
                "금액 입력",
                key=key,
                on_change=format_money_input,
                args=(key,),
                label_visibility="collapsed",
                placeholder="예: 5000 (= 5,000만원)",
            )
        
        preset_amounts = [
            ("100만", 100),
            ("500만", 500),
            ("1000만", 1000),
            ("5000만", 5000),
            ("1억", 10000),
        ]
        
        # 콜백 함수: 프리셋 값을 세션 상태에 저장
        def set_preset_value(manwon_val):
            st.session_state[key] = fmt_money(manwon_val)
        
        # 프리셋 버튼 처리
        for idx, (label, manwon_val) in enumerate(preset_amounts):
            col = input_cols[idx + 1]
            with col:
                st.button(
                    label,
                    key=f"{key}_btn_{idx}",
                    use_container_width=True,
                    on_click=set_preset_value,
                    args=(manwon_val,),
                )
        
        # 입력값 표시
        try:
            amount_won = int(format_won_from_manwon_key(key))
        except Exception as e:
            amount_won = 0
            
        display_manwon = st.session_state.get(key, "") or "0"
        
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius:12px;
                padding:18px 24px;
                margin:12px 0;
                display:flex;
                justify-content:space-between;
                align-items:center;
                box-shadow:0 4px 12px rgba(102, 126, 234, 0.3);
            ">
                <span style="color:#ffffff;font-size:0.95rem;font-weight:600;">💾 입력금액</span>
                <div style="text-align:right;">
                    <div style="color:#e0e7ff;font-size:0.9rem;font-weight:500;">
                        {display_manwon} 만원
                    </div>
                    <div style="color:#ffffff;font-size:1.6rem;font-weight:800;">
                        {fmt_money(amount_won)}원
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        return amount_won

    if "app_step" not in st.session_state:
        st.session_state["app_step"] = 1
        st.session_state["current_question"] = "G1"
        st.session_state["profile_responses"] = {}  # {Q_KEY: choice_index}
        st.session_state["profile_submitted"] = False
        st.session_state["profile_weights"] = None
        st.session_state["gemini_profile_questions"] = []
        st.session_state["gemini_profile_subquestions"] = {}
        st.session_state["profile_keyword_explanations_shown"] = set()
        st.session_state["profile_keyword_explanation_contexts"] = {}
        st.session_state["profile_ai_fallback"] = False
        st.session_state["selected_etfs"] = []
        st.session_state["analysis_mode"] = ANALYSIS_MODES[0]
        st.session_state["current_capital"] = 10000000.0
        st.session_state["monthly_contribution"] = 100000.0
        st.session_state["years"] = 10
        st.session_state["target_amount"] = 500000000.0
        st.session_state["target_years"] = 10
        st.session_state["invest_period_start"] = date.today()
        st.session_state["invest_period_end"] = add_calendar_years(date.today(), 10)
        st.session_state["sim_mode"] = INVESTMENT_MODES[1]
        st.session_state["simulation_results"] = None

    step_labels = {
        1: "STEP1: AI 성향 분석",
        2: "STEP2: ETF 추천",
        3: "STEP3: 모드 선택",
        4: "STEP4: 전략 추천",
        6: "STEP6: 결과 시각화",
    }
    if st.session_state["app_step"] == 5:
        st.session_state["app_step"] = 6
    current_step = st.session_state["app_step"]
    display_step = 5 if current_step == 6 else current_step
    total_steps = 5
    st.write(f"### {step_labels[current_step]} ({display_step}/{total_steps})")
    st.progress((display_step - 1) / (total_steps - 1))

    def move_step(new_step):
        st.session_state["app_step"] = new_step
        st.rerun()

    def reset_app_to_start():
        """앱을 최초 실행 상태(STEP1 · Q1)로 되돌립니다."""
        st.session_state.clear()
        st.rerun()

    def activate_admin_mode():
        """개발 확인용: 질문을 건너뛰고 랜덤 포트폴리오로 STEP2에 진입합니다."""
        rng = np.random.default_rng()
        random_profile = rng.dirichlet(np.ones(len(AI_PROFILE_CATEGORIES))) * 100
        profile_weights = {
            category: round(float(weight), 2)
            for category, weight in zip(AI_PROFILE_CATEGORIES, random_profile)
        }
        profile_weights["_leverage_allowed"] = True

        random_etfs = list(rng.choice(list(ETF_DATA.keys()), size=5, replace=False))
        random_weights = rng.dirichlet(np.ones(len(random_etfs)))

        st.session_state["profile_weights"] = profile_weights
        st.session_state["profile_responses"] = {"Q8": int(rng.integers(0, 3))}
        st.session_state["profile_submitted"] = True
        st.session_state["selected_etfs"] = random_etfs
        st.session_state["etf_weights"] = {
            ticker: float(weight)
            for ticker, weight in zip(random_etfs, random_weights)
        }
        st.session_state["app_step"] = 2
        st.rerun()

    def step_button_row(can_next=True, can_prev=True):
        cols = st.columns([2, 2, 1])
        if can_prev:
            nav_cols = cols[0].columns(2)
            if nav_cols[0].button("이전", key=f"prev_{current_step}"):
                move_step(4 if current_step == 6 else max(1, current_step - 1))
            if nav_cols[1].button("처음으로", key=f"home_{current_step}"):
                reset_app_to_start()
        if can_next and cols[2].button("다음", key=f"next_{current_step}"):
            move_step(6 if current_step == 4 else min(6, current_step + 1))

    profile_weights = st.session_state.get("profile_weights")
    selected_etfs = st.session_state.get("selected_etfs", [])
    simulation_results = st.session_state.get("simulation_results")

    if current_step == 1:
        st.write("쉬운 질문에 답하면 투자 성향에 맞는 상품 유형을 추천해 드립니다. (종목 코드는 결과 화면에서 확인할 수 있습니다.)")

        responses = st.session_state.get("profile_responses", {})
        if st.session_state.get("profile_ai_fallback", False) and _get_gemini_model() is not None and not responses:
            st.session_state["profile_ai_fallback"] = False
            st.session_state["current_question"] = "G1"
            st.session_state["gemini_profile_questions"] = []
            st.session_state["gemini_profile_subquestions"] = {}
            st.rerun()

        if not st.session_state.get("profile_ai_fallback", False):
            gemini_questions = st.session_state.setdefault("gemini_profile_questions", [])
            if not gemini_questions:
                try:
                    first_question = _generate_gemini_profile_question([], 1)
                    gemini_questions.append(first_question)
                    st.session_state["current_question"] = "G1"
                except Exception as exc:
                    _activate_fixed_profile_fallback()
                    st.warning(f"Gemini 질문 생성에 실패해 기존 고정 질문으로 진행합니다. ({type(exc).__name__}: {exc})")
                    st.rerun()

            current_q = st.session_state.get("current_question", "G1")
            if not str(current_q).startswith("G"):
                st.session_state["current_question"] = "G1"
                st.session_state["profile_responses"] = {}
                responses = st.session_state["profile_responses"]
                current_q = "G1"
            try:
                current_idx = max(0, int(str(current_q).replace("G", "")) - 1)
            except ValueError:
                current_idx = 0

            if current_idx >= len(gemini_questions):
                st.session_state["current_question"] = f"G{len(gemini_questions)}"
                st.rerun()

            q_data = gemini_questions[current_idx]
            q_choices = q_data["choices"]
            main_question_number = current_idx + 1
            can_have_subquestions = main_question_number in GEMINI_SUBQUESTION_RULES
            st.write(f"**Q{current_idx + 1}. {q_data['question']}**")
            st.write(f"진행 상황: {len(responses)}/{GEMINI_MIN_PROFILE_QUESTIONS} (최대 {GEMINI_MAX_PROFILE_QUESTIONS}문항)")
            _render_profile_keyword_explanations([q_data["question"], *q_choices], current_q)

            selected_index = st.radio(
                "선택지",
                range(len(q_choices)),
                format_func=lambda i: q_choices[i],
                index=responses.get(current_q) if current_q in responses else None,
                key=f"choice_{current_q}",
                label_visibility="collapsed",
            )

            subquestions = st.session_state.setdefault("gemini_profile_subquestions", {})
            sub_state = subquestions.get(current_q)
            sub_complete = selected_index is not None

            if selected_index is not None:
                previous_answer = st.session_state["profile_responses"].get(current_q)
                if previous_answer != selected_index:
                    st.session_state["profile_responses"][current_q] = selected_index
                    sub_state = {
                        "main_answer": selected_index,
                        "items": [],
                        "complete": not can_have_subquestions,
                    }
                    subquestions[current_q] = sub_state
                elif sub_state is None:
                    sub_state = {
                        "main_answer": selected_index,
                        "items": [],
                        "complete": not can_have_subquestions,
                    }
                    subquestions[current_q] = sub_state
                elif not can_have_subquestions:
                    sub_state["items"] = []
                    sub_state["complete"] = True

                if can_have_subquestions and not sub_state.get("complete", False):
                    items = sub_state.setdefault("items", [])
                    for sub_idx, sub_data in enumerate(items):
                        st.write(f"↳ **추가 질문 {sub_idx + 1}. {sub_data['question']}**")
                        saved_answer = sub_data.get("answer")
                        sub_selected = st.radio(
                            "하위 선택지",
                            range(len(sub_data["choices"])),
                            format_func=lambda i, choices=sub_data["choices"]: choices[i],
                            index=saved_answer if saved_answer is not None else None,
                            key=f"sub_choice_{current_q}_{sub_idx}",
                            label_visibility="collapsed",
                        )
                        if sub_selected is not None and sub_data.get("answer") != sub_selected:
                            sub_data["answer"] = sub_selected
                            subquestions[current_q] = sub_state
                            st.rerun()

                    all_sub_answered = all(item.get("answer") is not None for item in items)
                    if all_sub_answered:
                        if len(items) >= GEMINI_MAX_SUBQUESTIONS_PER_MAIN:
                            sub_state["complete"] = True
                        else:
                            answers = _profile_answers_for_gemini(
                                st.session_state["profile_responses"],
                                gemini_questions,
                                subquestions,
                            )
                            sub_answers = []
                            for item in items:
                                answer_idx = item.get("answer")
                                sub_answers.append({
                                    "question": item.get("question", ""),
                                    "answer": item["choices"][answer_idx],
                                })
                            try:
                                with st.spinner("추가 확인이 필요한지 확인 중입니다..."):
                                    next_sub = _generate_gemini_profile_subquestion(
                                        answers,
                                        main_question_number,
                                        q_data["question"],
                                        q_choices[selected_index],
                                        sub_answers,
                                    )
                                if next_sub.get("needs_sub_question"):
                                    items.append({
                                        "question": next_sub["question"],
                                        "choices": next_sub["choices"],
                                    })
                                    subquestions[current_q] = sub_state
                                    st.rerun()
                                else:
                                    sub_state["complete"] = True
                            except Exception as exc:
                                _activate_fixed_profile_fallback()
                                st.warning(f"Gemini 하위 질문 생성에 실패해 기존 고정 질문으로 진행합니다. ({type(exc).__name__}: {exc})")
                                st.rerun()

                sub_complete = bool(sub_state.get("complete", False)) and all(
                    item.get("answer") is not None for item in sub_state.get("items", [])
                )

            cols = st.columns([2, 2, 1])
            nav_cols = cols[0].columns(2)
            if nav_cols[0].button("이전", key="q_prev"):
                if current_q in st.session_state["profile_responses"]:
                    del st.session_state["profile_responses"][current_q]
                st.session_state.setdefault("gemini_profile_subquestions", {}).pop(current_q, None)
                if current_idx > 0:
                    st.session_state["current_question"] = f"G{current_idx}"
                    st.rerun()
            if nav_cols[1].button("처음으로", key="q_home"):
                reset_app_to_start()

            with cols[2]:
                if st.button("다음", key="q_next", disabled=not sub_complete):
                    st.session_state["profile_responses"][current_q] = selected_index
                    responses = st.session_state["profile_responses"]
                    answers = _profile_answers_for_gemini(
                        responses,
                        gemini_questions,
                        st.session_state.get("gemini_profile_subquestions", {}),
                    )
                    answered_count = len(answers)

                    try:
                        should_finish = answered_count >= GEMINI_MAX_PROFILE_QUESTIONS
                        next_question = None
                        if not should_finish and answered_count >= GEMINI_MIN_PROFILE_QUESTIONS:
                            next_question = _generate_gemini_profile_question(answers, answered_count + 1)
                            should_finish = next_question.get("is_complete", False)
                        elif not should_finish:
                            next_question = _generate_gemini_profile_question(answers, answered_count + 1)

                        if should_finish:
                            profile_weights, funding = _generate_gemini_profile_result(answers)
                            st.session_state["profile_weights"] = profile_weights
                            st.session_state["profile_responses"]["Q8"] = _funding_situation_to_q8(funding)
                            st.session_state["gemini_profile_result"] = {
                                "answers": answers,
                                "funding_situation": funding,
                                "buy_mode": funding.get("buy_mode", ""),
                            }
                            st.session_state["profile_submitted"] = True
                            st.session_state["selected_etfs"] = []
                            st.session_state["app_step"] = 2
                            st.rerun()

                        gemini_questions.append(next_question)
                        st.session_state["current_question"] = f"G{answered_count + 1}"
                        st.rerun()
                    except Exception as exc:
                        _activate_fixed_profile_fallback()
                        st.warning(f"Gemini 응답 처리에 실패해 기존 고정 질문으로 진행합니다. ({type(exc).__name__}: {exc})")
                        st.rerun()
        else:
            st.caption("Gemini를 사용할 수 없어 기존 고정 질문으로 진행합니다.")
            current_q = st.session_state.get("current_question", "Q1")
            
            # 현재 질문 텍스트와 선택지 가져오기
            q_text, q_choices = get_question_text_and_choices(current_q)
            
            if q_text is None:
                # 모든 질문 완료
                st.success("✅ 모든 질문을 완료했습니다!")
                profile_weights = calculate_profile_scores(responses)
                st.session_state["profile_weights"] = profile_weights
                st.session_state["profile_submitted"] = True
                
                # 분석 결과 표시
                st.write("### 투자 성향 분석 결과")
                render_profile_metrics(profile_weights)
                
                if st.button("STEP2로 이동", key="proceed_to_step2"):
                    st.session_state["app_step"] = 2
                    st.rerun()
            else:
                q_data = AI_PROFILE_QUESTIONS[current_q]
                is_multi = _is_multi_select_question(current_q)

                # 질문 제목과 텍스트
                st.write(q_text)
                _render_profile_keyword_explanations([q_text, *q_choices], current_q)
                
                # 진행 상황
                completed = len(responses)
                total_questions = count_total_profile_questions(responses)
                st.write(f"진행 상황: {completed}/{total_questions}")
                
                # 질문별 설명 또는 힌트
                if current_q == "Q1":
                    st.info("💡 이 질문은 당신의 가장 중요한 투자 목표를 파악합니다.")
                elif current_q == "Q3":
                    st.info("💡 현금이 들어오는 느낌과 자산이 커지는 느낌 중 어느 것이 더 좋은지 선택해 주세요.")
                elif current_q == "Q4":
                    st.warning("⚠️ 이 질문은 당신의 위험 감수 능력을 파악하는 중요한 질문입니다.")
                elif current_q == "Q5":
                    st.info("💡 Q4의 답변에 따라 나타나는 심화 질문입니다.")
                elif current_q == "Q7":
                    st.markdown(
                        """
                        <div style="background-color:#f0f7ff;padding:12px;border-left:4px solid #1f77b4;border-radius:4px;margin-bottom:16px;">
                        <strong>💡 알아두세요:</strong> <strong>배당집중</strong>은 월배당처럼 정기적인 현금흐름을 받는 전략입니다. 
                        <strong>레버리지 ETF</strong>는 2-3배 움직이는 고위험 상품입니다.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif current_q == "Q7_SUB":
                    st.markdown(
                        """
                        <div style="background-color:#fff3cd;padding:12px;border-left:4px solid #ff9800;border-radius:4px;margin-bottom:16px;">
                        <strong>⚠️ 레버리지 ETF 이해도 측정:</strong> 아래 질문을 통해 당신의 이해도를 파악하고 적절한 비중을 추천합니다.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.expander("📖 레버리지 ETF 간단 설명 보기"):
                        st.markdown(LEVERAGE_EXPLANATION)

                if is_multi:
                    max_select = q_data.get("max_select", PROFILE_MULTI_SELECT_MAX)
                    st.caption(f"해당되는 항목을 **최소 1개, 최대 {max_select}개**까지 선택하세요.")
                    saved = responses.get(current_q, [])
                    if not isinstance(saved, list):
                        saved = []
                    selected_indices = st.multiselect(
                        "선택 (복수)",
                        options=list(range(len(q_choices))),
                        default=saved,
                        format_func=lambda i: q_choices[i],
                        max_selections=max_select,
                        key=f"choice_{current_q}",
                        label_visibility="collapsed",
                    )
                else:
                    selected_indices = st.radio(
                        "선택지",
                        range(len(q_choices)),
                        format_func=lambda i: q_choices[i],
                        index=responses.get(current_q, 0) if current_q in responses else 0,
                        key=f"choice_{current_q}",
                        label_visibility="collapsed",
                    )

                cols = st.columns([2, 2, 1])
                nav_cols = cols[0].columns(2)
                if nav_cols[0].button("이전", key="q_prev"):
                    if current_q in st.session_state["profile_responses"]:
                        del st.session_state["profile_responses"][current_q]
                    question_order = get_profile_question_order(st.session_state["profile_responses"])
                    try:
                        current_idx = question_order.index(current_q)
                        if current_idx > 0:
                            st.session_state["current_question"] = question_order[current_idx - 1]
                            st.rerun()
                    except ValueError:
                        pass
                if nav_cols[1].button("처음으로", key="q_home"):
                    reset_app_to_start()

                with cols[2]:
                    if st.button("다음", key="q_next"):
                        if is_multi:
                            if not selected_indices:
                                st.warning("최소 1개 이상 선택해 주세요.")
                                st.stop()
                            st.session_state["profile_responses"][current_q] = list(selected_indices)
                        else:
                            st.session_state["profile_responses"][current_q] = selected_indices

                        next_q = get_next_question(current_q, st.session_state["profile_responses"])

                        if next_q is None:
                            profile_weights = calculate_profile_scores(st.session_state["profile_responses"])
                            st.session_state["profile_weights"] = profile_weights
                            st.session_state["profile_submitted"] = True
                            st.session_state["app_step"] = 2
                            st.rerun()
                        else:
                            st.session_state["current_question"] = next_q
                            st.rerun()

        st.markdown("---")
        if st.button("관리자모드", key="admin_mode_step1"):
            activate_admin_mode()

    elif current_step == 2:
        if not profile_weights:
            st.warning("STEP1을 먼저 완료해 주세요.")
            if st.button("STEP1로 이동", key="goto1"):
                move_step(1)
            return

        if not selected_etfs:
            selected_etfs = recommend_etfs(profile_weights, top_n=5)
            # 비중 계산
            etf_weights = recommend_etfs_with_weights(profile_weights, top_n=5)
            st.session_state["etf_weights"] = etf_weights
            st.session_state["selected_etfs"] = selected_etfs
        else:
            etf_weights = st.session_state.get("etf_weights", {})

        # 투자 성향 분석 결과 표시
        st.write("### 📊 당신의 투자 성향")
        render_profile_metrics(profile_weights)

        if profile_weights.get("배당집중", 0) >= 15:
            st.info(
                f"배당집중 성향이 **{profile_weights.get('배당집중', 0):.1f}%**입니다. "
                "월배당·프리미엄 수익을 노리는 ETF가 포함될 수 있습니다. "
                "위 **「배당집중이란?」** 안내를 참고해 주세요."
            )

        st.markdown("---")
        
        st.write("### 🎯 추천 포트폴리오 (5개 ETF)")
        st.write(f"아래는 당신의 투자 성향에 맞게 선별된 ETF들입니다. 각 ETF의 비중은 당신의 성향 점수에 따라 자동으로 배분되었습니다.")
        
        # 포트폴리오 테이블
        portfolio_data = []
        for idx, ticker in enumerate(selected_etfs, 1):
            info = ETF_DATA.get(ticker, {})
            weight = etf_weights.get(ticker, 1/len(selected_etfs))
            weight_pct = round(weight * 100, 1)
            
            portfolio_data.append({
                "순위": idx,
                "ETF": ticker,
                "이름": info.get("이름", ""),
                "카테고리": info.get("카테고리", ""),
                "비중": f"{weight_pct}%"
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 각 ETF 상세 설명
        st.write("### 📋 ETF 상세 정보")
        
        for ticker in selected_etfs:
            info = ETF_DATA.get(ticker, {})
            weight = etf_weights.get(ticker, 1/len(selected_etfs))
            weight_pct = round(weight * 100, 1)
            detail = ETF_DETAIL_GUIDE.get(ticker, {})
            live_snapshot = _get_live_etf_snapshot(ticker)
            usdkrw = _get_usdkrw_rate()
            annual_return = live_snapshot.get("annual_return_pct")
            if annual_return is None:
                annual_return = _get_etf_annual_return(ticker)
            current_price = live_snapshot.get("price_usd")
            dividend_yield = live_snapshot.get("dividend_yield_pct")
            if dividend_yield is None and info.get("배당"):
                dividend_yield = 2.0
            risk_stars = _get_risk_stars(ticker)
            risk_text = _format_risk_label(risk_stars)
            category_label = CATEGORY_DISPLAY_LABELS.get(info.get("카테고리", ""), info.get("카테고리", ""))
            
            with st.expander(f"**{ticker}** - {info.get('이름', '')} ({weight_pct}%)"):
                st.markdown(f"**{ETF_SUMMARY_DESCRIPTIONS.get(ticker, info.get('이름', ''))}**")
                if detail.get("intro"):
                    st.write(detail["intro"])

                st.write(f"• **분류(카테고리):** {category_label}")
                st.write(f"• **역사적 연평균 수익률(추정):** {annual_return:.1f}%")

                if current_price is not None:
                    krw_price = current_price * usdkrw
                    st.write(
                        f"• **현재 주가:** ${current_price:,.2f} (약 {int(krw_price):,}원)"
                    )
                else:
                    st.write("• **현재 주가:** 데이터를 불러오지 못했습니다.")

                if dividend_yield is not None:
                    st.write(f"• **배당수익률:** 연 {dividend_yield:.1f}%")
                else:
                    st.write("• **배당수익률:** 정보 없음")

                st.write(f"• **위험도:** {'★' * risk_stars}{'☆' * (5 - risk_stars)} ({risk_text})")
                st.write(f"• **🏦 운용보수:** 연 {info.get('보수율', 0)*100:.2f}%")

                recommended_for = detail.get("recommended_for", [])
                if recommended_for:
                    st.write("**이런 분께 추천:**")
                    for item in recommended_for:
                        st.write(f"→ {item}")

                top_holdings = detail.get("top_holdings", [])
                if top_holdings:
                    st.write("**대표 보유 종목:**")
                    st.write(f"→ {', '.join(top_holdings)}")

                caution = detail.get("caution")
                if caution:
                    st.write("**주의사항:**")
                    st.write(f"→ {caution}")
        
        st.markdown("---")

        st.write("### 🔄 종목 교체")
        st.caption(
            "교체는 **같은 카테고리** ETF끼리만 가능합니다. "
            "아래 목록에서 종목을 고른 뒤 **선택한 ETF로 교체**를 누르세요."
        )
        st.markdown(
            """
            <style>
            /* 종목 교체 라디오: 세로 레이아웃으로 가독성 향상 */
            div[data-testid="stRadio"] label {
                white-space: pre-line !important;
                line-height: 1.55 !important;
                padding: 1.0rem 0.75rem !important;
                align-items: flex-start !important;
                border-left: 3px solid #ddd !important;
                margin-bottom: 0.5rem !important;
                background-color: #f9f9f9 !important;
                border-radius: 4px !important;
            }
            div[data-testid="stRadio"] label:hover {
                background-color: #f0f0f0 !important;
                border-left-color: #1f77b4 !important;
            }
            div[data-testid="stRadio"] label p {
                white-space: pre-line !important;
                line-height: 1.55 !important;
                margin: 0 !important;
            }
            div[data-testid="stRadio"] > div {
                gap: 0.5rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for slot_idx, current_ticker in enumerate(selected_etfs):
            category = ETF_DATA.get(current_ticker, {}).get("카테고리", "")
            replace_options = get_same_category_replacement_options(current_ticker, selected_etfs)
            radio_key = _replace_radio_key(slot_idx)

            st.subheader(f"📌 {current_ticker} · {category}")
            st.write(f"포트폴리오 **{slot_idx + 1}번** 종목 — 같은 카테고리 ETF 중에서 선택")

            # 현재 종목의 설명 표시
            with st.expander(f"📊 {current_ticker} 상세 정보 보기"):
                current_info = ETF_DATA.get(current_ticker, {})
                st.write(f"**이름:** {current_info.get('이름', 'N/A')}")
                st.write(f"**카테고리:** {current_info.get('카테고리', 'N/A')}")
                st.write(f"**보수율:** {current_info.get('보수율', 0) * 100:.2f}%")
                st.write(f"**배당:** {'있음' if current_info.get('배당') else '없음'}")
                st.write(f"**레버리지:** {'있음 (위험)' if current_info.get('레버리지') else '없음'}")
                if current_ticker in ETF_SUMMARY_DESCRIPTIONS:
                    st.info(f"📝 {ETF_SUMMARY_DESCRIPTIONS[current_ticker]}")

            if len(replace_options) <= 1:
                st.info(f"**{category}** 카테고리에 선택 가능한 다른 ETF가 없습니다.")
                st.markdown("---")
                continue

            if radio_key not in st.session_state:
                default_idx = 1 if len(replace_options) > 1 else 0
                st.session_state[radio_key] = default_idx
            else:
                default_idx = 1 if len(replace_options) > 1 else 0
                st.session_state[radio_key] = _resolve_replace_index(
                    st.session_state.get(radio_key),
                    replace_options,
                    default_idx,
                )

            picked_idx = st.radio(
                "교체할 ETF",
                options=list(range(len(replace_options))),
                format_func=lambda i, opts=replace_options, cur=current_ticker: _format_replace_option_label(
                    opts[i], cur
                ),
                key=radio_key,
                label_visibility="collapsed",
            )
            picked_ticker = replace_options[picked_idx]

            # 교체 후보 ETF들의 비교 정보
            with st.expander("💡 교체 가능한 ETF 비교"):
                comparison_data = []
                for option_ticker in replace_options:
                    info = ETF_DATA.get(option_ticker, {})
                    comparison_data.append({
                        "종목": option_ticker,
                        "이름": info.get("이름", "")[:40] + "..." if len(info.get("이름", "")) > 40 else info.get("이름", ""),
                        "보수율(%)": f"{info.get('보수율', 0) * 100:.2f}",
                        "배당": "O" if info.get("배당") else "X",
                        "상태": "현재 보유" if option_ticker == current_ticker else "선택 가능"
                    })
                st.dataframe(comparison_data, use_container_width=True, hide_index=True)

            error_key = f"replace_error_{slot_idx}"
            if error_key in st.session_state:
                st.error(st.session_state.pop(error_key))

            msg_key = f"replace_msg_{slot_idx}"
            if msg_key in st.session_state:
                st.success(f"✅ {st.session_state.pop(msg_key)}")

            replace_disabled = picked_ticker == current_ticker
            if st.button(
                "선택한 ETF로 교체",
                key=f"replace_apply_{slot_idx}",
                type="primary",
                use_container_width=True,
                disabled=replace_disabled,
                help="같은 카테고리 ETF로만 교체됩니다.",
            ):
                _apply_etf_replace(slot_idx, current_ticker, replace_options)
                st.rerun()

            if replace_disabled:
                st.caption("현재 보유 종목과 같습니다. 목록에서 다른 ETF를 선택해 주세요.")
            else:
                st.caption(
                    f"선택됨: **{picked_ticker}** → **{current_ticker}** 자리에 반영됩니다."
                )

            st.markdown("---")

        step_button_row()

    elif current_step == 4:
        # 종합 투자 전략 보고서
        if not profile_weights:
            st.warning("STEP1을 먼저 완료해 주세요.")
            if st.button("STEP1로 이동", key="goto1c"):
                move_step(1)
            return
        if not selected_etfs:
            st.warning("STEP2에서 ETF 추천을 먼저 받아오세요.")
            if st.button("STEP2로 이동", key="goto2c"):
                move_step(2)
            return

        st.markdown("## 📋 종합 투자 전략 보고서")
        st.markdown("**STEP 1~3 분석 결과를 바탕으로 한 맞춤형 투자 전략입니다.**")
        st.markdown("---")

        # ===== 섹션 1: 투자 성향 분석 요약 =====
        st.markdown("### 📊 1. 투자 성향 분석 결과")
        
        col1, col2, col3, col4 = st.columns(4)
        for col, category in zip([col1, col2, col3, col4], AI_PROFILE_CATEGORIES):
            pct = profile_weights.get(category, 0)
            col.metric(category, f"{pct:.0f}%")
        
        top_categories = sorted(
            [(cat, profile_weights.get(cat, 0)) for cat in AI_PROFILE_CATEGORIES],
            key=lambda x: x[1],
            reverse=True
        )
        top_cat = top_categories[0][0]
        top_pct = top_categories[0][1]
        
        description = ""
        if top_cat == "지수추적":
            description = "📈 시장 지수를 따라가며 안정적인 장기 수익을 추구하는 성향입니다."
        elif top_cat == "배당성장":
            description = "💰 정기적인 배당 수익을 통해 안정적인 현금흐름을 원하는 성향입니다."
        elif top_cat == "배당집중":
            description = "📅 월배당·프리미엄 수익을 통해 규칙적인 추가 수익을 기대하는 성향입니다."
        elif top_cat == "레버리지":
            description = "🚀 높은 수익을 추구하되 높은 변동성을 감수할 수 있는 공격적 성향입니다."
        
        st.write(f"**최상위 성향:** {top_cat} ({top_pct:.0f}%)")
        st.write(description)

        st.markdown("---")

        # ===== 섹션 2: 선택된 포트폴리오 =====
        st.markdown("### 🎯 2. 추천 포트폴리오 구성")
        
        etf_weights = st.session_state.get("etf_weights", {})
        def fmt_money(n):
            try:
                f = float(n)
            except Exception:
                return "0"
            if f.is_integer():
                return f"{int(f):,}"
            s = f"{f:,}"
            if "." in s:
                s = s.rstrip('0').rstrip('.')
            return s
        
        portfolio_data = []
        for idx, ticker in enumerate(selected_etfs, 1):
            info = ETF_DATA.get(ticker, {})
            weight = etf_weights.get(ticker, 1/len(selected_etfs))
            weight_pct = round(weight * 100, 1)
            portfolio_data.append({
                "순위": idx,
                "ETF": ticker,
                "이름": info.get("이름", "")[:35],
                "카테고리": info.get("카테고리", ""),
                "비중": f"{weight_pct}%"
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        
        try:
            portfolio_weights = build_portfolio_weights(selected_etfs, etf_weights)
        except Exception:
            portfolio_weights = {t: 1.0 / max(1, len(selected_etfs)) for t in selected_etfs}
        
        portfolio_avg_return = sum(
            portfolio_weights.get(ticker, 0) * ETF_DATA[ticker].get("cagr", 0.10)
            for ticker in portfolio_weights
        ) or 0.06
        
        leverage_count = sum(1 for t in selected_etfs if ETF_DATA.get(t, {}).get("레버리지", False))
        dividend_count = sum(1 for t in selected_etfs if ETF_DATA.get(t, {}).get("배당", False))
        
        st.write(f"• **예상 연평균 수익률:** {portfolio_avg_return * 100:.2f}%")
        st.write(f"• **배당 ETF 포함:** {dividend_count}개")
        st.write(f"• **레버리지 ETF 포함:** {leverage_count}개")

        st.markdown("---")

        # ===== 섹션 3: 투자 목표 및 계획 =====
        st.markdown("### 🎲 3. 투자 목표 및 시나리오")
        
        target_won = int(st.session_state.get("target_amount", 0))
        current_capital = float(st.session_state.get("current_capital", 0.0))
        monthly_won = int(st.session_state.get("monthly_contribution", 0))
        selected_mode = st.session_state.get("analysis_mode")
        dividend_report_months = int(float(st.session_state.get("years", st.session_state.get("target_years", 10))) * 12)
        
        goal_col1, goal_col2 = st.columns(2)
        with goal_col1:
            st.metric("현재 보유 자금", f"{fmt_money(int(current_capital))}원")
        with goal_col2:
            st.metric("목표 금액", f"{fmt_money(target_won)}원")
        
        st.write(f"**분석 모드:** {selected_mode}")
        
        if selected_mode == "목표 금액 달성":
            required_years = estimate_years_to_target(
                target_won,
                monthly_won,
                current_capital,
                avg_return=portfolio_avg_return,
            )
            
            st.write(f"• **월 적립 계획:** {fmt_money(monthly_won)}원")
            
            if current_capital >= target_won:
                st.success("✅ 현재 보유 자금만으로도 목표 달성 완료!")
                dividend_report_months = 0
            elif monthly_won <= 0:
                st.warning("⚠️ 월 적립금이 0입니다. 목표 달성에 필요한 월 적립금을 설정해 주세요.")
            elif not np.isfinite(required_years) or required_years == float("inf"):
                st.warning("⚠️ 현재 설정으로는 목표 달성이 어렵습니다.")
            else:
                total_months = int(round(required_years * 12))
                dividend_report_months = total_months
                years_calc = total_months // 12
                months_remain = total_months % 12
                st.write(f"• **예상 달성 기간:** {years_calc}년 {months_remain}개월")
                
        elif selected_mode == "목표 기간 확인":
            target_years_val = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
            start_date = st.session_state.get("sim_period_mode3_start", st.session_state.get("invest_period_start", date.today()))
            end_date = st.session_state.get("sim_period_mode3_end", st.session_state.get("invest_period_end"))
            
            st.write(f"• **투자 기간:** {format_period_label(start_date, end_date)}")
            
            months = int(target_years_val * 12)
            dividend_report_months = months
            monthly_r = portfolio_avg_return / 12
            future_current_capital = current_capital * np.power(1 + monthly_r, months)
            
            if future_current_capital >= target_won:
                st.success("✅ 현재 보유 자금만으로도 목표 달성 가능!")
            else:
                required_monthly = estimate_required_contribution(
                    target_won,
                    target_years_val,
                    current_capital,
                    avg_return=portfolio_avg_return,
                )
                if not np.isfinite(required_monthly):
                    required_monthly_int = 0
                else:
                    required_monthly_int = int(np.ceil(required_monthly))
                st.write(f"• **필요 월 적립금:** {fmt_money(required_monthly_int)}원")

        st.markdown("---")

        # ===== 투자기간 후 월 배당금 카드 =====
        st.markdown("### 💵 투자기간 후 월 배당금 예상")

        MONTHLY_DIVIDEND_ETFS = {"JEPI", "JEPQ", "QYLD"}
        dividend_yields = {
            ticker: dividend_yield
            for ticker in selected_etfs
            if (dividend_yield := _get_etf_dividend_yield(ticker)) is not None and dividend_yield >= 0.01
        }

        if not dividend_yields:
            st.info("배당수익률 1% 이상 ETF가 없어 월 배당금 예상치를 표시하지 않습니다.")
        else:
            total_months = max(0, int(dividend_report_months))
            monthly_r = portfolio_avg_return / 12
            buy_mode = _recommended_investment_mode_from_profile()
            base_value = 0.0 if buy_mode == "적립형" else current_capital
            monthly_add = monthly_won if buy_mode in {"적립형", "혼합형"} else 0

            projected_value = base_value
            cumulative_dividend = 0
            final_monthly_total = 0
            final_dividend_asset_value = 0
            for month in range(total_months + 1):
                monthly_total = 0
                dividend_asset_value = 0
                for ticker, dividend_yield in dividend_yields.items():
                    etf_asset_won = projected_value * portfolio_weights.get(ticker, 0)
                    dividend_asset_value += etf_asset_won
                    annual_dividend_won = etf_asset_won * dividend_yield
                    if ticker in MONTHLY_DIVIDEND_ETFS:
                        monthly_dividend_won = annual_dividend_won / 12
                    else:
                        monthly_dividend_won = (annual_dividend_won / 4) / 3
                    monthly_total += monthly_dividend_won
                final_monthly_total = monthly_total
                final_dividend_asset_value = dividend_asset_value
                if month < total_months:
                    cumulative_dividend += monthly_total
                    projected_value *= (1 + monthly_r)
                    projected_value += monthly_add

            dividend_weight_sum = sum(portfolio_weights.get(ticker, 0) for ticker in dividend_yields)
            dividend_invested_amount = (base_value + monthly_add * total_months) * dividend_weight_sum
            dividend_return_pct = (
                ((final_dividend_asset_value + cumulative_dividend) / dividend_invested_amount - 1) * 100
                if dividend_invested_amount > 0
                else 0
            )
            report_years = total_months // 12
            report_months = total_months % 12
            period_label = f"{report_years}년 후" if report_months == 0 else f"{report_years}년 {report_months}개월 후"
            monthly_dividend_won = int(round(final_monthly_total))
            final_dividend_asset_won = int(round(final_dividend_asset_value))
            cumulative_dividend_won = int(round(cumulative_dividend))

            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,#2f9f63 0%,#4aa36b 52%,#2f7d6d 100%);
                    border-radius:0;
                    padding:26px 28px 24px 28px;
                    color:white;
                    box-shadow:0 8px 20px rgba(0,0,0,0.12);
                ">
                    <div style="font-size:1.05rem;font-weight:700;margin-bottom:8px;opacity:0.95;">💸 {period_label} 매달 받는 배당금</div>
                    <div style="font-size:3.2rem;font-weight:900;line-height:1.05;letter-spacing:-1px;">{monthly_dividend_won:,}원</div>
                    <div style="font-size:0.9rem;margin-top:8px;opacity:0.78;">배당금은 세전 기준입니다.</div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:26px;">
                        <div style="background:rgba(255,255,255,0.16);border-radius:999px;padding:8px 14px;font-weight:700;">💼 배당 ETF 총 자산 {final_dividend_asset_won:,}원</div>
                        <div style="background:rgba(255,255,255,0.16);border-radius:999px;padding:8px 14px;font-weight:700;">↗ 배당 ETF 수익률 {dividend_return_pct:,.1f}%</div>
                        <div style="background:rgba(255,255,255,0.16);border-radius:999px;padding:8px 14px;font-weight:700;">💸 총 배당 {cumulative_dividend_won:,}원</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ===== 섹션 4: AI 성향 분석 결과 =====
        st.markdown("### 💡 4. AI 성향 분석 결과")

        investment_method = _recommended_investment_mode_from_profile().replace("형", "식")
        analysis_years = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
        selected_etfs_for_prompt = [
            {
                "ETF": ticker,
                "비중": f"{portfolio_weights.get(ticker, 0) * 100:.1f}%",
            }
            for ticker in selected_etfs
        ]

        try:
            with st.spinner("AI 분석 중입니다. 잠시만 기다려 주세요..."):
                ai_analysis = _generate_gemini_step4_analysis(
                    profile_weights,
                    selected_etfs_for_prompt,
                    investment_method,
                    target_won,
                    analysis_years,
                )
        except Exception:
            ai_analysis = _default_step4_analysis_text(
                profile_weights,
                selected_etfs,
                investment_method,
                target_won,
                analysis_years,
            )

        st.info(ai_analysis)

        st.markdown("---")

        step_button_row()

    elif current_step == 3:
        if not profile_weights:
            st.warning("STEP1을 먼저 완료해 주세요.")
            if st.button("STEP1로 이동", key="goto1b"):
                move_step(1)
            return

        st.markdown("### 💼 투자 목표 설정 및 분석")
        st.markdown("**분석 모드를 먼저 선택한 뒤 필요한 투자 정보를 입력하세요.**")
        st.markdown("---")

        def fmt_money(n):
            try:
                f = float(n)
            except Exception:
                return "0"
            if f.is_integer():
                return f"{int(f):,}"
            s = f"{f:,}"
            if "." in s:
                s = s.rstrip('0').rstrip('.')
            return s

        step3_selected_etfs = st.session_state.get("selected_etfs", [])
        step3_etf_weights = st.session_state.get("etf_weights", {})
        step3_portfolio_weights = build_portfolio_weights(step3_selected_etfs, step3_etf_weights) if step3_selected_etfs else {}
        step3_avg_return = sum(
            step3_portfolio_weights.get(ticker, 0) * ETF_DATA[ticker].get("cagr", 0.10)
            for ticker in step3_portfolio_weights
        ) or 0.06

        # ===== 분석 모드 선택 =====
        st.markdown("#### 🎯 1단계: 분석 모드 선택")

        st.markdown(
            """
            <style>
            .analysis-mode-card {
                border: 1.5px solid #d9e2ec;
                border-radius: 18px;
                padding: 22px 20px;
                min-height: 168px;
                background: #ffffff;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
                transition: all 0.2s ease;
            }
            .analysis-mode-card.selected {
                border-color: #4f8cff;
                background: linear-gradient(135deg, #eef5ff 0%, #ffffff 100%);
                box-shadow: 0 10px 26px rgba(79, 140, 255, 0.18);
            }
            .analysis-mode-title {
                font-size: 1.15rem;
                font-weight: 800;
                margin-bottom: 10px;
                color: #172033;
            }
            .analysis-mode-question {
                font-size: 0.98rem;
                font-weight: 700;
                color: #315174;
                margin-bottom: 10px;
            }
            .analysis-mode-desc {
                font-size: 0.9rem;
                color: #526173;
                line-height: 1.55;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.caption("어떤 방식으로 목표를 확인하고 싶은지 선택해 주세요.")

        selected_analysis_mode = st.session_state.get("analysis_mode")
        mode_col1, mode_col2 = st.columns(2)

        with mode_col1:
            selected_class = " selected" if selected_analysis_mode == "목표 금액 달성" else ""
            st.markdown(
                f"""
                <div class="analysis-mode-card{selected_class}">
                    <div class="analysis-mode-title">🏁 목표 금액 달성</div>
                    <div class="analysis-mode-question">“원하는 금액을 만들려면?”</div>
                    <div class="analysis-mode-desc">
                        목표 금액까지 가려면 현재 자금과 월 적립금으로 얼마나 걸릴지 확인합니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("목표 금액 달성 선택", use_container_width=True, key="select_target_amount_mode"):
                selected_analysis_mode = "목표 금액 달성"

        with mode_col2:
            selected_class = " selected" if selected_analysis_mode == "목표 기간 확인" else ""
            st.markdown(
                f"""
                <div class="analysis-mode-card{selected_class}">
                    <div class="analysis-mode-title">📅 목표 기간 확인</div>
                    <div class="analysis-mode-question">“이 기간 동안 가능할까?”</div>
                    <div class="analysis-mode-desc">
                        정해진 투자 기간 동안 목표 금액에 가까워질 수 있는지 확인합니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("목표 기간 확인 선택", use_container_width=True, key="select_target_period_mode"):
                selected_analysis_mode = "목표 기간 확인"

        st.session_state["analysis_mode"] = selected_analysis_mode

        if selected_analysis_mode is None:
            st.info("분석 모드를 선택하면 입력창이 표시됩니다.")
            st.markdown("---")
            step_button_row()
            return

        st.markdown("---")

        # ===== 입력 섹션 (선택한 모드에 따라 표시) =====
        st.markdown("#### 📊 2단계: 투자 정보 입력")

        if selected_analysis_mode == "목표 금액 달성":
            col1, col2 = st.columns(2)
            with col1:
                target_won = render_manwon_amount_input(
                    "목표 금액",
                    "unified_target_amount_manwon",
                    int(st.session_state.get("target_amount", 500000000)),
                )
            with col2:
                monthly_won = render_manwon_amount_input(
                    "월 적립금",
                    "unified_monthly_contribution_manwon",
                    int(st.session_state.get("monthly_contribution", 100000)),
                )

            current_capital = render_manwon_amount_input(
                "현재 보유 자금",
                "unified_current_capital_manwon",
                int(st.session_state.get("current_capital", 0)),
            )

        elif selected_analysis_mode == "목표 기간 확인":
            st.markdown("#### ⏰ 투자 기간")
            target_years_val = render_investment_period_selector(
                key_prefix="unified_sim_period",
                section_title="투자 기간 설정",
                period_label="투자 기간",
            )

            col1, col2 = st.columns(2)
            with col1:
                monthly_won = render_manwon_amount_input(
                    "월 적립금",
                    "unified_monthly_contribution_manwon",
                    int(st.session_state.get("monthly_contribution", 100000)),
                )
            with col2:
                current_capital = render_manwon_amount_input(
                    "현재 보유 자금",
                    "unified_current_capital_manwon",
                    int(st.session_state.get("current_capital", 0)),
                )
        
        st.markdown("---")
        
        # ===== 분석 결과 표시 =====
        st.markdown("#### 📈 3단계: 분석 결과")
        
        # 기본 정보 표시
        if selected_analysis_mode == "목표 금액 달성":
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("목표 금액", f"{fmt_money(target_won)}원")
            with info_cols[1]:
                st.metric("월 적립금", f"{fmt_money(monthly_won)}원")
            with info_cols[2]:
                st.metric("현재 자금", f"{fmt_money(int(current_capital))}원")
        elif selected_analysis_mode == "목표 기간 확인":
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("투자 기간", f"{target_years_val:.1f}년")
            with info_cols[1]:
                st.metric("월 적립금", f"{fmt_money(monthly_won)}원")
            with info_cols[2]:
                st.metric("현재 자금", f"{fmt_money(int(current_capital))}원")

        st.markdown("")
        
        # 선택한 모드에 따라 분석
        if selected_analysis_mode == "목표 금액 달성":
            st.markdown("**📍 분석 모드: 목표 금액 달성**")
            st.write("월 적립금을 고정하고, 목표 금액 달성까지 걸리는 기간을 계산합니다.")
            
            required_years = estimate_years_to_target(
                target_won,
                monthly_won,
                current_capital,
                avg_return=step3_avg_return,
            )
            
            st.session_state["target_amount"] = target_won
            st.session_state["monthly_contribution"] = monthly_won
            st.session_state["current_capital"] = current_capital
            
            if current_capital >= target_won:
                st.success("✅ 현재 보유 자금만으로도 목표 달성 완료!")
            elif monthly_won <= 0:
                st.warning("⚠️ 월 적립금이 0입니다.")
            elif not np.isfinite(required_years) or required_years == float("inf"):
                st.warning("⚠️ 현재 설정으로는 목표 달성이 어렵습니다. 월 적립금을 증가시켜 주세요.")
            else:
                total_months = int(round(required_years * 12))
                years_calc = total_months // 12
                months_remain = total_months % 12
                
                st.session_state["years"] = total_months / 12.0
                
                st.markdown(
                    f"""
                    <div style="
                        background:linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        border-radius:12px;
                        padding:24px;
                        text-align:center;
                        color:white;
                    ">
                        <div style="font-size:0.95rem;opacity:0.9;margin-bottom:10px;font-weight:600;">📅 예상 달성 기간</div>
                        <div style="font-size:2.2rem;font-weight:800;margin-bottom:6px;">{years_calc}년 {months_remain}개월</div>
                        <div style="font-size:0.9rem;opacity:0.85;">연평균 {step3_avg_return*100:.2f}% 수익률 기준</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        elif selected_analysis_mode == "목표 기간 확인":
            st.markdown("**📍 분석 모드: 목표 기간 확인**")
            st.write("투자 기간을 고정하고, 해당 기간 후 달성 가능한 금액을 계산합니다.")
            
            st.session_state["monthly_contribution"] = monthly_won
            st.session_state["current_capital"] = current_capital
            st.session_state["target_years"] = target_years_val
            st.session_state["years"] = target_years_val
            
            months = int(target_years_val * 12)
            monthly_r = step3_avg_return / 12
            future_current_capital = current_capital * np.power(1 + monthly_r, months)
            future_monthly_contribution = monthly_won * (
                (np.power(1 + monthly_r, months) - 1) / monthly_r
                if monthly_r != 0
                else months
            )
            achievable_amount = int(future_current_capital + future_monthly_contribution)
            
            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                    border-radius:12px;
                    padding:24px;
                    text-align:center;
                    color:white;
                ">
                    <div style="font-size:0.95rem;opacity:0.9;margin-bottom:10px;font-weight:600;">💰 예상 달성 금액</div>
                    <div style="font-size:2.2rem;font-weight:800;margin-bottom:6px;">{fmt_money(achievable_amount)}원</div>
                    <div style="font-size:0.9rem;opacity:0.85;">연평균 {step3_avg_return*100:.2f}% 수익률 기준</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        step_button_row()

    elif current_step == 6:
        if not profile_weights:
            st.warning("STEP1을 먼저 완료해 주세요.")
            if st.button("STEP1로 이동", key="goto1d"):
                move_step(1)
            return
        if not selected_etfs:
            st.warning("STEP2에서 ETF 추천을 먼저 받아오세요.")
            if st.button("STEP2로 이동", key="goto2d"):
                move_step(2)
            return

        etf_weights = st.session_state.get("etf_weights", {})
        port_weights = build_portfolio_weights(selected_etfs, etf_weights)
        years_sim = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
        current_capital = float(st.session_state.get("current_capital", 0.0))
        monthly_won = int(st.session_state.get("monthly_contribution", 0))
        recommended_mode = _recommended_investment_mode_from_profile()
        start_date = st.session_state.get("unified_sim_period_start", st.session_state.get("invest_period_start", date.today()))
        if "unified_sim_period_end" in st.session_state:
            end_date = st.session_state["unified_sim_period_end"]
        else:
            end_date = add_calendar_years(start_date, max(1, int(np.ceil(years_sim))))
        st.write("### 결과 시각화")
        try:
            with st.spinner("시뮬레이션 계산 중..."):
                fig = plot_portfolio_vs_sp500_monte_carlo(
                    port_weights,
                    start_date,
                    end_date,
                    current_capital,
                    monthly_won,
                    recommended_mode,
                )
            render_start = time.perf_counter()
            st.plotly_chart(fig, use_container_width=True)
            # region agent log
            _agent_debug_log(
                "pre-fix",
                "H5",
                "app.py:run_streamlit_app:plotly_chart",
                "Plotly chart rendered",
                {"duration_ms": round((time.perf_counter() - render_start) * 1000, 2)},
            )
            # endregion
            st.info(
                "📌 급락 구간 기준:\n"
                "내 포트폴리오 중앙값이 전월 대비 15% 이상 \n"
                "하락한 구간을 급락 구간으로 표시합니다.\n\n"
                "💡 급락 구간은 추가 매수 기회입니다.\n"
                "여유 현금이 있다면 해당 시점에 추가 투자를 \n"
                "고려해보세요. 장기적으로 저점 매수 효과를 \n"
                "기대할 수 있습니다."
            )
        except RuntimeError as exc:
            st.error(str(exc))
        cols = st.columns([2, 2, 1])
        nav_cols = cols[0].columns(2)
        if nav_cols[0].button("이전", key="prev_6"):
            move_step(4)
        if nav_cols[1].button("처음으로", key="home_6"):
            reset_app_to_start()


if __name__ == "__main__":
    run_streamlit_app()

