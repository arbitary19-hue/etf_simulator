# -*- coding: utf-8 -*-
import calendar
import hashlib
import json
import os
import re
import time
from datetime import date
import numpy as np
import anthropic
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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

# region agent log
DEBUG_B8B036_LOG_PATH = "debug-b8b036.log"
DEBUG_B8B036_SESSION_ID = "b8b036"


def _debug_b8b036_log(run_id, hypothesis_id, location, message, data):
    try:
        payload = {
            "sessionId": DEBUG_B8B036_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        line = json.dumps(payload, ensure_ascii=False)
        print(f"[debug-b8b036] {line}", flush=True)
        with open(DEBUG_B8B036_LOG_PATH, "a", encoding="utf-8") as debug_file:
            debug_file.write(line + "\n")
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


def _close_series(hist):
    """yfinance history()의 Close 컬럼을 항상 1차원 numeric Series로 반환합니다.
    yfinance 버전에 따라 hist['Close']가 DataFrame으로 반환될 수 있어 squeeze() 처리합니다."""
    col = hist["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col.squeeze(), errors="coerce")


@st.cache_data
def load_etf_cagr():
    import yfinance as yf
    for ticker in ETF_DATA:
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period="max", auto_adjust=True)
            if hist is not None and len(hist) > 0:
                _close = _close_series(hist).dropna()
                if _close.empty:
                    continue
                start = _close.iloc[0]
                end = _close.iloc[-1]
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
LEVERAGE_MAX_PORTFOLIO_WEIGHT = 0.50


def _cap_profile_leverage_pct(profile_weights, max_share=LEVERAGE_MAX_PORTFOLIO_WEIGHT):
    """투자 성향 비율에서도 레버리지 성향이 상한을 넘지 않도록 조정합니다."""
    if not profile_weights:
        return profile_weights

    capped = dict(profile_weights)
    regular_weights = {
        cat: max(0.0, float(capped.get(cat, 0.0)))
        for cat in AI_PROFILE_CATEGORIES
    }
    total = sum(regular_weights.values())
    if total <= 0:
        return capped

    normalized = {cat: regular_weights[cat] / total * 100 for cat in AI_PROFILE_CATEGORIES}
    leverage_cap_pct = (max_share if capped.get("_leverage_allowed", True) else 0.0) * 100
    leverage_pct = normalized["레버리지"]
    if leverage_pct <= leverage_cap_pct:
        for cat in AI_PROFILE_CATEGORIES:
            capped[cat] = round(normalized[cat], 2)
        return capped

    freed_pct = leverage_pct - leverage_cap_pct
    normalized["레버리지"] = leverage_cap_pct
    non_leverage_categories = [cat for cat in AI_PROFILE_CATEGORIES if cat != "레버리지"]
    non_leverage_total = sum(normalized[cat] for cat in non_leverage_categories)
    if non_leverage_total > 0:
        for cat in non_leverage_categories:
            normalized[cat] += freed_pct * (normalized[cat] / non_leverage_total)
    else:
        per_category = freed_pct / len(non_leverage_categories)
        for cat in non_leverage_categories:
            normalized[cat] = per_category

    for cat in AI_PROFILE_CATEGORIES:
        capped[cat] = round(normalized[cat], 2)
    return capped


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

    return _cap_profile_leverage_pct(profile)


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
    weights = _cap_profile_leverage_pct(weights)
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


def render_profile_metrics(profile_weights, show_guides=True):
    """투자 성향 비율을 표시하고, 필요 시 배당집중·레버리지 안내를 함께 표시합니다."""
    profile_weights = _cap_profile_leverage_pct(profile_weights)
    cols = st.columns(len(AI_PROFILE_CATEGORIES))
    for i, cat in enumerate(AI_PROFILE_CATEGORIES):
        with cols[i]:
            st.metric(
                cat,
                f"{profile_weights.get(cat, 0)}%",
            )
    if show_guides:
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
    profile_weights = _cap_profile_leverage_pct(profile_weights)
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
    # 성향 비율이 양수인 카테고리는 최소 1개 ETF가 선택되도록 보정
    positive_categories = [
        c for c in AI_PROFILE_CATEGORIES
        if normalized.get(c, 0.0) > 0 and (c != "레버리지" or leverage_allowed)
    ]
    for category in positive_categories:
        category_counts[category] = max(1, int(category_counts.get(category, 0)))
    while sum(category_counts.values()) > top_n:
        reducible = [
            c for c in AI_PROFILE_CATEGORIES
            if category_counts.get(c, 0) > 1 and (c != "레버리지" or leverage_allowed)
        ]
        if not reducible:
            break
        drop_category = max(reducible, key=lambda c: category_counts[c])
        category_counts[drop_category] -= 1
    while sum(category_counts.values()) < top_n:
        add_category = max(
            [c for c in AI_PROFILE_CATEGORIES if c != "레버리지" or leverage_allowed],
            key=lambda c: normalized.get(c, 0.0),
        )
        category_counts[add_category] = int(category_counts.get(add_category, 0)) + 1
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

    return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))


def recommend_etfs(profile_weights, top_n=5):
    """투자 성향 비율을 받아 ETF를 추천합니다."""
    weights = recommend_etfs_with_weights(profile_weights, top_n)
    return list(weights.keys())[:top_n]


def _allocate_profile_weights_to_selected_etfs(selected_etfs, profile_weights):
    """현재 선택된 ETF 목록에 성향 비율을 카테고리별로 배분합니다."""
    if not selected_etfs:
        return {}

    normalized = _normalize_profile_weights(profile_weights)
    by_category = {}
    for ticker in selected_etfs:
        category = ETF_DATA.get(ticker, {}).get("카테고리")
        if category:
            by_category.setdefault(category, []).append(ticker)

    weights = {}
    for category, tickers in by_category.items():
        if not tickers:
            continue
        cat_weight = float(normalized.get(category, 0.0))
        per_ticker = cat_weight / len(tickers)
        for ticker in tickers:
            weights[ticker] = per_ticker

    total = sum(weights.values())
    if total <= 0:
        equal = 1.0 / len(selected_etfs)
        return _cap_leverage_weights({ticker: round(equal, 4) for ticker in selected_etfs})
    normalized_weights = {
        ticker: round(weights.get(ticker, 0.0) / total, 4)
        for ticker in selected_etfs
    }
    return _cap_leverage_weights(normalized_weights)


HISTORICAL_RETURN_LABEL = "과거 연환산 수익률"
SIMULATION_RETURN_LABEL = "시뮬레이션 가정 수익률"
SOURCE_YAHOO_INCEPTION = "Yahoo · 상장 후"
SOURCE_SIMULATION = "시뮬 가정"
HISTORICAL_RETURN_DISCLAIMER = "과거 수익률은 미래 수익을 보장하지 않습니다."
SIMULATION_RETURN_DISCLAIMER = (
    "시뮬레이션 가정 수익률은 Yahoo 상장 이후 전체 기간 CAGR을 사용합니다."
)


def _get_etf_annual_return(ticker):
    """시뮬레이션 fallback용 ETF 연 수익률(모델 가정)을 계산합니다."""
    mu, _ = _get_ticker_return_profile(ticker)
    return max(0, mu * 100)


def _calc_cagr_pct_from_hist(hist):
    if hist is None or hist.empty or "Close" not in hist:
        return None, None
    close = _close_series(hist).dropna()
    if close.empty:
        return None, None
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    elapsed_years = max((close.index[-1] - close.index[0]).days / 365.25, 1e-6)
    if (
        not np.isfinite(first_close)
        or not np.isfinite(last_close)
        or first_close <= 0
        or last_close <= 0
        or elapsed_years <= 0
    ):
        return None, None
    cagr_pct = ((last_close / first_close) ** (1 / elapsed_years) - 1) * 100
    if not np.isfinite(cagr_pct):
        return None, None
    return cagr_pct, elapsed_years


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _get_etf_historical_return_info(ticker):
    """표시용 과거 연환산 수익률(Yahoo 상장 이후 전체 기간). 모델 가정치는 사용하지 않습니다."""
    if yf is None:
        return {"pct": None, "source": None, "period_years": None}

    try:
        etf = yf.Ticker(ticker)
        hist_max = etf.history(period="max", auto_adjust=True)
        if hist_max is not None and not hist_max.empty and len(hist_max) >= 252:
            pct, years = _calc_cagr_pct_from_hist(hist_max)
            if pct is not None:
                return {
                    "pct": float(pct),
                    "source": SOURCE_YAHOO_INCEPTION,
                    "period_years": float(years),
                }
    except Exception:
        pass

    return {"pct": None, "source": None, "period_years": None}


def _format_historical_return_value(hist_info):
    pct = (hist_info or {}).get("pct")
    if pct is None or not np.isfinite(float(pct)):
        return "-"
    return f"{pct:.1f}%"


def _format_historical_return_line(hist_info):
    value = _format_historical_return_value(hist_info)
    return f"{HISTORICAL_RETURN_LABEL}: {value}"


def _get_etf_cagr_rate(ticker):
    hist_info = _get_etf_historical_return_info(ticker)
    pct = (hist_info or {}).get("pct")
    if pct is not None and np.isfinite(float(pct)):
        return float(pct) / 100
    fallback_rate = float(ETF_DATA.get(ticker, {}).get("cagr", 0.10))
    return fallback_rate if np.isfinite(fallback_rate) else 0.10


def _get_portfolio_simulation_return(portfolio_weights):
    """시뮬레이션용 포트폴리오 가정 수익률(상장 이후 전체 기간 CAGR 가중평균)."""
    if not portfolio_weights:
        return 0.06, 6.0, SOURCE_SIMULATION

    rate = sum(
        float(portfolio_weights.get(ticker, 0.0)) * _get_etf_cagr_rate(ticker)
        for ticker in portfolio_weights
    )
    sources = {
        (_get_etf_historical_return_info(ticker) or {}).get("source")
        for ticker in portfolio_weights
        if (_get_etf_historical_return_info(ticker) or {}).get("source")
    }
    if not np.isfinite(rate) or rate <= 0:
        rate = 0.06
    if sources == {SOURCE_YAHOO_INCEPTION}:
        source = SOURCE_YAHOO_INCEPTION
    else:
        source = SOURCE_SIMULATION
    return float(rate), float(rate * 100), source


def _format_simulation_return_line(rate_pct, source=SOURCE_SIMULATION):
    return f"{SIMULATION_RETURN_LABEL}: {rate_pct:.2f}% (출처: {source})"


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
        return float(_close_series(hist).dropna().iloc[-1])
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
        hist = etf.history(period="max", auto_adjust=True)
        if hist is not None and not hist.empty and len(hist) >= 252:
            _hist_close = _close_series(hist).dropna()
            first_close = float(_hist_close.iloc[0])
            last_close = float(_hist_close.iloc[-1])
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
    hist_info = _get_etf_historical_return_info(ticker)
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
        f"{_format_historical_return_line(hist_info)}</div></div>"
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


def _render_etf_replace_option_detail(ticker):
    """종목 교체 카드 안에서 보여줄 ETF 상세 정보."""
    info = ETF_DATA.get(ticker, {})
    detail = ETF_DETAIL_GUIDE.get(ticker, {})
    category_label = CATEGORY_DISPLAY_LABELS.get(info.get("카테고리", ""), info.get("카테고리", ""))

    st.markdown(f"**{info.get('이름', ticker)}**")
    st.caption(category_label)

    summary = ETF_SUMMARY_DESCRIPTIONS.get(ticker, "")
    if summary:
        st.write(summary)

    st.write(f"**보수율:** {info.get('보수율', 0) * 100:.2f}%")
    st.write(f"**배당:** {'있음' if info.get('배당') else '없음'}")
    st.write(f"**레버리지:** {'있음 (위험)' if info.get('레버리지') else '없음'}")

    if detail.get("intro"):
        st.write(detail["intro"])

    recommended_for = detail.get("recommended_for", [])
    if recommended_for:
        st.write("**이런 분께 추천:**")
        for item in recommended_for:
            st.write(f"→ {item}")

    caution = detail.get("caution")
    if caution:
        st.write("**주의사항:**")
        st.write(f"→ {caution}")


def _replace_option_card_html(option_ticker, category_label, is_current=False):
    border = "#2563eb" if is_current else "#d9e2ec"
    background = (
        "linear-gradient(135deg, #eef5ff 0%, #ffffff 100%)"
        if is_current
        else "#ffffff"
    )
    current_tag = (
        '<span style="margin-left:8px;font-size:0.78rem;font-weight:700;color:#1d4ed8;">'
        "현재 보유</span>"
        if is_current
        else ""
    )
    return (
        f'<div style="border:2px solid {border};background:{background};border-radius:12px;'
        f'padding:14px 12px 8px;min-height:58px;">'
        f'<div style="font-size:1.15rem;font-weight:800;color:#172033;line-height:1.3;">'
        f"{option_ticker}{current_tag}</div>"
        f'<div style="font-size:0.82rem;color:#64748b;margin-top:6px;">{category_label}</div>'
        f"</div>"
    )


def _build_etf_replace_comparison_df(replace_options, current_ticker):
    """교체 후보 ETF 비교용 데이터프레임을 만듭니다."""
    rows = []
    for option_ticker in replace_options:
        info = ETF_DATA.get(option_ticker, {})
        hist_info = _get_etf_historical_return_info(option_ticker)
        live_snapshot = _get_live_etf_snapshot(option_ticker)

        dividend_yield = live_snapshot.get("dividend_yield_pct")
        if dividend_yield is None and info.get("배당"):
            dividend_yield = 2.0

        risk_stars = _get_risk_stars(option_ticker)
        is_current = option_ticker == current_ticker
        rows.append({
            "종목": f"{option_ticker} · 현재" if is_current else option_ticker,
            HISTORICAL_RETURN_LABEL: _format_historical_return_value(hist_info),
            "위험도": f"{'★' * risk_stars}{'☆' * (5 - risk_stars)}",
            "배당수익률": f"{dividend_yield:.1f}%" if dividend_yield is not None else "-",
            "보수율": f"{info.get('보수율', 0) * 100:.2f}%",
            "_is_current": is_current,
        })
    return pd.DataFrame(rows)


def _render_etf_replace_comparison_table(replace_options, current_ticker):
    """카드 아래에 항상 보이는 교체 후보 비교표를 표시합니다."""
    if len(replace_options) <= 1:
        return

    comparison_df = _build_etf_replace_comparison_df(replace_options, current_ticker)
    display_cols = ["종목", HISTORICAL_RETURN_LABEL, "위험도", "배당수익률", "보수율"]

    def _highlight_current_row(row):
        is_current = bool(comparison_df.loc[row.name, "_is_current"])
        style = "background-color: #eef5ff; font-weight: 600"
        return [style if is_current else "" for _ in row]

    st.caption(
        "같은 카테고리 ETF 한눈에 비교 · "
        f"{HISTORICAL_RETURN_DISCLAIMER}"
    )
    styled_df = (
        comparison_df[display_cols]
        .style.apply(_highlight_current_row, axis=1)
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("font-size", "0.85rem"), ("text-align", "center")],
                },
                {
                    "selector": "td",
                    "props": [("font-size", "0.85rem"), ("text-align", "center")],
                },
            ]
        )
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def _render_horizontal_replace_selector(slot_idx, current_ticker, replace_options, radio_key):
    """같은 카테고리 교체 후보를 가로 카드로 표시하고, 카드 클릭 시 즉시 교체합니다."""
    st.markdown(
        """
        <style>
        div.replace-ticker-btn + div[data-testid="stVerticalBlock"] button {
            font-size: 1.15rem !important;
            font-weight: 800 !important;
            color: #172033 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            text-align: left !important;
            padding: 0 !important;
            min-height: 0 !important;
        }
        div.replace-ticker-btn + div[data-testid="stVerticalBlock"] button:hover {
            color: #2563eb !important;
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    option_cols = st.columns(len(replace_options))

    for i, option_ticker in enumerate(replace_options):
        info = ETF_DATA.get(option_ticker, {})
        is_current = option_ticker == current_ticker
        category_label = CATEGORY_DISPLAY_LABELS.get(info.get("카테고리", ""), info.get("카테고리", ""))

        with option_cols[i]:
            if is_current:
                st.markdown(
                    _replace_option_card_html(option_ticker, category_label, is_current=True),
                    unsafe_allow_html=True,
                )
            else:
                with st.container(border=True):
                    st.markdown('<div class="replace-ticker-btn"></div>', unsafe_allow_html=True)
                    if st.button(
                        option_ticker,
                        key=f"replace_pick_{slot_idx}_{i}",
                        use_container_width=True,
                        help=f"{option_ticker}로 교체합니다.",
                    ):
                        st.session_state[radio_key] = i
                        _apply_etf_replace(slot_idx, current_ticker, replace_options)
                        st.rerun()
                    st.caption(category_label)

            if hasattr(st, "popover"):
                with st.popover("상세 보기", use_container_width=True):
                    _render_etf_replace_option_detail(option_ticker)
            else:
                with st.expander("상세 보기", expanded=False):
                    _render_etf_replace_option_detail(option_ticker)

    _render_etf_replace_comparison_table(replace_options, current_ticker)
    return current_ticker


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


def _get_leverage_multiplier(ticker):
    if not ETF_DATA.get(ticker, {}).get("레버리지"):
        return 1.0
    if ticker in {"TQQQ", "UPRO", "SOXL"}:
        return 3.0
    if ticker in {"QLD", "SSO"}:
        return 2.0
    return 1.0


def get_current_exchange_rate():
    if yf is None:
        return 1400.0
    return float(yf.Ticker("USDKRW=X").fast_info["last_price"])


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_exchange_rate_params():
    if yf is None:
        return 0.0, 0.025

    hist = yf.download("USDKRW=X", start="2005-01-01", interval="1mo", auto_adjust=True, progress=False)
    if hist is None or hist.empty or "Close" not in hist:
        return 0.0, 0.025

    close = _close_series(hist).dropna()
    close = close[(close >= 500) & (close <= 3000)]
    monthly_returns = close.pct_change().dropna()
    monthly_returns = monthly_returns[np.isfinite(monthly_returns) & (monthly_returns.abs() < 0.5)]
    if monthly_returns.empty:
        return 0.0, 0.025

    mu_fx = 0.0006
    sigma_fx = float(monthly_returns.std(ddof=1))
    if not np.isfinite(mu_fx) or not np.isfinite(sigma_fx):
        return 0.0, 0.025
    return mu_fx, sigma_fx


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_portfolio_params(etf_weights):
    if yf is None or not etf_weights:
        return 0.0, 0.0

    normalized_weights = _normalize_weights(etf_weights)
    mu_portfolio = 0.0
    sigma_portfolio = 0.0

    for ticker, weight in normalized_weights.items():
        try:
            hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist:
                continue

            daily_returns = _close_series(hist).dropna().pct_change().dropna()
            if daily_returns.empty:
                continue

            mu_daily = float(daily_returns.mean())
            sigma_daily = float(daily_returns.std(ddof=1))
            if not np.isfinite(mu_daily) or not np.isfinite(sigma_daily):
                continue

            mu_portfolio += mu_daily * weight
            sigma_portfolio += sigma_daily * weight
        except Exception:
            continue

    return float(mu_portfolio), float(sigma_portfolio)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _get_live_usdkrw_rate():
    if yf is None:
        return 1400.0

    for symbol in ("USDKRW=X", "KRW=X"):
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", {}) or {}
            rate = (
                fast_info.get("lastPrice")
                or fast_info.get("regularMarketPrice")
                or fast_info.get("previousClose")
            )
            if rate is None:
                hist = ticker.history(period="5d", auto_adjust=True)
                if hist is not None and not hist.empty:
                    rate = _close_series(hist).dropna().iloc[-1]
            if rate is not None and float(rate) > 0:
                return float(rate)
        except Exception:
            continue

    return 1400.0


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _get_usdkrw_monthly_stats():
    if yf is None:
        return 0.0, 0.025

    for symbol in ("USDKRW=X", "KRW=X"):
        try:
            hist = yf.Ticker(symbol).history(period="10y", interval="1mo", auto_adjust=True)
            if hist is None or hist.empty:
                continue
            monthly_returns = _close_series(hist).dropna().pct_change().dropna()
            if len(monthly_returns) >= 24:
                return float(monthly_returns.mean()), float(monthly_returns.std(ddof=1))
        except Exception:
            continue

    return 0.0, 0.025


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _get_voo_daily_stats_since_inception():
    if yf is None:
        return 0.08, 0.16

    try:
        hist = yf.Ticker("VOO").history(period="max", auto_adjust=True)
        daily_returns = _close_series(hist).dropna().pct_change().dropna()
        if len(daily_returns) < 252:
            return 0.08, 0.16
        annual_mu = float(daily_returns.mean() * 252)
        annual_sigma = float(daily_returns.std(ddof=1) * np.sqrt(252))
        return annual_mu, annual_sigma
    except Exception:
        return 0.08, 0.16


def _get_portfolio_voo_based_stats(etf_weights):
    normalized_weights = _normalize_weights(etf_weights)
    base_mu, base_sigma = _get_voo_daily_stats_since_inception()

    portfolio_mu = 0.0
    portfolio_sigma = 0.0
    for ticker, weight in normalized_weights.items():
        leverage = _get_leverage_multiplier(ticker)
        expense_ratio = ETF_DATA.get(ticker, {}).get("보수율", 0.0)
        portfolio_mu += weight * (base_mu - expense_ratio)
        portfolio_sigma += weight * base_sigma * leverage

    return portfolio_mu, portfolio_sigma, normalized_weights


def _get_weighted_dividend_yield(etf_weights):
    normalized_weights = _normalize_weights(etf_weights)
    weighted_yield = 0.0
    for ticker, weight in normalized_weights.items():
        dividend_yield = _get_etf_dividend_yield(ticker)
        if dividend_yield is None:
            dividend_yield = DIVIDEND_YIELD_ESTIMATE if ETF_DATA.get(ticker, {}).get("배당") else 0.0
        weighted_yield += weight * max(float(dividend_yield), 0.0)
    return weighted_yield


@st.cache_data(ttl=60 * 60, show_spinner=False)
def run_simulation(
    initial_capital,
    monthly_contribution,
    years,
    investment_method,
    mu_portfolio,
    sigma_portfolio,
    mu_fx,
    sigma_fx,
    start_rate,
    n_simulations=100,
):
    """일별 ETF 수익률과 월별 환율 변동을 반영한 Monte Carlo 경로를 반환합니다."""
    months = max(int(round(float(years) * 12)), 0)
    n_simulations = max(int(n_simulations), 1)
    initial_capital = max(float(initial_capital), 0.0)
    monthly_contribution = max(float(monthly_contribution), 0.0)
    mu_portfolio = float(mu_portfolio)
    sigma_portfolio = max(float(sigma_portfolio), 0.0)
    mu_fx = float(mu_fx)
    sigma_fx = max(float(sigma_fx), 0.0)
    start_rate = max(float(start_rate), 1e-6)
    trading_days_per_month = 21

    # region agent log
    _debug_b8b036_log(
        "initial",
        "H1,H2,H3,H5",
        "app.py:run_simulation:entry",
        "run_simulation normalized inputs",
        {
            "initial_capital": initial_capital,
            "monthly_contribution": monthly_contribution,
            "years": years,
            "months": months,
            "investment_method": investment_method,
            "mu_portfolio_daily": mu_portfolio,
            "sigma_portfolio_daily": sigma_portfolio,
            "mu_fx_monthly": mu_fx,
            "sigma_fx_monthly": sigma_fx,
            "start_rate": start_rate,
            "n_simulations": n_simulations,
        },
    )
    # endregion

    # 두 값이 모두 제공된 경우 investment_method와 무관하게 혼합형으로 동작
    if initial_capital > 0 and monthly_contribution > 0:
        invest_initial = initial_capital
        invest_monthly = monthly_contribution
    elif investment_method in {"거치식", "거치형", "lump_sum"}:
        invest_initial = initial_capital
        invest_monthly = 0.0
    elif investment_method in {"적립식", "적립형", "dca"}:
        invest_initial = 0.0
        invest_monthly = monthly_contribution
    elif investment_method in {"혼합식", "혼합형", "mixed"}:
        invest_initial = initial_capital
        invest_monthly = monthly_contribution
    else:
        raise ValueError("investment_method는 '거치식', '적립식', '혼합식' 중 하나여야 합니다.")

    print(
        f"[run_simulation] method={investment_method!r} → "
        f"invest_initial={invest_initial:,.0f}원  invest_monthly={invest_monthly:,.0f}원",
        flush=True,
    )

    paths = np.zeros((n_simulations, months + 1), dtype=float)

    for sim_idx in range(n_simulations):
        exchange_rate = start_rate
        usd_value = invest_initial / start_rate
        paths[sim_idx, 0] = usd_value * exchange_rate

        for month in range(1, months + 1):
            fx_return = np.random.normal(mu_fx, sigma_fx)
            exchange_rate = max(exchange_rate * (1 + fx_return), 1e-6)

            if invest_monthly > 0:
                usd_value += invest_monthly / exchange_rate

            daily_returns = np.maximum(
                np.random.normal(mu_portfolio, sigma_portfolio, trading_days_per_month),
                -0.99,
            )
            usd_value *= np.prod(1 + daily_returns)
            paths[sim_idx, month] = usd_value * exchange_rate

            if sim_idx == 0 and month <= 12:
                print(
                    f"[run_simulation] sim=0 month={month:>2}  "
                    f"exchange_rate={exchange_rate:,.1f}  "
                    f"invest_monthly={invest_monthly:,.0f}원  "
                    f"usd_value={usd_value:.4f}$  "
                    f"won_value={paths[sim_idx, month]:>16,.0f}원",
                    flush=True,
                )

            if sim_idx == 0 and month <= 2:
                # region agent log
                _debug_b8b036_log(
                    "initial",
                    "H1,H2,H3,H5",
                    "app.py:run_simulation:first_path_month",
                    "first simulation monthly state",
                    {
                        "month": month,
                        "fx_return": float(fx_return),
                        "exchange_rate": float(exchange_rate),
                        "invest_monthly": float(invest_monthly),
                        "usd_value_after_return": float(usd_value),
                        "month_daily_return_min": float(np.min(daily_returns)),
                        "month_daily_return_max": float(np.max(daily_returns)),
                        "month_daily_return_mean": float(np.mean(daily_returns)),
                        "month_growth_factor": float(np.prod(1 + daily_returns)),
                        "won_path_value": float(paths[sim_idx, month]),
                    },
                )
                # endregion

    median_path = np.percentile(paths, 50, axis=0)
    upper_path = np.percentile(paths, 90, axis=0)
    lower_path = np.percentile(paths, 10, axis=0)

    # region agent log
    _debug_b8b036_log(
        "initial",
        "H2,H3,H4,H5",
        "app.py:run_simulation:exit",
        "run_simulation output summary",
        {
            "median_first_values": [float(v) for v in median_path[: min(5, len(median_path))]],
            "upper_first_values": [float(v) for v in upper_path[: min(5, len(upper_path))]],
            "lower_first_values": [float(v) for v in lower_path[: min(5, len(lower_path))]],
            "paths_shape": list(paths.shape),
            "paths_min": float(np.min(paths)) if paths.size else None,
            "paths_max": float(np.max(paths)) if paths.size else None,
        },
    )
    # endregion

    return {
        "paths": paths.tolist(),
        "median_path": median_path.tolist(),
        "upper_path": upper_path.tolist(),
        "lower_path": lower_path.tolist(),
    }


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

    hist = yf.Ticker("VOO").history(period="max", auto_adjust=True)
    if hist.empty or "Close" not in hist:
        raise RuntimeError("VOO 가격 데이터를 불러오지 못했습니다.")

    daily_returns = _close_series(hist).dropna().pct_change().dropna()
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
    rng = np.random.default_rng(
        _stable_simulation_seed(
            annual_return,
            annual_volatility,
            month_end_date_keys,
            initial_capital,
            monthly_contribution,
            mode,
            trials,
        )
    )

    values = paths[:, 0].copy()
    for month, days in enumerate(business_days_by_month, start=1):
        if mode in {"적립형", "혼합형"}:
            values += monthly_contribution
        portfolio_daily_returns = np.maximum(
            rng.normal(daily_mu, daily_sigma, size=(trials, days)),
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
        weight * _get_etf_cagr_rate(ticker)
        for ticker, weight in normalized_weights.items()
    )
    annual_volatility = sum(
        weight * _get_ticker_return_profile(ticker)[1]
        for ticker, weight in normalized_weights.items()
    )
    return float(annual_return), float(annual_volatility)


def _detect_drawdown_marker_indices(values, threshold=-0.10):
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


def _count_monthly_crashes(values, threshold=-0.10):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0
    monthly_changes = np.diff(values) / np.maximum(values[:-1], 1)
    return int(np.sum(monthly_changes <= threshold))


def _target_crash_count_for_months(months):
    if months <= 60:
        return 1
    if months <= 120:
        return 2
    return 3


def _select_representative_crash_path(paths, median_path, months, threshold=-0.10):
    """Step5 시각화용: 급락이 포함되면서 최종값은 중앙값에 가까운 경로를 고릅니다."""
    paths = np.asarray(paths, dtype=float)
    median_path = np.asarray(median_path, dtype=float)
    if paths.ndim != 2 or paths.shape[0] == 0 or paths.shape[1] == 0:
        return median_path, 0, 0, False

    plot_len = min(paths.shape[1], len(median_path), max(int(months), 0) + 1)
    if plot_len <= 0:
        return median_path, 0, 0, False

    candidate_paths = paths[:, :plot_len]
    target_count = _target_crash_count_for_months(max(plot_len - 1, 0))
    crash_counts = np.array([
        _count_monthly_crashes(path, threshold)
        for path in candidate_paths
    ])
    final_median = float(median_path[min(plot_len - 1, len(median_path) - 1)]) if len(median_path) else 0.0
    final_distances = np.abs(candidate_paths[:, -1] - final_median)

    eligible = np.where(crash_counts >= target_count)[0]
    if len(eligible) > 0:
        selected_idx = int(eligible[np.argmin(final_distances[eligible])])
    else:
        max_crashes = int(np.max(crash_counts)) if len(crash_counts) else 0
        closest = np.where(crash_counts == max_crashes)[0]
        selected_idx = int(closest[np.argmin(final_distances[closest])])

    selected_path = candidate_paths[selected_idx]
    selected_count = int(crash_counts[selected_idx])
    return selected_path, selected_count, target_count, len(eligible) > 0


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
    voo_return = _get_etf_cagr_rate("VOO")
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
        if monthly_change <= -0.10:
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


def _stable_simulation_seed(*parts):
    seed_text = "|".join(str(part) for part in parts)
    return int(hashlib.md5(seed_text.encode("utf-8")).hexdigest()[:8], 16)


def _sync_unified_sim_period_end_from_years(years_value):
    start_date = st.session_state.get(
        "unified_sim_period_start",
        st.session_state.get("invest_period_start", date.today()),
    )
    st.session_state["unified_sim_period_end"] = add_calendar_years(
        start_date,
        max(1, int(np.ceil(float(years_value)))),
    )


def _resolve_simulation_period():
    """STEP3 분석 결과와 STEP6 차트가 같은 투자 기간을 쓰도록 맞춥니다."""
    analysis_mode = st.session_state.get("analysis_mode", "목표 금액 달성")
    start_date = st.session_state.get(
        "unified_sim_period_start",
        st.session_state.get("invest_period_start", date.today()),
    )

    if analysis_mode == "목표 기간 확인" and "unified_sim_period_end" in st.session_state:
        end_date = st.session_state["unified_sim_period_end"]
        if end_date < start_date:
            years_sim = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
            end_date = add_calendar_years(start_date, max(1, int(np.ceil(years_sim))))
        else:
            years_sim = period_between_dates_years(start_date, end_date)
    else:
        years_sim = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
        end_date = add_calendar_years(start_date, max(1, int(np.ceil(years_sim))))

    return start_date, end_date, years_sim


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
    
    # etf_weights가 있으면 사용, 없으면 균등 분배
    if etf_weights:
        raw_weights = {ticker: float(etf_weights.get(ticker, 0.0)) for ticker in etfs}
        if sum(raw_weights.values()) <= 0:
            raw_weights = {ticker: 1.0 / len(etfs) for ticker in etfs}
    else:
        raw_weights = {ticker: 1.0 / len(etfs) for ticker in etfs}

    normalized_weights = _normalize_weights(raw_weights)
    # 추천 이후 상태 변경(교체/수동 조정 등)에서도 레버리지 상한을 일관 적용
    return _cap_leverage_weights(normalized_weights)


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


def _request_scroll_to_top():
    st.session_state["_scroll_to_top"] = True


def _render_scroll_to_top():
    if not st.session_state.pop("_scroll_to_top", False):
        return
    components.html(
        """
        <script>
            (function () {
                function scrollToTop() {
                    const parent = window.parent;
                    const doc = parent.document;
                    const selectors = [
                        'section[data-testid="stMain"]',
                        'section.main',
                        '[data-testid="stAppViewContainer"]',
                        '.main',
                    ];
                    selectors.forEach(function (selector) {
                        const el = doc.querySelector(selector);
                        if (el) {
                            el.scrollTop = 0;
                            if (el.scrollTo) {
                                el.scrollTo({ top: 0, left: 0, behavior: "auto" });
                            }
                        }
                    });
                    parent.scrollTo(0, 0);
                }
                scrollToTop();
                setTimeout(scrollToTop, 50);
                setTimeout(scrollToTop, 200);
            })();
        </script>
        """,
        height=0,
        width=0,
    )


def _set_app_step(new_step, *, scroll_to_top=True):
    if scroll_to_top and new_step != st.session_state.get("app_step"):
        _request_scroll_to_top()
    st.session_state["app_step"] = new_step
    st.rerun()


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
        """프리셋 버튼 + 카드 UI 내 직접 입력 방식의 금액 입력"""
        init_man_amount_key(key, default_won)

        st.markdown(f"#### 💰 {title}")

        preset_amounts = [
            ("100만", 100),
            ("500만", 500),
            ("1000만", 1000),
            ("5000만", 5000),
            ("1억", 10000),
        ]

        def set_preset_value(manwon_val):
            st.session_state[key] = fmt_money(manwon_val)

        preset_cols = st.columns(len(preset_amounts))
        for idx, (label, manwon_val) in enumerate(preset_amounts):
            with preset_cols[idx]:
                st.button(
                    label,
                    key=f"{key}_btn_{idx}",
                    use_container_width=True,
                    on_click=set_preset_value,
                    args=(manwon_val,),
                )

        with st.container(border=True):
            st.text_input(
                "금액 입력 (만원)",
                key=key,
                on_change=format_money_input,
                args=(key,),
                placeholder="예: 5000 (= 5,000만원)",
            )

            try:
                amount_won = int(format_won_from_manwon_key(key))
            except Exception:
                amount_won = 0

            display_manwon = st.session_state.get(key, "") or "0"
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    padding: 14px 18px;
                    margin-top: 8px;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
                ">
                    <div style="color:#e8eeff;font-size:0.88rem;font-weight:600;margin-bottom:6px;">
                        입력 금액
                    </div>
                    <div style="color:#ffffff;font-size:1.45rem;font-weight:800;line-height:1.3;">
                        {display_manwon} 만원
                    </div>
                    <div style="
                        color:#ffffff;
                        font-size:1.05rem;
                        font-weight:700;
                        text-align:right;
                        margin-top:10px;
                        padding-top:10px;
                        border-top:1px solid rgba(255,255,255,0.28);
                    ">
                        {fmt_money(amount_won)}원
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
        st.session_state["simulation_result"] = None

    step_labels = {
        1: "STEP1: AI 성향 분석",
        2: "STEP2: ETF 추천",
        3: "STEP3: 모드 선택",
        4: "STEP4: 전략 추천",
        6: "STEP5: 결과 시각화",
    }
    if st.session_state["app_step"] == 5:
        st.session_state["app_step"] = 6
    current_step = st.session_state["app_step"]
    display_step = 5 if current_step == 6 else current_step
    total_steps = 5
    st.write(f"### {step_labels[current_step]} ({display_step}/{total_steps})")
    st.progress((display_step - 1) / (total_steps - 1))
    _render_scroll_to_top()

    def move_step(new_step):
        _set_app_step(new_step)

    def reset_app_to_start():
        """앱을 최초 실행 상태(STEP1 · Q1)로 되돌립니다."""
        st.session_state.clear()
        _request_scroll_to_top()
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
        _set_app_step(2)

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
    if profile_weights:
        profile_weights = _cap_profile_leverage_pct(profile_weights)
        st.session_state["profile_weights"] = profile_weights
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
                            _set_app_step(2)

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
                    _set_app_step(2)
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
                            _set_app_step(2)
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

        # 프로필이 바뀐 경우에만 자동 재추천하고, 그 외에는 수동 교체 상태를 유지
        _profile_signature = str(
            sorted(
                (cat, float(profile_weights.get(cat, 0.0)))
                for cat in AI_PROFILE_CATEGORIES + ["_leverage_allowed"]
            )
        )
        _last_profile_signature = st.session_state.get("step2_profile_signature")
        _should_recommend = (not selected_etfs) or (_last_profile_signature != _profile_signature)

        if _should_recommend:
            selected_etfs = recommend_etfs(profile_weights, top_n=5)
            st.session_state["selected_etfs"] = selected_etfs
            st.session_state["step2_profile_signature"] = _profile_signature

        # 비중은 현재 성향(profile_weights) 기준으로 항상 재계산
        etf_weights = _allocate_profile_weights_to_selected_etfs(selected_etfs, profile_weights)
        st.session_state["etf_weights"] = etf_weights

        # 투자 성향 분석 결과 표시
        st.write("### 📊 당신의 투자 성향")
        render_profile_metrics(profile_weights, show_guides=False)

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
            hist_info = _get_etf_historical_return_info(ticker)
            live_snapshot = _get_live_etf_snapshot(ticker)
            usdkrw = _get_usdkrw_rate()
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
                st.write(f"• **{_format_historical_return_line(hist_info)}**")

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
            "교체할 ETF **카드를 클릭**하면 바로 반영됩니다."
        )

        for slot_idx, current_ticker in enumerate(selected_etfs):
            category = ETF_DATA.get(current_ticker, {}).get("카테고리", "")
            replace_options = get_same_category_replacement_options(current_ticker, selected_etfs)
            radio_key = _replace_radio_key(slot_idx)

            st.subheader(f"📌 {current_ticker} · {category}")
            st.write(f"포트폴리오 **{slot_idx + 1}번** 종목 — 같은 카테고리 ETF 중에서 선택")

            if len(replace_options) <= 1:
                st.info(f"**{category}** 카테고리에 선택 가능한 다른 ETF가 없습니다.")
                st.markdown("---")
                continue

            _render_horizontal_replace_selector(
                slot_idx,
                current_ticker,
                replace_options,
                radio_key,
            )

            error_key = f"replace_error_{slot_idx}"
            if error_key in st.session_state:
                st.error(st.session_state.pop(error_key))

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
        simulation_result = st.session_state.get("simulation_result")
        if not simulation_result:
            st.warning("STEP3에서 분석 시작 버튼을 눌러 시뮬레이션을 먼저 실행해 주세요.")
            if st.button("STEP3로 이동", key="goto3c"):
                move_step(3)
            return

        # ── 공통 데이터 ──────────────────────────────────────────────────────
        median_path = np.asarray(simulation_result.get("median_path", []), dtype=float)
        target_won = int(st.session_state.get("target_amount", 0))
        current_capital = float(st.session_state.get("current_capital", 0.0))
        monthly_won = int(st.session_state.get("monthly_contribution", 0))
        selected_mode = simulation_result.get("sim_mode", st.session_state.get("analysis_mode"))
        etf_weights = st.session_state.get("etf_weights", {})
        achievement_prob = float(simulation_result.get("achievement_prob", 0.0))

        try:
            portfolio_weights = build_portfolio_weights(selected_etfs, etf_weights)
        except Exception:
            portfolio_weights = {t: 1.0 / max(1, len(selected_etfs)) for t in selected_etfs}

        def fmt_money(n):
            try:
                f = float(n)
            except Exception:
                return "0"
            if f.is_integer():
                return f"{int(f):,}"
            s = f"{f:,}"
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s

        # 참조 환율 (시뮬레이션 시작 기준 / 배당 환전용)
        try:
            ref_exchange_rate = get_current_exchange_rate()
        except Exception:
            ref_exchange_rate = 1400.0

        st.markdown("## 📋 종합 투자 전략 보고서")
        st.markdown("**STEP 1~3 분석 결과를 바탕으로 한 맞춤형 투자 전략입니다.**")
        st.markdown("---")

        # ===== 섹션 1: 투자 성향 분석 결과 =====
        st.markdown("### 📊 1. 투자 성향 분석 결과")

        col1, col2, col3, col4 = st.columns(4)
        for col, category in zip([col1, col2, col3, col4], AI_PROFILE_CATEGORIES):
            pct = profile_weights.get(category, 0)
            col.metric(category, f"{pct:.0f}%")

        top_categories = sorted(
            [(cat, profile_weights.get(cat, 0)) for cat in AI_PROFILE_CATEGORIES],
            key=lambda x: x[1],
            reverse=True,
        )
        top_cat = top_categories[0][0]
        top_pct = top_categories[0][1]

        _desc_map = {
            "지수추적": "📈 시장 지수를 따라가며 안정적인 장기 수익을 추구하는 성향입니다.",
            "배당성장": "💰 정기적인 배당 수익을 통해 안정적인 현금흐름을 원하는 성향입니다.",
            "배당집중": "📅 월배당·프리미엄 수익을 통해 규칙적인 추가 수익을 기대하는 성향입니다.",
            "레버리지": "🚀 높은 수익을 추구하되 높은 변동성을 감수할 수 있는 공격적 성향입니다.",
        }
        st.write(f"**최상위 성향:** {top_cat} ({top_pct:.0f}%)")
        st.write(_desc_map.get(top_cat, ""))

        st.markdown("---")

        # ===== 섹션 2: 추천 포트폴리오 구성 =====
        st.markdown("### 🎯 2. 추천 포트폴리오 구성")

        portfolio_data = []
        for idx, ticker in enumerate(selected_etfs, 1):
            info = ETF_DATA.get(ticker, {})
            weight = portfolio_weights.get(ticker, 1 / len(selected_etfs))
            portfolio_data.append({
                "순위": idx,
                "ETF": ticker,
                "이름": info.get("이름", "")[:35],
                "카테고리": info.get("카테고리", ""),
                "비중": f"{round(weight * 100, 1)}%",
            })

        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

        leverage_count = sum(1 for t in selected_etfs if ETF_DATA.get(t, {}).get("레버리지", False))
        dividend_count = sum(1 for t in selected_etfs if ETF_DATA.get(t, {}).get("배당", False))

        st.write("• **STEP3 Monte Carlo 시뮬레이션 결과 사용**")
        st.write(f"• **배당 ETF 포함:** {dividend_count}개")
        st.write(f"• **레버리지 ETF 포함:** {leverage_count}개")

        st.markdown("---")

        # ===== 섹션 3: 투자 목표 및 시나리오 =====
        st.markdown("### 🎲 3. 투자 목표 및 시나리오")

        # 포트폴리오 예상 연 수익률 — 실제 시뮬레이션에 사용된 값을 표시
        _annual_return_pct = simulation_result.get("sim_annual_rate")
        _annual_return_source = "시뮬레이션 실행 시점"
        if _annual_return_pct is None:
            try:
                _, _annual_return_pct, _annual_return_source = _get_portfolio_simulation_return(portfolio_weights)
            except Exception:
                _annual_return_pct = None
                _annual_return_source = None

        _expected_amount_label = "목표 금액"
        _expected_amount_value = f"{fmt_money(target_won)}원"
        if selected_mode == "목표 기간 확인" and len(median_path):
            _expected_amount_label = "예상 달성 금액"
            _expected_amount_value = f"{fmt_money(int(float(median_path[-1])))}원"

        goal_col1, goal_col2, goal_col3, goal_col4 = st.columns(4)
        with goal_col1:
            st.metric("현재 보유 자금", f"{fmt_money(int(current_capital))}원")
        with goal_col2:
            st.metric(_expected_amount_label, _expected_amount_value)
        with goal_col3:
            st.metric("시작 환율", f"{ref_exchange_rate:,.0f}원/달러")
        with goal_col4:
            if _annual_return_pct is not None:
                st.metric("포트폴리오 예상 연 수익률", f"{_annual_return_pct:.1f}%")
            else:
                st.metric("포트폴리오 예상 연 수익률", "산출 불가")
        if _annual_return_source:
            st.caption(f"수익률 기준: {_annual_return_source} CAGR 가중평균")

        st.write(f"**분석 모드:** {selected_mode}")

        # dividend_report_months: 배당 계산에 사용할 기간
        dividend_report_months = int(float(st.session_state.get("years", st.session_state.get("target_years", 10))) * 12)

        if selected_mode == "목표 금액 달성":
            st.write(f"• **월 적립 계획:** {fmt_money(monthly_won)}원")
            target_months_stored = simulation_result.get("target_months")
            if target_months_stored is None:
                st.warning("⚠️ 중앙값 기준으로는 시뮬레이션 기간 내 목표 금액에 도달하지 못했습니다.")
                dividend_report_months = max(0, len(median_path) - 1)
            else:
                _tm = int(target_months_stored)
                dividend_report_months = _tm
                st.write(f"• **예상 달성 기간:** {_tm // 12}년 {_tm % 12}개월")
                st.write(f"• **목표 달성 확률:** {achievement_prob:.1f}%")

        elif selected_mode == "목표 기간 확인":
            dividend_report_months = max(0, len(median_path) - 1)
            if len(median_path):
                _final_median_val = float(median_path[-1])
                _eok = int(_final_median_val) // 100_000_000
                _man = (int(_final_median_val) % 100_000_000) // 10_000
                if _eok > 0 and _man > 0:
                    _amt = f"{_eok}억 {_man:,}만원"
                elif _eok > 0:
                    _amt = f"{_eok}억원"
                else:
                    _amt = f"{_man:,}만원"
                st.write(f"• **예상 달성 금액:** {_amt}")
                if target_won > 0 and _final_median_val >= target_won:
                    st.success("✅ 중앙값 기준으로 목표 금액 달성이 가능합니다.")

        st.markdown("---")

        # ===== 섹션 4: 투자기간 후 월 배당금 예상 =====
        st.markdown("### 💵 4. 투자기간 후 월 배당금 예상")

        MONTHLY_DIVIDEND_ETFS = {"JEPI", "JEPQ", "QYLD"}
        dividend_yields = {
            ticker: dy
            for ticker in selected_etfs
            if ETF_DATA.get(ticker, {}).get("카테고리") in {"배당성장", "배당집중"}
            and (dy := _get_etf_dividend_yield(ticker)) is not None
            and 0.01 <= dy <= 0.20
        }

        if not dividend_yields:
            st.info("배당성장/배당집중 ETF가 없어 월 배당금 예상치를 표시하지 않습니다.")
        else:
            _dm = min(max(0, int(dividend_report_months)), max(0, len(median_path) - 1))

            # 목표 시점 예측 환율: 현재 환율에 월별 환율 변화 기댓값(mu_fx)을 복리로 적용
            _mu_fx_div = 0.0
            try:
                _mu_fx_div, _ = get_exchange_rate_params()
                _final_exchange_rate = ref_exchange_rate * ((1 + _mu_fx_div) ** _dm)
            except Exception:
                _final_exchange_rate = ref_exchange_rate

            # 배당 ETF는 전체 포트폴리오 성장률이 아니라 ETF별 자체 CAGR로 별도 추정
            _cumulative_div_usd = 0.0
            _final_monthly_div_usd = 0.0
            _final_div_asset_usd = 0.0
            _dividend_asset_usd = {
                _ticker: (current_capital * portfolio_weights.get(_ticker, 0)) / ref_exchange_rate
                for _ticker in dividend_yields
            }
            _monthly_growth_rates = {
                _ticker: (1 + max(_get_etf_cagr_rate(_ticker), -0.99)) ** (1 / 12) - 1
                for _ticker in dividend_yields
            }

            for _month in range(1, _dm + 1):
                _month_exchange_rate = ref_exchange_rate * ((1 + _mu_fx_div) ** _month)
                _month_div_usd = 0.0
                for _ticker, _dy in dividend_yields.items():
                    _weight = portfolio_weights.get(_ticker, 0)
                    if monthly_won > 0 and _weight > 0:
                        _dividend_asset_usd[_ticker] += (monthly_won * _weight) / _month_exchange_rate
                    _dividend_asset_usd[_ticker] *= max(0.0, 1 + _monthly_growth_rates.get(_ticker, 0.0))
                    _ann_div_usd = _dividend_asset_usd[_ticker] * _dy
                    _mon_div_usd = _ann_div_usd / 12 if _ticker in MONTHLY_DIVIDEND_ETFS else (_ann_div_usd / 4) / 3
                    _month_div_usd += _mon_div_usd
                _cumulative_div_usd += _month_div_usd
                if _month == _dm:
                    _final_monthly_div_usd = _month_div_usd

            _final_div_asset_usd = sum(_dividend_asset_usd.values())

            # 목표 시점 예측 환율 기준 원화 환전
            _monthly_div_won = int(round(_final_monthly_div_usd * _final_exchange_rate))
            _final_div_asset_won = _final_div_asset_usd * _final_exchange_rate
            _final_div_asset_won_int = int(round(_final_div_asset_won))
            _cumulative_div_won = int(round(_cumulative_div_usd * _final_exchange_rate))

            _div_weight_sum = sum(portfolio_weights.get(t, 0) for t in dividend_yields)
            _invested_won = (current_capital + monthly_won * _dm) * _div_weight_sum
            _div_return_pct = (
                ((_final_div_asset_won + _cumulative_div_usd * _final_exchange_rate) / _invested_won - 1) * 100
                if _invested_won > 0
                else 0.0
            )

            _rep_y = _dm // 12
            _rep_m = _dm % 12
            _period_lbl = f"{_rep_y}년 후" if _rep_m == 0 else f"{_rep_y}년 {_rep_m}개월 후"

            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,#2f9f63 0%,#4aa36b 52%,#2f7d6d 100%);
                    border-radius:0;
                    padding:26px 28px 24px 28px;
                    color:white;
                    box-shadow:0 8px 20px rgba(0,0,0,0.12);
                ">
                    <div style="font-size:1.05rem;font-weight:700;margin-bottom:8px;opacity:0.95;">💸 {_period_lbl} 매달 받는 배당금</div>
                    <div style="font-size:3.2rem;font-weight:900;line-height:1.05;letter-spacing:-1px;">{_monthly_div_won:,}원</div>
                    <div style="font-size:0.9rem;margin-top:8px;opacity:0.78;">배당금은 세전 기준 · 달러 누적 후 목표 시점 예측 환율 {_final_exchange_rate:,.0f}원/$ 기준 환전</div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:26px;">
                        <div style="background:rgba(255,255,255,0.16);border-radius:999px;padding:8px 14px;font-weight:700;">💼 배당 ETF 총 자산 {_final_div_asset_won_int:,}원</div>
                        <div style="background:rgba(255,255,255,0.16);border-radius:999px;padding:8px 14px;font-weight:700;">↗ 배당 ETF 수익률 {_div_return_pct:,.1f}%</div>
                        <div style="background:rgba(255,255,255,0.16);border-radius:999px;padding:8px 14px;font-weight:700;">💸 총 배당 {_cumulative_div_won:,}원</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ===== 섹션 5: AI 성향 분석 결과 =====
        st.markdown("### 💡 5. AI 성향 분석 결과")

        _investment_method = _recommended_investment_mode_from_profile().replace("형", "식")
        _analysis_years = float(st.session_state.get("years", st.session_state.get("target_years", 10)))
        _etfs_for_prompt = [
            {"ETF": t, "비중": f"{portfolio_weights.get(t, 0) * 100:.1f}%"}
            for t in selected_etfs
        ]

        # session_state 캐시 — 입력이 바뀌지 않으면 Gemini를 재호출하지 않음
        _ai_cache_key = str((
            sorted(profile_weights.items()),
            sorted(portfolio_weights.items()),
            _investment_method,
            target_won,
            int(_analysis_years),
            selected_mode,
        ))
        _cached = st.session_state.get("step4_ai_analysis")
        _cached_key = st.session_state.get("step4_ai_cache_key")

        if _cached is None or _cached_key != _ai_cache_key:
            try:
                with st.spinner("AI 분석 중입니다. 잠시만 기다려 주세요..."):
                    _ai_text = _generate_gemini_step4_analysis(
                        profile_weights,
                        _etfs_for_prompt,
                        _investment_method,
                        target_won,
                        _analysis_years,
                    )
            except Exception:
                _ai_text = _default_step4_analysis_text(
                    profile_weights,
                    selected_etfs,
                    _investment_method,
                    target_won,
                    _analysis_years,
                )
            st.session_state["step4_ai_analysis"] = _ai_text
            st.session_state["step4_ai_cache_key"] = _ai_cache_key
        else:
            _ai_text = _cached

        st.info(_ai_text)

        if st.button("🔄 AI 분석 다시 생성", key="regen_ai_step4"):
            st.session_state.pop("step4_ai_analysis", None)
            st.session_state.pop("step4_ai_cache_key", None)
            st.rerun()

        st.markdown("---")

        step_button_row(can_prev=False)

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
                st.session_state["simulation_result"] = None

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
                st.session_state["simulation_result"] = None

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
        st.caption("분석 시작 버튼을 누르면 환율과 ETF 일별 변동성을 반영한 Monte Carlo 시뮬레이션을 실행합니다.")
        
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
        
        if selected_analysis_mode == "목표 금액 달성":
            target_years_val = 100.0
            st.session_state["target_amount"] = target_won
        else:
            target_won = int(st.session_state.get("target_amount", 0))

        if st.button("분석 시작", use_container_width=True, key="start_step3_simulation"):
            st.session_state["monthly_contribution"] = monthly_won
            st.session_state["current_capital"] = current_capital
            st.session_state["target_years"] = target_years_val
            st.session_state["years"] = target_years_val

            try:
                with st.spinner("환율과 ETF 변동성을 반영해 시뮬레이션 중입니다..."):
                    start_rate = get_current_exchange_rate()
                    mu_fx, sigma_fx = get_exchange_rate_params()
                    _portfolio_annual_rate, _, _ = _get_portfolio_simulation_return(step3_portfolio_weights)
                    # 기대수익률은 CAGR(기하평균) 기준을 일별 로그수익률로 변환해 과대추정을 완화
                    mu_portfolio = np.log1p(max(_portfolio_annual_rate, -0.99)) / 252
                    _, sigma_portfolio = get_portfolio_params(step3_portfolio_weights)
                    # region agent log
                    _debug_b8b036_log(
                        "initial",
                        "H1,H3,H5",
                        "app.py:step3:before_run_simulation",
                        "STEP3 values before simulation",
                        {
                            "selected_analysis_mode": selected_analysis_mode,
                            "target_won": int(target_won),
                            "current_capital": float(current_capital),
                            "monthly_won": int(monthly_won),
                            "target_years_val": float(target_years_val),
                            "investment_method": _recommended_investment_mode_from_profile(),
                            "step3_portfolio_weights": step3_portfolio_weights,
                            "start_rate": float(start_rate),
                            "mu_fx": float(mu_fx),
                            "sigma_fx": float(sigma_fx),
                            "mu_portfolio": float(mu_portfolio),
                            "sigma_portfolio": float(sigma_portfolio),
                        },
                    )
                    # endregion
                    simulation_result = run_simulation(
                        current_capital,
                        monthly_won,
                        target_years_val,
                        _recommended_investment_mode_from_profile(),
                        mu_portfolio,
                        sigma_portfolio,
                        mu_fx,
                        sigma_fx,
                        start_rate,
                        n_simulations=100,
                    )
                    median_path = np.asarray(simulation_result.get("median_path", []), dtype=float)
                    paths = np.asarray(simulation_result.get("paths", []), dtype=float)

                    # 목표 금액 달성 모드에서만: 중앙값 3개월 연속 목표 이상 유지 첫 시점
                    target_months = None
                    achievement_prob = 0.0
                    if selected_analysis_mode == "목표 금액 달성":
                        _SUSTAIN = 3
                        if len(median_path) >= _SUSTAIN:
                            for _ti in range(len(median_path) - _SUSTAIN + 1):
                                if all(median_path[_ti + k] >= target_won for k in range(_SUSTAIN)):
                                    target_months = _ti
                                    break

                        # 달성 확률: 100개 경로 중 최종값 ≥ 목표 금액 비율
                        if paths.size and target_won > 0:
                            final_values = paths[:, -1]
                            reached_count = int(np.sum(final_values >= target_won))
                            achievement_prob = reached_count / len(final_values) * 100
                            # region agent log
                            print(f"[debug-b8b036] final_values sample (first 10): {[round(v) for v in final_values[:10]]}", flush=True)
                            print(f"[debug-b8b036] target_won={target_won} reached={reached_count}/{len(final_values)} prob={achievement_prob:.1f}%", flush=True)
                            # endregion

                    final_median = float(median_path[-1]) if len(median_path) else 0.0
                    sim_months = max(1, len(median_path) - 1)
                    required_monthly = monthly_won
                    if target_won > final_median:
                        required_monthly += int(np.ceil((target_won - final_median) / sim_months))

                    simulation_result["target_months"] = target_months
                    simulation_result["achievement_prob"] = achievement_prob
                    simulation_result["sim_mode"] = selected_analysis_mode
                    simulation_result["sim_annual_rate"] = float(_portfolio_annual_rate * 100)
                    simulation_result["required_monthly_contribution"] = int(required_monthly)
                    # STEP6 비교선에서도 동일한 환율 조건을 재사용하기 위해 저장
                    simulation_result["sim_start_rate"] = float(start_rate)
                    simulation_result["sim_mu_fx"] = float(mu_fx)
                    simulation_result["sim_sigma_fx"] = float(sigma_fx)
                    # region agent log
                    _debug_b8b036_log(
                        "initial",
                        "H4,H5",
                        "app.py:step3:after_target_metrics",
                        "STEP3 target metrics after simulation",
                        {
                            "target_won": int(target_won),
                            "target_hits_first_5": [target_months] if target_months is not None else [],
                            "target_months": target_months,
                            "achievement_prob": achievement_prob,
                            "final_median": final_median,
                            "required_monthly_contribution": int(required_monthly),
                            "median_first_values": [float(v) for v in median_path[: min(5, len(median_path))]],
                            "paths_shape": list(paths.shape),
                            "paths_over_target_count": int(np.sum(np.any(paths >= target_won, axis=1))) if paths.size and target_won > 0 else 0,
                        },
                    )
                    # endregion
                    st.session_state["simulation_result"] = simulation_result
            except Exception as exc:
                st.session_state["simulation_result"] = None
                st.error(f"시뮬레이션 실행 중 오류가 발생했습니다: {exc}")

        simulation_result = st.session_state.get("simulation_result")
        if not simulation_result:
            st.info("입력값을 확인한 뒤 **분석 시작** 버튼을 눌러 주세요.")
        else:
            median_path = np.asarray(simulation_result.get("median_path", []), dtype=float)
            target_months = simulation_result.get("target_months")
            achievement_prob = float(simulation_result.get("achievement_prob", 0.0))

            if selected_analysis_mode == "목표 금액 달성":
                st.markdown("**📍 분석 모드: 목표 금액 달성**")
                if target_months is None:
                    st.warning("⚠️ 중앙값 기준으로는 시뮬레이션 기간 내 목표 금액에 도달하지 못했습니다.")
                    st.metric("달성 확률", f"{achievement_prob:.1f}%")
                else:
                    total_months = int(target_months)
                    years_calc = total_months // 12
                    months_remain = total_months % 12
                    if total_months > 0:
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
                            <div style="font-size:0.9rem;opacity:0.85;">Monte Carlo 100회 중앙값 기준 · 달성 확률 {achievement_prob:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("**📍 분석 모드: 목표 기간 확인**")
                if len(median_path) == 0:
                    st.warning("⚠️ 시뮬레이션 결과를 표시할 수 없습니다.")
                else:
                    _final_median_val = float(median_path[-1])
                    _eok = int(_final_median_val) // 100_000_000
                    _man = (int(_final_median_val) % 100_000_000) // 10_000
                    if _eok > 0 and _man > 0:
                        _amount_str = f"{_eok}억 {_man:,}만원"
                    elif _eok > 0:
                        _amount_str = f"{_eok}억원"
                    else:
                        _amount_str = f"{_man:,}만원"
                    st.markdown(
                        f"""
                        <div style="
                            background:linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            border-radius:12px;
                            padding:24px;
                            text-align:center;
                            color:white;
                        ">
                            <div style="font-size:0.95rem;opacity:0.9;margin-bottom:10px;font-weight:600;">예상 달성 금액</div>
                            <div style="font-size:2.2rem;font-weight:800;margin-bottom:6px;">{_amount_str}</div>
                            <div style="font-size:0.9rem;opacity:0.85;">Monte Carlo 100회 중앙값 기준 · 달성 확률 {achievement_prob:.1f}%</div>
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

        simulation_result = st.session_state.get("simulation_result")
        if not simulation_result:
            st.warning("STEP3에서 분석 시작 버튼을 눌러 시뮬레이션을 먼저 실행해 주세요.")
            if st.button("STEP3로 이동", key="goto3d"):
                move_step(3)
            return

        # ── session_state에서 데이터 로드 (재계산 없음) ─────────────────────
        median_path = np.asarray(simulation_result.get("median_path", []), dtype=float)
        sim_paths = np.asarray(simulation_result.get("paths", []), dtype=float)

        # "목표 금액 달성" 모드는 최대 100년 시뮬레이션 후 달성 시점만 저장
        # → median_path를 달성 시점까지만 잘라낸다
        _s_mode = simulation_result.get("sim_mode", st.session_state.get("analysis_mode"))
        _target_months_stored = simulation_result.get("target_months")
        if _s_mode == "목표 금액 달성" and _target_months_stored is not None:
            _cutoff = int(_target_months_stored) + 1
            median_path = median_path[:_cutoff]
            if sim_paths.ndim == 2:
                sim_paths = sim_paths[:, :_cutoff]

        current_capital = float(st.session_state.get("current_capital", 0.0))
        monthly_won = int(st.session_state.get("monthly_contribution", 0))
        recommended_mode = _recommended_investment_mode_from_profile()
        start_date, end_date, years_sim = _resolve_simulation_period()

        st.write("### 결과 시각화")

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("투자 기간", f"{max(0, len(median_path) - 1)}개월")
        with summary_cols[1]:
            st.metric("현재 자금", f"{fmt_money(int(current_capital))}원")
        with summary_cols[2]:
            st.metric("월 적립금", f"{fmt_money(monthly_won)}원")
        with summary_cols[3]:
            st.metric("투자 방식", recommended_mode)


        if len(median_path) == 0:
            st.warning("표시할 시뮬레이션 경로가 없습니다. STEP3에서 다시 분석을 실행해 주세요.")
        else:
            # 정규 길이: median_path 기준 (STEP3 투자 기간)
            _n_months = len(median_path) - 1

            # ── VOO 기준선: 동일 조건, 종목만 VOO 100% ─────────────────────
            _sim_start_rate = float(simulation_result.get("sim_start_rate", get_current_exchange_rate()))
            _sim_mu_fx = float(simulation_result.get("sim_mu_fx", get_exchange_rate_params()[0]))
            _sim_sigma_fx = float(simulation_result.get("sim_sigma_fx", get_exchange_rate_params()[1]))
            _voo_cache_key = str((
                "representative_path_v1",
                int(current_capital),
                monthly_won,
                recommended_mode,
                _n_months,
                round(_sim_start_rate, 6),
                round(_sim_mu_fx, 8),
                round(_sim_sigma_fx, 8),
            ))
            _voo_median_raw = st.session_state.get("step5_voo_median")
            _voo_key_raw = st.session_state.get("step5_voo_cache_key")

            if _voo_median_raw is None or _voo_key_raw != _voo_cache_key:
                try:
                    _voo_annual_rate, _, _ = _get_portfolio_simulation_return({"VOO": 1.0})
                    _voo_mu = np.log1p(max(_voo_annual_rate, -0.99)) / 252
                    _, _voo_sigma = get_portfolio_params({"VOO": 1.0})
                    _voo_sim = run_simulation(
                        current_capital,
                        monthly_won,
                        _n_months / 12.0,       # median_path 개월 수와 정확히 일치
                        recommended_mode,
                        _voo_mu,
                        _voo_sigma,
                        _sim_mu_fx,
                        _sim_sigma_fx,
                        _sim_start_rate,
                        n_simulations=100,
                    )
                    _voo_median = np.asarray(_voo_sim.get("median_path", []), dtype=float)
                    _voo_paths = np.asarray(_voo_sim.get("paths", []), dtype=float)
                except Exception:
                    _voo_median = np.array([])
                    _voo_paths = np.array([])
                st.session_state["step5_voo_median"] = _voo_median.tolist()
                st.session_state["step5_voo_paths"] = _voo_paths.tolist()
                st.session_state["step5_voo_cache_key"] = _voo_cache_key
            else:
                _voo_median = np.asarray(_voo_median_raw, dtype=float)
                _voo_paths = np.asarray(st.session_state.get("step5_voo_paths", []), dtype=float)

            # "목표 금액 달성" 모드에서는 마지막 시점에 목표를 달성한 경로만 후보로 사용
            _target_won_filter = int(st.session_state.get("target_amount", 0))
            _sim_paths_for_repr = sim_paths
            if (
                _s_mode == "목표 금액 달성"
                and _target_won_filter > 0
                and sim_paths.ndim == 2
                and sim_paths.shape[0] > 0
            ):
                _final_vals = sim_paths[:, -1]
                _achieved_mask = _final_vals >= _target_won_filter
                if np.sum(_achieved_mask) > 0:
                    _sim_paths_for_repr = sim_paths[_achieved_mask]

            _representative_path, _selected_crashes, _target_crashes, _target_met = _select_representative_crash_path(
                _sim_paths_for_repr,
                median_path,
                _n_months,
                threshold=-0.15,
            )
            _voo_representative_path, _, _, _ = _select_representative_crash_path(
                _voo_paths,
                _voo_median,
                _n_months,
                threshold=-0.15,
            )

            # 두 선을 동일한 길이로 맞춤 (STEP3 투자 기간 = _n_months + 1 포인트)
            _plot_len = _n_months + 1
            _portfolio_plot = _representative_path[:_plot_len]
            _voo_plot = _voo_representative_path[:_plot_len] if len(_voo_representative_path) >= _plot_len else (
                _voo_representative_path if len(_voo_representative_path) > 0 else np.array([])
            )
            _actual_len = min(len(_portfolio_plot), len(_voo_plot)) if len(_voo_plot) > 0 else len(_portfolio_plot)
            _portfolio_plot = _portfolio_plot[:_actual_len]
            _voo_plot = _voo_plot[:_actual_len] if len(_voo_plot) > 0 else np.array([])

            # 공통 X축 (두 선 동일 범위)
            _x_vals = [pd.Timestamp(start_date) + pd.DateOffset(months=i) for i in range(_actual_len)]

            # ── 목표 금액 달성 모드: 대표 경로(파란 선) 기준으로 3개월 연속 목표 유지 시점 재계산 ──
            if _s_mode == "목표 금액 달성":
                _crash_target_won = int(st.session_state.get("target_amount", 0))
                _crash_achieved_at = None
                if _crash_target_won > 0 and len(_portfolio_plot) >= 3:
                    for _ti in range(1, len(_portfolio_plot) - 2):
                        if all(_portfolio_plot[_ti + k] >= _crash_target_won for k in range(3)):
                            _crash_achieved_at = _ti
                            break
                if _crash_achieved_at is not None:
                    _trim = _crash_achieved_at + 3  # 달성 후 3개월까지 보여줌
                    _portfolio_plot = _portfolio_plot[:_trim]
                    _voo_plot = _voo_plot[:_trim] if len(_voo_plot) >= _trim else _voo_plot
                    _actual_len = len(_portfolio_plot)
                    _x_vals = _x_vals[:_actual_len]
                    _target_months_stored = _crash_achieved_at  # 밴드차트·📅·세금 계산에 반영

            # ── 급락 구간 탐지: 전월 대비 -15% 이상 하락 ──────────────────
            _crash_months = [
                i for i in range(1, len(_portfolio_plot))
                if _portfolio_plot[i - 1] > 0 and _portfolio_plot[i] / _portfolio_plot[i - 1] < 0.85
            ]

            # ── 원금 누적 계산 ──────────────────────────────────────────────
            _principal_plot = [current_capital + monthly_won * i for i in range(_actual_len)]

            # ── Plotly 차트 ────────────────────────────────────────────────
            fig = go.Figure()

            # 원금 누적선 (가장 먼저 그려 다른 선 아래에 위치)
            fig.add_trace(go.Scatter(
                x=_x_vals,
                y=_principal_plot,
                mode="lines",
                name="원금 누적",
                line=dict(color="#F59E0B", width=2),
                hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}원<extra></extra>",
            ))

            # S&P500 기준선
            if len(_voo_plot) > 0:
                fig.add_trace(go.Scatter(
                    x=_x_vals,
                    y=_voo_plot,
                    mode="lines",
                    name="S&P500 기준선 (VOO 대표 경로)",
                    line=dict(color="#6EE7B7", width=2),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}원<extra></extra>",
                ))

            # 내 포트폴리오
            fig.add_trace(go.Scatter(
                x=_x_vals,
                y=_portfolio_plot,
                mode="lines",
                name="내 포트폴리오 (대표 경로)",
                line=dict(color="#2563EB", width=3),
                hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}원<extra></extra>",
            ))

            _target_won_hline = 0 if _s_mode == "목표 기간 확인" else int(st.session_state.get("target_amount", 0))
            if _target_won_hline > 0:
                fig.add_hline(
                    y=_target_won_hline,
                    line_dash="dot",
                    line_color="red",
                    opacity=0.7,
                    annotation_text="목표 금액",
                    annotation_position="right",
                )

            _y_data_max = float(np.max(_portfolio_plot)) if len(_portfolio_plot) else 0.0
            if len(_voo_plot) > 0:
                _y_data_max = max(_y_data_max, float(np.max(_voo_plot)))
            _y_axis_top = _y_data_max * 1.12 if _y_data_max > 0 else 1.0
            if _target_won_hline > 0:
                _y_axis_top = max(_y_axis_top, _target_won_hline * 1.08)
            _n_y_ticks = 6
            _y_tick_values = (
                [float(i * _y_axis_top / (_n_y_ticks - 1)) for i in range(_n_y_ticks)]
                if _y_axis_top > 0
                else [0.0]
            )
            _y_tick_text = [f"{v / 100_000_000:.1f}억" for v in _y_tick_values]

            # 급락 구간 Annotation (상단 근처는 핀을 아래로 배치해 그래프 밖으로 나가지 않게 함)
            _annotations = []
            _has_upper_crash_pin = False
            for _cm in _crash_months:
                if _cm < len(_x_vals):
                    _crash_y = float(_portfolio_plot[_cm])
                    _near_top = _y_axis_top > 0 and (_crash_y / _y_axis_top) >= 0.72
                    if _near_top:
                        _has_upper_crash_pin = True
                        _pin_ay = 34
                    else:
                        _pin_ay = -34
                    _annotations.append(dict(
                        x=_x_vals[_cm],
                        y=_crash_y,
                        xref="x",
                        yref="y",
                        text="📌",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#ef4444",
                        arrowwidth=1.5,
                        ax=0,
                        ay=_pin_ay,
                        font=dict(size=20, color="#ef4444"),
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        borderpad=2,
                    ))

            _plot_margin_top = 90 if _has_upper_crash_pin else 60

            fig.update_layout(
                title=f"Monte Carlo 시뮬레이션 경로 ({recommended_mode})",
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="white",
                font=dict(family="Malgun Gothic", size=13),
                margin=dict(l=80, r=110, t=_plot_margin_top, b=60),
                hovermode="x unified",
                legend=dict(
                    x=0.01,
                    y=0.99,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)",
                ),
                xaxis=dict(
                    title="투자 기간",
                    tickformat="%Y년",
                    dtick="M12",
                    ticklabelmode="period",
                    showgrid=False,
                ),
                yaxis=dict(
                    title="자산 금액",
                    range=[0, _y_axis_top],
                    tickvals=_y_tick_values,
                    ticktext=_y_tick_text,
                    gridcolor="#E5E7EB",
                    gridwidth=0.5,
                    showgrid=True,
                ),
                annotations=_annotations,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── 3. 급락 구간 설명 ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 급락 구간 설명")
            _crash_info_rows = [
                ("📌", "급락 구간: 전월 대비 15% 하락한 구간입니다."),
                ("💡", "급락 시 여유자금이 있다면 추가 매수를 고려해 보세요."),
            ]
            if _s_mode == "목표 금액 달성":
                if _target_months_stored is not None:
                    _tm_date_label = (
                        pd.Timestamp(start_date) + pd.DateOffset(months=int(_target_months_stored))
                    ).strftime("%Y년 %m월")
                    _calendar_text = (
                        f"목표 달성 시점: {_tm_date_label} "
                        f"(중앙값 기준 목표 금액을 3개월 연속 유지하는 첫 시점)"
                    )
                else:
                    _period_end = _x_vals[-1] if _x_vals else pd.Timestamp(end_date)
                    _calendar_text = (
                        f"목표 달성 시점: 시뮬레이션 기간({_period_end.strftime('%Y년 %m월')}) 내 "
                        f"중앙값 기준 목표 금액 3개월 연속 유지 미달성"
                    )
                _crash_info_rows.append(("📅", _calendar_text))
            _crash_info_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:0.4rem;'
                f'margin:{"0.35rem 0 0 0" if i else "0"};">'
                f'<span style="flex-shrink:0;line-height:1.6;">{icon}</span>'
                f'<span style="line-height:1.6;">{text}</span></div>'
                for i, (icon, text) in enumerate(_crash_info_rows)
            )
            st.markdown(
                f'<div style="padding:0.75rem 1rem;background-color:rgba(28,131,225,0.12);'
                f'border-radius:0.5rem;line-height:1.6;">{_crash_info_html}</div>',
                unsafe_allow_html=True,
            )
            if not _crash_months:
                st.warning(
                    "급락 구간 없음\n\n"
                    "이번 대표 경로에서는 전월 대비 15% 이상 하락 구간이 발생하지 않았습니다."
                )

            # ── 4. Monte Carlo 불확실성 구간 (10~90% 밴드) ─────────────────
            st.markdown("---")
            st.markdown("#### 📊 Monte Carlo 불확실성 구간")

            _band_median = np.asarray(simulation_result.get("median_path", []), dtype=float)
            _band_upper = np.asarray(simulation_result.get("upper_path", []), dtype=float)
            _band_lower = np.asarray(simulation_result.get("lower_path", []), dtype=float)
            if _s_mode == "목표 금액 달성" and _target_months_stored is not None:
                _band_cutoff = int(_target_months_stored) + 1
                _band_median = _band_median[:_band_cutoff]
                _band_upper = _band_upper[:_band_cutoff]
                _band_lower = _band_lower[:_band_cutoff]

            _band_len = min(
                len(_band_median),
                len(_band_upper) if len(_band_upper) else len(_band_median),
                len(_band_lower) if len(_band_lower) else len(_band_median),
            )
            if _band_len > 0:
                _band_median = _band_median[:_band_len]
                _band_upper = _band_upper[:_band_len]
                _band_lower = _band_lower[:_band_len]
                _x_band = [
                    pd.Timestamp(start_date) + pd.DateOffset(months=i)
                    for i in range(_band_len)
                ]

                fig_band = go.Figure()
                fig_band.add_trace(go.Scatter(
                    x=list(_x_band) + list(_x_band[::-1]),
                    y=list(_band_upper) + list(_band_lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(37, 99, 235, 0.18)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="10~90% 구간",
                    hoverinfo="skip",
                    showlegend=True,
                ))
                fig_band.add_trace(go.Scatter(
                    x=_x_band,
                    y=_band_lower,
                    mode="lines",
                    name="하위 10%",
                    line=dict(color="rgba(37, 99, 235, 0.35)", width=1, dash="dot"),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}원<extra></extra>",
                ))
                fig_band.add_trace(go.Scatter(
                    x=_x_band,
                    y=_band_upper,
                    mode="lines",
                    name="상위 10%",
                    line=dict(color="rgba(37, 99, 235, 0.35)", width=1, dash="dot"),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}원<extra></extra>",
                ))
                fig_band.add_trace(go.Scatter(
                    x=_x_band,
                    y=_band_median,
                    mode="lines",
                    name="중앙값 (50%)",
                    line=dict(color="#2563EB", width=2.5),
                    hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}원<extra></extra>",
                ))

                if _target_won_hline > 0:
                    fig_band.add_hline(
                        y=_target_won_hline,
                        line_dash="dot",
                        line_color="red",
                        opacity=0.7,
                        annotation_text="목표 금액",
                        annotation_position="right",
                    )

                _band_y_max = float(
                    max(
                        np.max(_band_upper) if len(_band_upper) else 0.0,
                        np.max(_band_median) if len(_band_median) else 0.0,
                    )
                )
                _band_y_top = _band_y_max * 1.12 if _band_y_max > 0 else 1.0
                if _target_won_hline > 0:
                    _band_y_top = max(_band_y_top, _target_won_hline * 1.08)
                _band_n_ticks = 6
                _band_tick_values = (
                    [float(i * _band_y_top / (_band_n_ticks - 1)) for i in range(_band_n_ticks)]
                    if _band_y_top > 0
                    else [0.0]
                )
                _band_tick_text = [f"{v / 100_000_000:.1f}억" for v in _band_tick_values]

                fig_band.update_layout(
                    title="Monte Carlo 100회 시뮬레이션 · 불확실성 범위",
                    plot_bgcolor="#f8f9fa",
                    paper_bgcolor="white",
                    font=dict(family="Malgun Gothic", size=13),
                    margin=dict(l=80, r=110, t=60, b=60),
                    hovermode="x unified",
                    legend=dict(
                        x=0.01,
                        y=0.99,
                        xanchor="left",
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.85)",
                    ),
                    xaxis=dict(
                        title="투자 기간",
                        tickformat="%Y년",
                        dtick="M12",
                        ticklabelmode="period",
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title="자산 금액",
                        range=[0, _band_y_top],
                        tickvals=_band_tick_values,
                        ticktext=_band_tick_text,
                        gridcolor="#E5E7EB",
                        gridwidth=0.5,
                        showgrid=True,
                    ),
                )
                st.plotly_chart(fig_band, use_container_width=True)
                st.caption(
                    "음영은 100회 시뮬 중 하위 10%~상위 10% 구간이고, "
                    "파란 실선은 중앙값(50%)입니다. 대표 경로 그래프와 달리 "
                    "개별 시나리오가 아니라 결과가 퍼질 수 있는 범위를 보여줍니다."
                )
            else:
                st.info("불확실성 구간을 표시할 시뮬레이션 데이터가 없습니다.")

            # ── 세금 영향 분석 ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💰 세금 고려 시 예상 자산")

            _CAPITAL_GAINS_TAX_RATE = 0.22
            _DIVIDEND_INCOME_TAX_RATE = 0.154
            _MONTHLY_DIVIDEND_ETFS = {"JEPI", "JEPQ", "QYLD"}
            _goal_month = max(0, len(_portfolio_plot) - 1)
            _goal_exchange_rate = _sim_start_rate * ((1 + _sim_mu_fx) ** _goal_month)

            if _target_months_stored is not None and _s_mode == "목표 금액 달성":
                _tm_goal = int(_target_months_stored)
                _goal_period_label = (
                    f"{_tm_goal // 12}년 {_tm_goal % 12}개월"
                    if _tm_goal % 12
                    else f"{_tm_goal // 12}년"
                )
            elif _goal_month < len(_x_vals):
                _goal_period_label = _x_vals[_goal_month].strftime("%Y년 %m월")
            else:
                _goal_period_label = f"{_goal_month}개월"

            _stock_pretax = float(_portfolio_plot[_goal_month])
            _total_invested = current_capital + monthly_won * _goal_month
            _capital_gain = max(0.0, _stock_pretax - _total_invested)
            _cg_tax = int(round(_capital_gain * _CAPITAL_GAINS_TAX_RATE))
            _stock_after = int(round(_stock_pretax - _cg_tax))

            _step5_selected_etfs = st.session_state.get("selected_etfs", [])
            _step5_etf_weights = st.session_state.get("etf_weights", {})
            try:
                _step5_portfolio_weights = build_portfolio_weights(_step5_selected_etfs, _step5_etf_weights)
            except Exception:
                _step5_portfolio_weights = {
                    t: 1.0 / max(1, len(_step5_selected_etfs)) for t in _step5_selected_etfs
                }

            _dividend_yields = {
                _ticker: _dy
                for _ticker in _step5_selected_etfs
                if ETF_DATA.get(_ticker, {}).get("카테고리") in {"배당성장", "배당집중"}
                and (_dy := _get_etf_dividend_yield(_ticker)) is not None
                and 0.01 <= _dy <= 0.20
            }

            _cumulative_div_pretax_won = 0.0
            _cumulative_div_tax_won = 0.0
            if _dividend_yields and _goal_month > 0:
                _dividend_asset_usd = {
                    _ticker: (current_capital * _step5_portfolio_weights.get(_ticker, 0)) / _sim_start_rate
                    for _ticker in _dividend_yields
                }
                _monthly_growth_rates = {
                    _ticker: (1 + max(_get_etf_cagr_rate(_ticker), -0.99)) ** (1 / 12) - 1
                    for _ticker in _dividend_yields
                }
                for _month in range(1, _goal_month + 1):
                    _month_exchange_rate = _sim_start_rate * ((1 + _sim_mu_fx) ** _month)
                    _month_div_usd = 0.0
                    for _ticker, _dy in _dividend_yields.items():
                        _weight = _step5_portfolio_weights.get(_ticker, 0)
                        if monthly_won > 0 and _weight > 0:
                            _dividend_asset_usd[_ticker] += (monthly_won * _weight) / _month_exchange_rate
                        _dividend_asset_usd[_ticker] *= max(0.0, 1 + _monthly_growth_rates.get(_ticker, 0.0))
                        _ann_div_usd = _dividend_asset_usd[_ticker] * _dy
                        _mon_div_usd = (
                            _ann_div_usd / 12
                            if _ticker in _MONTHLY_DIVIDEND_ETFS
                            else (_ann_div_usd / 4) / 3
                        )
                        _month_div_usd += _mon_div_usd
                    _month_div_won = _month_div_usd * _month_exchange_rate
                    _cumulative_div_pretax_won += _month_div_won
                    _cumulative_div_tax_won += _month_div_won * _DIVIDEND_INCOME_TAX_RATE

            _div_pretax = int(round(_cumulative_div_pretax_won))
            _div_tax = int(round(_cumulative_div_tax_won))
            _div_after = int(round(_cumulative_div_pretax_won - _cumulative_div_tax_won))
            _final_holding = _stock_after + _div_after

            st.html(
                f"""
<div style="font-family:'Malgun Gothic',sans-serif;max-width:680px;margin:0 auto;">
  <p style="color:#6B7280;font-size:0.82rem;margin:0 0 12px 2px;letter-spacing:0.02em;">
    목표 달성: {_goal_period_label} &nbsp;·&nbsp; 환율 {_goal_exchange_rate:,.0f}원/달러
  </p>
  <div style="background:linear-gradient(135deg,#1D4ED8 0%,#2563EB 100%);border-radius:16px;padding:28px 32px;text-align:center;margin-bottom:14px;box-shadow:0 4px 18px rgba(37,99,235,0.25);">
    <div style="color:rgba(255,255,255,0.75);font-size:0.85rem;letter-spacing:0.08em;margin-bottom:6px;">최종 보유액 (세후 주식 + 세후 배당)</div>
    <div style="color:#FFFFFF;font-size:2.15rem;font-weight:700;line-height:1.15;">{fmt_money(_final_holding)}원</div>
  </div>
  <div style="display:flex;gap:12px;margin-bottom:14px;">
    <div style="flex:1;background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:18px 20px;">
      <div style="color:#1D4ED8;font-size:0.78rem;font-weight:600;letter-spacing:0.06em;margin-bottom:8px;">📈 주식 자산</div>
      <div style="color:#6B7280;font-size:0.8rem;text-decoration:line-through;margin-bottom:4px;">세전 {fmt_money(int(round(_stock_pretax)))}원</div>
      <div style="color:#1E3A8A;font-size:1.35rem;font-weight:700;margin-bottom:8px;">{fmt_money(_stock_after)}원</div>
      <div style="color:#EF4444;font-size:0.75rem;background:#FEE2E2;border-radius:6px;padding:3px 8px;display:inline-block;">양도소득세 22% -{fmt_money(_cg_tax)}원</div>
    </div>
    <div style="flex:1;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:12px;padding:18px 20px;">
      <div style="color:#15803D;font-size:0.78rem;font-weight:600;letter-spacing:0.06em;margin-bottom:8px;">💰 배당금 누적</div>
      <div style="color:#6B7280;font-size:0.8rem;text-decoration:line-through;margin-bottom:4px;">세전 {fmt_money(_div_pretax)}원</div>
      <div style="color:#14532D;font-size:1.35rem;font-weight:700;margin-bottom:8px;">{fmt_money(_div_after)}원</div>
      <div style="color:#EF4444;font-size:0.75rem;background:#FEE2E2;border-radius:6px;padding:3px 8px;display:inline-block;">배당소득세 15.4% -{fmt_money(_div_tax)}원</div>
    </div>
  </div>
  <p style="color:#9CA3AF;font-size:0.75rem;line-height:1.6;margin:0 2px;">
    양도소득세는 투입 원금 대비 이익분에 22%를 적용했고, 배당소득세는 매월 배당 수령 시 15.4%를 누적 반영했습니다.
    환율은 시뮬레이션 시작 환율과 월별 환율 변동 기댓값을 복리 적용한 추정치입니다.
  </p>
</div>
                """
            )

        cols = st.columns([2, 2, 1])
        if cols[0].button("이전", key="prev_6"):
            move_step(4)


if __name__ == "__main__":
    run_streamlit_app()

