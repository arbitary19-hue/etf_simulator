# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import anthropic
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ETF 포트폴리오 시뮬레이션 - app.py
# ============================================================

# ============================================================
# 섹션 1. ETF 데이터 정의
# ============================================================

ETF_DATA = {
    "VOO":  {"이름": "Vanguard S&P 500",         "카테고리": "지수추적", "보수율": 0.0003, "배당": True,  "레버리지": False},
    "QQQ":  {"이름": "Invesco Nasdaq 100",        "카테고리": "지수추적", "보수율": 0.0020, "배당": True,  "레버리지": False},
    "VTI":  {"이름": "Vanguard Total Stock Market", "카테고리": "지수추적", "보수율": 0.0003, "배당": True,  "레버리지": False},
    "SCHD": {"이름": "Schwab U.S. Dividend Equity", "카테고리": "배당형",   "보수율": 0.0006, "배당": True,  "레버리지": False},
    "DGRO": {"이름": "iShares Core Dividend Growth", "카테고리": "배당형",   "보수율": 0.0008, "배당": True,  "레버리지": False},
    "VYM":  {"이름": "Vanguard High Dividend Yield", "카테고리": "배당형",   "보수율": 0.0006, "배당": True,  "레버리지": False},
    "JEPI": {"이름": "JPMorgan Equity Premium Income", "카테고리": "금융형", "보수율": 0.0035, "배당": True,  "레버리지": False},
    "JEPQ": {"이름": "JPMorgan Nasdaq Equity Premium", "카테고리": "금융형", "보수율": 0.0035, "배당": True,  "레버리지": False},
    "QYLD": {"이름": "Global X Nasdaq 100 Covered Call", "카테고리": "금융형", "보수율": 0.0060, "배당": True,  "레버리지": False},
    "QLD":  {"이름": "ProShares Ultra QQQ",         "카테고리": "레버리지", "보수율": 0.0095, "배당": False, "레버리지": True},
    "TQQQ": {"이름": "ProShares UltraPro QQQ",      "카테고리": "레버리지", "보수율": 0.0088, "배당": False, "레버리지": True},
    "SSO":  {"이름": "ProShares Ultra S&P 500",     "카테고리": "레버리지", "보수율": 0.0089, "배당": False, "레버리지": True},
    "UPRO": {"이름": "ProShares UltraPro S&P 500",  "카테고리": "레버리지", "보수율": 0.0091, "배당": False, "레버리지": True},
    "SOXL": {"이름": "Direxion Daily Semiconductor Bull 3x", "카테고리": "레버리지", "보수율": 0.0075, "배당": False, "레버리지": True},
    "VGT":  {"이름": "Vanguard Information Technology", "카테고리": "기술",     "보수율": 0.0010, "배당": True,  "레버리지": False},
    "SOXX": {"이름": "iShares Semiconductor ETF",  "카테고리": "기술",     "보수율": 0.0035, "배당": True,  "레버리지": False},
    "SMH":  {"이름": "VanEck Semiconductor ETF",   "카테고리": "기술",     "보수율": 0.0035, "배당": True,  "레버리지": False},
    "DRAM": {"이름": "Global X Memory & Storage ETF", "카테고리": "기술", "보수율": 0.0050, "배당": True,  "레버리지": False},
    "XLV":  {"이름": "Health Care Select Sector SPDR", "카테고리": "안정형", "보수율": 0.0010, "배당": True,  "레버리지": False},
    "SGOV": {"이름": "iShares Short Treasury Bond ETF", "카테고리": "안정형", "보수율": 0.0007, "배당": True,  "레버리지": False},
}

# ============================================================
# 섹션 2. AI 성향 분석 함수
# ============================================================

AI_PROFILE_CATEGORIES = ["성장형", "배당형", "금융형", "기술", "안정형"]

# 점수 계산에 사용될 모든 카테고리 (레버리지 포함)
ALL_SCORE_CATEGORIES = ["성장형", "배당형", "금융형", "기술", "안정형", "레버리지"]

# 새로운 질문 구조: 조건부 질문 지원
AI_PROFILE_QUESTIONS = {
    "Q1": {
        "text": "1단계: 투자 기본 환경\n\n**Q1. 이번에 굴리실 투자 자금의 주된 목적과 목표 기간은 어떻게 되시나요?**",
        "choices": [
            "① 1~3년 내에 써야 하는 전세금 / 결혼 자금 (단기)",
            "② 5~10년 뒤 집 장만을 위한 목돈 마련 (중기)",
            "③ 15년 이상 장기 노후 자금 준비 (장기)"
        ],
        "scores": [
            {"안정형": 3},
            {"배당형": 2, "성장형": 1},
            {"성장형": 3, "레버리지": 1}
        ]
    },
    "Q2": {
        "text": "**Q2. 매달 추가로 저축(적립식 투자)을 하실 여력이 있으신가요?**",
        "choices": [
            "① 매달 일정 금액을 꾸준히 저축할 수 있습니다.",
            "② 지금 가진 목돈을 한 번에 묻어두고 싶습니다."
        ],
        "scores": [
            {"성장형": 1},
            {}
        ]
    },
    "Q3": {
        "text": "2단계: 수익 보상 선호도\n\n**Q3. 투자로 얻는 이익 중 어떤 형태를 더 선호하시나요?**",
        "choices": [
            "① 월별로 꾸준히 나오는 배당/금리 수익 (현금흐름)",
            "② 매달 나오진 않지만 기업이 성장하면서 주가가 크게 오르는 것",
            "③ 주가 성장도 적당히 하면서, 매년 배당금도 늘려주는 것"
        ],
        "scores": [
            {"배당형": 2, "금융형": 2},
            {"성장형": 3},
            {"배당형": 2, "성장형": 1}
        ]
    },
    "Q4": {
        "text": "3단계: 위험 감수 성향 (가장 중요)\n\n**Q4. 만약 내가 1,000만 원을 투자했는데, 세계 경제 위기로 한 달 만에 30% 손실이 난다면 심정이 어떠실 것 같나요?**",
        "choices": [
            "① 정신이 번쩍 든다. 손실이 두렵고 안정자산으로 옮기고 싶다.",
            "② 속은 쓰리지만 시장은 우상향할 거니 꾹 참고 버틴다.",
            "③ 어차피 장기 투자다. 바겐세일 기간이니 돈을 더 끌어와서 추가 매수한다."
        ],
        "scores": [
            {"안정형": 4},
            {"성장형": 2, "배당형": 1},
            {"성장형": 3, "레버리지": 2}
        ],
        "follow_up": {
            3: "Q5"  # Q4에서 ③번 선택 시 Q5로 이동
        }
    },
    "Q5": {
        "text": "**Q5. (심화 질문) 시장이 내 예상과 반대로 갈 때 3배 빠르게 자산이 녹아내리는 극단적인 변동성(레버리지)도 감당할 준비가 되셨나요?**",
        "choices": [
            "① 아, 3배는 너무 무섭네요. 일반 지수 추종이 좋겠습니다.",
            "② 하이 리스크 하이 리턴! 화끈하게 감수하겠습니다."
        ],
        "scores": [
            {"성장형": 1},
            {"레버리지": 3, "성장형": 1}
        ],
        "condition": "previous_q4_choice == 2"
    },
    "Q6": {
        "text": "4단계: 관심 분야\n\n**Q6. 투자 시 특정 산업을 선택할 때, 어떤 방식을 선호하시나요?**",
        "choices": [
            "① 미국 전체 시장(대기업 500개~전체)에 분산 투자하고 싶다.",
            "② AI, 반도체, IT 기술 기업들의 미래를 강력하게 믿는다.",
            "③ 고령화 시대에 헬스케어/바이오/의약품 분야가 유망하다고 본다."
        ],
        "scores": [
            {"성장형": 1},
            {"기술": 3, "성장형": 1},
            {"안정형": 2, "기술": 1}
        ]
    },
    "Q7": {
        "text": "5단계: 배당/금융형 선호도\n\n**Q7. 배당 수익이 중요하다면, 어떤 방식을 원하시나요?**",
        "choices": [
            "① 매년 1-2회 정도 정해진 시기에 받는 배당금 (연 배당)",
            "② 예측 불가능하지만 자주 받는 배당/수익 (월 배당)",
            "③ 잘 모르겠는데, 배당이 중요하지는 않습니다."
        ],
        "scores": [
            {"배당형": 3},
            {"금융형": 3, "배당형": 1},
            {}
        ]
    },
    "Q8": {
        "text": "6단계: 기술/반도체 집중도\n\n**Q8. 기술 섹터 내에서 어떤 전략을 선호하시나요?**",
        "choices": [
            "① 기술/IT 전체에 분산 투자 (Apple, NVIDIA, Google, Microsoft 등 다양)",
            "② 반도체 기업 집중 (NVIDIA, TSMC, Intel, Broadcom 등)",
            "③ 반도체에 매우 집중 + 3배 레버리지 추구"
        ],
        "scores": [
            {"기술": 2},
            {"기술": 3},
            {"기술": 3, "레버리지": 2}
        ]
    }
}

# 질문 순서 (조건부 질문 제외)
AI_PROFILE_BASE_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q6", "Q7", "Q8"]


def get_next_question(current_q, responses):
    """현재 질문 다음에 표시할 질문을 결정합니다."""
    # Q5 다음은 항상 Q6
    if current_q == "Q5":
        return "Q6"
    
    # Q4 → Q5로의 조건부 분기
    if current_q == "Q4" and responses.get("Q4") == 2:  # Q4에서 ③번(인덱스 2) 선택
        return "Q5"
    
    # Q4에서 ③번이 아니면 Q6으로
    if current_q == "Q4" and responses.get("Q4") != 2:
        return "Q6"
    
    # 기본 순서로 진행
    base_order = ["Q1", "Q2", "Q3", "Q4", "Q6", "Q7", "Q8"]  # Q5는 조건부
    try:
        current_idx = base_order.index(current_q)
        if current_idx + 1 < len(base_order):
            return base_order[current_idx + 1]
    except ValueError:
        pass
    
    return None


def calculate_profile_scores(responses):
    """사용자 응답을 기반으로 카테고리별 점수를 계산합니다."""
    scores = {cat: 0 for cat in AI_PROFILE_CATEGORIES}
    leverage_score = 0  # 레버리지 점수 별도 추적
    
    for q_key, choice_idx in responses.items():
        if q_key not in AI_PROFILE_QUESTIONS:
            continue
        
        q_data = AI_PROFILE_QUESTIONS[q_key]
        if choice_idx is None or choice_idx >= len(q_data["scores"]):
            continue
            
        choice_scores = q_data["scores"][choice_idx]
        for category, points in choice_scores.items():
            # 레버리지 점수는 별도로 추적
            if category == "레버리지":
                leverage_score += points
            elif category in scores:
                scores[category] += points
    
    # 정규화 (합계 100%)
    total = sum(scores.values())
    if total == 0:
        profile = {cat: 100 / len(AI_PROFILE_CATEGORIES) for cat in AI_PROFILE_CATEGORIES}
    else:
        profile = {cat: round((scores[cat] / total) * 100, 2) for cat in AI_PROFILE_CATEGORIES}
    
    # 레버리지 점수를 별도 필드로 저장 (성장형에 영향)
    profile["_leverage_score"] = leverage_score
    
    return profile


def get_question_text_and_choices(q_key):
    """질문 키에 해당하는 텍스트와 선택지를 반환합니다."""
    if q_key not in AI_PROFILE_QUESTIONS:
        return None, None
    
    q_data = AI_PROFILE_QUESTIONS[q_key]
    return q_data["text"], q_data["choices"]

# ============================================================
# 섹션 3. ETF 추천 함수
# ============================================================

ETF_CATEGORY_MAP = {
    "지수추적": "성장형",
    "배당형": "배당형",
    "금융형": "금융형",
    "레버리지": "성장형",
    "기술": "기술",
    "안정형": "안정형",
}

CATEGORY_TO_ETFS = {
    "성장형": ["VOO", "QQQ", "VTI"],  # 지수추적만
    "배당형": ["SCHD", "DGRO", "VYM"],
    "금융형": ["JEPI", "JEPQ", "QYLD"],
    "기술": ["VGT", "SOXX", "SMH", "DRAM"],
    "안정형": ["SGOV", "XLV"],
    "레버리지": ["TQQQ", "UPRO", "SOXL", "QLD", "SSO"],
}


def _normalize_profile_weights(profile_weights):
    """레버리지 점수를 제외한 카테고리만 정규화합니다."""
    regular_weights = {cat: profile_weights.get(cat, 0) for cat in AI_PROFILE_CATEGORIES}
    total = sum(regular_weights.values())
    if total <= 0:
        raise ValueError("프로필 가중치 합계는 0보다 커야 합니다.")
    return {cat: regular_weights[cat] / total for cat in AI_PROFILE_CATEGORIES}


def recommend_etfs_with_weights(profile_weights, top_n=5):
    """투자 성향 비율을 받아 ETF와 비중을 함께 추천합니다.
    
    반환: {ticker: weight} 딕셔너리
    """
    normalized = _normalize_profile_weights(profile_weights)
    leverage_score = profile_weights.get("_leverage_score", 0)
    
    # 레버리지 점수를 성장형에 통합하여 비중 계산 (0~1 사이 정규화)
    # 예: 레버리지 0.3은 성장형의 30% 가중치로 추가
    growth_with_leverage = normalized["성장형"] + (leverage_score * normalized["성장형"])
    
    selected = []
    
    # 각 카테고리의 할당 개수 계산 (점수에 비례)
    # 성장형 비중이 높으면 레버리지 선택 가능성도 높아짐
    category_counts = {}
    for category in AI_PROFILE_CATEGORIES:
        if category == "성장형":
            allocation = int(round(growth_with_leverage * top_n))
        else:
            allocation = int(round(normalized[category] * top_n))
        category_counts[category] = max(0, allocation)
    
    # 반올림으로 인해 합계가 top_n과 안 맞을 수 있으니 조정
    total_allocated = sum(category_counts.values())
    if total_allocated < top_n:
        # 가장 높은 스코어의 카테고리에 추가
        top_category = "성장형" if growth_with_leverage >= normalized["성장형"] else max(AI_PROFILE_CATEGORIES, key=lambda c: normalized[c])
        category_counts[top_category] += top_n - total_allocated
    elif total_allocated > top_n:
        # 가장 낮은 스코어의 카테고리에서 제거
        lowest_category = min(AI_PROFILE_CATEGORIES, key=lambda c: normalized[c])
        category_counts[lowest_category] = max(0, category_counts[lowest_category] - (total_allocated - top_n))
    
    # 카테고리별로 정렬 (높은 점수 순)
    sorted_categories = sorted(AI_PROFILE_CATEGORIES, key=lambda c: normalized[c], reverse=True)
    
    # ETF 선택 + 비중 저장
    etf_to_category = {}  # ticker -> category 매핑
    etf_to_weight_category = {}  # ticker -> 비중 계산용 카테고리 (레버리지용 별도 처리)
    
    # 성장형에서 레버리지 선택 개수 결정
    growth_allocation = category_counts["성장형"]
    leverage_count = 0
    if leverage_score > 0 and growth_allocation > 0:
        # 레버리지 스코어 비율에 따라 일부를 레버리지 ETF로 대체
        leverage_ratio = leverage_score / (normalized["성장형"] + leverage_score)
        leverage_count = max(1, int(round(growth_allocation * leverage_ratio)))
    
    # 각 카테고리에서 필요한 개수만큼 ETF 선택
    for category in sorted_categories:
        needed = category_counts[category]
        if needed <= 0:
            continue
        
        # 성장형이면서 레버리지 할당이 있는 경우
        if category == "성장형" and leverage_count > 0:
            # 일부는 레버리지에서, 일부는 성장형에서 선택
            leverage_etfs = CATEGORY_TO_ETFS.get("레버리지", [])[:leverage_count]
            growth_etfs = CATEGORY_TO_ETFS.get(category, [])[leverage_count:leverage_count + (needed - leverage_count)]
            
            for ticker in leverage_etfs:
                if ticker not in selected and len(selected) < top_n:
                    selected.append(ticker)
                    etf_to_category[ticker] = "성장형"
                    etf_to_weight_category[ticker] = "레버리지"  # 비중 계산용
            
            for ticker in growth_etfs:
                if ticker not in selected and len(selected) < top_n:
                    selected.append(ticker)
                    etf_to_category[ticker] = "성장형"
                    etf_to_weight_category[ticker] = "성장형"
        else:
            etf_list = CATEGORY_TO_ETFS.get(category, [])[:needed]
            for ticker in etf_list:
                if ticker not in selected and len(selected) < top_n:
                    selected.append(ticker)
                    etf_to_category[ticker] = category
                    etf_to_weight_category[ticker] = category
    
    # 부족한 경우 더 추가
    if len(selected) < top_n:
        all_etfs = []
        for cat_etfs in CATEGORY_TO_ETFS.values():
            all_etfs.extend(cat_etfs)
        
        for ticker in all_etfs:
            if ticker not in selected and len(selected) < top_n:
                selected.append(ticker)
                etf_category = etf_to_category.get(ticker, ETF_CATEGORY_MAP.get(ETF_DATA[ticker]["카테고리"], "성장형"))
                etf_to_category[ticker] = etf_category
                etf_to_weight_category[ticker] = etf_category
    
    # 비중 계산: 카테고리별 점수에 따라 분배
    # 레버리지는 성장형 비중의 일부로 계산
    weights = {}
    
    # 레버리지 ETF와 일반 성장형 ETF를 구분해서 비중 계산
    leverage_etfs = [t for t in selected if etf_to_weight_category.get(t) == "레버리지"]
    growth_etfs = [t for t in selected if etf_to_weight_category.get(t) == "성장형"]
    
    # 성장형 비중을 레버리지와 성장형으로 분배
    if leverage_etfs and growth_etfs:
        leverage_ratio = leverage_score / (normalized["성장형"] + leverage_score)
        growth_allocation_weight = normalized["성장형"] * (1 - leverage_ratio)
        leverage_allocation_weight = normalized["성장형"] * leverage_ratio
    else:
        growth_allocation_weight = normalized["성장형"]
        leverage_allocation_weight = 0
    
    for ticker in selected:
        weight_category = etf_to_weight_category.get(ticker, etf_to_category.get(ticker, "성장형"))
        
        if weight_category == "레버리지":
            category_weight = leverage_allocation_weight
            same_category_etfs = leverage_etfs
        elif weight_category == "성장형":
            category_weight = growth_allocation_weight
            same_category_etfs = growth_etfs
        else:
            category_weight = normalized.get(weight_category, 0)
            same_category_etfs = [t for t in selected if etf_to_weight_category.get(t) == weight_category]
        
        weights[ticker] = category_weight / len(same_category_etfs) if same_category_etfs else 1 / len(selected)
    
    # 정규화
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {ticker: round(w / total_weight, 4) for ticker, w in weights.items()}
    
    return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))


def recommend_etfs(profile_weights, top_n=5):
    """투자 성향 비율을 받아 ETF를 추천합니다."""
    weights = recommend_etfs_with_weights(profile_weights, top_n)
    return list(weights.keys())[:top_n]


def detect_sector_overlap(selected_etfs):
    """선택 ETF 중 카테고리 중복 여부를 확인합니다."""
    category_map = {}
    overlaps = []

    for ticker in selected_etfs:
        category = ETF_DATA.get(ticker, {}).get("카테고리", "정보없음")
        category_map.setdefault(category, []).append(ticker)

    for category, tickers in category_map.items():
        if len(tickers) > 1:
            alternatives = [
                t for t, info in ETF_DATA.items()
                if info["카테고리"] != category and t not in selected_etfs
            ][:3]
            overlaps.append({
                "중복카테고리": category,
                "중복ETF": tickers,
                "대체ETF": alternatives,
            })

    return overlaps

# ============================================================
# 섹션 4. 시뮬레이션 함수
# ============================================================

SIMULATION_BASE_STATS = {
    "지수추적": {"mu": 0.075, "sigma": 0.15},
    "배당형": {"mu": 0.055, "sigma": 0.12},
    "금융형": {"mu": 0.050, "sigma": 0.10},
    "기술":   {"mu": 0.080, "sigma": 0.18},
    "안정형": {"mu": 0.020, "sigma": 0.05},
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
    if total <= 0:
        raise ValueError("포트폴리오 가중치 합계는 0보다 커야 합니다.")
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


def build_strategy_reason(profile_weights, selected_etfs, analysis_mode):
    reasons = []
    if profile_weights["성장형"] >= max(profile_weights.values()):
        reasons.append("성장형 성향을 반영하여 기술/테크 중심 ETF를 추천합니다.")
    if profile_weights["배당형"] >= max(profile_weights.values()):
        reasons.append("배당 성향이 높아 안정적인 배당 ETF를 포함했습니다.")
    if profile_weights["금융형"] >= max(profile_weights.values()):
        reasons.append("금융형 중심 ETF로 포트폴리오 안정성을 보완합니다.")
    if profile_weights["기술"] >= max(profile_weights.values()):
        reasons.append("기술 집중 전략으로 성장 기회를 노립니다.")
    if profile_weights["안정형"] >= max(profile_weights.values()):
        reasons.append("안정형 비중이 높아 변동성을 낮추는 전략을 권장합니다.")
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
        parsed = parse_int(raw_value)
        formatted = fmt_money(parsed) if raw_value.strip() != "" else ""
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

    if "app_step" not in st.session_state:
        st.session_state["app_step"] = 1
        st.session_state["current_question"] = "Q1"
        st.session_state["profile_responses"] = {}  # {Q_KEY: choice_index}
        st.session_state["profile_submitted"] = False
        st.session_state["profile_weights"] = None
        st.session_state["selected_etfs"] = []
        st.session_state["overlap_info"] = []
        st.session_state["analysis_mode"] = ANALYSIS_MODES[0]
        st.session_state["current_capital"] = 10000000.0
        st.session_state["monthly_contribution"] = 100000.0
        st.session_state["years"] = 10
        st.session_state["target_amount"] = 500000000.0
        st.session_state["target_years"] = 10
        st.session_state["sim_mode"] = INVESTMENT_MODES[1]
        st.session_state["simulation_results"] = None

    step_labels = [
        "STEP1: AI 성향 분석",
        "STEP2: ETF 추천",
        "STEP3: 모드 선택",
        "STEP4: 전략 추천",
        "STEP5: 시뮬레이션 실행",
        "STEP6: 결과 시각화",
    ]
    current_step = st.session_state["app_step"]
    st.write(f"### {step_labels[current_step - 1]} ({current_step}/6)")
    st.progress((current_step - 1) / 5)

    def move_step(new_step):
        st.session_state["app_step"] = new_step
        st.rerun()

    def step_button_row(can_next=True, can_prev=True):
        cols = st.columns([1, 1, 1])
        if can_prev and cols[0].button("이전", key=f"prev_{current_step}"):
            move_step(max(1, current_step - 1))
        cols[1].write("")
        if can_next and cols[2].button("다음", key=f"next_{current_step}"):
            move_step(min(6, current_step + 1))

    profile_weights = st.session_state.get("profile_weights")
    selected_etfs = st.session_state.get("selected_etfs", [])
    overlap_info = st.session_state.get("overlap_info", [])
    simulation_results = st.session_state.get("simulation_results")

    if current_step == 1:
        st.write("간단한 질문에 답해 투자 성향을 분석합니다.")
        
        current_q = st.session_state.get("current_question", "Q1")
        responses = st.session_state.get("profile_responses", {})
        
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
            cols = st.columns(5)
            for i, (cat, pct) in enumerate(profile_weights.items()):
                with cols[i]:
                    st.metric(cat, f"{pct}%")
            
            if st.button("STEP2로 이동", key="proceed_to_step2"):
                st.session_state["app_step"] = 2
                st.rerun()
        else:
            # 질문 표시
            st.write(q_text)
            
            # 진행 상황 계산 (정확하게)
            # 현재까지 응답한 질문 개수
            completed = len(responses)
            
            # 전체 질문 개수 결정
            if "Q5" in AI_PROFILE_QUESTIONS and responses.get("Q4") == 2:
                # Q5가 포함될 경우
                total_questions = len(AI_PROFILE_BASE_QUESTIONS) + 1
            else:
                # Q5가 미포함될 경우
                total_questions = len(AI_PROFILE_BASE_QUESTIONS)
            
            st.write(f"진행 상황: {completed}/{total_questions}")
            
            # 선택지 표시 (라디오 버튼)
            selected_idx = st.radio(
                "선택지",
                range(len(q_choices)),
                format_func=lambda i: q_choices[i],
                key=f"choice_{current_q}"
            )
            
            # 이전/다음 버튼
            cols = st.columns([1, 1, 1])
            
            # 이전 버튼
            with cols[0]:
                if st.button("이전", key="q_prev"):
                    # 현재 질문의 응답 삭제 (뒤로 갔을 때 진행상황 업데이트)
                    if current_q in st.session_state["profile_responses"]:
                        del st.session_state["profile_responses"][current_q]
                    
                    # 현재 질문 이전으로 돌아가기
                    all_questions = AI_PROFILE_BASE_QUESTIONS.copy()
                    # Q5가 응답에 있으면 포함
                    if "Q5" in st.session_state["profile_responses"]:
                        all_questions.insert(all_questions.index("Q6"), "Q5")
                    
                    try:
                        current_idx = all_questions.index(current_q)
                        if current_idx > 0:
                            prev_q = all_questions[current_idx - 1]
                            st.session_state["current_question"] = prev_q
                            st.rerun()
                    except ValueError:
                        pass
            
            # 다음 버튼
            with cols[2]:
                if st.button("다음", key="q_next"):
                    # 현재 선택지 저장
                    st.session_state["profile_responses"][current_q] = selected_idx
                    
                    # 다음 질문 결정
                    next_q = get_next_question(current_q, st.session_state["profile_responses"])
                    
                    if next_q is None:
                        # 모든 질문 완료 - 프로필 계산
                        profile_weights = calculate_profile_scores(st.session_state["profile_responses"])
                        st.session_state["profile_weights"] = profile_weights
                        st.session_state["profile_submitted"] = True
                        st.session_state["app_step"] = 2
                        st.rerun()
                    else:
                        st.session_state["current_question"] = next_q
                        st.rerun()

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
            st.session_state["overlap_info"] = detect_sector_overlap(selected_etfs)
            overlap_info = st.session_state["overlap_info"]
        else:
            etf_weights = st.session_state.get("etf_weights", {})
        
        # 투자 성향 분석 결과 표시
        st.write("### 📊 당신의 투자 성향")
        visible_categories = [cat for cat in profile_weights if cat != "_leverage_score"]
        cols = st.columns(len(visible_categories))
        for i, cat in enumerate(visible_categories):
            pct = profile_weights[cat]
            with cols[i]:
                st.metric(cat, f"{pct}%")
        
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
            
            with st.expander(f"**{ticker}** - {info.get('이름', '')} ({weight_pct}%)"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**기본 정보**")
                    st.write(f"• **카테고리**: {info.get('카테고리', '')}")
                    st.write(f"• **보수율**: {info.get('보수율', 0)*100:.2f}%")
                    st.write(f"• **배당**: {'있음 💰' if info.get('배당') else '없음'}")
                    st.write(f"• **레버리지**: {'3배 🚀' if info.get('레버리지') else '없음'}")
                
                with col2:
                    st.write("**설명**")
                    descriptions = {
                        "VOO": "S&P 500 (미국 500대 기업)\n안정적 성장형의 대표",
                        "QQQ": "나스닥 100 (미국 기술기업)\n기술주 중심 성장형",
                        "VTI": "미국 전체 주식시장\n최대 분산의 지수 추적",
                        "SCHD": "배당성장 ETF\n안정적 배당 수익",
                        "DGRO": "배당성장 ETF\n매년 배당금 증가",
                        "VYM": "고배당 수익 ETF\n높은 배당 수익률",
                        "JEPI": "월배당 프리미엄 인컴 ETF\n매월 배당금 수령",
                        "JEPQ": "나스닥 월배당 ETF\n기술주 + 월배당",
                        "QYLD": "나스닥 월배당 커버드콜\n높은 월 배당 수익",
                        "QLD": "나스닥 2배 레버리지\n2배 수익/손실",
                        "TQQQ": "나스닥 3배 레버리지\n3배 수익/손실 (위험 높음)",
                        "SSO": "S&P500 2배 레버리지\n2배 수익/손실",
                        "UPRO": "S&P500 3배 레버리지\n3배 수익/손실 (위험 높음)",
                        "SOXL": "반도체 3배 레버리지\n반도체 + 3배 수익/손실 (최고 위험)",
                        "VGT": "기술섹터 종합 ETF\n다양한 IT기업 투자",
                        "SOXX": "반도체 전문 ETF\nNVIDIA, TSMC 등",
                        "SMH": "반도체 전문 ETF\nIntel, Broadcom 등",
                        "DRAM": "반도체 메모리 전문\nDRAM, NAND 기업",
                        "XLV": "헬스케어 섹터 ETF\n의약품, 의료기기 기업",
                        "SGOV": "초단기 미국채 ETF\n극도로 안전한 자산"
                    }
                    st.write(descriptions.get(ticker, ""))
        
        st.markdown("---")
        
        if overlap_info:
            st.warning("⚠️ 섹터/카테고리가 중복되는 ETF가 있습니다. 대체 ETF를 확인해보세요.")
            overlap_df = pd.DataFrame([
                {
                    "중복카테고리": item["중복카테고리"],
                    "중복ETF": ", ".join(item["중복ETF"]),
                    "대체ETF": ", ".join(item["대체ETF"])
                }
                for item in overlap_info
            ])
            st.dataframe(overlap_df, use_container_width=True)
        
        step_button_row()

    elif current_step == 5:
        # 시뮬레이션 실행 화면
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

        st.write("### 시뮬레이션 실행")
        st.write("선택한 모드에 따라 입력창이 표시됩니다. 입력 후 '시뮬레이션 실행'을 클릭하세요.")

        etf_weights = st.session_state.get("etf_weights", {})
        port_weights = build_portfolio_weights(selected_etfs, etf_weights)

        sim_container = st.container()
        if st.session_state.get("analysis_mode") == "목표 금액 달성":
            with sim_container:
                t_amt_str = st.text_input("목표 금액 (원)", value=fmt_money(st.session_state.get("target_amount", 500000000)), key="sim_target_ok")
                monthly_str = st.text_input("월 적립금 (원)", value=fmt_money(st.session_state.get("monthly_contribution", 100000)), key="sim_monthly_man")
                years_sim = st.number_input("투자 기간(년)", min_value=1, max_value=50, value=st.session_state.get("years", 10), key="sim_years")

                # parse
                t_val = parse_number_input(t_amt_str)
                m_val = parse_number_input(monthly_str)
                target_won = int(t_val)
                monthly_won = int(m_val)

                if st.button("시뮬레이션 실행", key="run_sim_mode2"):
                    st.session_state["target_amount"] = target_won
                    st.session_state["monthly_contribution"] = monthly_won
                    st.session_state["years"] = int(years_sim)

                    # summary simulation
                    summary = simulate_portfolio(port_weights, years=st.session_state["years"], mode=st.session_state.get("sim_mode","적립형"), initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=st.session_state["monthly_contribution"], exchange_rate=1300.0)

                    # path curves
                    path_curves = simulate_portfolio_paths(port_weights, years=st.session_state["years"], mode=st.session_state.get("sim_mode","적립형"), initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=st.session_state["monthly_contribution"], exchange_rate=1300.0, trials=300)

                    mode_paths = {
                        "거치형": simulate_portfolio_paths(port_weights, years=st.session_state["years"], mode="거치형", initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=st.session_state["monthly_contribution"], exchange_rate=1300.0, trials=200),
                        "적립형": simulate_portfolio_paths(port_weights, years=st.session_state["years"], mode="적립형", initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=st.session_state["monthly_contribution"], exchange_rate=1300.0, trials=200),
                        "혼합형": simulate_portfolio_paths(port_weights, years=st.session_state["years"], mode="혼합형", initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=st.session_state["monthly_contribution"], exchange_rate=1300.0, trials=200),
                    }

                    rebalance_summary = compare_rebalancing_strategies(port_weights, years=st.session_state["years"], initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=st.session_state["monthly_contribution"], frequency="연간", exchange_rate=1300.0, trials=200)

                    st.session_state["simulation_results"] = {
                        "summary": summary,
                        "path_curves": path_curves,
                        "mode_paths": mode_paths,
                        "rebalance_summary": rebalance_summary,
                    }
                    st.success("시뮬레이션이 완료되었습니다. STEP6에서 결과를 확인할 수 있습니다.")

        elif st.session_state.get("analysis_mode") == "목표 기간 확인":
            with sim_container:
                t_amt_str = st.text_input("목표 금액 (억원)", value=fmt_money(float(st.session_state.get("target_amount", 500000000)) / 1e8), key="sim_target_ok_mode3")
                target_years = st.number_input("목표 기간 (년)", min_value=1, max_value=50, value=st.session_state.get("target_years",10), key="sim_target_years_mode3")

                t_val = parse_number_input(t_amt_str)
                target_won = int(t_val * 1e8)

                if st.button("시뮬레이션 실행", key="run_sim_mode3"):
                    st.session_state["target_amount"] = target_won
                    st.session_state["target_years"] = int(target_years)

                    required_monthly = estimate_required_contribution(st.session_state["target_amount"], st.session_state["target_years"], st.session_state.get("current_capital",0.0))
                    st.session_state["monthly_contribution"] = int(required_monthly)

                    st.info(f"역산된 필요 월적립금: {fmt_money(int(required_monthly))}원 ({fmt_money(int(np.ceil(required_monthly/1e4)))}만원/월)")

                    # simulate growth using required_monthly
                    path_curves = simulate_portfolio_paths(port_weights, years=st.session_state["target_years"], mode=st.session_state.get("sim_mode","적립형"), initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=int(required_monthly), exchange_rate=1300.0, trials=300)

                    summary = simulate_portfolio(port_weights, years=st.session_state["target_years"], mode=st.session_state.get("sim_mode","적립형"), initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=int(required_monthly), exchange_rate=1300.0)

                    mode_paths = {
                        "거치형": simulate_portfolio_paths(port_weights, years=st.session_state["target_years"], mode="거치형", initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=int(required_monthly), exchange_rate=1300.0, trials=200),
                        "적립형": simulate_portfolio_paths(port_weights, years=st.session_state["target_years"], mode="적립형", initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=int(required_monthly), exchange_rate=1300.0, trials=200),
                        "혼합형": simulate_portfolio_paths(port_weights, years=st.session_state["target_years"], mode="혼합형", initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=int(required_monthly), exchange_rate=1300.0, trials=200),
                    }

                    rebalance_summary = compare_rebalancing_strategies(port_weights, years=st.session_state["target_years"], initial_capital=st.session_state.get("current_capital",0.0), monthly_contribution=int(required_monthly), frequency="연간", exchange_rate=1300.0, trials=200)

                    st.session_state["simulation_results"] = {
                        "summary": summary,
                        "path_curves": path_curves,
                        "mode_paths": mode_paths,
                        "rebalance_summary": rebalance_summary,
                    }
                    st.success("시뮬레이션이 완료되었습니다. STEP6에서 결과를 확인할 수 있습니다.")

        step_button_row()
    elif current_step == 4:
        # 전략 추천 화면
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

        st.write("### 매수 전략 추천")
        st.write("선택된 ETF 및 투자 성향 기반으로 DCA/일시투자(LOC)/리밸런싱 주기와 추천 이유를 제시합니다.")

        # 포트폴리오 가중치 (점수에 따른 비중 적용)
        etf_weights = st.session_state.get("etf_weights", {})
        try:
            portfolio_weights = build_portfolio_weights(selected_etfs, etf_weights)
        except Exception:
            portfolio_weights = {t: 1.0 / max(1, len(selected_etfs)) for t in selected_etfs}

        # 변동성 지표: 선택 ETF의 평균 sigma
        sigmas = []
        leverage_present = False
        for t in selected_etfs:
            mu, sigma = _get_ticker_return_profile(t)
            sigmas.append(sigma)
            if ETF_DATA.get(t, {}).get("레버리지", False):
                leverage_present = True
        avg_sigma = float(np.mean(sigmas)) if sigmas else 0.0

        # 매수 방식 추천: 변동성 기준으로 DCA vs LOC
        if leverage_present or avg_sigma >= 0.15:
            buy_rec = "DCA"
            buy_reason = "변동성(또는 레버리지 포함)이 높아 분할매수(DCA)를 추천합니다."
        elif avg_sigma < 0.10:
            buy_rec = "LOC"
            buy_reason = "변동성이 낮아 일시투자(LOC)가 효율적일 수 있습니다."
        else:
            buy_rec = "혼합(일부 LOC + DCA)"
            buy_reason = "중간 수준 변동성으로 일부는 일시투자, 일부는 DCA를 권장합니다."

        # 리밸런싱 주기 추천
        dominant = max(profile_weights, key=lambda k: profile_weights.get(k, 0))
        if dominant in ["기술", "성장형"]:
            rebalance_rec = "분기"
            rebalance_reason = "성장/기술 중심이면 더 자주 리밸런싱하여 비중 관리를 권장합니다."
        elif dominant in ["배당형", "금융형"]:
            rebalance_rec = "반기"
            rebalance_reason = "배당/금융형은 반기 리밸런싱으로 배당 수익과 안정성 균형을 맞추기 좋습니다."
        else:
            rebalance_rec = "연간"
            rebalance_reason = "안정형 비중이 높으면 연간 리밸런싱으로 트랜잭션 비용을 절감하세요."

        # 추가 권장 이유 텍스트
        strategy_reasons = build_strategy_reason(profile_weights, selected_etfs, st.session_state.get("analysis_mode"))
        extra_notes = []
        if leverage_present:
            extra_notes.append("포트폴리오에 레버리지 ETF가 포함되어 있습니다. 레버리지 상품은 변동성이 크므로 분할매수(예: 3~6회)와 포지션 제한을 권장합니다.")

        # 출력
        cols = st.columns([1, 1])
        with cols[0]:
            st.subheader("추천 매수 방식")
            st.write(f"**권장 방식:** {buy_rec}")
            st.write(f"**이유:** {buy_reason}")
            if extra_notes:
                for note in extra_notes:
                    st.info(note)

        with cols[1]:
            st.subheader("리밸런싱")
            st.write(f"**권장 주기:** {rebalance_rec}")
            st.write(f"**이유:** {rebalance_reason}")

        st.subheader("추천 이유 상세")
        st.write(strategy_reasons)

        st.subheader("선택 ETF 요약")
        df_sel = pd.DataFrame([{"티커": t, "비중": f"{portfolio_weights.get(t,0)*100:.2f}%", "카테고리": ETF_DATA[t]["카테고리"], "레버리지": ETF_DATA[t]["레버리지"]} for t in selected_etfs])
        st.dataframe(df_sel, use_container_width=True)

        step_button_row()

    elif current_step == 3:
        if not profile_weights:
            st.warning("STEP1을 먼저 완료해 주세요.")
            if st.button("STEP1로 이동", key="goto1b"):
                move_step(1)
            return
        st.write("투자 방식과 목표값을 선택하세요.")
        selected_analysis_mode = st.selectbox(
            "STEP3: 모드 선택",
            ANALYSIS_MODES,
            index=ANALYSIS_MODES.index(st.session_state["analysis_mode"]),
            key="analysis_mode_select"
        )
        st.session_state["analysis_mode"] = selected_analysis_mode

        # 유틸: 금액 콤마 포맷 및 파서 (천단위 콤마, 소수 .00 제거)
        def fmt_money(n):
            try:
                f = float(n)
            except Exception:
                return "0"
            if f.is_integer():
                return f"{int(f):,}"
            # 소수점이 유의미할 경우 소수부의 불필요한 0 제거
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

        # MODE: 목표 금액 달성 (입력 단위: 목표 금액=억원, 월 적립금=만원, 투자 기간=년)
        mode_container = st.container()
        if selected_analysis_mode == "목표 금액 달성":
            with mode_container:
                if "target_amount_man" not in st.session_state:
                    st.session_state["target_amount_man"] = fmt_money(int(st.session_state.get("target_amount", 500000000) / 10000))
                if "monthly_contribution_man" not in st.session_state:
                    st.session_state["monthly_contribution_man"] = fmt_money(int(st.session_state.get("monthly_contribution", 100000) / 10000))

                st.markdown("#### 목표 금액")
                target_cols = st.columns([1, 1])
                with target_cols[0]:
                    target_man_str = st.text_input(
                        "목표 금액 (만원)",
                        value=st.session_state["target_amount_man"],
                        key="target_amount_man",
                        on_change=format_money_input,
                        args=("target_amount_man",),
                    )
                with target_cols[1]:
                    target_won_value = format_won_from_manwon_key("target_amount_man")
                    st.markdown(f"### {fmt_money(target_won_value)}원")

                target_button_cols = st.columns([1, 1, 1, 1, 1, 1])
                for idx, inc in enumerate([50, 100, 500, 1000, 5000]):
                    target_button_cols[idx].button(
                        f"+{fmt_money(inc)}만원",
                        key=f"target_add_{inc}",
                        on_click=change_man_amount,
                        args=("target_amount_man", inc),
                    )
                target_button_cols[-1].button("다시입력", key="target_reset", on_click=reset_man_amount, args=("target_amount_man",))

                st.write("---")
                st.markdown("#### 월 적립금")
                monthly_cols = st.columns([1, 1])
                with monthly_cols[0]:
                    monthly_man_str = st.text_input(
                        "월 적립금 (만원)",
                        value=st.session_state["monthly_contribution_man"],
                        key="monthly_contribution_man",
                        on_change=format_money_input,
                        args=("monthly_contribution_man",),
                    )
                with monthly_cols[1]:
                    monthly_won_value = format_won_from_manwon_key("monthly_contribution_man")
                    st.markdown(f"### {fmt_money(monthly_won_value)}원")

                monthly_button_cols = st.columns([1, 1, 1, 1, 1, 1])
                for idx, inc in enumerate([50, 100, 500, 1000, 5000]):
                    monthly_button_cols[idx].button(
                        f"+{fmt_money(inc)}만원",
                        key=f"monthly_add_{inc}",
                        on_click=change_man_amount,
                        args=("monthly_contribution_man", inc),
                    )
                monthly_button_cols[-1].button("다시입력", key="monthly_reset", on_click=reset_man_amount, args=("monthly_contribution_man",))

                target_won = parse_int(st.session_state.get("target_amount_man", 0)) * 10000
                monthly_won = parse_int(st.session_state.get("monthly_contribution_man", 0)) * 10000
                st.session_state["target_amount"] = target_won
                st.session_state["monthly_contribution"] = monthly_won
                years_mode2 = st.number_input("투자 기간(년)", min_value=1, max_value=50, value=st.session_state.get("years", 10), key="years_mode2_input")
                st.session_state["years"] = int(years_mode2)

                # 계산: 예상 성장 경로 및 달성 여부 (6개월 간격)
                growth = generate_growth_path(
                    st.session_state["current_capital"], st.session_state["monthly_contribution"], st.session_state["years"], interval_months=6
                )
                final_value = int(growth[-1])
                target_val = int(st.session_state["target_amount"])
                if final_value >= target_val:
                    st.success(f"목표 달성 가능: 예상 최종자산 {fmt_money(final_value)}원 >= 목표 {fmt_money(target_val)}원")
                else:
                    st.warning(f"목표 미달성 가능성: 예상 최종자산 {fmt_money(final_value)}원 < 목표 {fmt_money(target_val)}원")

                # 그래프 (현재 날짜 기준 6개월 간격 x축)
                from datetime import datetime

                total_points = len(growth)
                start_date = datetime.now()
                date_labels = []
                for i in range(total_points):
                    months = i * 6
                    year = start_date.year + (start_date.month - 1 + months) // 12
                    month = (start_date.month - 1 + months) % 12 + 1
                    date_labels.append(f"{year}-{month:02d}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=date_labels, y=[int(v) for v in growth], mode="lines+markers", name="예상자산"))
                fig.update_layout(title=f"예상 자산 성장 곡선 (평균 수익률 6%)", xaxis_title="날짜", yaxis_title="자산(원)")
                fig.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)

        # MODE: 목표 기간 확인 — 목표 금액 입력 UI를 '목표 금액 달성'과 동일하게 만원/원 표시로 교체
        elif selected_analysis_mode == "목표 기간 확인":
            with mode_container:
                if "target_amount_man" not in st.session_state:
                    st.session_state["target_amount_man"] = fmt_money(int(st.session_state.get("target_amount", 500000000) / 10000))

                st.markdown("#### 목표 금액")
                tcols = st.columns([1, 1])
                with tcols[0]:
                    # 만원 단위 입력 (콤마 포함 문자열)
                    ta = st.text_input(
                        "목표 금액 (만원)",
                        value=st.session_state["target_amount_man"],
                        key="target_amount_man",
                        on_change=format_money_input,
                        args=("target_amount_man",),
                    )
                with tcols[1]:
                    twon = format_won_from_manwon_key("target_amount_man")
                    st.markdown(f"### {fmt_money(twon)}원")

                btns = st.columns([1, 1, 1, 1, 1, 1])
                for idx, inc in enumerate([50, 100, 500, 1000, 5000]):
                    btns[idx].button(f"+{fmt_money(inc)}만원", key=f"mode3_target_add_{inc}", on_click=change_man_amount, args=("target_amount_man", inc))
                btns[-1].button("다시입력", key="mode3_target_reset", on_click=reset_man_amount, args=("target_amount_man",))

                # 목표 기간 입력
                target_years = st.number_input("목표 기간 (년)", min_value=1, max_value=50, value=st.session_state.get("target_years", 10), key="target_years_mode3_input")

                # 내부값 업데이트
                target_won2 = parse_int(st.session_state.get("target_amount_man", "0")) * 10000
                st.session_state["target_amount"] = target_won2
                st.session_state["target_years"] = int(target_years)

                required_monthly = estimate_required_contribution(
                    st.session_state["target_amount"], st.session_state["target_years"], st.session_state["current_capital"]
                )

                st.write(f"필요 월적립금: {fmt_money(int(required_monthly))}원 ({fmt_money(int(np.ceil(required_monthly/1e4)))}만원/월)")

                # 그래프: 해당 월적립금으로 성장 경로 (6개월 간격)
                growth_req = generate_growth_path(
                    st.session_state["current_capital"], int(required_monthly), st.session_state["target_years"], interval_months=6
                )
                from datetime import datetime
                total_points2 = len(growth_req)
                start_date2 = datetime.now()
                date_labels2 = []
                for i in range(total_points2):
                    months = i * 6
                    year = start_date2.year + (start_date2.month - 1 + months) // 12
                    month = (start_date2.month - 1 + months) % 12 + 1
                    date_labels2.append(f"{year}-{month:02d}")

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=date_labels2, y=[int(v) for v in growth_req], mode="lines+markers", name="필요적립금 성장"))
                fig2.update_layout(title=f"필요 적립금으로 예상 자산 성장 (평균 수익률 6%)", xaxis_title="날짜", yaxis_title="자산(원)")
                fig2.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig2, use_container_width=True)

        step_button_row()

    elif current_step == 6:
        if not simulation_results:
            st.warning("STEP5에서 시뮬레이션을 먼저 실행해 주세요.")
            if st.button("STEP5로 이동", key="goto5"):
                move_step(5)
            return
        st.write("### 결과 시각화")
        st.plotly_chart(plot_growth_curves(simulation_results["path_curves"], years=st.session_state["years"]), use_container_width=True)
        st.plotly_chart(plot_mode_comparison(simulation_results["mode_paths"], years=st.session_state["years"]), use_container_width=True)
        st.plotly_chart(plot_rebalance_comparison(simulation_results["rebalance_summary"]), use_container_width=True)
        cols = st.columns([1, 1, 1])
        if cols[0].button("이전", key="prev_6"):
            move_step(5)
        cols[1].write("")
        if cols[2].button("처음으로", key="restart"):
            st.session_state.clear()
            st.rerun()


if __name__ == "__main__":
    run_streamlit_app()

