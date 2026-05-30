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

AI_PROFILE_QUESTIONS = [
    "1. ETF 투자 목표는 무엇인가요?",
    "2. 투자 기간은 얼마나 되나요?",
    "3. 손실이 발생하면 어떻게 하시겠습니까?",
    "4. 배당 수익은 얼마나 중요한가요?",
    "5. 특정 섹터 집중 투자를 원하시나요?",
    "6. 레버리지 ETF를 포함할 때 위험 허용 수준은 어느 정도인가요?",
    "7. 자산은 성장 중심, 균형형, 안정 중심 중 어디에 가깝습니까?",
]

AI_PROFILE_CHOICES = [
    ["단기 성장 중심", "배당 중심", "금융형 중심", "기술 집중"],
    ["단기", "중기", "장기", "초장기"],
    ["손실에도 추가 매수", "손실 시 보유", "손실 시 부분 매도", "손실 시 전량 매도"],
    ["매우 중요", "중요", "보통", "덜 중요"],
    ["기술/IT", "테크", "반도체", "섹터 분산"],
    ["공격적으로 위험 감수", "보통 수준의 위험 감수", "중립적", "안정 지향"],
    ["성장 중심", "균형형", "안정 중심", "매우 안정 중심"],
]


def get_adaptive_question(responses):
    """응답에 따라 다음 질문을 결정합니다."""
    index = len(responses)
    if index >= len(AI_PROFILE_QUESTIONS):
        return None

    next_question = AI_PROFILE_QUESTIONS[index]
    if index == 4 and any(keyword in responses[-1].lower() for keyword in ["기술", "테크", "반도체", "금융", "에너지", "소비"]):
        return "5. 선택하신 섹터/자산 비중 중 기대 수익률과 위험 허용 범위는 어느 정도인가요?"
    return next_question


def build_claude_prompt(responses):
    """Claude에게 보낼 프롬프트를 만듭니다."""
    response_lines = "\n".join(f"Q{i+1}: {answer}" for i, answer in enumerate(responses))
    return (
        "다음 고객 응답을 바탕으로 5개 자산 비율을 계산해 주세요. "
        "각 카테고리별 비율 합이 100%가 되도록 작성해 주세요. "
        "응답 성향에 따라 성장형, 배당형, 금융형, 기술, 안정형을 고려해 주세요.\n\n"
        f"{response_lines}\n\n"
        "응답은 반드시 JSON 형식으로 아래와 같이 출력해 주세요:\n"
        "{\n"
        "  \"성장형\": 0,\n"
        "  \"배당형\": 0,\n"
        "  \"금융형\": 0,\n"
        "  \"기술\": 0,\n"
        "  \"안정형\": 0\n"
        "}\n"
        "추가 설명은 생략해 주세요."
    )


def parse_claude_profile_response(raw_text):
    """Claude 응답에서 JSON을 추출합니다."""
    json_text = None
    match = re.search(r"\{[\s\S]*?\}", raw_text)
    if match:
        json_text = match.group(0)

    if not json_text:
        raise ValueError("Claude 응답에서 JSON을 찾을 수 없습니다.")

    try:
        profile = {
            "성장형": 0.0,
            "배당형": 0.0,
            "금융형": 0.0,
            "기술": 0.0,
            "안정형": 0.0,
        }
        cleaned = json_text.replace("\n", " ").replace("\r", " ")
        for category in profile:
            pattern = rf'"{category}"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
            found = re.search(pattern, cleaned)
            if found:
                profile[category] = float(found.group(1))
        total = sum(profile.values())
        if total == 0:
            raise ValueError("Claude가 0 비율 응답을 반환했습니다.")
        return {k: round((v / total) * 100, 2) for k, v in profile.items()}
    except Exception as exc:
        raise ValueError(f"Claude 응답 파싱 오류: {exc}") from exc


def analyze_investor_profile_claude(responses, api_key=None):
    """Claude API를 호출해 투자 성향 비율을 계산합니다."""
    if len(responses) != len(AI_PROFILE_QUESTIONS):
        raise ValueError(f"응답은 {len(AI_PROFILE_QUESTIONS)}개여야 합니다. 현재: {len(responses)}")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 환경 변수가 필요합니다.")

    client = anthropic.Client(api_key=api_key)
    prompt = build_claude_prompt(responses)
    completion = client.completions.create(
        model="claude-3.5-mini",
        prompt=anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT,
        max_tokens_to_sample=300,
        temperature=0.0,
    )

    raw_output = completion.get("completion", "")
    return parse_claude_profile_response(raw_output)

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
    "성장형": ["VOO", "QQQ", "VTI", "VGT", "SSO", "UPRO"],
    "배당형": ["SCHD", "DGRO", "VYM"],
    "금융형": ["JEPI", "JEPQ", "QYLD"],
    "기술": ["VGT", "SOXX", "SMH", "DRAM", "XLV"],
    "안정형": ["SGOV"],
}


def _normalize_profile_weights(profile_weights):
    total = sum(profile_weights.get(cat, 0) for cat in AI_PROFILE_CATEGORIES)
    if total <= 0:
        raise ValueError("프로필 가중치 합계는 0보다 커야 합니다.")
    return {cat: profile_weights.get(cat, 0) / total for cat in AI_PROFILE_CATEGORIES}


def recommend_etfs(profile_weights, top_n=5):
    """투자 성향 비율을 받아 ETF를 추천합니다."""
    normalized = _normalize_profile_weights(profile_weights)
    selected = []

    for category in sorted(AI_PROFILE_CATEGORIES, key=lambda c: normalized[c], reverse=True):
        if normalized[category] <= 0:
            continue
        for ticker in CATEGORY_TO_ETFS.get(category, []):
            if ticker not in selected:
                selected.append(ticker)
                break
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        remaining = [ticker for ticker in ETF_DATA.keys() if ticker not in selected]
        scored = sorted(
            remaining,
            key=lambda t: normalized.get(ETF_CATEGORY_MAP.get(ETF_DATA[t]["카테고리"], "성장형"), 0),
            reverse=True,
        )
        for ticker in scored:
            if len(selected) >= top_n:
                break
            selected.append(ticker)

    return selected[:top_n]


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
    score = {cat: 0 for cat in AI_PROFILE_CATEGORIES}
    for answer in responses:
        text = answer.lower()
        if any(token in text for token in ["성장", "테크", "기술", "qqq", "vti", "upro", "tqqq"]):
            score["성장형"] += 2
        if any(token in text for token in ["배당", "dividend", "schd", "dgro", "vym"]):
            score["배당형"] += 2
        if any(token in text for token in ["금융", "jepi", "jepq", "qyld"]):
            score["금융형"] += 2
        if any(token in text for token in ["기술", "semiconductor", "테크", "반도체"]):
            score["기술"] += 2
        if any(token in text for token in ["안정", "채권", "low risk", "stable", "sgov"]):
            score["안정형"] += 2
    total = sum(score.values())
    if total == 0:
        score = {cat: 1 for cat in AI_PROFILE_CATEGORIES}
        total = len(score)
    return {cat: round(score[cat] / total * 100, 2) for cat in AI_PROFILE_CATEGORIES}


def build_portfolio_weights(etfs):
    if not etfs:
        raise ValueError("추천 ETF 목록이 비어 있습니다.")
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
        st.session_state["profile_step"] = 0
        st.session_state["profile_answers"] = [None] * len(AI_PROFILE_QUESTIONS)
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
        profile_step = st.session_state["profile_step"]
        st.write(f"**질문 {profile_step + 1} / {len(AI_PROFILE_QUESTIONS)}**")
        st.write(AI_PROFILE_QUESTIONS[profile_step])
        selected_choice = st.radio(
            "보기 선택", AI_PROFILE_CHOICES[profile_step], key=f"choice_{profile_step}"
        )
        cols = st.columns([1, 1, 1])
        if cols[0].button("이전", disabled=profile_step == 0, key="q_prev"):
            st.session_state["profile_step"] = profile_step - 1
            st.rerun()
        if cols[2].button("다음", key="q_next"):
            st.session_state["profile_answers"][profile_step] = selected_choice
            if profile_step + 1 < len(AI_PROFILE_QUESTIONS):
                st.session_state["profile_step"] = profile_step + 1
                st.rerun()
            else:
                responses = st.session_state["profile_answers"]
                try:
                    profile_weights = analyze_investor_profile_claude(responses)
                except Exception as exc:
                    st.warning(f"Claude API 호출 실패: {exc}. 로컬 추정 알고리즘으로 대체합니다.")
                    profile_weights = estimate_profile_locally(responses)
                st.session_state["profile_weights"] = profile_weights
                st.session_state["profile_submitted"] = True
                st.session_state["app_step"] = 2
                st.rerun()

    elif current_step == 2:
        if not profile_weights:
            st.warning("STEP1을 먼저 완료해 주세요.")
            if st.button("STEP1로 이동", key="goto1"):
                move_step(1)
            return

        if not selected_etfs:
            selected_etfs = recommend_etfs(profile_weights, top_n=5)
            st.session_state["selected_etfs"] = selected_etfs
            st.session_state["overlap_info"] = detect_sector_overlap(selected_etfs)
            overlap_info = st.session_state["overlap_info"]

        st.write("추천 ETF 목록")
        card_rows = [selected_etfs[i:i + 3] for i in range(0, len(selected_etfs), 3)]
        for row in card_rows:
            cols = st.columns(len(row))
            for col, ticker in zip(cols, row):
                info = ETF_DATA[ticker]
                fee_text = f"{info['보수율'] * 100:.2f}%"
                dividend_text = "배당 있음" if info["배당"] else "배당 없음"
                leverage_text = "레버리지 ETF" if info["레버리지"] else "일반 ETF"
                category_label = info["카테고리"]
                summary_text = {
                    "지수추적": "시장 전체를 추적하는 안정적 성장형 ETF입니다.",
                    "배당형": "배당 수익과 안정성을 동시에 기대할 수 있는 ETF입니다.",
                    "금융형": "금융 섹터 배당과 변동성 완화를 목표로 합니다.",
                    "기술": "기술 섹터 집중 투자를 통해 성장 기회를 노립니다.",
                    "안정형": "저변동성 안정 자산으로 포트폴리오를 보완합니다.",
                    "레버리지": "단기 성장 기회를 노리고 변동성에 대비하는 전략에 적합합니다.",
                }.get(category_label, "균형 있는 포트폴리오 구성을 위해 추천된 ETF입니다.")
                with col:
                    st.markdown(
                        f"**{ticker} · {info['이름']}**  \
                        카테고리: {category_label}  \
                        보수율: {fee_text}  \
                        {dividend_text} · {leverage_text}  \
                        \n"
                        f"{summary_text}"
                    )
                    st.markdown("---")
        if overlap_info:
            st.warning("섹터/카테고리가 중복되는 ETF가 있습니다. 대체 ETF를 확인해보세요.")
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

        port_weights = build_portfolio_weights(selected_etfs)

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

        # 포트폴리오 가중치(단순 균등)
        try:
            portfolio_weights = build_portfolio_weights(selected_etfs)
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

