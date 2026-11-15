import hashlib
import ipaddress
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from config import config
from engine import (
    calculate_credit_spread_candidates,
    calculate_depth_imbalance,
    calculate_iv_percentile,
    calculate_iv_rank,
    calculate_max_pain,
    calculate_oi_change,
    calculate_pcr,
    calculate_rolling_correlation,
)

logger = logging.getLogger(__name__)
MIN_IV_HISTORY_SESSIONS = 20
OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "drivers": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "drivers", "caveats"],
    "additionalProperties": False,
}


class SuggestedStrike(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_key: str = Field(min_length=1, max_length=120)
    side: Literal["BUY", "SELL"]
    option_type: Literal["CE", "PE"]
    strike: float = Field(gt=0, allow_inf_nan=False)


class StrategyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_type: Literal[
        "NO_TRADE",
        "Bull put credit spread",
        "Bear call credit spread",
    ]
    market_regime: Literal[
        "TRENDING",
        "RANGE_BOUND",
        "HIGH_VOLATILITY",
        "MIXED",
        "UNDETERMINED",
    ]
    suggested_strikes: list[SuggestedStrike] = Field(max_length=2)
    entry_credit_debit: float = Field(ge=0, allow_inf_nan=False)
    target_profit: float = Field(ge=0, allow_inf_nan=False)
    stop_loss: float = Field(ge=0, allow_inf_nan=False)
    confidence: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_trade_shape(self):
        if self.strategy_type == "NO_TRADE":
            if self.suggested_strikes or any((
                self.entry_credit_debit,
                self.target_profit,
                self.stop_loss,
            )):
                raise ValueError("NO_TRADE must not contain legs or trade values.")
        elif len(self.suggested_strikes) != 2 or any((
            self.entry_credit_debit <= 0,
            self.target_profit <= 0,
            self.stop_loss <= 0,
        )):
            raise ValueError("A spread proposal requires two legs and positive values.")
        return self


@dataclass(frozen=True)
class AgentResult:
    agent: str
    status: str
    summary: str
    data: dict = field(default_factory=dict)


def _age_seconds(timestamp, now):
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds())


def _aligned_prices(left_rows, right_rows):
    def keyed(rows):
        values = {}
        for row in rows:
            if row.timestamp is None or row.ltp is None:
                continue
            timestamp = row.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            minute = timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
            values[minute] = row.ltp
        return values

    left = keyed(left_rows)
    right = keyed(right_rows)
    common = sorted(left.keys() & right.keys())
    return [left[key] for key in common], [right[key] for key in common]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validate_loopback_url(base_url):
    parsed = urlsplit(base_url)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        is_loopback = address.is_loopback
    except ValueError:
        is_loopback = parsed.hostname == "localhost"
    if (
        parsed.scheme != "http"
        or not is_loopback
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama must use a loopback HTTP URL.")


def _request_ollama_content(agent_name, base_url, model, timeout, messages, schema):
    try:
        _validate_loopback_url(base_url)
    except ValueError as error:
        return None, AgentResult(agent_name, "ERROR", str(error))
    if not model:
        return None, AgentResult(agent_name, "ERROR", "OLLAMA_MODEL is not configured.")

    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": schema,
                    "options": {"temperature": 0.1},
                },
                timeout=timeout,
            )
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning("%s request failed: %s", agent_name, error)
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        message = (
            f"Ollama returned HTTP {status_code}; verify the local service and model."
            if status_code is not None
            else "Could not reach the local Ollama service."
        )
        return None, AgentResult(agent_name, "UNAVAILABLE", message)

    try:
        content = response.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as error:
        logger.warning("%s returned an invalid response envelope: %s", agent_name, error)
        return None, AgentResult(
            agent_name,
            "ERROR",
            "Ollama returned an invalid structured response.",
        )
    if not isinstance(content, str):
        return None, AgentResult(
            agent_name,
            "ERROR",
            "Ollama returned an invalid structured response.",
        )
    return content, None


class MarketDataAgent:
    name = "Market data validation"

    def run(self, snapshot, now=None):
        observed_at = now or datetime.now(timezone.utc)
        tick = snapshot.get("tick")
        chain = snapshot.get("chain")
        spot_age = _age_seconds(tick.timestamp if tick else None, observed_at)
        chain_age = _age_seconds(chain.timestamp if chain else None, observed_at)
        spot_fresh = (
            spot_age is not None
            and spot_age <= config.FEED_STALE_AFTER_SECONDS
            and tick.ltp is not None
            and tick.ltp > 0
        )
        chain_fresh = (
            chain_age is not None
            and chain_age <= config.OPTION_CHAIN_REFRESH_SECONDS * 2
            and chain is not None
        )
        history_count = sum(
            row.atm_iv is not None for row in snapshot.get("history", [])
        )
        ready = spot_fresh and chain_fresh
        if ready:
            summary = "Underlying and option-chain snapshots are fresh."
            status = "READY"
        else:
            missing = []
            if not spot_fresh:
                missing.append("fresh underlying quote")
            if not chain_fresh:
                missing.append("fresh option-chain snapshot")
            summary = f"Waiting for {', '.join(missing)}."
            status = "BLOCKED"
        return AgentResult(
            self.name,
            status,
            summary,
            {
                "ready": ready,
                "spot_fresh": spot_fresh,
                "chain_fresh": chain_fresh,
                "spot_age_seconds": spot_age,
                "chain_age_seconds": chain_age,
                "iv_history_count": history_count,
            },
        )


class FactorAnalysisAgent:
    name = "Factor analysis"

    def run(self, snapshot):
        chain = snapshot.get("chain")
        quotes = snapshot.get("quotes", [])
        quote_rows = [
            {
                "strike": quote.strike,
                "option_type": quote.option_type,
                "oi": quote.oi,
                "previous_oi": quote.previous_oi,
                "bid_depth": quote.bid_depth or [],
                "ask_depth": quote.ask_depth or [],
            }
            for quote in quotes
        ]
        history = snapshot.get("history", [])
        history_ivs = [row.atm_iv for row in history if row.atm_iv is not None]
        current_iv = chain.atm_iv if chain is not None else None
        enough_history = len(history_ivs) >= MIN_IV_HISTORY_SESSIONS
        iv_rank = calculate_iv_rank(current_iv, history_ivs) if enough_history else None
        iv_percentile = (
            calculate_iv_percentile(current_iv, history_ivs)
            if enough_history
            else None
        )
        calls = [
            row for row in quote_rows
            if row["option_type"] == "CE" and row["oi"] is not None
        ]
        puts = [
            row for row in quote_rows
            if row["option_type"] == "PE" and row["oi"] is not None
        ]
        call_wall = max(calls, key=lambda row: row["oi"]) if calls else None
        put_wall = max(puts, key=lambda row: row["oi"]) if puts else None
        spot = (
            snapshot["tick"].ltp
            if snapshot.get("tick") is not None
            else chain.underlying_spot if chain is not None else None
        )
        nearest = (
            min(quotes, key=lambda quote: abs(quote.strike - spot))
            if quotes and spot is not None
            else quotes[0] if quotes else None
        )
        oi_change, oi_change_pct = (
            calculate_oi_change(nearest.oi, nearest.previous_oi)
            if nearest is not None
            else (None, None)
        )
        depth_imbalance = (
            calculate_depth_imbalance(nearest.bid_depth, nearest.ask_depth)
            if nearest is not None
            else None
        )
        oi_build_up = []
        for row in quote_rows:
            change, change_pct = calculate_oi_change(row["oi"], row["previous_oi"])
            if (
                isinstance(change, (int, float))
                and not isinstance(change, bool)
                and math.isfinite(change)
            ):
                oi_build_up.append((
                    abs(float(change)),
                    {
                        "strike": row["strike"],
                        "option_type": row["option_type"],
                        "oi_change": change,
                        "oi_change_percent": (
                            change_pct
                            if isinstance(change_pct, (int, float))
                            and math.isfinite(change_pct)
                            else None
                        ),
                    },
                ))
        oi_build_up.sort(key=lambda item: item[0], reverse=True)

        correlation = None
        market_history = snapshot.get("market_history", {})
        symbol = snapshot.get("symbol")
        underlying_history = market_history.get(symbol, [])
        vix_history = market_history.get("INDIA_VIX", [])
        if underlying_history and vix_history:
            underlying_prices, vix_prices = _aligned_prices(
                list(reversed(underlying_history)),
                list(reversed(vix_history)),
            )
            correlation = calculate_rolling_correlation(underlying_prices, vix_prices)

        session_prices = [
            row
            for row in underlying_history
            if row.timestamp is not None
            and isinstance(row.ltp, (int, float))
            and math.isfinite(row.ltp)
            and row.ltp > 0
        ]
        session_prices.sort(key=lambda row: row.timestamp)
        if session_prices:
            latest_session = session_prices[-1].timestamp.date()
            session_prices = [
                row for row in session_prices
                if row.timestamp.date() == latest_session
            ][-60:]
        price_change_pct = None
        trend_efficiency = None
        if len(session_prices) >= 2:
            first_price = session_prices[0].ltp
            last_price = session_prices[-1].ltp
            price_path = sum(
                abs(right.ltp - left.ltp)
                for left, right in zip(session_prices, session_prices[1:])
            )
            price_change_pct = (last_price / first_price - 1) * 100
            if price_path > 0:
                trend_efficiency = abs(last_price - first_price) / price_path

        historical_volatility_status = (
            "Unavailable: stored market history does not provide daily close "
            "observations for a defensible HV estimate."
        )
        values = {
            "current_iv": current_iv,
            "iv_history_count": len(history_ivs),
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "pcr": calculate_pcr(quote_rows),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "max_pain": calculate_max_pain(quote_rows),
            "oi_change": oi_change,
            "oi_change_pct": oi_change_pct,
            "depth_imbalance": depth_imbalance,
            "correlation": correlation,
            "nearest_quote": nearest,
            "oi_build_up": [row for _, row in oi_build_up[:5]],
            "historical_volatility": None,
            "historical_volatility_status": historical_volatility_status,
            "market_regime_inputs": {
                "same_session_sample_count": len(session_prices),
                "same_session_price_change_percent": price_change_pct,
                "same_session_trend_efficiency": trend_efficiency,
                "india_vix": (
                    snapshot["vix"].ltp
                    if snapshot.get("vix") is not None
                    else None
                ),
                "iv_percentile": iv_percentile,
                "historical_volatility": None,
                "historical_volatility_status": historical_volatility_status,
            },
        }
        status = "READY" if chain is not None and quotes else "PARTIAL"
        summary = (
            "Computed factors from the selected option chain and stored history."
            if status == "READY"
            else "Some factors are unavailable until option-chain data arrives."
        )
        return AgentResult(self.name, status, summary, values)


class StrategyAgent:
    name = "Strategy candidate"

    def run(self, snapshot, factors, market_data_result, minimum_iv_percentile, now=None):
        if not market_data_result.data["ready"]:
            return AgentResult(
                self.name,
                "BLOCKED",
                "Fresh underlying and option-chain data are required.",
                {"candidates": []},
            )
        history_count = factors.data["iv_history_count"]
        iv_percentile = factors.data["iv_percentile"]
        if history_count < MIN_IV_HISTORY_SESSIONS or iv_percentile is None:
            return AgentResult(
                self.name,
                "BLOCKED",
                f"At least {MIN_IV_HISTORY_SESSIONS} daily IV observations are required.",
                {"candidates": []},
            )
        if iv_percentile < minimum_iv_percentile:
            return AgentResult(
                self.name,
                "BLOCKED",
                f"IV percentile {iv_percentile:.1f}% is below the "
                f"{minimum_iv_percentile}% threshold.",
                {"candidates": []},
            )

        tick = snapshot.get("tick")
        candidates = calculate_credit_spread_candidates(
            snapshot.get("quotes", []),
            snapshot.get("contracts", {}),
            tick.ltp if tick is not None else None,
            iv_percentile,
            minimum_iv_percentile,
            now=now,
            max_quote_age_seconds=config.FEED_STALE_AFTER_SECONDS,
        )
        if not candidates:
            return AgentResult(
                self.name,
                "NO_CANDIDATE",
                "No fresh two-sided short/hedge pair met the spread rules.",
                {"candidates": []},
            )
        return AgentResult(
            self.name,
            "READY",
            f"Found {len(candidates)} defined-risk spread candidate(s).",
            {"candidates": candidates},
        )


class RiskReviewAgent:
    name = "Risk review"

    def run(self, candidates):
        if not candidates:
            return AgentResult(
                self.name,
                "SKIPPED",
                "No strategy candidates were available to review.",
                {"approved": [], "rejected": []},
            )

        approved = []
        rejected = []
        for candidate in candidates:
            legs = candidate.get("legs", [])
            if not isinstance(legs, list) or len(legs) != 2 or not all(
                isinstance(leg, dict) for leg in legs
            ):
                rejected.append({
                    "strategy": candidate.get("strategy", "Unknown"),
                    "reason": "Spread legs or payoff values failed risk validation.",
                })
                continue

            short_legs = [leg for leg in legs if leg.get("side") == "SELL"]
            long_legs = [leg for leg in legs if leg.get("side") == "BUY"]
            numeric_values = (
                candidate.get("entry_credit_total"),
                candidate.get("entry_credit_points"),
                candidate.get("max_profit"),
                candidate.get("max_loss"),
                candidate.get("breakeven"),
                candidate.get("spread_width"),
                candidate.get("lot_size"),
                candidate.get("target_exit_debit"),
                candidate.get("target_profit"),
                candidate.get("stop_exit_debit"),
                candidate.get("stop_loss"),
                short_legs[0].get("strike") if len(short_legs) == 1 else None,
                long_legs[0].get("strike") if len(long_legs) == 1 else None,
            )
            values_are_finite = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in numeric_values
            )
            option_type = candidate.get("option_type")
            valid = (
                len(short_legs) == 1
                and len(long_legs) == 1
                and option_type in {"CE", "PE"}
                and {leg.get("side") for leg in legs} == {"BUY", "SELL"}
                and legs[0].get("instrument_key") != legs[1].get("instrument_key")
                and all(leg.get("instrument_key") for leg in legs)
                and bool(candidate.get("expiry"))
                and all(
                    leg.get("option_type") == option_type
                    and leg.get("expiry") == candidate.get("expiry")
                    for leg in legs
                )
                and values_are_finite
                and candidate.get("entry_credit_total", 0) > 0
                and candidate.get("entry_credit_points", 0) > 0
                and candidate.get("max_profit", 0) > 0
                and candidate.get("max_loss", 0) > 0
                and candidate.get("spread_width", 0) > candidate.get("entry_credit_points", 0)
                and candidate.get("lot_size", 0) > 0
                and math.isclose(
                    candidate.get("entry_credit_total", 0),
                    candidate.get("entry_credit_points", 0) * candidate.get("lot_size", 0),
                    rel_tol=1e-9,
                )
                and math.isclose(
                    candidate.get("max_profit", 0),
                    candidate.get("entry_credit_total", 0),
                    rel_tol=1e-9,
                )
                and math.isclose(
                    candidate.get("target_exit_debit", 0),
                    candidate.get("entry_credit_points", 0) * 0.5,
                    rel_tol=1e-9,
                )
                and math.isclose(
                    candidate.get("target_profit", 0),
                    (
                        candidate.get("entry_credit_points", 0)
                        - candidate.get("target_exit_debit", 0)
                    )
                    * candidate.get("lot_size", 0),
                    rel_tol=1e-9,
                )
                and math.isclose(
                    candidate.get("stop_exit_debit", 0),
                    min(
                        candidate.get("entry_credit_points", 0) * 1.5,
                        candidate.get("spread_width", 0),
                    ),
                    rel_tol=1e-9,
                )
                and math.isclose(
                    candidate.get("stop_loss", 0),
                    (
                        candidate.get("stop_exit_debit", 0)
                        - candidate.get("entry_credit_points", 0)
                    )
                    * candidate.get("lot_size", 0),
                    rel_tol=1e-9,
                )
                and math.isclose(
                    abs(short_legs[0]["strike"] - long_legs[0]["strike"]),
                    candidate.get("spread_width", 0),
                    rel_tol=1e-9,
                )
                and (
                    long_legs[0]["strike"] < short_legs[0]["strike"]
                    if option_type == "PE"
                    else long_legs[0]["strike"] > short_legs[0]["strike"]
                )
                and math.isclose(
                    candidate.get("breakeven", 0),
                    (
                        short_legs[0]["strike"] - candidate.get("entry_credit_points", 0)
                        if option_type == "PE"
                        else short_legs[0]["strike"] + candidate.get("entry_credit_points", 0)
                    ),
                    rel_tol=1e-9,
                )
                and math.isclose(
                    candidate.get("max_loss", 0),
                    (
                        candidate.get("spread_width", 0)
                        - candidate.get("entry_credit_points", 0)
                    )
                    * candidate.get("lot_size", 0),
                    rel_tol=1e-9,
                )
            )
            if valid:
                approved.append(candidate)
            else:
                rejected.append({
                    "strategy": candidate.get("strategy", "Unknown"),
                    "reason": "Spread legs or payoff values failed risk validation.",
                })

        status = "READY" if approved else "BLOCKED"
        summary = (
            f"Validated {len(approved)} candidate(s); maximum loss is defined by the hedge leg."
            if approved
            else "All candidates failed the payoff/leg validation checks."
        )
        return AgentResult(
            self.name,
            status,
            summary,
            {"approved": approved, "rejected": rejected},
        )

    def validate_proposal(self, proposal, candidates):
        reviewed = self.run(candidates)
        if reviewed.status != "READY":
            return None, reviewed.summary

        proposal_legs = proposal["suggested_strikes"]
        matching = next(
            (
                candidate
                for candidate in reviewed.data["approved"]
                if candidate["strategy"] == proposal["strategy_type"]
                and len(candidate["legs"]) == len(proposal_legs)
                and all(
                    proposed["instrument_key"] == actual["instrument_key"]
                    and proposed["side"] == actual["side"]
                    and proposed["option_type"] == actual["option_type"]
                    and math.isclose(
                        proposed["strike"],
                        actual["strike"],
                        rel_tol=0,
                        abs_tol=1e-9,
                    )
                    for proposed, actual in zip(proposal_legs, candidate["legs"])
                )
                and math.isclose(
                    proposal["entry_credit_debit"],
                    candidate["entry_credit_total"],
                    rel_tol=1e-3,
                    abs_tol=0.01,
                )
                and math.isclose(
                    proposal["target_profit"],
                    candidate["target_profit"],
                    rel_tol=1e-3,
                    abs_tol=0.01,
                )
                and math.isclose(
                    proposal["stop_loss"],
                    candidate["stop_loss"],
                    rel_tol=1e-3,
                    abs_tol=0.01,
                )
            ),
            None,
        )
        if matching is None:
            return None, (
                "The proposed legs or payoff values did not match a risk-approved "
                "candidate and its current quote-derived values."
            )
        return matching, None


class OllamaStrategyReasoningAgent:
    name = "Strategy reasoning"

    def __init__(self, base_url=None, model=None, timeout_seconds=None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout_seconds = timeout_seconds or config.OLLAMA_TIMEOUT_SECONDS

    def run(self, context, correction_feedback=None):
        user_input = {"snapshot": context}
        if correction_feedback:
            user_input["risk_rejection"] = correction_feedback

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a recommendation-only options strategy analyst. Use only the "
                    "supplied snapshot and candidate list. Assess the market regime from "
                    "the supplied features; use UNDETERMINED when evidence is insufficient. "
                    "Select exactly one listed candidate or return NO_TRADE. Never invent "
                    "or modify contracts, strikes, prices, targets, or stops. Copy the "
                    "candidate's legs and quote-derived values exactly. If retrying after "
                    "risk rejection, correct the mismatch or choose NO_TRADE. Do not "
                    "forecast, request credentials, use tools, or suggest executing an "
                    "order. Return only the required JSON object."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    _json_safe(user_input),
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        ]
        content, error_result = _request_ollama_content(
            self.name,
            self.base_url,
            self.model,
            self.timeout_seconds,
            messages,
            StrategyProposal.model_json_schema(),
        )
        if error_result is not None:
            return error_result
        if content is None:
            return AgentResult(self.name, "ERROR", "Ollama returned an empty response.")

        try:
            proposal = StrategyProposal.model_validate_json(content)
        except ValidationError as error:
            logger.warning("Ollama strategy proposal failed schema validation: %s", error)
            return AgentResult(
                self.name,
                "ERROR",
                "Ollama returned a proposal that did not match the strategy schema.",
            )
        return AgentResult(
            self.name,
            "READY",
            proposal.rationale,
            {"proposal": proposal.model_dump()},
        )


class BoundedAdvisoryLoop:
    name = "Agentic strategy advisor"

    def __init__(self, reasoning_agent=None, risk_agent=None):
        self.reasoning_agent = reasoning_agent or OllamaStrategyReasoningAgent()
        self.risk_agent = risk_agent or RiskReviewAgent()

    def run(self, context, candidates):
        if not candidates:
            return AgentResult(
                self.name,
                "BLOCKED",
                "No deterministic risk-approved candidates are available.",
                {"iterations": 0, "risk_status": "SKIPPED"},
            )

        feedback = None
        last_proposal = None
        for iteration in range(2):
            reasoning = self.reasoning_agent.run(context, correction_feedback=feedback)
            if reasoning.status != "READY":
                return AgentResult(
                    self.name,
                    reasoning.status,
                    reasoning.summary,
                    {
                        "iterations": iteration + 1,
                        "risk_status": "NOT_REACHED",
                    },
                )

            proposal = reasoning.data["proposal"]
            last_proposal = proposal
            if proposal["strategy_type"] == "NO_TRADE":
                return AgentResult(
                    self.name,
                    "NO_TRADE",
                    proposal["rationale"],
                    {
                        "proposal": proposal,
                        "selected_candidate": None,
                        "iterations": iteration + 1,
                        "risk_status": "NOT_REQUIRED",
                    },
                )

            candidate, rejection = self.risk_agent.validate_proposal(
                proposal,
                candidates,
            )
            if candidate is not None:
                return AgentResult(
                    self.name,
                    "READY",
                    proposal["rationale"],
                    {
                        "proposal": proposal,
                        "selected_candidate": candidate,
                        "iterations": iteration + 1,
                        "risk_status": "APPROVED",
                    },
                )

            feedback = rejection

        return AgentResult(
            self.name,
            "BLOCKED",
            "Risk review rejected the proposal after one constrained revision.",
            {
                "proposal": last_proposal,
                "selected_candidate": None,
                "iterations": 2,
                "risk_status": "REJECTED",
                "rejection": feedback,
            },
        )


class OllamaNarrativeAgent:
    name = "Ollama explanation"

    def __init__(self, base_url=None, model=None, timeout_seconds=None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout_seconds = timeout_seconds or config.OLLAMA_TIMEOUT_SECONDS

    def run(self, context):
        messages = [
            {
                "role": "system",
                "content": (
                    "Explain the supplied options-market snapshot neutrally. Use only values "
                    "in the JSON. Do not forecast prices, invent missing values, recommend "
                    "placing a trade, or describe this as financial advice. Explain that "
                    "all candidates are advisory and no orders are executed. Return only "
                    "the requested JSON object."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    _json_safe(context),
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        ]
        content, error_result = _request_ollama_content(
            self.name,
            self.base_url,
            self.model,
            self.timeout_seconds,
            messages,
            OLLAMA_RESPONSE_SCHEMA,
        )
        if error_result is not None:
            return error_result
        if content is None:
            return AgentResult(self.name, "ERROR", "Ollama returned an empty response.")

        try:
            result = json.loads(content)
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("Ollama returned an invalid structured response: %s", error)
            return AgentResult(
                self.name,
                "ERROR",
                "Ollama returned an invalid JSON explanation.",
            )

        summary = result.get("summary")
        drivers = result.get("drivers")
        caveats = result.get("caveats")
        if (
            not isinstance(summary, str)
            or not summary.strip()
            or len(summary) > 600
            or not isinstance(drivers, list)
            or not isinstance(caveats, list)
            or len(drivers) > 5
            or len(caveats) > 5
            or any(not isinstance(item, str) or len(item) > 240 for item in drivers + caveats)
        ):
            return AgentResult(
                self.name,
                "ERROR",
                "Ollama's explanation did not match the required response schema.",
            )
        return AgentResult(
            self.name,
            "READY",
            summary.strip(),
            {
                "summary": summary.strip(),
                "drivers": [item.strip() for item in drivers],
                "caveats": [item.strip() for item in caveats],
            },
        )


class DashboardAgentOrchestrator:
    def __init__(self, narrative_agent=None, strategy_reasoning_agent=None):
        self.market_data_agent = MarketDataAgent()
        self.factor_agent = FactorAnalysisAgent()
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskReviewAgent()
        self.narrative_agent = narrative_agent or OllamaNarrativeAgent()
        self.advisory_loop = BoundedAdvisoryLoop(
            reasoning_agent=strategy_reasoning_agent,
            risk_agent=self.risk_agent,
        )

    def analyze(self, snapshot, minimum_iv_percentile, now=None):
        market_data = self.market_data_agent.run(snapshot, now=now)
        factors = self.factor_agent.run(snapshot)
        strategy = self.strategy_agent.run(
            snapshot,
            factors,
            market_data,
            minimum_iv_percentile,
            now=now,
        )
        risk = self.risk_agent.run(strategy.data["candidates"])
        if risk.status != "READY":
            strategy = AgentResult(
                strategy.agent,
                "BLOCKED" if strategy.data["candidates"] else strategy.status,
                risk.summary if strategy.data["candidates"] else strategy.summary,
                {"candidates": []},
            )
        return {
            "market_data": market_data,
            "factors": factors,
            "strategy": strategy,
            "risk": risk,
            "narrative": AgentResult(
                self.narrative_agent.name,
                "NOT_RUN",
                "Run local Ollama only when you request an explanation.",
            ),
        }

    def run_advisory_loop(self, context, candidates):
        return self.advisory_loop.run(context, candidates)

    def narrative_context(self, snapshot, analysis):
        factors = analysis["factors"].data
        tick = snapshot.get("tick")
        risk = analysis["risk"]
        return _json_safe({
            "symbol": snapshot.get("symbol"),
            "expiry": snapshot.get("expiry"),
            "spot": tick.ltp if tick is not None else None,
            "atm_iv": factors.get("current_iv"),
            "iv_rank": factors.get("iv_rank"),
            "iv_percentile": factors.get("iv_percentile"),
            "pcr": factors.get("pcr"),
            "call_oi_wall": (
                factors["call_wall"]["strike"]
                if factors.get("call_wall")
                else None
            ),
            "put_oi_wall": (
                factors["put_wall"]["strike"]
                if factors.get("put_wall")
                else None
            ),
            "max_pain": factors.get("max_pain"),
            "india_vix": (
                snapshot["vix"].ltp
                if snapshot.get("vix") is not None
                else None
            ),
            "spot_vix_correlation": factors.get("correlation"),
            "top_oi_build_up": factors.get("oi_build_up", []),
            "market_regime_inputs": factors.get("market_regime_inputs", {}),
            "historical_volatility": factors.get("historical_volatility"),
            "historical_volatility_status": factors.get(
                "historical_volatility_status",
                "Unavailable: no daily underlying history is stored.",
            ),
            "factor_status": analysis["factors"].status,
            "market_data_status": analysis["market_data"].status,
            "risk_review_status": risk.status,
            "spread_candidates": [
                {
                    "strategy": candidate["strategy"],
                    "bias": candidate["bias"],
                    "legs": [
                        {
                            "instrument_key": leg["instrument_key"],
                            "side": leg["side"],
                            "option_type": leg["option_type"],
                            "contract": leg["trading_symbol"],
                            "strike": leg["strike"],
                            "midpoint": leg["midpoint"],
                            "implied_volatility": leg["implied_volatility"],
                            "delta": leg["delta"],
                            "gamma": leg["gamma"],
                            "theta": leg["theta"],
                            "vega": leg["vega"],
                        }
                        for leg in candidate["legs"]
                    ],
                    "entry_credit_points": candidate["entry_credit_points"],
                    "entry_credit_per_lot": candidate["entry_credit_total"],
                    "max_profit_per_lot": candidate["max_profit"],
                    "max_loss_per_lot": candidate["max_loss"],
                    "target_profit_per_lot": candidate["target_profit"],
                    "stop_loss_per_lot": candidate["stop_loss"],
                    "breakeven": candidate["breakeven"],
                }
                for candidate in risk.data["approved"]
            ],
        })


def narrative_fingerprint(context):
    serialized = json.dumps(
        _json_safe(context),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def advisory_snapshot_fingerprint(context):
    def bucket(value, size):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            return None
        return round(value / size)

    spot = context.get("spot")
    spot_bucket_size = (
        max(round(abs(spot) * 0.0005 / 5) * 5, 5)
        if isinstance(spot, (int, float))
        and not isinstance(spot, bool)
        and math.isfinite(spot)
        else 1
    )
    regime_inputs = context.get("market_regime_inputs", {})
    return narrative_fingerprint({
        "symbol": context.get("symbol"),
        "expiry": context.get("expiry"),
        "spot_bucket": bucket(spot, spot_bucket_size),
        "atm_iv_bucket": bucket(context.get("atm_iv"), 0.5),
        "iv_percentile_bucket": bucket(context.get("iv_percentile"), 5),
        "pcr_bucket": bucket(context.get("pcr"), 0.05),
        "call_oi_wall": context.get("call_oi_wall"),
        "put_oi_wall": context.get("put_oi_wall"),
        "max_pain": context.get("max_pain"),
        "india_vix_bucket": bucket(context.get("india_vix"), 0.5),
        "correlation_bucket": bucket(context.get("spot_vix_correlation"), 0.1),
        "oi_build_up": [
            {
                "strike": row["strike"],
                "option_type": row["option_type"],
                "change_bucket": bucket(row["oi_change"], 500),
                "change_percent_bucket": bucket(row["oi_change_percent"], 5),
            }
            for row in context.get("top_oi_build_up", [])
        ],
        "trend_efficiency_bucket": bucket(
            regime_inputs.get("same_session_trend_efficiency"),
            0.1,
        ),
        "same_session_change_bucket": bucket(
            regime_inputs.get("same_session_price_change_percent"),
            0.1,
        ),
        "spread_candidates": [
            {
                "strategy": candidate["strategy"],
                "legs": [
                    (
                        leg["instrument_key"],
                        leg["side"],
                        leg["option_type"],
                        leg["strike"],
                    )
                    for leg in candidate["legs"]
                ],
                "entry_credit_bucket": bucket(
                    candidate["entry_credit_points"],
                    0.5,
                ),
            }
            for candidate in context.get("spread_candidates", [])
        ],
    })


def should_start_advisory_cycle(
    fingerprint,
    last_fingerprint,
    last_started_at,
    now,
    cooldown_seconds,
    pending,
    force=False,
):
    if pending:
        return False
    if force:
        return True
    if fingerprint == last_fingerprint:
        return False
    if last_started_at is not None and now - last_started_at < cooldown_seconds:
        return False
    return True


def result_as_dict(result):
    return asdict(result)
