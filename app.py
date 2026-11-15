import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from agents import (
    DashboardAgentOrchestrator,
    advisory_snapshot_fingerprint,
    narrative_fingerprint,
    result_as_dict,
    should_start_advisory_cycle,
)
from config import config
from database import (
    FeedStatusModel,
    MarketTickSnapshotModel,
    OptionChainSnapshotModel,
    OptionContractModel,
    OptionQuoteModel,
    PositionAlertModel,
    SessionLocal,
    TickDataModel,
    TradePositionLegModel,
    TradePositionModel,
    init_db,
)
from engine import calculate_oi_change, calculate_quote_mark
from tracker import ExitTracker

logger = logging.getLogger(__name__)
SYMBOLS = ["NIFTY", "BANKNIFTY"]
MARKET_SYMBOLS = [*SYMBOLS, "INDIA_VIX"]
FEED_STALE_AFTER_SECONDS = config.FEED_STALE_AFTER_SECONDS
INDIA_TZ = ZoneInfo("Asia/Kolkata")

st.set_page_config(
    page_title="Trade recommendations | NSE",
    page_icon=":material/query_stats:",
    layout="wide",
)


def tick_age_seconds(timestamp: datetime | None) -> float | None:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds())


def format_currency(value: float | None) -> str:
    return f"₹ {value:,.2f}" if value is not None else "—"


@st.cache_resource(max_entries=1)
def agent_loop_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="advisory-loop")


def advance_advisory_loop(orchestrator, context, candidates, fingerprint, force=False):
    state_key = f"advisory-loop-{context['symbol']}-{context['expiry']}"
    state = st.session_state.setdefault(state_key, {
        "future": None,
        "submitted_fingerprint": None,
        "last_fingerprint": None,
        "last_started_at": None,
        "result": None,
        "result_fingerprint": None,
    })
    future = state["future"]
    if future is not None and future.done():
        result = result_as_dict(future.result())
        if result["status"] == "READY":
            proposal = result["data"]["proposal"]
            current_candidate, rejection = orchestrator.risk_agent.validate_proposal(
                proposal,
                candidates,
            )
            if current_candidate is None:
                result.update(
                    status="BLOCKED",
                    summary=(
                        "The snapshot changed while reasoning was in progress; "
                        f"the deterministic risk gate rejected the stale proposal. {rejection}"
                    ),
                )
                result["data"].update(
                    selected_candidate=None,
                    risk_status="REJECTED_AS_STALE",
                )
            else:
                result["data"]["selected_candidate"] = current_candidate
        state["result"] = result
        state["result_fingerprint"] = state["submitted_fingerprint"]
        state["future"] = None
        state["submitted_fingerprint"] = None
        future = None

    now = time.monotonic()
    if candidates and should_start_advisory_cycle(
        fingerprint,
        state["last_fingerprint"],
        state["last_started_at"],
        now,
        config.AGENT_LOOP_COOLDOWN_SECONDS,
        pending=future is not None,
        force=force,
    ):
        state["future"] = agent_loop_executor().submit(
            orchestrator.run_advisory_loop,
            context,
            candidates,
        )
        state["submitted_fingerprint"] = fingerprint
        state["last_fingerprint"] = fingerprint
        state["last_started_at"] = now
        future = state["future"]

    if not candidates:
        return {
            "status": "BLOCKED",
            "summary": "No fresh, deterministic risk-approved candidates are available.",
            "result": None,
            "pending": future is not None,
        }
    if future is not None:
        same_snapshot = state["submitted_fingerprint"] == fingerprint
        return {
            "status": "RUNNING" if same_snapshot else "WAITING",
            "summary": (
                "Ollama is evaluating this fresh snapshot."
                if same_snapshot
                else "A previous snapshot is still being evaluated."
            ),
            "result": None,
            "pending": True,
        }
    if state["result_fingerprint"] == fingerprint:
        result = state["result"]
        return {
            "status": result["status"],
            "summary": result["summary"],
            "result": result,
            "pending": False,
        }

    last_started_at = state["last_started_at"]
    cooldown_remaining = (
        max(
            0.0,
            config.AGENT_LOOP_COOLDOWN_SECONDS - (now - last_started_at),
        )
        if last_started_at is not None
        else 0.0
    )
    return {
        "status": "COOLDOWN" if cooldown_remaining else "WAITING",
        "summary": (
            f"Waiting {cooldown_remaining:.0f}s before analyzing the changed snapshot."
            if cooldown_remaining
            else "The next analysis cycle will start on the next refresh."
        ),
        "result": None,
        "pending": False,
    }


def load_market_data(symbol: str, expiry: str):
    with SessionLocal() as db:
        tick = (
            db.query(TickDataModel)
            .filter(TickDataModel.symbol == symbol)
            .order_by(TickDataModel.id.desc())
            .first()
        )
        vix = (
            db.query(TickDataModel)
            .filter(TickDataModel.symbol == "INDIA_VIX")
            .order_by(TickDataModel.id.desc())
            .first()
        )
        chain = (
            db.query(OptionChainSnapshotModel)
            .filter(
                OptionChainSnapshotModel.underlying_symbol == symbol,
                OptionChainSnapshotModel.expiry == expiry,
            )
            .order_by(OptionChainSnapshotModel.timestamp.desc())
            .first()
        )
        history = (
            db.query(OptionChainSnapshotModel)
            .filter(OptionChainSnapshotModel.underlying_symbol == symbol)
            .order_by(OptionChainSnapshotModel.timestamp.desc())
            .limit(252)
            .all()
        )
        option_quotes = (
            db.query(OptionQuoteModel)
            .filter(
                OptionQuoteModel.underlying_symbol == symbol,
                OptionQuoteModel.expiry == expiry,
            )
            .order_by(OptionQuoteModel.strike.asc(), OptionQuoteModel.option_type.asc())
            .all()
        )
        contracts = {
            row.instrument_key: row
            for row in db.query(OptionContractModel)
            .filter(
                OptionContractModel.underlying_symbol == symbol,
                OptionContractModel.expiry == expiry,
                OptionContractModel.active.is_(True),
                OptionContractModel.live_subscribed.is_(True),
            )
            .all()
        }
        market_history = {
            market_symbol: (
                db.query(MarketTickSnapshotModel)
                .filter(MarketTickSnapshotModel.symbol == market_symbol)
                .order_by(MarketTickSnapshotModel.timestamp.desc())
                .limit(600)
                .all()
            )
            for market_symbol in (symbol, "INDIA_VIX")
        }
        feed_status = {row.source: row for row in db.query(FeedStatusModel).all()}
    return {
        "tick": tick,
        "vix": vix,
        "chain": chain,
        "history": list(reversed(history)),
        "quotes": option_quotes,
        "contracts": contracts,
        "market_history": market_history,
        "feed_status": feed_status,
    }


try:
    init_db()
    with SessionLocal() as db:
        expiry_rows = (
            db.query(
                OptionChainSnapshotModel.underlying_symbol,
                OptionChainSnapshotModel.expiry,
            )
            .distinct()
            .order_by(OptionChainSnapshotModel.expiry.asc())
            .all()
        )
    expiries_by_symbol = {symbol: [] for symbol in SYMBOLS}
    for underlying_symbol, expiry_value in expiry_rows:
        expiries_by_symbol.setdefault(underlying_symbol, []).append(expiry_value)
except SQLAlchemyError:
    logger.exception("Unable to initialize the market database")
    st.error("The market database is unavailable. Check the SQLite file and restart the app.")
    st.stop()

with st.sidebar:
    st.title("Market controls", anchor=False)
    symbol = st.segmented_control(
        "Underlying",
        SYMBOLS,
        default="NIFTY",
        key="underlying_symbol",
    ) or "NIFTY"
    expiry_options = sorted(expiries_by_symbol.get(symbol) or [config.UPSTOX_OPTION_EXPIRY])
    today = datetime.now(INDIA_TZ).date()
    upcoming_expiries = []
    for expiry_option in expiry_options:
        try:
            expiry_date = datetime.strptime(expiry_option, "%Y-%m-%d").date()
        except ValueError:
            continue
        if expiry_date >= today:
            upcoming_expiries.append((expiry_date, expiry_option))
    default_expiry = (
        min(upcoming_expiries)[1]
        if upcoming_expiries
        else expiry_options[-1]
    )
    expiry = st.selectbox(
        "Option expiry",
        expiry_options,
        index=expiry_options.index(default_expiry),
        key="expiry",
    )
    min_ivp = st.slider("Minimum IV percentile", 0, 100, 60, key="min_ivp")
    if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
        st.rerun()

    st.badge("Advisory only", icon=":material/shield:", color="blue")
    st.caption("Read-only market analysis. No orders are placed or routed.")
    st.caption("Feed process: `python3 upstox_ws.py`")
    st.caption(
        f"Option subscription: nearest {config.OPTION_STRIKE_WINDOW} strikes on either side of spot."
    )

st.title("Trade recommendation & factor analysis", icon=":material/query_stats:")
st.caption(
    f"NSE derivatives · {symbol} · expiry {expiry} · "
    "Rule-based defined-risk spread candidates; not trade instructions."
)


def _candidate_block(
    candidate,
    iv_percentile,
    pcr,
    call_wall,
    put_wall,
    agent_selected=False,
):
    with st.container(border=True):
        st.subheader(candidate["strategy"], anchor=False)
        if agent_selected:
            st.badge(
                "Agent-selected · risk-approved",
                icon=":material/verified:",
                color="green",
            )
        st.caption(
            f"{candidate['bias']} · {candidate['expiry']} · "
            f"one lot ({candidate['lot_size']} units)"
        )
        entry_col, profit_col, loss_col, breakeven_col = st.columns(4)
        entry_col.metric(
            "Estimated entry credit / lot",
            format_currency(candidate["entry_credit_total"]),
        )
        profit_col.metric("Maximum profit / lot", format_currency(candidate["max_profit"]))
        loss_col.metric("Maximum loss / lot", format_currency(candidate["max_loss"]))
        breakeven_col.metric(
            "Expiry breakeven",
            f"{candidate['breakeven']:,.2f}",
        )

        leg_rows = [
            {
                "Action": leg["side"],
                "Contract": leg["trading_symbol"],
                "Strike": leg["strike"],
                "Quote midpoint": leg["midpoint"],
            }
            for leg in candidate["legs"]
        ]
        st.dataframe(
            pd.DataFrame(leg_rows),
            hide_index=True,
            alt=f"Option legs for the {candidate['strategy']}",
        )

        target_col, stop_col = st.columns(2)
        target_col.metric(
            "Profit target",
            f"Close at debit ≤ {format_currency(candidate['target_exit_debit'])} / unit",
            delta=f"Estimated profit {format_currency(candidate['target_profit'])} / lot",
        )
        stop_col.metric(
            "Stop threshold",
            f"Close at debit ≥ {format_currency(candidate['stop_exit_debit'])} / unit",
            delta=f"Estimated loss {format_currency(candidate['stop_loss'])} / lot",
            delta_color="inverse",
        )
        wall = put_wall if candidate["option_type"] == "PE" else call_wall
        wall_name = "put" if candidate["option_type"] == "PE" else "call"
        rationale = [
            f"IV percentile {iv_percentile:.1f}% meets the configured threshold.",
            f"Short-leg absolute delta is {abs(candidate['short_delta']):.2f}.",
            "Entry estimate uses fresh two-sided quote midpoints.",
        ]
        if wall is not None:
            rationale.append(
                f"The largest {wall_name} OI concentration is at {wall['strike']:,.0f}; "
                "OI concentration is context, not a guaranteed support or resistance level."
            )
        if pcr is not None:
            rationale.append(f"Chain put-call OI ratio is {pcr:.3f}.")
        st.caption(" · ".join(rationale))
        st.caption(
            "Gross one-lot estimates exclude fees, slippage, margin, and execution effects. "
            "The maximum loss assumes both legs are held as a spread."
        )


@st.fragment(run_every="5s")
def render_recommendation_and_factors(
    selected_symbol: str,
    selected_expiry: str,
    ivp_threshold: int,
) -> None:
    try:
        data = load_market_data(selected_symbol, selected_expiry)
    except SQLAlchemyError:
        logger.exception("Unable to refresh recommendation and factor data")
        st.error("Market data could not be refreshed from SQLite.")
        return

    snapshot = {
        **data,
        "symbol": selected_symbol,
        "expiry": selected_expiry,
    }
    orchestrator = DashboardAgentOrchestrator()
    agent_results = orchestrator.analyze(snapshot, ivp_threshold)
    market_agent = agent_results["market_data"]
    factor_agent = agent_results["factors"]
    strategy_agent = agent_results["strategy"]
    risk_agent = agent_results["risk"]
    factors = factor_agent.data

    tick = data["tick"]
    vix = data["vix"]
    chain = data["chain"]
    option_quotes = data["quotes"]
    current_iv = factors["current_iv"]
    iv_rank = factors["iv_rank"]
    iv_percentile = factors["iv_percentile"]
    pcr = factors["pcr"]
    call_wall = factors["call_wall"]
    put_wall = factors["put_wall"]
    max_pain = factors["max_pain"]
    nearest = factors["nearest_quote"]
    oi_change = factors["oi_change"]
    oi_change_pct = factors["oi_change_pct"]
    oi_build_up = factors["oi_build_up"]
    imbalance = factors["depth_imbalance"]
    correlation = factors["correlation"]
    spot = tick.ltp if tick is not None else None
    underlying_age = market_agent.data["spot_age_seconds"]
    chain_age = market_agent.data["chain_age_seconds"]
    underlying_fresh = market_agent.data["spot_fresh"]
    chain_fresh = market_agent.data["chain_fresh"]

    st.subheader("Live market snapshot", anchor=False)
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        f"{selected_symbol} Spot LTP",
        format_currency(tick.ltp if tick else None),
        border=True,
    )
    metric_columns[1].metric(
        "ATM implied volatility",
        f"{current_iv:.2f}%" if current_iv is not None else "—",
        border=True,
    )
    metric_columns[2].metric(
        "IV percentile",
        f"{iv_percentile:.1f}%" if iv_percentile is not None else "—",
        border=True,
        help="Requires at least 20 daily ATM-IV observations.",
    )
    metric_columns[3].metric(
        "PCR from OI",
        f"{pcr:.3f}" if pcr is not None else "—",
        border=True,
    )
    metric_columns[4].metric(
        "India VIX",
        f"{vix.ltp:.2f}" if vix and vix.ltp is not None else "—",
        border=True,
    )

    if tick is None:
        st.warning(
            "Waiting for an underlying quote. Start the market-data feed to populate this view.",
            icon=":material/wifi_off:",
        )
    elif not underlying_fresh:
        st.warning(
            f"Underlying quote is stale ({underlying_age:.0f}s old). "
            "Recommendations are withheld.",
            icon=":material/schedule:",
        )
    else:
        st.badge("Underlying feed receiving data", icon=":material/wifi:", color="green")

    option_status = data["feed_status"].get("upstox_options")
    st.caption(
        f"Option feed: {option_status.status} · {option_status.updated_at} · "
        f"{option_status.message or ''}"
        if option_status
        else "Option feed status will appear after the Upstox collector starts."
    )
    if chain is None:
        st.info("No option-chain snapshot is available for this expiry.")
    elif not chain_fresh:
        st.warning(
            f"Option-chain snapshot is stale ({chain_age:.0f}s old); "
            "OI factors and recommendations may be unreliable."
        )

    st.subheader("Trade recommendation", anchor=False)
    candidates = risk_agent.data["approved"]
    narrative_context = orchestrator.narrative_context(snapshot, agent_results)
    loop_fingerprint = advisory_snapshot_fingerprint(narrative_context)
    force_agent_run = st.button(
        "Run strategy analysis now",
        key=f"agent-loop-run-{selected_symbol}-{selected_expiry}",
        icon=":material/psychology:",
        disabled=not candidates,
    )
    loop_state = advance_advisory_loop(
        orchestrator,
        narrative_context,
        candidates,
        loop_fingerprint,
        force=force_agent_run,
    )
    loop_result = loop_state["result"]
    selected_candidate = (
        loop_result["data"]["selected_candidate"]
        if loop_result and loop_state["status"] == "READY"
        else None
    )

    if (
        loop_state["status"] == "READY"
        and loop_result
        and selected_candidate is not None
    ):
        proposal = loop_result["data"]["proposal"]
        st.success(
            f"Agent-selected: {selected_candidate['strategy']}. "
            "The deterministic risk gate approved the candidate.",
            title="Advisory selection",
            icon=":material/verified:",
        )
        st.caption(
            f"Model-assessed regime: {proposal['market_regime']} · "
            f"confidence: {proposal['confidence']}% · "
            f"risk-gate iterations: {loop_result['data']['iterations']}"
        )
        st.text(proposal["rationale"])
    elif loop_state["status"] == "READY":
        st.error(
            "The agent loop returned READY without a validated candidate; "
            "no selection is shown."
        )
    elif loop_state["status"] == "NO_TRADE" and loop_result:
        st.info("The advisory agent recommends no trade for this snapshot.")
        st.text(loop_result["data"]["proposal"]["rationale"])
    elif loop_state["status"] in {"RUNNING", "WAITING", "COOLDOWN"}:
        st.info(loop_state["summary"])
    elif loop_state["status"] == "BLOCKED" and candidates:
        st.warning(loop_state["summary"])
    elif loop_state["status"] not in {"BLOCKED", "WAITING"}:
        st.warning(loop_state["summary"])
    elif candidates:
        st.info(loop_state["summary"])
    st.caption(
        f"Automatic analysis runs once per materially changed fresh snapshot, with a "
        f"{config.AGENT_LOOP_COOLDOWN_SECONDS:.0f}s minimum interval. "
        "The model can select only a deterministic candidate or abstain; it cannot "
        "place orders or override the risk gate."
    )

    if candidates:
        candidate_columns = st.columns(len(candidates))
        for column, candidate in zip(candidate_columns, candidates):
            with column:
                is_selected = (
                    selected_candidate is not None
                    and candidate["strategy"] == selected_candidate["strategy"]
                    and [
                        leg["instrument_key"] for leg in candidate["legs"]
                    ] == [
                        leg["instrument_key"]
                        for leg in selected_candidate["legs"]
                    ]
                )
                _candidate_block(
                    candidate,
                    iv_percentile,
                    pcr,
                    call_wall,
                    put_wall,
                    agent_selected=is_selected,
                )
    else:
        st.info(
            f"No trade candidate: {strategy_agent.summary} "
            "No values are inferred from missing or stale data.",
            icon=":material/info:",
        )

    st.subheader("Agent run status", anchor=False)
    current_fingerprint = narrative_fingerprint(narrative_context)
    narrative_state_key = f"ollama-explanation-{selected_symbol}-{selected_expiry}"
    saved_narrative = st.session_state.get(narrative_state_key)
    if st.button(
        "Ask local Ollama to explain this snapshot",
        key=f"ollama-explain-button-{selected_symbol}-{selected_expiry}",
        icon=":material/psychology:",
    ):
        with st.spinner("Requesting a local Ollama explanation…"):
            result = orchestrator.narrative_agent.run(narrative_context)
        saved_narrative = {
            "fingerprint": current_fingerprint,
            "result": result_as_dict(result),
        }
        st.session_state[narrative_state_key] = saved_narrative

    narrative_result = result_as_dict(agent_results["narrative"])
    if (
        saved_narrative
        and saved_narrative.get("fingerprint") == current_fingerprint
    ):
        narrative_result = saved_narrative["result"]
    status_rows = [
        {
            "Agent": result.agent,
            "Status": result.status,
            "Finding": result.summary,
        }
        for result in (
            market_agent,
            factor_agent,
            strategy_agent,
            risk_agent,
        )
    ]
    status_rows.append({
        "Agent": "Agentic strategy advisor",
        "Status": loop_state["status"],
        "Finding": loop_state["summary"],
    })
    status_rows.append({
        "Agent": narrative_result["agent"],
        "Status": narrative_result["status"],
        "Finding": narrative_result["summary"],
    })
    st.dataframe(
        pd.DataFrame(status_rows),
        hide_index=True,
        alt="Status and findings from the dashboard's specialist agents",
    )
    if (
        saved_narrative
        and saved_narrative.get("fingerprint") != current_fingerprint
    ):
        st.caption("Market data changed; request a fresh Ollama explanation.")
    elif narrative_result["status"] == "READY":
        st.text(f"Local explanation: {narrative_result['data']['summary']}")
        if narrative_result["data"]["drivers"]:
            st.text("Observed drivers:\n" + "\n".join(
                f"- {driver}" for driver in narrative_result["data"]["drivers"]
            ))
        if narrative_result["data"]["caveats"]:
            st.warning(
                "\n".join(narrative_result["data"]["caveats"]),
                title="Model caveats",
            )
    elif narrative_result["status"] != "NOT_RUN":
        st.warning(narrative_result["summary"])
    else:
        st.caption(
            f"Optional local model: {config.OLLAMA_MODEL} at {config.OLLAMA_BASE_URL}. "
            "Strategy analysis runs automatically under the bounded loop; this separate "
            "explanation request is made only when you press its button."
        )

    st.subheader("Quantitative factor analysis", anchor=False)
    market_history = data["market_history"]

    factor_rows = [
        {
            "Factor": "Historical volatility",
            "Reading": factors["historical_volatility_status"],
            "Interpretation": "Daily close history is insufficient for a defensible HV estimate.",
        },
        {
            "Factor": "IV rank",
            "Reading": f"{iv_rank:.1f}%" if iv_rank is not None else "Insufficient history",
            "Interpretation": "Current IV within its historical high-low range.",
        },
        {
            "Factor": "IV percentile",
            "Reading": f"{iv_percentile:.1f}%" if iv_percentile is not None else "Insufficient history",
            "Interpretation": "Share of stored daily ATM-IV observations at or below current IV.",
        },
        {
            "Factor": "Put-call OI ratio",
            "Reading": f"{pcr:.3f}" if pcr is not None else "Unavailable",
            "Interpretation": "Aggregate put OI divided by call OI in the selected chain.",
        },
        {
            "Factor": "Call OI concentration",
            "Reading": f"{call_wall['strike']:,.0f}" if call_wall else "Unavailable",
            "Interpretation": "Strike with the largest observed call open interest.",
        },
        {
            "Factor": "Put OI concentration",
            "Reading": f"{put_wall['strike']:,.0f}" if put_wall else "Unavailable",
            "Interpretation": "Strike with the largest observed put open interest.",
        },
        {
            "Factor": "Max pain",
            "Reading": f"{max_pain:,.0f}" if max_pain is not None else "Unavailable",
            "Interpretation": "Settlement strike with the lowest aggregate intrinsic OI payout.",
        },
        {
            "Factor": "Nearest contract OI change",
            "Reading": f"{oi_change:,.0f}" if oi_change is not None else "Unavailable",
            "Interpretation": "Change versus exchange-provided previous OI.",
        },
        {
            "Factor": "Nearest contract OI change %",
            "Reading": f"{oi_change_pct:+.2f}%" if oi_change_pct is not None else "Unavailable",
            "Interpretation": "Percentage change versus previous OI when defined.",
        },
        {
            "Factor": "Largest strike-wise OI changes",
            "Reading": (
                " · ".join(
                    f"{row['option_type']} {row['strike']:,.0f}: "
                    f"{row['oi_change']:+,} ({row['oi_change_percent']:+.1f}%)"
                    if row["oi_change_percent"] is not None
                    else f"{row['option_type']} {row['strike']:,.0f}: "
                    f"{row['oi_change']:+,}"
                    for row in oi_build_up
                )
                if oi_build_up
                else "Unavailable"
            ),
            "Interpretation": "Largest observed changes versus exchange-provided previous OI; not confirmed institutional flow.",
        },
        {
            "Factor": "Available depth imbalance",
            "Reading": f"{imbalance:+.3f}" if imbalance is not None else "Unavailable",
            "Interpretation": "Bid/ask quantity imbalance across levels supplied (up to five).",
        },
        {
            "Factor": "Spot / India VIX return correlation",
            "Reading": f"{correlation:.3f}" if correlation is not None else "Insufficient aligned history",
            "Interpretation": "Correlation of sampled one-minute returns; not a causal signal.",
        },
        {
            "Factor": "Nearest contract Greeks",
            "Reading": (
                f"Δ {nearest.delta if nearest.delta is not None else '—'} · "
                f"Γ {nearest.gamma if nearest.gamma is not None else '—'} · "
                f"Θ {nearest.theta if nearest.theta is not None else '—'} · "
                f"V {nearest.vega if nearest.vega is not None else '—'}"
                if nearest is not None
                else "Unavailable"
            ),
            "Interpretation": "Greeks reported by the option feed for the nearest strike.",
        },
    ]
    st.dataframe(
        pd.DataFrame(factor_rows),
        hide_index=True,
        alt=f"Live volatility, positioning, liquidity, and Greeks factors for {selected_symbol}",
    )

    history_rows = [
        {
            "Timestamp": row.timestamp,
            "Spot": row.ltp,
        }
        for row in market_history.get(selected_symbol, [])
        if row.timestamp is not None and row.ltp is not None
    ]
    vix_history_rows = [
        {
            "Timestamp": row.timestamp,
            "India VIX": row.ltp,
        }
        for row in market_history.get("INDIA_VIX", [])
        if row.timestamp is not None and row.ltp is not None
    ]
    if history_rows or vix_history_rows:
        chart_columns = st.columns(2)
        if history_rows:
            with chart_columns[0]:
                st.line_chart(
                    pd.DataFrame(history_rows).sort_values("Timestamp"),
                    x="Timestamp",
                    y="Spot",
                    alt=f"{selected_symbol} sampled spot-price history",
                )
        if vix_history_rows:
            with chart_columns[1]:
                st.line_chart(
                    pd.DataFrame(vix_history_rows).sort_values("Timestamp"),
                    x="Timestamp",
                    y="India VIX",
                    alt="Sampled India VIX history",
                )

    st.subheader(f"Option chain · {selected_symbol} · {selected_expiry}", anchor=False)
    if not option_quotes:
        st.info("Option-chain quotes will appear after the collector receives a valid chain.")
    else:
        option_rows = []
        for quote in option_quotes:
            change, change_pct = calculate_oi_change(quote.oi, quote.previous_oi)
            option_rows.append({
                "Contract": quote.trading_symbol or quote.instrument_key,
                "Strike": quote.strike,
                "Type": quote.option_type,
                "LTP": quote.ltp,
                "Bid": quote.bid_price,
                "Bid qty": quote.bid_quantity,
                "Ask": quote.ask_price,
                "Ask qty": quote.ask_quantity,
                "OI": quote.oi,
                "OI change": change,
                "OI change %": change_pct,
                "IV": quote.implied_volatility,
                "Delta": quote.delta,
                "Gamma": quote.gamma,
                "Theta": quote.theta,
                "Vega": quote.vega,
                "Depth levels": min(
                    len(quote.bid_depth or []),
                    len(quote.ask_depth or []),
                ),
                "Instrument key": quote.instrument_key,
            })
        st.dataframe(
            pd.DataFrame(option_rows),
            hide_index=True,
            height=390,
            alt=f"Live {selected_symbol} option-chain quotes, open interest, Greeks, and depth",
        )

        contract_labels = {
            quote.instrument_key: quote.trading_symbol or quote.instrument_key
            for quote in option_quotes
        }
        selected_contract = st.selectbox(
            "Depth contract",
            list(contract_labels),
            format_func=lambda key: contract_labels.get(key, key),
            key="depth_contract",
        )
        selected_quote = next(
            quote for quote in option_quotes if quote.instrument_key == selected_contract
        )
        depth_rows = []
        for level in range(5):
            bid = (
                (selected_quote.bid_depth or [])[level]
                if level < len(selected_quote.bid_depth or [])
                else {}
            )
            ask = (
                (selected_quote.ask_depth or [])[level]
                if level < len(selected_quote.ask_depth or [])
                else {}
            )
            depth_rows.append({
                "Level": level + 1,
                "Bid price": bid.get("price"),
                "Bid quantity": bid.get("quantity"),
                "Ask price": ask.get("price"),
                "Ask quantity": ask.get("quantity"),
            })
        st.dataframe(
            pd.DataFrame(depth_rows),
            hide_index=True,
            alt="Top-five bid and ask depth for the selected option contract",
        )


render_recommendation_and_factors(symbol, expiry, min_ivp)

st.header("Live trade lifecycle", anchor=False)
st.caption(
    "Only positions you confirm as filled are monitored. Alerts are advisory; "
    "the tracker never closes or routes an order."
)


@st.fragment(run_every="2s")
def render_tracked_positions(selected_symbol: str) -> None:
    try:
        ExitTracker().check_tracked_positions()
        with SessionLocal() as db:
            positions = (
                db.query(TradePositionModel)
                .options(selectinload(TradePositionModel.legs))
                .filter(
                    TradePositionModel.underlying_symbol == selected_symbol,
                    TradePositionModel.status != "CLOSED",
                )
                .order_by(TradePositionModel.id.desc())
                .all()
            )
            keys = {
                leg.instrument_key for position in positions for leg in position.legs
            }
            current_quotes = {
                quote.instrument_key: quote
                for quote in db.query(OptionQuoteModel)
                .filter(OptionQuoteModel.instrument_key.in_(keys))
                .all()
            } if keys else {}
            alerts = (
                db.query(PositionAlertModel)
                .join(TradePositionModel)
                .filter(
                    TradePositionModel.underlying_symbol == selected_symbol,
                    PositionAlertModel.acknowledged_at.is_(None),
                )
                .order_by(PositionAlertModel.timestamp.desc())
                .limit(10)
                .all()
            )
    except SQLAlchemyError:
        logger.exception("Unable to refresh confirmed-position monitoring")
        st.error("Position monitoring could not be refreshed from SQLite.")
        return

    for alert in alerts:
        st.warning(
            alert.message,
            title=f"Position {alert.position_id} · {alert.alert_type}",
        )
    if not positions:
        st.info("No confirmed positions are being monitored for this underlying.")
        return

    for position in positions:
        with st.container(border=True):
            st.subheader(
                f"{position.strategy_name} · position {position.id}",
                anchor=False,
            )
            marks = []
            stale_or_missing = False
            pnl = 0.0
            for leg in position.legs:
                quote = current_quotes.get(leg.instrument_key)
                age = tick_age_seconds(quote.timestamp if quote else None)
                mark = calculate_quote_mark(quote)
                if (
                    quote is None
                    or mark is None
                    or age is None
                    or age > FEED_STALE_AFTER_SECONDS
                ):
                    stale_or_missing = True
                    leg_pnl = None
                else:
                    direction = 1 if leg.side == "BUY" else -1
                    leg_pnl = direction * (mark - leg.entry_price) * leg.quantity
                    pnl += leg_pnl
                marks.append({
                    "Contract": leg.trading_symbol,
                    "Side": leg.side,
                    "Quantity (units)": leg.quantity,
                    "Entry": leg.entry_price,
                    "Mark": mark,
                    "Unrealized P&L": leg_pnl,
                    "Quote age (s)": round(age, 1) if age is not None else None,
                })

            guidance = (
                "WAITING_FOR_FRESH_QUOTES"
                if stale_or_missing
                else "EXIT_ALERT"
                if position.status == "EXIT_ALERT"
                else "HOLD"
            )
            if stale_or_missing:
                st.warning(
                    "One or more option quotes are missing or stale. "
                    "Current P&L and exit guidance are withheld."
                )
            metrics = st.columns(4)
            metrics[0].metric(
                "Unrealized P&L",
                format_currency(None if stale_or_missing else pnl),
            )
            metrics[1].metric("Target profit", format_currency(position.target_profit))
            metrics[2].metric("Maximum loss", format_currency(position.max_loss))
            metrics[3].metric("Guidance", guidance)
            if position.alert_reason:
                st.warning(position.alert_reason)
            st.dataframe(
                pd.DataFrame(marks),
                hide_index=True,
                alt=f"Live marks and unrealized P&L for position {position.id}",
            )
            if st.button(
                "Mark as closed",
                key=f"close-position-{position.id}",
                icon=":material/check:",
            ):
                try:
                    ExitTracker().close_position(position.id)
                    st.toast("Position marked closed. No order was placed.")
                except (SQLAlchemyError, ValueError):
                    logger.exception("Unable to close tracked position %s", position.id)
                    st.error("The position could not be marked closed.")


render_tracked_positions(symbol)

with st.expander("Track a confirmed fill", icon=":material/add_circle:"):
    try:
        with SessionLocal() as db:
            contracts = (
                db.query(OptionContractModel)
                .filter(
                    OptionContractModel.underlying_symbol == symbol,
                    OptionContractModel.expiry == expiry,
                    OptionContractModel.active.is_(True),
                    OptionContractModel.live_subscribed.is_(True),
                )
                .order_by(
                    OptionContractModel.strike.asc(),
                    OptionContractModel.option_type.asc(),
                )
                .all()
            )
            current_chain = (
                db.query(OptionChainSnapshotModel)
                .filter(
                    OptionChainSnapshotModel.underlying_symbol == symbol,
                    OptionChainSnapshotModel.expiry == expiry,
                )
                .order_by(OptionChainSnapshotModel.timestamp.desc())
                .first()
            )
    except SQLAlchemyError:
        logger.exception("Unable to load contracts for position tracking")
        st.error("Subscribed contracts could not be loaded.")
        contracts = []
        current_chain = None

    if not contracts:
        st.info("No live-subscribed option contracts are available for this expiry.")
    else:
        contract_keys = [contract.instrument_key for contract in contracts]
        contracts_by_key = {contract.instrument_key: contract for contract in contracts}
        contract_labels = {
            contract.instrument_key: contract.trading_symbol or contract.instrument_key
            for contract in contracts
        }
        with st.form("confirmed-position-form"):
            strategy_name = st.selectbox(
                "Strategy name",
                [
                    "Individual short call",
                    "Individual short put",
                    "Credit spread",
                    "Iron condor",
                    "Custom multi-leg",
                ],
                key="tracked_strategy_name",
            )
            initial_legs = pd.DataFrame([{
                "instrument_key": contract_keys[0],
                "side": "SELL",
                "quantity": 1,
                "entry_price": 0.01,
            }])
            edited_legs = st.data_editor(
                initial_legs,
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "instrument_key": st.column_config.SelectboxColumn(
                        "Option contract",
                        options=contract_keys,
                        format_func=lambda key: contract_labels.get(key, key),
                        required=True,
                    ),
                    "side": st.column_config.SelectboxColumn(
                        "Position side",
                        options=["BUY", "SELL"],
                        required=True,
                    ),
                    "quantity": st.column_config.NumberColumn(
                        "Quantity (units)",
                        min_value=1,
                        step=1,
                        required=True,
                    ),
                    "entry_price": st.column_config.NumberColumn(
                        "Confirmed entry price",
                        min_value=0.01,
                        step=0.05,
                        format="%.2f",
                        required=True,
                    ),
                },
                key="confirmed_position_legs",
                alt="Enter actual filled option legs for a confirmed position",
            )
            risk_columns = st.columns(2)
            target_profit = risk_columns[0].number_input(
                "Target profit (₹)",
                min_value=0.0,
                value=500.0,
                step=100.0,
                key="position_target_profit",
            )
            max_loss = risk_columns[1].number_input(
                "Maximum loss threshold (₹)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                key="position_max_loss",
            )
            confirmed = st.checkbox(
                "I confirm these are actual filled positions and entry prices.",
                key="confirm_filled_position",
            )
            submitted = st.form_submit_button(
                "Track confirmed trade",
                icon=":material/visibility:",
                type="primary",
            )

        if submitted:
            clean_legs = []
            invalid_leg = False
            for _, row in edited_legs.iterrows():
                instrument_key = row.get("instrument_key")
                side = row.get("side")
                try:
                    quantity_value = float(row.get("quantity"))
                    entry_value = float(row.get("entry_price"))
                except (TypeError, ValueError):
                    invalid_leg = True
                    st.error(
                        "Every leg needs a subscribed contract, side, positive quantity, and entry price."
                    )
                    break
                if (
                    pd.isna(instrument_key)
                    or instrument_key not in contracts_by_key
                    or side not in {"BUY", "SELL"}
                    or not math.isfinite(quantity_value)
                    or not math.isfinite(entry_value)
                    or quantity_value < 1
                    or not quantity_value.is_integer()
                    or entry_value <= 0
                ):
                    invalid_leg = True
                    st.error(
                        "Every leg needs a subscribed contract, side, positive quantity, and entry price."
                    )
                    break
                clean_legs.append({
                    "contract": contracts_by_key[instrument_key],
                    "side": side,
                    "quantity": int(quantity_value),
                    "entry_price": entry_value,
                })

            if not confirmed:
                st.error("Confirm the actual filled position before tracking it.")
            elif invalid_leg:
                st.caption("No position was saved.")
            elif not clean_legs:
                st.error("Add at least one valid position leg.")
            elif target_profit <= 0 or max_loss <= 0:
                st.error("Set positive target-profit and maximum-loss thresholds.")
            else:
                try:
                    with SessionLocal() as db:
                        position = TradePositionModel(
                            underlying_symbol=symbol,
                            expiry=expiry,
                            strategy_name=strategy_name,
                            status="TRACKED",
                            target_profit=float(target_profit),
                            max_loss=float(max_loss),
                            entry_iv=(
                                current_chain.atm_iv
                                if current_chain is not None
                                else None
                            ),
                        )
                        position.legs = [
                            TradePositionLegModel(
                                instrument_key=leg["contract"].instrument_key,
                                trading_symbol=leg["contract"].trading_symbol,
                                option_type=leg["contract"].option_type,
                                strike=leg["contract"].strike,
                                side=leg["side"],
                                quantity=leg["quantity"],
                                entry_price=leg["entry_price"],
                            )
                            for leg in clean_legs
                        ]
                        db.add(position)
                        db.commit()
                    st.success("Position is now tracked. No broker order was placed.")
                except SQLAlchemyError:
                    logger.exception("Unable to save confirmed position")
                    st.error("The confirmed position could not be saved.")

st.caption(
    "Risk disclaimer: Options selling can result in substantial or unlimited losses. "
    "Spread figures are estimates from current quotes, exclude costs, and can become stale. "
    "This dashboard is informational only and never executes orders."
)
