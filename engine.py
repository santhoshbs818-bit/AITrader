import pandas as pd
import numpy as np
import logging
import math
from datetime import datetime, timezone

from sqlalchemy.exc import SQLAlchemyError

from config import config
from database import SessionLocal, RecommendationModel, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def calculate_iv_percentile(current_iv, historical_ivs):
    current = _finite_float(current_iv)
    history = [value for value in (_finite_float(iv) for iv in historical_ivs) if value is not None]
    if current is None or not history:
        return None
    return sum(iv <= current for iv in history) / len(history) * 100.0


def calculate_iv_rank(current_iv, historical_ivs):
    current = _finite_float(current_iv)
    history = [value for value in (_finite_float(iv) for iv in historical_ivs) if value is not None]
    if current is None or not history:
        return None
    low, high = min(history), max(history)
    if high <= low:
        return None
    return min(100.0, max(0.0, (current - low) / (high - low) * 100.0))


def calculate_pcr(option_quotes):
    call_oi = 0.0
    put_oi = 0.0
    for quote in option_quotes:
        oi = _finite_float(quote.get("oi"))
        if oi is None or oi < 0:
            continue
        if quote.get("option_type") == "CE":
            call_oi += oi
        elif quote.get("option_type") == "PE":
            put_oi += oi
    return put_oi / call_oi if call_oi > 0 else None


def calculate_oi_change(current_oi, previous_oi):
    current = _finite_float(current_oi)
    previous = _finite_float(previous_oi)
    if current is None or previous is None:
        return None, None
    difference = current - previous
    percentage = difference / previous * 100.0 if previous > 0 else None
    return difference, percentage


def calculate_depth_imbalance(bid_depth, ask_depth):
    bid_quantity = sum(
        value for value in (_finite_float(level.get("quantity")) for level in bid_depth or [])
        if value is not None and value > 0
    )
    ask_quantity = sum(
        value for value in (_finite_float(level.get("quantity")) for level in ask_depth or [])
        if value is not None and value > 0
    )
    total = bid_quantity + ask_quantity
    return (bid_quantity - ask_quantity) / total if total else None


def calculate_max_pain(option_quotes):
    open_interest = {}
    for quote in option_quotes:
        strike = _finite_float(quote.get("strike"))
        oi = _finite_float(quote.get("oi"))
        option_type = quote.get("option_type")
        if strike is None or oi is None or oi < 0 or option_type not in {"CE", "PE"}:
            continue
        row = open_interest.setdefault(strike, {"CE": 0.0, "PE": 0.0})
        row[option_type] += oi
    if not open_interest:
        return None

    def payout(settlement):
        return sum(
            oi["CE"] * max(settlement - strike, 0)
            + oi["PE"] * max(strike - settlement, 0)
            for strike, oi in open_interest.items()
        )

    return min(open_interest, key=payout)


def calculate_rolling_correlation(underlying_prices, vix_prices):
    underlying_returns = pd.Series(underlying_prices, dtype="float64").pct_change()
    vix_returns = pd.Series(vix_prices, dtype="float64").pct_change()
    paired = pd.concat([underlying_returns, vix_returns], axis=1).dropna()
    if len(paired) < 2:
        return None
    correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return float(correlation) if math.isfinite(float(correlation)) else None


def calculate_quote_mark(quote):
    if quote is None:
        return None

    bid = _finite_float(quote.bid_price)
    ask = _finite_float(quote.ask_price)
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (bid + ask) / 2

    ltp = _finite_float(quote.ltp)
    return ltp if ltp is not None and ltp > 0 else None


def calculate_position_pnl(legs, current_quotes):
    total_pnl = 0.0
    for leg in legs:
        quote = current_quotes.get(leg.instrument_key)
        mark = calculate_quote_mark(quote)
        if mark is None:
            return None
        direction = 1 if leg.side == "BUY" else -1
        quantity = leg.quantity
        total_pnl += direction * (mark - leg.entry_price) * quantity
    return total_pnl


def calculate_credit_spread_candidates(
    option_quotes,
    contracts_by_key,
    spot,
    iv_percentile,
    minimum_iv_percentile,
    now=None,
    max_quote_age_seconds=30,
    target_delta=0.25,
):
    spot = _finite_float(spot)
    iv_percentile = _finite_float(iv_percentile)
    if (
        spot is None
        or spot <= 0
        or iv_percentile is None
        or iv_percentile < minimum_iv_percentile
    ):
        return []

    observed_at = now or datetime.now(timezone.utc)
    quotes_by_type = {"CE": [], "PE": []}
    for quote in option_quotes:
        contract = contracts_by_key.get(quote.instrument_key)
        if (
            contract is None
            or not contract.live_subscribed
            or quote.option_type not in quotes_by_type
            or quote.timestamp is None
        ):
            continue
        timestamp = quote.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_seconds = (observed_at - timestamp.astimezone(timezone.utc)).total_seconds()
        if age_seconds < 0 or age_seconds > max_quote_age_seconds:
            continue

        bid = _finite_float(quote.bid_price)
        ask = _finite_float(quote.ask_price)
        delta = _finite_float(quote.delta)
        if (
            bid is None
            or ask is None
            or bid <= 0
            or ask < bid
            or delta is None
            or _finite_float(contract.lot_size) is None
            or contract.lot_size <= 0
        ):
            continue
        midpoint = (bid + ask) / 2
        quotes_by_type[quote.option_type].append(
            (quote, contract, midpoint, abs(delta))
        )

    candidates = []
    for option_type in ("PE", "CE"):
        short_candidates = [
            row for row in quotes_by_type[option_type]
            if (
                row[0].strike < spot if option_type == "PE"
                else row[0].strike > spot
            )
            and 0.10 <= row[3] <= 0.40
        ]
        short_candidates.sort(key=lambda row: abs(row[3] - target_delta))

        for short_quote, short_contract, short_midpoint, short_delta in short_candidates:
            hedges = [
                row for row in quotes_by_type[option_type]
                if (
                    row[0].strike < short_quote.strike if option_type == "PE"
                    else row[0].strike > short_quote.strike
                )
                and row[1].lot_size == short_contract.lot_size
            ]
            if not hedges:
                continue
            long_quote, long_contract, long_midpoint, _ = min(
                hedges,
                key=lambda row: abs(row[0].strike - short_quote.strike),
            )
            width = abs(short_quote.strike - long_quote.strike)
            entry_credit = short_midpoint - long_midpoint
            if width <= 0 or entry_credit <= 0 or entry_credit >= width:
                continue

            lot_size = int(short_contract.lot_size)
            target_exit_debit = entry_credit * 0.5
            stop_exit_debit = min(entry_credit * 1.5, width)
            is_put_spread = option_type == "PE"
            candidates.append({
                "strategy": (
                    "Bull put credit spread"
                    if is_put_spread
                    else "Bear call credit spread"
                ),
                "bias": "Bullish to neutral" if is_put_spread else "Bearish to neutral",
                "expiry": short_quote.expiry,
                "option_type": option_type,
                "short_delta": short_quote.delta,
                "entry_credit_points": entry_credit,
                "entry_credit_total": entry_credit * lot_size,
                "max_profit": entry_credit * lot_size,
                "max_loss": (width - entry_credit) * lot_size,
                "breakeven": (
                    short_quote.strike - entry_credit
                    if is_put_spread
                    else short_quote.strike + entry_credit
                ),
                "target_exit_debit": target_exit_debit,
                "target_profit": (entry_credit - target_exit_debit) * lot_size,
                "stop_exit_debit": stop_exit_debit,
                "stop_loss": (stop_exit_debit - entry_credit) * lot_size,
                "lot_size": lot_size,
                "spread_width": width,
                "legs": [
                    {
                        "instrument_key": short_quote.instrument_key,
                        "trading_symbol": short_quote.trading_symbol,
                        "side": "SELL",
                        "option_type": short_quote.option_type,
                        "expiry": short_quote.expiry,
                        "strike": short_quote.strike,
                        "midpoint": short_midpoint,
                        "implied_volatility": getattr(
                            short_quote,
                            "implied_volatility",
                            None,
                        ),
                        "delta": getattr(short_quote, "delta", None),
                        "gamma": getattr(short_quote, "gamma", None),
                        "theta": getattr(short_quote, "theta", None),
                        "vega": getattr(short_quote, "vega", None),
                    },
                    {
                        "instrument_key": long_quote.instrument_key,
                        "trading_symbol": long_quote.trading_symbol,
                        "side": "BUY",
                        "option_type": long_quote.option_type,
                        "expiry": long_quote.expiry,
                        "strike": long_quote.strike,
                        "midpoint": long_midpoint,
                        "implied_volatility": getattr(
                            long_quote,
                            "implied_volatility",
                            None,
                        ),
                        "delta": getattr(long_quote, "delta", None),
                        "gamma": getattr(long_quote, "gamma", None),
                        "theta": getattr(long_quote, "theta", None),
                        "vega": getattr(long_quote, "vega", None),
                    },
                ],
            })
            break
    return candidates


class OptionSellingDecisionEngine:
    def __init__(self):
        self.min_ivp = config.MIN_IV_PERCENTILE

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates EMA, VWAP, and ATR using pure pandas/numpy without external build issues."""
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        vwap_numerator = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        cumulative_volume = df['volume'].cumsum().replace(0, np.nan)
        df['VWAP'] = vwap_numerator / cumulative_volume
        
        # True Range & ATR (14 periods)
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        return df

    def calculate_iv_percentile(self, current_iv: float, historical_ivs: list) -> float | None:
        """Return historical IV percentile, or None until real observations exist."""
        return calculate_iv_percentile(current_iv, historical_ivs)

    def evaluate_strategy(self, current_data: dict, historical_ivs: list) -> dict:
        """
        Confluence Gatekeeper: Triggers 'Short Straddle/Strangle' recommendation 
        only when IV Percentile > 60 and market structure is range-bound.
        """
        current_iv = _finite_float(current_data.get("iv"))
        historical_ivs = [
            value for value in (_finite_float(iv) for iv in historical_ivs)
            if value is not None
        ]
        iv_p = self.calculate_iv_percentile(current_iv, historical_ivs)

        ltp = _finite_float(current_data.get("ltp")) or 0.0
        atr = _finite_float(current_data.get("atr"))

        # Missing or invalid price/ATR data must not pass the range-bound gate.
        is_range_bound = (
            atr is not None and atr >= 0 and ltp > 0 and atr < (ltp * 0.01)
        )

        symbol = current_data.get("symbol") or config.SYMBOL
        signal = {
            "trigger": False,
            "strategy": "None",
            "symbol": symbol,
            "iv_percentile": iv_p,
            "entry_credit": 0.0,
            "target_price": 0.0,
            "stop_loss_price": 0.0,
            "reason": "Confluence checks not met",
        }

        # Do not create a trade recommendation without an actual market premium.
        entry_credit = _finite_float(current_data.get("entry_credit"))

        confluence_passed = iv_p is not None and iv_p >= self.min_ivp and is_range_bound
        premium_is_valid = entry_credit is not None and entry_credit > 0
        if confluence_passed and premium_is_valid:
            signal = {
                "trigger": True,
                "strategy": current_data.get("strategy_name") or "Short Premium Strategy",
                "symbol": symbol,
                "iv_percentile": iv_p,
                "entry_credit": entry_credit,
                "target_price": entry_credit * (1 - config.PROFIT_TARGET_DECAY),
                "stop_loss_price": entry_credit * config.STOP_LOSS_MULTIPLIER,
                "reason": "Confluence checks passed",
            }
            
            # Persist recommendation to SQLite database
            self.save_recommendation(signal)
            logger.info(f"SIGNAL TRIGGERED: {signal['strategy']} | IVP: {iv_p:.2f}%")
        elif confluence_passed:
            signal["reason"] = "A positive finite market premium is required"
        elif iv_p is None:
            signal["reason"] = "Historical IV data is required to calculate IV percentile"

        return signal

    def save_recommendation(self, signal: dict):
        init_db()
        try:
            with SessionLocal() as db:
                rec = RecommendationModel(
                    timestamp=datetime.now(timezone.utc),
                    strategy_name=signal["strategy"],
                    symbol=signal.get("symbol", config.SYMBOL),
                    entry_credit=signal["entry_credit"],
                    target_price=signal["target_price"],
                    stop_loss_price=signal["stop_loss_price"],
                    iv_percentile=signal["iv_percentile"],
                    status="SIGNAL",
                )
                db.add(rec)
                db.commit()
        except SQLAlchemyError:
            logger.exception("Failed to save recommendation")

if __name__ == "__main__":
    # Test runner for engine module
    engine = OptionSellingDecisionEngine()
    sample_tick = {"ltp": 22500.0, "iv": 18.5, "atr": 150.0}
    mock_history = [12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    result = engine.evaluate_strategy(sample_tick, mock_history)
    print("Engine Evaluation Result:", result)