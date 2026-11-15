import logging
import math
import os
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import or_

try:
    import upstox_client
except ImportError:  # pragma: no cover - optional dependency during local development
    upstox_client = None

from config import config
from database import (
    FeedStatusModel,
    MarketTickSnapshotModel,
    OptionChainSnapshotModel,
    OptionContractModel,
    OptionQuoteModel,
    OptionQuoteSnapshotModel,
    SessionLocal,
    TickDataModel,
    init_db,
)

logger = logging.getLogger(__name__)
streamer = None

UNDERLYING_KEYS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
}
INSTRUMENT_SYMBOLS = {
    **{instrument_key: symbol for symbol, instrument_key in UNDERLYING_KEYS.items()},
    "NSE_INDEX|India VIX": "INDIA_VIX",
}
OPTION_CHAIN_URL = "https://api.upstox.com/v2/option/chain"
OPTION_CONTRACTS_URL = "https://api.upstox.com/v2/option/contract"
REQUEST_TIMEOUT_SECONDS = 10
RELATIVE_EXPIRIES = {
    "current_week",
    "next_week",
    "far_week",
    "current_month",
    "next_month",
    "far_month",
}
OPTION_QUOTE_FIELDS = (
    "timestamp",
    "underlying_symbol",
    "expiry",
    "trading_symbol",
    "strike",
    "option_type",
    "ltp",
    "volume",
    "oi",
    "previous_oi",
    "bid_price",
    "bid_quantity",
    "ask_price",
    "ask_quantity",
    "bid_depth",
    "ask_depth",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "source",
)

_state_lock = threading.RLock()
_connected = False
_stop_refresh = threading.Event()
_option_instruments = {}
_option_subscriptions = set()
_last_option_history = {}
_last_market_history = {}
_last_history_pruned_at = None


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _integer(value):
    number = _finite_float(value)
    return int(number) if number is not None and number >= 0 else None


def _utc_timestamp(value=None):
    number = _finite_float(value)
    if number is None:
        return datetime.now(timezone.utc)
    epoch_seconds = number / 1000 if number > 10_000_000_000 else number
    try:
        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return datetime.now(timezone.utc)


def _market_payload(feed):
    if not isinstance(feed, dict):
        return {}
    full_feed = _first(feed, "fullFeed", "full_feed") or {}
    if not isinstance(full_feed, dict):
        return {}
    return _first(full_feed, "marketFF", "market_ff", "indexFF", "index_ff") or {}


def _depth_levels(market_feed):
    market_level = _first(market_feed, "marketLevel", "market_level") or {}
    quotes = _first(market_level, "bidAskQuote", "bid_ask_quote") or []
    if not isinstance(quotes, list):
        return [], []

    bids = []
    asks = []
    for quote in quotes[:5]:
        if not isinstance(quote, dict):
            continue
        bid_price = _finite_float(_first(quote, "bidP", "bid_p", "bp"))
        bid_quantity = _integer(_first(quote, "bidQ", "bid_q", "bq"))
        ask_price = _finite_float(_first(quote, "askP", "ask_p", "ap"))
        ask_quantity = _integer(_first(quote, "askQ", "ask_q", "aq"))
        if bid_price is not None and bid_quantity is not None:
            bids.append({"price": bid_price, "quantity": bid_quantity})
        if ask_price is not None and ask_quantity is not None:
            asks.append({"price": ask_price, "quantity": ask_quantity})
    return bids, asks


def extract_market_ticks(message):
    """Extract supported underlying index quotes from an Upstox V3 message."""
    if not isinstance(message, dict):
        return []

    received_at = _utc_timestamp(message.get("currentTs"))
    ticks = []
    for instrument_key, feed in message.get("feeds", {}).items():
        symbol = INSTRUMENT_SYMBOLS.get(instrument_key)
        if symbol is None:
            continue
        market_feed = _market_payload(feed)
        ltpc = _first(market_feed, "ltpc") or _first(feed, "ltpc") or {}
        if not isinstance(ltpc, dict):
            continue

        ltp = _finite_float(ltpc.get("ltp"))
        if ltp is None or ltp <= 0:
            continue
        volume = _finite_float(_first(market_feed, "vtt", "volume"))
        ticks.append({
            "symbol": symbol,
            "timestamp": received_at,
            "ltp": ltp,
            "iv": _finite_float(_first(market_feed, "iv")),
            "volume": int(volume) if volume is not None and volume >= 0 else None,
        })
    return ticks


def fetch_option_chain(instrument_key, expiry, access_token):
    response = requests.get(
        OPTION_CHAIN_URL,
        params={"instrument_key": instrument_key, "expiry_date": expiry},
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError(f"Upstox returned an unsuccessful option-chain response for {instrument_key}.")
    if not isinstance(payload.get("data"), list):
        raise ValueError(f"Upstox option-chain response has no data list for {instrument_key}.")
    return payload


def fetch_option_contracts(instrument_key, expiry, access_token):
    params = {"instrument_key": instrument_key}
    if expiry is not None:
        params["expiry_date"] = expiry
    response = requests.get(
        OPTION_CONTRACTS_URL,
        params=params,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError(f"Upstox returned unsuccessful contract metadata for {instrument_key}.")
    contracts = payload.get("data")
    if not isinstance(contracts, list):
        raise ValueError(f"Upstox contract response has no data list for {instrument_key}.")
    return {
        contract["instrument_key"]: contract
        for contract in contracts
        if isinstance(contract, dict) and contract.get("instrument_key")
    }


def resolve_chain_expiry(requested_expiry, contract_metadata, today=None):
    """Translate relative Upstox expiry keywords to the ISO date expected by the chain request."""
    if requested_expiry not in RELATIVE_EXPIRIES:
        return requested_expiry

    expiries = {
        contract.get("expiry")
        for contract in contract_metadata.values()
        if isinstance(contract, dict) and contract.get("expiry")
    }
    if not expiries:
        raise ValueError(
            f"Upstox returned no contract expiry for relative expiry '{requested_expiry}'."
        )
    if len(expiries) > 1 and requested_expiry == "current_week":
        today = today or datetime.now(ZoneInfo("Asia/Kolkata")).date()
        future_expiries = sorted(
            expiry_date
            for expiry in expiries
            if _valid_expiry_date(expiry) is not None
            for expiry_date in [_valid_expiry_date(expiry)]
            if expiry_date >= today
        )
        if future_expiries:
            return future_expiries[0].isoformat()
        raise ValueError("Upstox returned no unexpired contracts for 'current_week'.")
    if len(expiries) > 1:
        raise ValueError(
            f"Upstox returned multiple contract expiries for '{requested_expiry}'."
        )
    return next(iter(expiries))


def _valid_expiry_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def parse_option_chain_response(symbol, response, contract_metadata=None):
    """Normalize Upstox option-chain data into quote rows and factor inputs."""
    contract_metadata = contract_metadata or {}
    chain_rows = response.get("data")
    if not isinstance(chain_rows, list) or not chain_rows:
        raise ValueError(f"Option-chain response for {symbol} contains no strikes.")

    expiry = next(
        (row.get("expiry") for row in chain_rows if isinstance(row, dict) and row.get("expiry")),
        None,
    )
    if not expiry:
        raise ValueError(f"Option-chain response for {symbol} has no expiry.")

    spot = next(
        (
            value
            for value in (
                _finite_float(row.get("underlying_spot_price"))
                for row in chain_rows
                if isinstance(row, dict)
            )
            if value is not None and value > 0
        ),
        None,
    )
    option_quotes = []
    metadata = {}
    call_oi_total = 0.0
    put_oi_total = 0.0
    atm_iv_values = []
    closest_strike_distance = None

    for chain_row in chain_rows:
        if not isinstance(chain_row, dict):
            continue
        strike = _finite_float(chain_row.get("strike_price"))
        if strike is None:
            continue
        distance = abs(strike - spot) if spot is not None else None
        for option_type, option_key in (("CE", "call_options"), ("PE", "put_options")):
            option = chain_row.get(option_key)
            if not isinstance(option, dict):
                continue
            instrument = option.get("instrument_key")
            market_data = option.get("market_data")
            greeks = option.get("option_greeks")
            if not instrument or not isinstance(market_data, dict):
                continue
            if not isinstance(greeks, dict):
                greeks = {}

            contract = contract_metadata.get(instrument, {})
            trading_symbol = contract.get("trading_symbol") or (
                f"{symbol} {strike:g} {option_type} {expiry}"
            )
            bid_price = _finite_float(market_data.get("bid_price"))
            bid_quantity = _integer(market_data.get("bid_qty"))
            ask_price = _finite_float(market_data.get("ask_price"))
            ask_quantity = _integer(market_data.get("ask_qty"))
            oi = _finite_float(market_data.get("oi"))
            previous_oi = _finite_float(market_data.get("prev_oi"))
            iv = _finite_float(greeks.get("iv"))
            if option_type == "CE" and oi is not None:
                call_oi_total += oi
            if option_type == "PE" and oi is not None:
                put_oi_total += oi
            if distance is not None and (closest_strike_distance is None or distance < closest_strike_distance):
                closest_strike_distance = distance
                atm_iv_values = [iv] if iv is not None and iv > 0 else []
            elif distance is not None and distance == closest_strike_distance and iv is not None and iv > 0:
                atm_iv_values.append(iv)

            quote = {
                "timestamp": datetime.now(timezone.utc),
                "underlying_symbol": symbol,
                "expiry": expiry,
                "trading_symbol": trading_symbol,
                "strike": strike,
                "option_type": option_type,
                "ltp": _finite_float(market_data.get("ltp")),
                "volume": _integer(market_data.get("volume")),
                "oi": oi,
                "previous_oi": previous_oi,
                "bid_price": bid_price,
                "bid_quantity": bid_quantity,
                "ask_price": ask_price,
                "ask_quantity": ask_quantity,
                "bid_depth": (
                    [{"price": bid_price, "quantity": bid_quantity}]
                    if bid_price is not None and bid_quantity is not None
                    else []
                ),
                "ask_depth": (
                    [{"price": ask_price, "quantity": ask_quantity}]
                    if ask_price is not None and ask_quantity is not None
                    else []
                ),
                "implied_volatility": iv,
                "delta": _finite_float(greeks.get("delta")),
                "gamma": _finite_float(greeks.get("gamma")),
                "theta": _finite_float(greeks.get("theta")),
                "vega": _finite_float(greeks.get("vega")),
                "source": "REST_CHAIN",
                "instrument_key": instrument,
            }
            option_quotes.append(quote)
            metadata[instrument] = {
                "underlying_symbol": symbol,
                "expiry": expiry,
                "trading_symbol": trading_symbol,
                "strike": strike,
                "option_type": option_type,
                "lot_size": _integer(contract.get("lot_size")),
                "tick_size": _finite_float(contract.get("tick_size")),
            }

    pcr = put_oi_total / call_oi_total if call_oi_total > 0 else None
    atm_iv = sum(atm_iv_values) / len(atm_iv_values) if atm_iv_values else None
    return {
        "expiry": expiry,
        "spot": spot,
        "pcr": pcr,
        "atm_iv": atm_iv,
        "quotes": option_quotes,
        "metadata": metadata,
    }


def select_option_subscription_keys(option_quotes, spot, strike_window=None):
    """Select both sides of the nearest strikes around spot for live depth updates."""
    if not option_quotes or spot is None or spot <= 0:
        return []
    window = config.OPTION_STRIKE_WINDOW if strike_window is None else strike_window
    strikes = sorted({row["strike"] for row in option_quotes if row.get("strike") is not None})
    selected_strikes = set(sorted(strikes, key=lambda strike: abs(strike - spot))[: 2 * max(0, window) + 1])
    return sorted({
        row["instrument_key"]
        for row in option_quotes
        if row.get("strike") in selected_strikes and row.get("instrument_key")
    })


def _persist_feed_status(source, status, message=None):
    with SessionLocal() as db:
        row = db.get(FeedStatusModel, source)
        if row is None:
            row = FeedStatusModel(source=source, status=status)
        row.status = status
        row.updated_at = datetime.now(timezone.utc)
        row.message = message
        db.add(row)
        db.commit()


def prune_old_history():
    global _last_history_pruned_at
    now = datetime.now(timezone.utc)
    if (
        _last_history_pruned_at is not None
        and (now - _last_history_pruned_at).total_seconds() < 3600
    ):
        return
    cutoff = now.timestamp() - config.OPTION_HISTORY_RETENTION_DAYS * 86400
    cutoff_datetime = datetime.fromtimestamp(cutoff, tz=timezone.utc)
    with SessionLocal() as db:
        db.query(OptionQuoteSnapshotModel).filter(
            OptionQuoteSnapshotModel.timestamp < cutoff_datetime
        ).delete(synchronize_session=False)
        db.query(MarketTickSnapshotModel).filter(
            MarketTickSnapshotModel.timestamp < cutoff_datetime
        ).delete(synchronize_session=False)
        db.query(OptionChainSnapshotModel).filter(
            OptionChainSnapshotModel.timestamp < cutoff_datetime
        ).delete(synchronize_session=False)
        db.commit()
    _last_history_pruned_at = now


def persist_market_ticks(ticks):
    """Update latest underlying quotes and retain sampled history."""
    if not ticks:
        return

    now = datetime.now(timezone.utc)
    history_keys = []
    with SessionLocal() as db:
        for tick in ticks:
            latest = (
                db.query(TickDataModel)
                .filter(TickDataModel.symbol == tick["symbol"])
                .order_by(TickDataModel.id.desc())
                .first()
            )
            if latest is None:
                latest = TickDataModel(symbol=tick["symbol"])
            latest.timestamp = tick.get("timestamp") or now
            latest.ltp = tick["ltp"]
            latest.iv = tick["iv"]
            latest.volume = tick["volume"]
            db.add(latest)

            previous_history = _last_market_history.get(tick["symbol"])
            if (
                previous_history is None
                or (now - previous_history).total_seconds() >= config.MARKET_HISTORY_INTERVAL_SECONDS
            ):
                db.add(MarketTickSnapshotModel(
                    timestamp=latest.timestamp,
                    symbol=tick["symbol"],
                    ltp=tick["ltp"],
                    iv=tick["iv"],
                    volume=tick["volume"],
                ))
                history_keys.append(tick["symbol"])
        db.commit()
    for symbol in history_keys:
        _last_market_history[symbol] = now


def persist_option_ticks(ticks):
    """Upsert current option quotes and append timestamped history at a bounded rate."""
    if not ticks:
        return

    now = datetime.now(timezone.utc)
    snapshot_keys = []
    instrument_keys = {tick["instrument_key"] for tick in ticks}
    with SessionLocal() as db:
        current_rows = {
            row.instrument_key: row
            for row in db.query(OptionQuoteModel)
            .filter(OptionQuoteModel.instrument_key.in_(instrument_keys))
            .all()
        }
        for tick in ticks:
            existing = current_rows.get(tick["instrument_key"])
            timestamp = tick.get("timestamp") or now
            if (
                existing is not None
                and existing.timestamp is not None
                and timestamp < existing.timestamp.replace(tzinfo=timezone.utc)
            ):
                continue

            row = existing or OptionQuoteModel(instrument_key=tick["instrument_key"])
            snapshot = {"instrument_key": tick["instrument_key"]}
            for field in OPTION_QUOTE_FIELDS:
                value = tick.get(field)
                if value is None and existing is not None and field not in {"timestamp", "source"}:
                    value = getattr(existing, field)
                setattr(row, field, value)
                snapshot[field] = value
            db.add(row)

            last_saved = _last_option_history.get(tick["instrument_key"])
            if (
                last_saved is None
                or (now - last_saved).total_seconds() >= config.OPTION_HISTORY_INTERVAL_SECONDS
            ):
                db.add(OptionQuoteSnapshotModel(**snapshot))
                snapshot_keys.append(tick["instrument_key"])
        db.commit()
    for instrument_key in snapshot_keys:
        _last_option_history[instrument_key] = now


def persist_option_chain(symbol, chain, subscribed_keys):
    """Store a REST chain snapshot and update contract/quote caches."""
    quote_rows = chain["quotes"]
    if quote_rows:
        persist_option_ticks(quote_rows)

    session_date = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    with SessionLocal() as db:
        db.query(OptionContractModel).filter(
            OptionContractModel.underlying_symbol == symbol,
            OptionContractModel.active.is_(True),
        ).update({OptionContractModel.active: False}, synchronize_session=False)
        for instrument_key, metadata in chain["metadata"].items():
            contract = db.get(OptionContractModel, instrument_key)
            if contract is None:
                contract = OptionContractModel(instrument_key=instrument_key)
            for field, value in metadata.items():
                setattr(contract, field, value)
            contract.updated_at = datetime.now(timezone.utc)
            contract.active = True
            contract.live_subscribed = instrument_key in subscribed_keys
            db.add(contract)
        snapshot = (
            db.query(OptionChainSnapshotModel)
            .filter(
                OptionChainSnapshotModel.underlying_symbol == symbol,
                OptionChainSnapshotModel.session_date == session_date,
            )
            .first()
        )
        if snapshot is None:
            snapshot = OptionChainSnapshotModel(
                underlying_symbol=symbol,
                session_date=session_date,
            )
        snapshot.timestamp = datetime.now(timezone.utc)
        snapshot.expiry = chain["expiry"]
        snapshot.underlying_spot = chain["spot"]
        snapshot.pcr = chain["pcr"]
        snapshot.atm_iv = chain["atm_iv"]
        db.add(snapshot)
        db.commit()


def _fetch_and_store_symbol_chain(symbol, underlying_key, access_token):
    requested_expiry = config.UPSTOX_OPTION_EXPIRY
    contract_metadata = {}
    chain_expiry = requested_expiry
    if requested_expiry in RELATIVE_EXPIRIES:
        try:
            contract_metadata = fetch_option_contracts(
                underlying_key,
                requested_expiry,
                access_token,
            )
            if not contract_metadata and requested_expiry == "current_week":
                contract_metadata = fetch_option_contracts(
                    underlying_key,
                    None,
                    access_token,
                )
        except (requests.RequestException, ValueError) as error:
            logger.warning(
                "Could not resolve relative expiry for %s; trying the option-chain endpoint directly: %s",
                symbol,
                error,
            )
        else:
            chain_expiry = resolve_chain_expiry(requested_expiry, contract_metadata)
            contract_metadata = {
                instrument_key: contract
                for instrument_key, contract in contract_metadata.items()
                if contract.get("expiry") == chain_expiry
            }

    chain_response = fetch_option_chain(
        underlying_key,
        chain_expiry,
        access_token,
    )
    response_data = chain_response["data"]
    expiry = next(
        (row.get("expiry") for row in response_data if isinstance(row, dict) and row.get("expiry")),
        None,
    )
    if not expiry:
        raise ValueError(f"Upstox returned no expiry for {symbol}.")

    if not contract_metadata:
        try:
            contract_metadata = fetch_option_contracts(underlying_key, expiry, access_token)
        except (requests.RequestException, ValueError) as error:
            logger.warning(
                "Contract metadata unavailable for %s; using option-chain keys: %s",
                symbol,
                error,
            )
    chain = parse_option_chain_response(symbol, chain_response, contract_metadata)
    keys = select_option_subscription_keys(chain["quotes"], chain["spot"])
    persist_option_chain(symbol, chain, set(keys))
    prune_old_history()
    return keys, chain["metadata"]


def refresh_option_subscriptions(access_token):
    """Refresh active expiry contracts, persist chain data, and roll WebSocket keys."""
    global _option_instruments, _option_subscriptions
    with _state_lock:
        previous_metadata = dict(_option_instruments)
        previous_subscriptions = set(_option_subscriptions)

    refreshed_metadata = {}
    refreshed_subscriptions = set()
    successful_symbols = set()
    for symbol, underlying_key in UNDERLYING_KEYS.items():
        try:
            keys, metadata = _fetch_and_store_symbol_chain(
                symbol,
                underlying_key,
                access_token,
            )
        except (requests.RequestException, ValueError, SQLAlchemyError) as error:
            logger.error("Could not refresh %s option chain: %s", symbol, error)
            for instrument_key, row in previous_metadata.items():
                if row["underlying_symbol"] == symbol:
                    refreshed_metadata[instrument_key] = row
            refreshed_subscriptions.update(
                instrument_key
                for instrument_key in previous_subscriptions
                if instrument_key in refreshed_metadata
            )
            continue
        refreshed_metadata.update(metadata)
        refreshed_subscriptions.update(keys)
        successful_symbols.add(symbol)

    with _state_lock:
        removed = _option_subscriptions - refreshed_subscriptions
        added = refreshed_subscriptions - _option_subscriptions
        _option_instruments = refreshed_metadata
        _option_subscriptions = refreshed_subscriptions
        is_connected = _connected
        active_streamer = streamer

    if is_connected and active_streamer is not None:
        try:
            if removed:
                active_streamer.unsubscribe(sorted(removed))
            if added:
                active_streamer.subscribe(sorted(added), "full")
        except Exception:
            logger.exception("Upstox option subscription update failed")
            _persist_feed_status("upstox_options", "degraded", "Subscription update failed")
            return

    if successful_symbols:
        _persist_feed_status(
            "upstox_options",
            "connected" if is_connected else "ready",
            f"Option chain refreshed for {', '.join(sorted(successful_symbols))}",
        )
    else:
        _persist_feed_status(
            "upstox_options",
            "degraded",
            "No option-chain response was available; retained prior subscriptions if present.",
        )


def _refresh_option_subscriptions_periodically(access_token):
    while not _stop_refresh.wait(config.OPTION_CHAIN_REFRESH_SECONDS):
        try:
            refresh_option_subscriptions(access_token)
        except SQLAlchemyError:
            logger.exception("Unable to persist the refreshed option universe")
        except requests.RequestException:
            logger.exception("Upstox option-chain refresh request failed")


def extract_option_ticks(message, instrument_metadata=None):
    """Extract subscribed option contracts, including top-five depth and Greeks."""
    if not isinstance(message, dict):
        return []
    if instrument_metadata is None:
        with _state_lock:
            instrument_metadata = dict(_option_instruments)

    received_at = _utc_timestamp(message.get("currentTs"))
    ticks = []
    for instrument_key, feed in message.get("feeds", {}).items():
        metadata = instrument_metadata.get(instrument_key)
        if metadata is None or not isinstance(feed, dict):
            continue
        market_feed = _market_payload(feed)
        ltpc = _first(market_feed, "ltpc") or _first(feed, "ltpc") or {}
        greeks = _first(market_feed, "optionGreeks", "option_greeks") or {}
        if not isinstance(ltpc, dict):
            ltpc = {}
        if not isinstance(greeks, dict):
            greeks = {}
        bid_depth, ask_depth = _depth_levels(market_feed)
        best_bid = bid_depth[0] if bid_depth else {}
        best_ask = ask_depth[0] if ask_depth else {}
        ltp = _finite_float(ltpc.get("ltp"))
        oi = _finite_float(_first(market_feed, "oi"))
        volume = _integer(_first(market_feed, "vtt", "volume"))
        if all(value is None for value in (ltp, oi, volume, best_bid.get("price"), best_ask.get("price"))):
            continue
        ticks.append({
            "timestamp": received_at,
            "underlying_symbol": metadata["underlying_symbol"],
            "expiry": metadata["expiry"],
            "trading_symbol": metadata["trading_symbol"],
            "strike": metadata["strike"],
            "option_type": metadata["option_type"],
            "instrument_key": instrument_key,
            "ltp": ltp if ltp is not None and ltp > 0 else None,
            "volume": volume,
            "oi": oi,
            "previous_oi": None,
            "bid_price": best_bid.get("price"),
            "bid_quantity": best_bid.get("quantity"),
            "ask_price": best_ask.get("price"),
            "ask_quantity": best_ask.get("quantity"),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "implied_volatility": _finite_float(_first(market_feed, "iv")),
            "delta": _finite_float(greeks.get("delta")),
            "gamma": _finite_float(greeks.get("gamma")),
            "theta": _finite_float(greeks.get("theta")),
            "vega": _finite_float(greeks.get("vega")),
            "source": "WEBSOCKET",
        })
    return ticks


def on_message(message):
    """Persist decoded market updates without logging potentially sensitive payloads."""
    if not isinstance(message, dict):
        logger.warning("Ignoring an unsupported Upstox WebSocket message type")
        return
    try:
        persist_market_ticks(extract_market_ticks(message))
        persist_option_ticks(extract_option_ticks(message))
    except SQLAlchemyError:
        logger.exception("Failed to persist an Upstox market-data message")


def on_open():
    global _connected
    if streamer is None:
        raise RuntimeError("Upstox streamer is not initialized before the websocket opens.")

    with _state_lock:
        instrument_keys = list(INSTRUMENT_SYMBOLS)
        instrument_keys.extend(sorted(_option_subscriptions))
    if len(instrument_keys) > 1500:
        raise ValueError(
            "The requested full-feed subscription exceeds Upstox's 1,500 combined-key limit."
        )
    streamer.subscribe(instrument_keys, "full")
    with _state_lock:
        _connected = True
    logger.info(
        "Connected to Upstox WebSocket; subscribed to %d index and option instruments",
        len(instrument_keys),
    )
    _persist_feed_status("upstox_market_data", "connected")
    _persist_feed_status("upstox_options", "connected")


def on_error(*args):
    error = str(args[-1]) if args else "unknown error"
    access_token = getattr(config, "UPSTOX_ACCESS_TOKEN", "")
    if access_token:
        error = error.replace(access_token, "[redacted]")
    logger.error("Upstox WebSocket error: %s", error)
    try:
        _persist_feed_status("upstox_market_data", "error", error[:500])
    except SQLAlchemyError:
        logger.exception("Could not persist Upstox connection error status")


def on_close(*args):
    global _connected
    with _state_lock:
        _connected = False
    status_code = args[-2] if len(args) >= 2 else "unknown"
    close_message = args[-1] if args else "no close message"
    logger.warning("Upstox WebSocket closed (%s): %s", status_code, close_message)
    _persist_feed_status("upstox_market_data", "disconnected", str(close_message)[:500])
    _persist_feed_status("upstox_options", "disconnected", str(close_message)[:500])


def main():
    if upstox_client is None:
        raise RuntimeError(
            "The Upstox Python SDK is not installed. Install 'upstox-python-sdk' or add it to requirements.txt."
        )

    access_token = getattr(config, "UPSTOX_ACCESS_TOKEN", "") or os.getenv("UPSTOX_ACCESS_TOKEN", "")
    if not access_token:
        raise ValueError(
            "UPSTOX_ACCESS_TOKEN is missing. Set it in the environment or in config.py before starting the websocket."
        )
    if config.OPTION_STRIKE_WINDOW < 0:
        raise ValueError("OPTION_STRIKE_WINDOW must be zero or greater.")
    if config.OPTION_CHAIN_REFRESH_SECONDS < 30:
        raise ValueError("OPTION_CHAIN_REFRESH_SECONDS must be at least 30 seconds.")
    if config.OPTION_HISTORY_INTERVAL_SECONDS <= 0:
        raise ValueError("OPTION_HISTORY_INTERVAL_SECONDS must be greater than zero.")
    if config.MARKET_HISTORY_INTERVAL_SECONDS <= 0:
        raise ValueError("MARKET_HISTORY_INTERVAL_SECONDS must be greater than zero.")
    if config.OPTION_HISTORY_RETENTION_DAYS < 1:
        raise ValueError("OPTION_HISTORY_RETENTION_DAYS must be at least one day.")

    init_db()
    refresh_option_subscriptions(access_token)

    configuration = upstox_client.Configuration()
    configuration.access_token = access_token

    global streamer
    streamer = upstox_client.MarketDataStreamerV3(
        upstox_client.ApiClient(configuration)
    )
    streamer.auto_reconnect(True, 10, 3)
    streamer.on("open", on_open)
    streamer.on("message", on_message)
    streamer.on("error", on_error)
    streamer.on("close", on_close)

    _stop_refresh.clear()
    threading.Thread(
        target=_refresh_option_subscriptions_periodically,
        args=(access_token,),
        name="upstox-option-chain-refresh",
        daemon=True,
    ).start()
    streamer.connect()


if __name__ == "__main__":
    main()
