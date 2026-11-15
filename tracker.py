import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from config import config
from database import (
    OptionChainSnapshotModel,
    OptionQuoteModel,
    PositionAlertModel,
    SessionLocal,
    TradePositionModel,
    init_db,
    utcnow,
)
from engine import calculate_position_pnl

logger = logging.getLogger(__name__)
INDIA_TZ = ZoneInfo("Asia/Kolkata")


def _is_stale(timestamp, now):
    if timestamp is None:
        return True
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (now - timestamp.astimezone(timezone.utc)).total_seconds() > config.FEED_STALE_AFTER_SECONDS


def _expiry_warning_due(expiry, now):
    try:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    local_now = now.astimezone(INDIA_TZ)
    return local_now.date() == expiry_date and (local_now.hour, local_now.minute) >= (14, 30)


def _record_alert(db, position, alert_type, message, now):
    existing = (
        db.query(PositionAlertModel)
        .filter(
            PositionAlertModel.position_id == position.id,
            PositionAlertModel.alert_type == alert_type,
        )
        .first()
    )
    if existing is None:
        db.add(PositionAlertModel(
            position_id=position.id,
            timestamp=now,
            alert_type=alert_type,
            message=message,
        ))


class ExitTracker:
    """Signal-only position monitor. It never submits or routes an order."""

    def check_tracked_positions(self, now=None):
        init_db()
        observed_at = now or utcnow()
        results = []
        try:
            with SessionLocal() as db:
                positions = (
                    db.query(TradePositionModel)
                    .options(selectinload(TradePositionModel.legs))
                    .filter(TradePositionModel.status.in_(("TRACKED", "EXIT_ALERT")))
                    .order_by(TradePositionModel.id.asc())
                    .all()
                )
                instrument_keys = {
                    leg.instrument_key for position in positions for leg in position.legs
                }
                quotes = {
                    quote.instrument_key: quote
                    for quote in db.query(OptionQuoteModel)
                    .filter(OptionQuoteModel.instrument_key.in_(instrument_keys))
                    .all()
                } if instrument_keys else {}
                latest_chain = {}
                for position in positions:
                    chain = (
                        db.query(OptionChainSnapshotModel)
                        .filter(
                            OptionChainSnapshotModel.underlying_symbol
                            == position.underlying_symbol
                        )
                        .order_by(OptionChainSnapshotModel.timestamp.desc())
                        .first()
                    )
                    latest_chain[position.id] = chain

                for position in positions:
                    position_quotes = [quotes.get(leg.instrument_key) for leg in position.legs]
                    missing_or_stale = any(
                        quote is None or _is_stale(quote.timestamp, observed_at)
                        for quote in position_quotes
                    )
                    if missing_or_stale:
                        results.append({
                            "position_id": position.id,
                            "status": position.status,
                            "pnl": None,
                            "guidance": "WAITING_FOR_FRESH_QUOTES",
                        })
                        continue

                    pnl = calculate_position_pnl(position.legs, quotes)
                    if pnl is None:
                        results.append({
                            "position_id": position.id,
                            "status": position.status,
                            "pnl": None,
                            "guidance": "WAITING_FOR_FRESH_QUOTES",
                        })
                        continue

                    position.last_pnl = pnl
                    guidance = "HOLD"
                    if position.target_profit is not None and pnl >= position.target_profit:
                        guidance = "EXIT_TARGET"
                    elif position.max_loss is not None and pnl <= -abs(position.max_loss):
                        guidance = "EXIT_STOP"

                    if guidance in {"EXIT_TARGET", "EXIT_STOP"}:
                        alert_type = "TARGET" if guidance == "EXIT_TARGET" else "STOP"
                        message = (
                            f"{alert_type.title()} threshold reached; unrealized P&L is ₹{pnl:,.2f}. "
                            "Review the position manually."
                        )
                        position.status = "EXIT_ALERT"
                        position.alert_reason = message
                        _record_alert(db, position, alert_type, message, observed_at)
                    else:
                        chain = latest_chain[position.id]
                        current_iv = chain.atm_iv if chain is not None else None
                        if (
                            position.entry_iv is not None
                            and current_iv is not None
                            and current_iv
                            >= position.entry_iv * (1 + config.POSITION_IV_SPIKE_PERCENT / 100)
                        ):
                            guidance = "IV_SPIKE_WARNING"
                            message = (
                                f"ATM IV expanded by at least {config.POSITION_IV_SPIKE_PERCENT:.0f}% "
                                "from position entry. Review risk manually."
                            )
                            position.alert_reason = message
                            _record_alert(db, position, "IV_SPIKE", message, observed_at)
                        if _expiry_warning_due(position.expiry, observed_at):
                            if guidance == "HOLD":
                                guidance = "EXPIRY_GAMMA_WARNING"
                            message = (
                                "Expiry-day gamma risk window: review or close the position manually."
                            )
                            position.alert_reason = message
                            _record_alert(db, position, "EXPIRY_GAMMA", message, observed_at)

                    results.append({
                        "position_id": position.id,
                        "status": position.status,
                        "pnl": pnl,
                        "guidance": guidance,
                    })
                db.commit()
        except SQLAlchemyError:
            logger.exception("Could not update tracked position metrics")
            raise
        return results

    def close_position(self, position_id, now=None):
        init_db()
        with SessionLocal() as db:
            position = db.get(TradePositionModel, position_id)
            if position is None:
                raise ValueError(f"Tracked position {position_id} was not found.")
            if position.status == "CLOSED":
                return position
            position.status = "CLOSED"
            position.closed_at = now or utcnow()
            position.updated_at = position.closed_at
            position.alert_reason = None
            db.add(position)
            db.commit()
            db.refresh(position)
            return position
