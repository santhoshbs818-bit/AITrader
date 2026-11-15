import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import (
    Base,
    OptionQuoteModel,
    TradePositionLegModel,
    TradePositionModel,
)
from tracker import ExitTracker


class ExitTrackerTests(unittest.TestCase):
    def test_tracker_uses_fresh_midpoint_quotes_for_confirmed_positions(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        try:
            now = datetime.now(timezone.utc)
            with session_factory() as db:
                position = TradePositionModel(
                    underlying_symbol="BANKNIFTY",
                    expiry="2099-12-31",
                    strategy_name="Short call",
                    status="TRACKED",
                    target_profit=1000,
                    max_loss=2000,
                )
                position.legs = [TradePositionLegModel(
                    instrument_key="NFO|BANKNIFTY",
                    trading_symbol="BANKNIFTY CE",
                    option_type="CE",
                    strike=50000,
                    side="SELL",
                    quantity=50,
                    entry_price=100,
                )]
                db.add(position)
                db.commit()

                db.add(OptionQuoteModel(
                    instrument_key="NFO|BANKNIFTY",
                    timestamp=now,
                    underlying_symbol="BANKNIFTY",
                    expiry="2099-12-31",
                    strike=50000,
                    option_type="CE",
                    ltp=None,
                    bid_price=70,
                    ask_price=80,
                    source="WEBSOCKET",
                ))
                db.commit()

            with patch("tracker.SessionLocal", session_factory), patch("tracker.init_db"):
                results = ExitTracker().check_tracked_positions(now=now)

            self.assertEqual(results[0]["guidance"], "EXIT_TARGET")
            self.assertEqual(results[0]["pnl"], 1250)
            with session_factory() as db:
                position = db.query(TradePositionModel).one()
                self.assertEqual(position.status, "EXIT_ALERT")
                self.assertEqual(position.last_pnl, 1250)

            with patch("tracker.SessionLocal", session_factory), patch("tracker.init_db"):
                results = ExitTracker().check_tracked_positions(now=now + timedelta(seconds=60))
            self.assertEqual(results[0]["guidance"], "WAITING_FOR_FRESH_QUOTES")
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
