import importlib
import os
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, TickDataModel


class UpstoxWebSocketTests(unittest.TestCase):
    def test_config_reads_upstox_access_token_from_env(self):
        original = os.environ.get("UPSTOX_ACCESS_TOKEN")
        try:
            os.environ["UPSTOX_ACCESS_TOKEN"] = "token-123"
            import config

            importlib.reload(config)
            self.assertEqual(config.config.UPSTOX_ACCESS_TOKEN, "token-123")
        finally:
            if original is None:
                os.environ.pop("UPSTOX_ACCESS_TOKEN", None)
            else:
                os.environ["UPSTOX_ACCESS_TOKEN"] = original
            importlib.reload(config)

    def test_main_requires_sdk_and_token(self):
        import upstox_ws

        with patch.object(upstox_ws, "upstox_client", None):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                upstox_ws.main()

        class FakeConfiguration:
            def __init__(self):
                self.access_token = None

        class FakeStreamer:
            def __init__(self, client):
                self.client = client

            def auto_reconnect(self, *args, **kwargs):
                return None

            def on(self, *args, **kwargs):
                return None

            def connect(self):
                return None

        class FakeClientModule:
            Configuration = FakeConfiguration
            ApiClient = staticmethod(lambda configuration: configuration)
            MarketDataStreamerV3 = FakeStreamer

        with patch.object(upstox_ws, "upstox_client", FakeClientModule), patch.object(
            upstox_ws.config, "UPSTOX_ACCESS_TOKEN", ""
        ), patch.dict(os.environ, {"UPSTOX_ACCESS_TOKEN": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "UPSTOX_ACCESS_TOKEN"):
                upstox_ws.main()

    def test_upstox_index_messages_parse_and_update_latest_tick(self):
        import upstox_ws

        message = {
            "feeds": {
                "NSE_INDEX|Nifty 50": {
                    "fullFeed": {
                        "indexFF": {
                            "ltpc": {"ltp": 22345.5},
                            "marketOHLC": {"ohlc": []},
                        }
                    }
                },
                "unrecognized-instrument": {
                    "fullFeed": {"indexFF": {"ltpc": {"ltp": 100.0}}}
                },
            }
        }
        ticks = upstox_ws.extract_market_ticks(message)
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0]["symbol"], "NIFTY")
        self.assertEqual(ticks[0]["ltp"], 22345.5)
        self.assertIsNone(ticks[0]["iv"])
        self.assertIsNone(ticks[0]["volume"])
        self.assertIsNotNone(ticks[0]["timestamp"])

        test_engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        try:
            Base.metadata.create_all(test_engine)
            session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
            with patch.object(upstox_ws, "SessionLocal", session_factory), patch.object(
                upstox_ws, "init_db"
            ):
                upstox_ws.persist_market_ticks(ticks)
                upstox_ws.persist_market_ticks([{**ticks[0], "ltp": 22346.0}])

            with session_factory() as db:
                saved_ticks = db.query(TickDataModel).all()
                self.assertEqual(len(saved_ticks), 1)
                self.assertEqual(saved_ticks[0].symbol, "NIFTY")
                self.assertEqual(saved_ticks[0].ltp, 22346.0)
        finally:
            test_engine.dispose()

    def test_option_ticks_extract_greeks_and_top_five_depth(self):
        import upstox_ws

        metadata = {
            "NFO|NIFTY": {
                "underlying_symbol": "NIFTY",
                "expiry": "2099-12-31",
                "trading_symbol": "NIFTY CE",
                "strike": 22000,
                "option_type": "CE",
            }
        }
        message = {
            "currentTs": "1700000000000",
            "feeds": {
                "NFO|NIFTY": {
                    "fullFeed": {
                        "marketFF": {
                            "ltpc": {"ltp": 105.0},
                            "oi": "1200",
                            "iv": 18.5,
                            "optionGreeks": {"delta": 0.4, "theta": -2.0},
                            "marketLevel": {
                                "bidAskQuote": [
                                    {"bidP": 104, "bidQ": 10, "askP": 106, "askQ": 12},
                                    {"bidP": 103, "bidQ": 20, "askP": 107, "askQ": 15},
                                ]
                            },
                        }
                    }
                }
            },
        }

        ticks = upstox_ws.extract_option_ticks(message, metadata)

        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0]["oi"], 1200)
        self.assertEqual(ticks[0]["delta"], 0.4)
        self.assertEqual(ticks[0]["bid_depth"], [
            {"price": 104, "quantity": 10},
            {"price": 103, "quantity": 20},
        ])
        self.assertEqual(ticks[0]["ask_depth"][0], {"price": 106, "quantity": 12})

    def test_relative_expiry_is_resolved_before_option_chain_request(self):
        import upstox_ws

        expiry = upstox_ws.resolve_chain_expiry(
            "current_week",
            {
                "NFO|1": {"expiry": "2099-12-31"},
                "NFO|2": {"expiry": "2099-12-31"},
            },
        )

        self.assertEqual(expiry, "2099-12-31")
        self.assertEqual(
            upstox_ws.resolve_chain_expiry(
                "current_week",
                {
                    "NFO|PAST": {"expiry": "2099-12-24"},
                    "NFO|FUTURE": {"expiry": "2099-12-31"},
                    "NFO|LATER": {"expiry": "2100-01-07"},
                },
                today=date(2099, 12, 25),
            ),
            "2099-12-31",
        )
        with self.assertRaisesRegex(ValueError, "no contract expiry"):
            upstox_ws.resolve_chain_expiry("current_week", {})

    def test_option_chain_parser_computes_pcr_and_selects_near_atm_contracts(self):
        import upstox_ws

        def option(instrument_key, oi, iv):
            return {
                "instrument_key": instrument_key,
                "market_data": {
                    "ltp": 100,
                    "bid_price": 99,
                    "bid_qty": 10,
                    "ask_price": 101,
                    "ask_qty": 12,
                    "oi": oi,
                    "prev_oi": oi - 10,
                    "volume": 500,
                },
                "option_greeks": {"iv": iv, "delta": 0.5},
            }

        chain = upstox_ws.parse_option_chain_response("NIFTY", {
            "data": [{
                "expiry": "2099-12-31",
                "strike_price": 22000,
                "underlying_spot_price": 22010,
                "call_options": option("NFO|CALL", 100, 20),
                "put_options": option("NFO|PUT", 200, 22),
            }]
        })

        self.assertEqual(chain["pcr"], 2)
        self.assertEqual(chain["atm_iv"], 21)
        self.assertEqual(
            set(upstox_ws.select_option_subscription_keys(chain["quotes"], 22010, 0)),
            {"NFO|CALL", "NFO|PUT"},
        )

    def test_option_quote_persists_current_state_and_timestamped_history(self):
        import upstox_ws
        from database import OptionQuoteModel, OptionQuoteSnapshotModel

        test_engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        try:
            Base.metadata.create_all(test_engine)
            session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
            tick = {
                "timestamp": datetime.now(timezone.utc),
                "underlying_symbol": "NIFTY",
                "expiry": "2099-12-31",
                "trading_symbol": "NIFTY CE",
                "strike": 22000,
                "option_type": "CE",
                "ltp": 100,
                "volume": 500,
                "oi": 1200,
                "previous_oi": 1100,
                "bid_price": 99,
                "bid_quantity": 10,
                "ask_price": 101,
                "ask_quantity": 12,
                "bid_depth": [{"price": 99, "quantity": 10}],
                "ask_depth": [{"price": 101, "quantity": 12}],
                "implied_volatility": 20,
                "delta": 0.5,
                "gamma": 0.01,
                "theta": -2,
                "vega": 1,
                "source": "WEBSOCKET",
                "instrument_key": "NFO|NIFTY",
            }
            with (
                patch.object(upstox_ws, "SessionLocal", session_factory),
                patch.object(upstox_ws, "_last_option_history", {}),
            ):
                upstox_ws.persist_option_ticks([tick])
                upstox_ws.persist_option_ticks([{**tick, "ltp": 101}])

            with session_factory() as db:
                quote = db.get(OptionQuoteModel, "NFO|NIFTY")
                snapshots = db.query(OptionQuoteSnapshotModel).all()
                self.assertEqual(quote.ltp, 101)
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(snapshots[0].ltp, 100)
                self.assertIsNotNone(snapshots[0].timestamp)
        finally:
            test_engine.dispose()


if __name__ == "__main__":
    unittest.main()
