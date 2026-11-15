import unittest
from unittest.mock import patch

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from engine import (
    OptionSellingDecisionEngine,
    calculate_credit_spread_candidates,
    calculate_position_pnl,
    calculate_quote_mark,
)


class OptionSellingDecisionEngineTests(unittest.TestCase):
    def test_position_pnl_uses_midpoint_when_ltp_is_missing(self):
        leg = SimpleNamespace(
            instrument_key="NFO|NIFTY",
            side="SELL",
            quantity=50,
            entry_price=100.0,
        )
        quote = SimpleNamespace(ltp=None, bid_price=70.0, ask_price=80.0)

        self.assertEqual(calculate_quote_mark(quote), 75.0)
        self.assertEqual(calculate_position_pnl([leg], {"NFO|NIFTY": quote}), 1250.0)

    def test_credit_spreads_have_defined_payoffs_and_require_fresh_high_iv_quotes(self):
        now = datetime.now(timezone.utc)
        quotes = [
            SimpleNamespace(
                instrument_key="NFO|PUT_SHORT",
                trading_symbol="NIFTY 21900 PE",
                option_type="PE",
                expiry="2099-12-31",
                strike=21900,
                delta=-0.25,
                bid_price=10,
                ask_price=12,
                timestamp=now,
            ),
            SimpleNamespace(
                instrument_key="NFO|PUT_LONG",
                trading_symbol="NIFTY 21800 PE",
                option_type="PE",
                expiry="2099-12-31",
                strike=21800,
                delta=-0.10,
                bid_price=4,
                ask_price=6,
                timestamp=now,
            ),
            SimpleNamespace(
                instrument_key="NFO|CALL_SHORT",
                trading_symbol="NIFTY 22100 CE",
                option_type="CE",
                expiry="2099-12-31",
                strike=22100,
                delta=0.25,
                bid_price=10,
                ask_price=12,
                timestamp=now,
            ),
            SimpleNamespace(
                instrument_key="NFO|CALL_LONG",
                trading_symbol="NIFTY 22200 CE",
                option_type="CE",
                expiry="2099-12-31",
                strike=22200,
                delta=0.10,
                bid_price=4,
                ask_price=6,
                timestamp=now,
            ),
        ]
        contracts = {
            quote.instrument_key: SimpleNamespace(lot_size=50, live_subscribed=True)
            for quote in quotes
        }

        candidates = calculate_credit_spread_candidates(
            quotes, contracts, 22000, 80, 60, now=now
        )

        self.assertEqual(
            [candidate["strategy"] for candidate in candidates],
            ["Bull put credit spread", "Bear call credit spread"],
        )
        self.assertEqual(candidates[0]["entry_credit_total"], 300)
        self.assertEqual(candidates[0]["max_loss"], 4700)
        self.assertEqual(candidates[0]["breakeven"], 21894)
        self.assertEqual(candidates[0]["target_exit_debit"], 3)
        self.assertEqual(candidates[0]["stop_exit_debit"], 9)

        stale = [
            SimpleNamespace(**{
                **quote.__dict__,
                "timestamp": now - timedelta(seconds=31),
            })
            for quote in quotes
        ]
        self.assertEqual(
            calculate_credit_spread_candidates(stale, contracts, 22000, 80, 60, now=now),
            [],
        )
        self.assertEqual(
            calculate_credit_spread_candidates(quotes, contracts, 22000, 50, 60, now=now),
            [],
        )

    def test_requires_real_market_premium_before_triggering_a_recommendation(self):
        engine = OptionSellingDecisionEngine()
        market_data = {"symbol": "BANKNIFTY", "ltp": 50000, "iv": 20, "atr": 200}
        history = [12, 14, 15, 16, 17, 18, 19]

        with patch.object(engine, "save_recommendation") as save_recommendation:
            signal = engine.evaluate_strategy(market_data, history)
            self.assertFalse(signal["trigger"])
            self.assertEqual(signal["reason"], "A positive finite market premium is required")
            save_recommendation.assert_not_called()

            market_data["entry_credit"] = 450
            signal = engine.evaluate_strategy({**market_data, "atr": None}, history)
            self.assertFalse(signal["trigger"])
            save_recommendation.assert_not_called()

            signal = engine.evaluate_strategy(market_data, history)
            self.assertTrue(signal["trigger"])
            self.assertEqual(signal["symbol"], "BANKNIFTY")
            self.assertEqual(signal["entry_credit"], 450)
            self.assertEqual(signal["target_price"], 225)
            self.assertEqual(signal["stop_loss_price"], 675)
            save_recommendation.assert_called_once_with(signal)


if __name__ == "__main__":
    unittest.main()
