import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

import database
from agents import AgentResult
from database import (
    OptionChainSnapshotModel,
    OptionContractModel,
    OptionQuoteModel,
    TickDataModel,
)


class AppStartupTests(unittest.TestCase):
    def test_local_explanation_runs_only_after_user_click(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        narrative = AgentResult(
            agent="Ollama explanation",
            status="READY",
            summary="Explained the current snapshot.",
            data={
                "summary": "IV is elevated in the supplied snapshot.",
                "drivers": ["IV percentile is high."],
                "caveats": ["This is not a forecast."],
            },
        )
        try:
            with (
                patch("database.engine", engine),
                patch("database.SessionLocal", session_factory),
                patch("tracker.SessionLocal", session_factory),
                patch("tracker.init_db"),
                patch(
                    "agents.OllamaNarrativeAgent.run",
                    return_value=narrative,
                ) as run_narrative,
            ):
                database.init_db()
                app = AppTest.from_file(str(app_path)).run(timeout=30)
                self.assertFalse(app.exception)
                run_narrative.assert_not_called()

                explanation_button = next(
                    button
                    for button in app.button
                    if button.label == "Ask local Ollama to explain this snapshot"
                )
                explanation_button.click().run()

                self.assertFalse(app.exception)
                run_narrative.assert_called_once()
                self.assertTrue(any(
                    "IV is elevated in the supplied snapshot." in item.value
                    for item in app.text
                ))
        finally:
            engine.dispose()

    def test_app_starts_with_a_fresh_database_and_scopes_selected_symbol(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        try:
            with (
                patch("database.engine", engine),
                patch("database.SessionLocal", session_factory),
                patch("tracker.SessionLocal", session_factory),
                patch("tracker.init_db"),
            ):
                database.init_db()
                app = AppTest.from_file(str(app_path)).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(any(metric.label == "NIFTY Spot LTP" and metric.value == "—" for metric in app.metric))
                self.assertTrue(any(metric.label == "IV percentile" and metric.value == "—" for metric in app.metric))
                self.assertTrue(any(metric.label == "India VIX" and metric.value == "—" for metric in app.metric))

                with session_factory() as db:
                    db.add_all([
                        TickDataModel(symbol="NIFTY", ltp=22000, iv=15, volume=10),
                        TickDataModel(symbol="BANKNIFTY", ltp=50000, iv=20, volume=10),
                    ])
                    db.commit()

                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any(metric.label == "India VIX" and metric.value == "—" for metric in app.metric))
                self.assertTrue(any(metric.label == "NIFTY Spot LTP" and metric.value == "₹ 22,000.00" for metric in app.metric))
                self.assertTrue(any(metric.label == "IV percentile" and metric.value == "—" for metric in app.metric))

                app.segmented_control(key="underlying_symbol").set_value("BANKNIFTY").run()
                self.assertFalse(app.exception)
                self.assertTrue(any(metric.label == "BANKNIFTY Spot LTP" and metric.value == "₹ 50,000.00" for metric in app.metric))
                self.assertTrue(any(metric.label == "IV percentile" and metric.value == "—" for metric in app.metric))
                self.assertEqual(len(app.tabs), 0)
                self.assertTrue(any(item.value == "Trade recommendation" for item in app.subheader))
                self.assertTrue(any(item.value == "Quantitative factor analysis" for item in app.subheader))
                self.assertTrue(any(item.value == "Live trade lifecycle" for item in app.header))
        finally:
            engine.dispose()

    def test_dashboard_renders_live_credit_spread_candidates_from_fresh_data(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        expiry = "2099-12-31"
        now = datetime.now(timezone.utc)
        try:
            with (
                patch("database.engine", engine),
                patch("database.SessionLocal", session_factory),
                patch("tracker.SessionLocal", session_factory),
                patch("tracker.init_db"),
                patch(
                    "agents.DashboardAgentOrchestrator.run_advisory_loop",
                    return_value=AgentResult(
                        agent="Agentic strategy advisor",
                        status="UNAVAILABLE",
                        summary="Mocked local model unavailable.",
                    ),
                ),
            ):
                database.init_db()
                with session_factory() as db:
                    db.add(TickDataModel(
                        timestamp=now,
                        symbol="NIFTY",
                        ltp=22000,
                        iv=20,
                        volume=100,
                    ))
                    for age in range(20):
                        snapshot_time = now - timedelta(days=age)
                        db.add(OptionChainSnapshotModel(
                            timestamp=snapshot_time,
                            session_date=snapshot_time.date().isoformat(),
                            underlying_symbol="NIFTY",
                            expiry=expiry,
                            underlying_spot=22000,
                            pcr=1.1,
                            atm_iv=34 - age,
                        ))

                    rows = [
                        ("PUT_SHORT", "NIFTY 21900 PE", "PE", 21900, -0.25, 10, 12, 1000),
                        ("PUT_LONG", "NIFTY 21800 PE", "PE", 21800, -0.10, 4, 6, 800),
                        ("CALL_SHORT", "NIFTY 22100 CE", "CE", 22100, 0.25, 10, 12, 1200),
                        ("CALL_LONG", "NIFTY 22200 CE", "CE", 22200, 0.10, 4, 6, 900),
                    ]
                    for key, label, option_type, strike, delta, bid, ask, oi in rows:
                        instrument_key = f"NFO|{key}"
                        db.add(OptionContractModel(
                            instrument_key=instrument_key,
                            underlying_symbol="NIFTY",
                            expiry=expiry,
                            trading_symbol=label,
                            strike=strike,
                            option_type=option_type,
                            lot_size=50,
                            active=True,
                            live_subscribed=True,
                        ))
                        db.add(OptionQuoteModel(
                            instrument_key=instrument_key,
                            timestamp=now,
                            underlying_symbol="NIFTY",
                            expiry=expiry,
                            trading_symbol=label,
                            strike=strike,
                            option_type=option_type,
                            ltp=(bid + ask) / 2,
                            oi=oi,
                            previous_oi=oi - 100,
                            bid_price=bid,
                            bid_quantity=20,
                            ask_price=ask,
                            ask_quantity=25,
                            bid_depth=[{"price": bid, "quantity": 20}],
                            ask_depth=[{"price": ask, "quantity": 25}],
                            implied_volatility=20,
                            delta=delta,
                            gamma=0.01,
                            theta=-2,
                            vega=1,
                            source="WEBSOCKET",
                        ))
                    db.commit()

                app = AppTest.from_file(str(app_path)).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(any(
                    item.value == "Bull put credit spread" for item in app.subheader
                ))
                self.assertTrue(any(
                    item.value == "Bear call credit spread" for item in app.subheader
                ))
                self.assertTrue(any(
                    metric.label == "Maximum profit / lot" and metric.value == "₹ 300.00"
                    for metric in app.metric
                ))
                self.assertTrue(any(
                    metric.label == "Maximum loss / lot" and metric.value == "₹ 4,700.00"
                    for metric in app.metric
                ))
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
