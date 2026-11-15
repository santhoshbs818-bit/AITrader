import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import requests

from agents import (
    AgentResult,
    BoundedAdvisoryLoop,
    DashboardAgentOrchestrator,
    OllamaNarrativeAgent,
    OllamaStrategyReasoningAgent,
    RiskReviewAgent,
    advisory_snapshot_fingerprint,
    should_start_advisory_cycle,
)


def build_market_snapshot(now=None):
    now = now or datetime.now(timezone.utc)
    quote_specs = [
        ("PUT_SHORT", "NIFTY 21900 PE", "PE", 21900, -0.25, 10, 12, 1000),
        ("PUT_LONG", "NIFTY 21800 PE", "PE", 21800, -0.10, 4, 6, 800),
        ("CALL_SHORT", "NIFTY 22100 CE", "CE", 22100, 0.25, 10, 12, 1200),
        ("CALL_LONG", "NIFTY 22200 CE", "CE", 22200, 0.10, 4, 6, 900),
    ]
    quotes = []
    contracts = {}
    for key, label, option_type, strike, delta, bid, ask, oi in quote_specs:
        instrument_key = f"NFO|{key}"
        quotes.append(SimpleNamespace(
            instrument_key=instrument_key,
            trading_symbol=label,
            option_type=option_type,
            expiry="2099-12-31",
            strike=strike,
            delta=delta,
            gamma=0.01,
            theta=-2,
            vega=1,
            bid_price=bid,
            ask_price=ask,
            bid_depth=[{"price": bid, "quantity": 20}],
            ask_depth=[{"price": ask, "quantity": 25}],
            oi=oi,
            previous_oi=oi - 100,
            timestamp=now,
        ))
        contracts[instrument_key] = SimpleNamespace(
            lot_size=50,
            live_subscribed=True,
        )
    return {
        "symbol": "NIFTY",
        "expiry": "2099-12-31",
        "tick": SimpleNamespace(timestamp=now, ltp=22000),
        "vix": SimpleNamespace(ltp=14),
        "chain": SimpleNamespace(
            timestamp=now,
            underlying_spot=22000,
            atm_iv=34,
        ),
        "history": [
            SimpleNamespace(
                atm_iv=15 + index,
                timestamp=now - timedelta(days=19 - index),
            )
            for index in range(20)
        ],
        "quotes": quotes,
        "contracts": contracts,
        "market_history": {"NIFTY": [], "INDIA_VIX": []},
    }


def mock_session(response=None, post_error=None):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session.post.return_value = response
    if post_error is not None:
        session.post.side_effect = post_error
    return session


def proposal_for(candidate, **overrides):
    proposal = {
        "strategy_type": candidate["strategy"],
        "market_regime": "RANGE_BOUND",
        "suggested_strikes": [
            {
                "instrument_key": leg["instrument_key"],
                "side": leg["side"],
                "option_type": leg["option_type"],
                "strike": leg["strike"],
            }
            for leg in candidate["legs"]
        ],
        "entry_credit_debit": candidate["entry_credit_total"],
        "target_profit": candidate["target_profit"],
        "stop_loss": candidate["stop_loss"],
        "confidence": 82,
        "rationale": "The supplied factors support this defined-risk candidate.",
    }
    proposal.update(overrides)
    return proposal


class StubReasoningAgent:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.feedback = []

    def run(self, context, correction_feedback=None):
        self.feedback.append(correction_feedback)
        return AgentResult(
            agent="Strategy reasoning",
            status="READY",
            summary="Mocked proposal.",
            data={"proposal": self.proposals.pop(0)},
        )


class DashboardAgentTests(unittest.TestCase):
    def test_orchestrator_runs_deterministic_agents_before_risk_approval(self):
        now = datetime.now(timezone.utc)
        analysis = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )

        self.assertEqual(analysis["market_data"].status, "READY")
        self.assertEqual(analysis["factors"].status, "READY")
        self.assertEqual(analysis["strategy"].status, "READY")
        self.assertEqual(analysis["risk"].status, "READY")
        self.assertEqual(
            len(analysis["risk"].data["approved"]),
            2,
        )

    def test_stale_market_data_blocks_strategy_agent(self):
        now = datetime.now(timezone.utc)
        snapshot = build_market_snapshot(now - timedelta(seconds=60))

        analysis = DashboardAgentOrchestrator().analyze(
            snapshot,
            minimum_iv_percentile=60,
            now=now,
        )

        self.assertEqual(analysis["market_data"].status, "BLOCKED")
        self.assertEqual(analysis["strategy"].status, "BLOCKED")
        self.assertEqual(analysis["risk"].status, "SKIPPED")

    def test_risk_review_rejects_invalid_legs_and_payoff(self):
        now = datetime.now(timezone.utc)
        analysis = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )
        original = analysis["strategy"].data["candidates"][0]
        invalid_candidates = []
        for mutation in (
            lambda candidate: candidate.update(max_loss=1),
            lambda candidate: candidate.update(target_profit=1),
            lambda candidate: candidate.update(stop_loss=1),
            lambda candidate: candidate["legs"][0].update(expiry="2099-12-30"),
            lambda candidate: candidate["legs"][0].update(option_type="CE"),
            lambda candidate: candidate["legs"][1].update(strike=22000),
        ):
            candidate = json.loads(json.dumps(original))
            mutation(candidate)
            invalid_candidates.append(candidate)

        result = RiskReviewAgent().run(invalid_candidates)

        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.data["approved"], [])
        self.assertEqual(len(result.data["rejected"]), 6)

    def test_loop_approves_only_a_matching_model_proposal(self):
        now = datetime.now(timezone.utc)
        analysis = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )
        candidates = analysis["risk"].data["approved"]
        reasoning = StubReasoningAgent([proposal_for(candidates[0])])
        loop = BoundedAdvisoryLoop(reasoning, RiskReviewAgent())

        result = loop.run({"symbol": "NIFTY"}, candidates)

        self.assertEqual(result.status, "READY")
        self.assertEqual(result.data["iterations"], 1)
        self.assertEqual(
            result.data["selected_candidate"]["strategy"],
            candidates[0]["strategy"],
        )
        self.assertEqual(result.data["risk_status"], "APPROVED")

    def test_loop_skips_model_when_candidates_are_unavailable(self):
        reasoning = StubReasoningAgent([])

        result = BoundedAdvisoryLoop(reasoning, RiskReviewAgent()).run(
            {"symbol": "NIFTY"},
            [],
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.data["iterations"], 0)
        self.assertEqual(reasoning.feedback, [])

    def test_loop_respects_a_valid_no_trade_decision_without_retrying(self):
        no_trade = {
            "strategy_type": "NO_TRADE",
            "market_regime": "UNDETERMINED",
            "suggested_strikes": [],
            "entry_credit_debit": 0,
            "target_profit": 0,
            "stop_loss": 0,
            "confidence": 95,
            "rationale": "The supplied factors do not justify a candidate.",
        }
        reasoning = StubReasoningAgent([no_trade])
        now = datetime.now(timezone.utc)
        candidates = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )["risk"].data["approved"]

        result = BoundedAdvisoryLoop(reasoning, RiskReviewAgent()).run(
            {"symbol": "NIFTY"},
            candidates,
        )

        self.assertEqual(result.status, "NO_TRADE")
        self.assertEqual(result.data["iterations"], 1)
        self.assertEqual(len(reasoning.feedback), 1)

    def test_loop_allows_only_one_revision_after_risk_rejection(self):
        now = datetime.now(timezone.utc)
        analysis = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )
        candidates = analysis["risk"].data["approved"]
        first_proposal = proposal_for(candidates[0], entry_credit_debit=999999)
        second_proposal = proposal_for(candidates[1])
        reasoning = StubReasoningAgent([first_proposal, second_proposal])
        loop = BoundedAdvisoryLoop(reasoning, RiskReviewAgent())

        result = loop.run({"symbol": "NIFTY"}, candidates)

        self.assertEqual(result.status, "READY")
        self.assertEqual(result.data["iterations"], 2)
        self.assertIsNone(reasoning.feedback[0])
        self.assertIn("did not match", reasoning.feedback[1])

    def test_loop_stops_after_second_risk_rejection(self):
        now = datetime.now(timezone.utc)
        analysis = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )
        candidates = analysis["risk"].data["approved"]
        invalid = proposal_for(candidates[0], stop_loss=999999)
        reasoning = StubReasoningAgent([invalid, invalid])

        result = BoundedAdvisoryLoop(reasoning, RiskReviewAgent()).run(
            {"symbol": "NIFTY"},
            candidates,
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.data["iterations"], 2)
        self.assertEqual(len(reasoning.feedback), 2)

    def test_cycle_gate_deduplicates_and_enforces_cooldown(self):
        context = {"symbol": "NIFTY", "expiry": "2099-12-31", "spot": 22000}
        fingerprint = advisory_snapshot_fingerprint(context)
        self.assertEqual(
            fingerprint,
            advisory_snapshot_fingerprint({**context, "spot": 22002}),
        )
        self.assertNotEqual(
            fingerprint,
            advisory_snapshot_fingerprint({**context, "spot": 22020}),
        )
        self.assertFalse(should_start_advisory_cycle(
            fingerprint,
            fingerprint,
            100,
            200,
            60,
            pending=False,
        ))
        self.assertFalse(should_start_advisory_cycle(
            "new",
            fingerprint,
            180,
            200,
            60,
            pending=False,
        ))
        self.assertTrue(should_start_advisory_cycle(
            "new",
            fingerprint,
            100,
            200,
            60,
            pending=False,
        ))
        self.assertFalse(should_start_advisory_cycle(
            "newer",
            fingerprint,
            100,
            200,
            60,
            pending=True,
            force=True,
        ))

    def test_strategy_reasoner_validates_pydantic_schema(self):
        now = datetime.now(timezone.utc)
        analysis = DashboardAgentOrchestrator().analyze(
            build_market_snapshot(now),
            minimum_iv_percentile=60,
            now=now,
        )
        candidate = analysis["risk"].data["approved"][0]
        response = Mock()
        response.json.return_value = {
            "message": {"content": json.dumps(proposal_for(candidate))}
        }
        session = mock_session(response)
        agent = OllamaStrategyReasoningAgent()
        with patch("agents.requests.Session", return_value=session):
            result = agent.run({"spread_candidates": [candidate]})

        self.assertEqual(result.status, "READY")
        self.assertEqual(result.data["proposal"]["strategy_type"], candidate["strategy"])
        schema = session.post.call_args.kwargs["json"]["format"]
        self.assertIn("suggested_strikes", schema["properties"])
        self.assertIn("entry_credit_debit", schema["required"])
        self.assertFalse(schema["additionalProperties"])

    def test_strategy_reasoner_rejects_invalid_structured_output(self):
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "strategy_type": "Unknown spread",
                    "suggested_strikes": [],
                    "entry_credit_debit": 0,
                    "target_profit": 0,
                    "stop_loss": 0,
                    "confidence": 50,
                    "rationale": "Not a valid strategy.",
                })
            }
        }
        session = mock_session(response)
        with patch("agents.requests.Session", return_value=session):
            result = OllamaStrategyReasoningAgent().run({})

        self.assertEqual(result.status, "ERROR")
        self.assertIn("strategy schema", result.summary)

    def test_ollama_uses_local_schema_constrained_request(self):
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "IV is elevated in the supplied snapshot.",
                    "drivers": ["IV percentile is 100%."],
                    "caveats": ["This is not a forecast."],
                })
            }
        }
        agent = OllamaNarrativeAgent(
            base_url="http://127.0.0.1:11434",
            model="test-model",
            timeout_seconds=4,
        )
        session = mock_session(response)
        with patch("agents.requests.Session", return_value=session):
            result = agent.run({"symbol": "NIFTY", "iv_percentile": 100})

        self.assertEqual(result.status, "READY")
        self.assertEqual(result.data["drivers"], ["IV percentile is 100%."])
        request_body = session.post.call_args.kwargs["json"]
        self.assertFalse(request_body["stream"])
        self.assertIsInstance(request_body["format"], dict)
        self.assertEqual(session.post.call_args.kwargs["timeout"], 4)
        self.assertFalse(session.trust_env)

    def test_ollama_failure_is_reported_without_fabricated_commentary(self):
        agent = OllamaNarrativeAgent()
        session = mock_session(
            post_error=requests.ConnectionError("local service unavailable")
        )
        with patch("agents.requests.Session", return_value=session):
            result = agent.run({"symbol": "NIFTY"})

        self.assertEqual(result.status, "UNAVAILABLE")
        self.assertEqual(result.data, {})
        self.assertIn("local Ollama service", result.summary)

    def test_ollama_rejects_malformed_response(self):
        response = Mock()
        response.json.return_value = {"message": {"content": "not-json"}}
        agent = OllamaNarrativeAgent()
        session = mock_session(response)
        with patch("agents.requests.Session", return_value=session):
            result = agent.run({"symbol": "NIFTY"})

        self.assertEqual(result.status, "ERROR")
        self.assertEqual(result.data, {})
        self.assertIn("invalid JSON", result.summary)

    def test_ollama_refuses_non_loopback_hosts(self):
        agent = OllamaNarrativeAgent(base_url="https://example.com")
        with patch("agents.requests.Session") as session:
            result = agent.run({"symbol": "NIFTY"})

        self.assertEqual(result.status, "ERROR")
        self.assertIn("loopback", result.summary)
        session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
