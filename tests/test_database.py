import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, text

import database


class DatabaseInitializationTests(unittest.TestCase):
    def test_init_db_creates_tables_and_migrates_legacy_symbols(self):
        test_engine = create_engine("sqlite://")
        try:
            with test_engine.begin() as connection:
                connection.execute(text("""
                    CREATE TABLE ticks (
                        id INTEGER PRIMARY KEY, timestamp DATETIME, symbol VARCHAR,
                        ltp FLOAT, iv FLOAT, volume INTEGER
                    )
                """))
                connection.execute(text("""
                    CREATE TABLE recommendations (
                        id INTEGER PRIMARY KEY, timestamp DATETIME, strategy_name VARCHAR,
                        entry_credit FLOAT, target_price FLOAT, stop_loss_price FLOAT,
                        iv_percentile FLOAT, status VARCHAR
                    )
                """))
                connection.execute(text("INSERT INTO ticks (id) VALUES (1)"))
                connection.execute(text("INSERT INTO recommendations (id) VALUES (1)"))

            with patch.object(database, "engine", test_engine):
                database.init_db()
                database.init_db()

            tables = set(inspect(test_engine).get_table_names())
            self.assertIn("greeks", tables)
            self.assertTrue({
                "option_quotes",
                "option_quote_snapshots",
                "option_chain_snapshots",
                "trade_positions",
                "trade_position_legs",
                "position_alerts",
            }.issubset(tables))
            self.assertIn(
                "symbol",
                {column["name"] for column in inspect(test_engine).get_columns("recommendations")},
            )
            self.assertIn(
                "ix_recommendations_symbol",
                {index["name"] for index in inspect(test_engine).get_indexes("recommendations")},
            )
            with test_engine.connect() as connection:
                self.assertEqual(
                    connection.execute(text("SELECT symbol FROM ticks")).scalar_one(),
                    database.config.SYMBOL,
                )
                self.assertEqual(
                    connection.execute(text("SELECT symbol FROM recommendations")).scalar_one(),
                    database.config.SYMBOL,
                )
        finally:
            test_engine.dispose()


if __name__ == "__main__":
    unittest.main()
