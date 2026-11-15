from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import config

engine = create_engine(config.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class TickDataModel(Base):
    __tablename__ = "ticks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utcnow, index=True)
    symbol = Column(String, index=True)
    ltp = Column(Float)
    iv = Column(Float)
    volume = Column(Integer)


class MarketTickSnapshotModel(Base):
    __tablename__ = "market_tick_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    symbol = Column(String, nullable=False, index=True)
    ltp = Column(Float)
    iv = Column(Float)
    volume = Column(Integer)

    __table_args__ = (Index("ix_market_tick_symbol_timestamp", "symbol", "timestamp"),)


class OptionGreekModel(Base):
    __tablename__ = "greeks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utcnow)
    strike = Column(Float)
    option_type = Column(String)
    delta = Column(Float)
    theta = Column(Float)
    gamma = Column(Float)
    vega = Column(Float)
    implied_volatility = Column(Float)


class OptionContractModel(Base):
    __tablename__ = "option_contracts"
    instrument_key = Column(String, primary_key=True)
    underlying_symbol = Column(String, nullable=False, index=True)
    expiry = Column(String, nullable=False, index=True)
    trading_symbol = Column(String, nullable=False)
    strike = Column(Float, nullable=False)
    option_type = Column(String, nullable=False)
    lot_size = Column(Integer)
    tick_size = Column(Float)
    updated_at = Column(DateTime, default=utcnow, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    live_subscribed = Column(Boolean, default=False, nullable=False)


class OptionQuoteFields:
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    underlying_symbol = Column(String, nullable=False, index=True)
    expiry = Column(String, nullable=False, index=True)
    trading_symbol = Column(String)
    strike = Column(Float, nullable=False)
    option_type = Column(String, nullable=False)
    ltp = Column(Float)
    volume = Column(Integer)
    oi = Column(Float)
    previous_oi = Column(Float)
    bid_price = Column(Float)
    bid_quantity = Column(Integer)
    ask_price = Column(Float)
    ask_quantity = Column(Integer)
    bid_depth = Column(JSON)
    ask_depth = Column(JSON)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    source = Column(String, nullable=False)


class OptionQuoteModel(OptionQuoteFields, Base):
    __tablename__ = "option_quotes"
    instrument_key = Column(String, primary_key=True)


class OptionQuoteSnapshotModel(OptionQuoteFields, Base):
    __tablename__ = "option_quote_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_key = Column(String, nullable=False)

    __table_args__ = (
        Index("ix_option_snapshot_instrument_time", "instrument_key", "timestamp"),
        Index("ix_option_snapshot_underlying_expiry_time", "underlying_symbol", "expiry", "timestamp"),
    )


class OptionChainSnapshotModel(Base):
    __tablename__ = "option_chain_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    session_date = Column(String, nullable=False)
    underlying_symbol = Column(String, nullable=False)
    expiry = Column(String, nullable=False)
    underlying_spot = Column(Float)
    pcr = Column(Float)
    atm_iv = Column(Float)

    __table_args__ = (
        UniqueConstraint("underlying_symbol", "session_date", name="uq_chain_symbol_session"),
        Index("ix_chain_symbol_time", "underlying_symbol", "timestamp"),
    )


class FeedStatusModel(Base):
    __tablename__ = "feed_status"
    source = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    updated_at = Column(DateTime, default=utcnow, nullable=False)
    message = Column(String)


class RecommendationModel(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utcnow)
    strategy_name = Column(String)
    symbol = Column(String, index=True, default=config.SYMBOL)
    entry_credit = Column(Float)
    target_price = Column(Float)
    stop_loss_price = Column(Float)
    iv_percentile = Column(Float)
    status = Column(String, default="SIGNAL")


class TradePositionModel(Base):
    __tablename__ = "trade_positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    closed_at = Column(DateTime)
    underlying_symbol = Column(String, nullable=False, index=True)
    expiry = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    status = Column(String, default="TRACKED", nullable=False, index=True)
    entry_iv = Column(Float)
    target_profit = Column(Float)
    max_loss = Column(Float)
    last_pnl = Column(Float)
    alert_reason = Column(String)
    legs = relationship(
        "TradePositionLegModel",
        back_populates="position",
        cascade="all, delete-orphan",
    )


class TradePositionLegModel(Base):
    __tablename__ = "trade_position_legs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(
        Integer, ForeignKey("trade_positions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_key = Column(String, nullable=False, index=True)
    trading_symbol = Column(String, nullable=False)
    option_type = Column(String, nullable=False)
    strike = Column(Float, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    position = relationship("TradePositionModel", back_populates="legs")


class PositionAlertModel(Base):
    __tablename__ = "position_alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(
        Integer, ForeignKey("trade_positions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    alert_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    acknowledged_at = Column(DateTime)


def init_db():
    Base.metadata.create_all(bind=engine)

    # create_all does not add columns to tables created by earlier versions.
    inspector = inspect(engine)
    rec_columns = {column["name"] for column in inspector.get_columns("recommendations")}
    rec_indexes = {index["name"] for index in inspector.get_indexes("recommendations")}
    with engine.begin() as connection:
        if "symbol" not in rec_columns:
            connection.execute(text("ALTER TABLE recommendations ADD COLUMN symbol VARCHAR"))
        if "ix_recommendations_symbol" not in rec_indexes:
            connection.execute(
                text("CREATE INDEX ix_recommendations_symbol ON recommendations (symbol)")
            )
        connection.execute(
            text("UPDATE recommendations SET symbol = :symbol WHERE symbol IS NULL"),
            {"symbol": config.SYMBOL},
        )
        connection.execute(
            text("UPDATE ticks SET symbol = :symbol WHERE symbol IS NULL"),
            {"symbol": config.SYMBOL},
        )


if __name__ == "__main__":
    init_db()
