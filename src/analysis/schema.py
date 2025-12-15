from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String)
    strategy: Mapped[str] = mapped_column(String)
    regime_at_entry: Mapped[str] = mapped_column(String)
    regime_at_exit: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    side: Mapped[str] = mapped_column(String)  # long/short
    qty: Mapped[float] = mapped_column(Float)
    entry_time: Mapped[datetime] = mapped_column(DateTime)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_estimate: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_gross: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_net: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    initial_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mae_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mfe_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    holding_period_seconds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    features_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    correlation_id: Mapped[str] = mapped_column(String, index=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String)
    strategy: Mapped[str] = mapped_column(String)
    regime: Mapped[str] = mapped_column(String)
    time: Mapped[datetime] = mapped_column(DateTime)
    raw_side: Mapped[str] = mapped_column(String)
    raw_size: Mapped[float] = mapped_column(Float)
    accepted: Mapped[bool] = mapped_column(Boolean)
    rejection_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    meta_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    correlation_id: Mapped[str] = mapped_column(String, index=True)
    trade_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("trades.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String)
    side: Mapped[str] = mapped_column(String)  # buy/sell
    qty: Mapped[float] = mapped_column(Float)
    type: Mapped[str] = mapped_column(String)  # market/limit
    limit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String)
    time_placed: Mapped[datetime] = mapped_column(DateTime)
    time_last_update: Mapped[datetime] = mapped_column(DateTime)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    meta_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    fills: Mapped[List["Fill"]] = relationship(back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    fill_time: Mapped[datetime] = mapped_column(DateTime)
    fill_price: Mapped[float] = mapped_column(Float)
    fill_qty: Mapped[float] = mapped_column(Float)

    order: Mapped["Order"] = relationship(back_populates="fills")


class RegimeHistory(Base):
    __tablename__ = "regime_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    regime: Mapped[str] = mapped_column(String)
    index_symbol: Mapped[str] = mapped_column(String)
    index_price: Mapped[float] = mapped_column(Float)
    cum_ret: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trend_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ScannerSnapshot(Base):
    __tablename__ = "scanner_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    regime: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String)
    scanner_score: Mapped[float] = mapped_column(Float)
    strategies_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    features_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class StrategyStatsDaily(Base):
    __tablename__ = "strategy_stats_daily"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime)
    strategy: Mapped[str] = mapped_column(String)
    regime: Mapped[str] = mapped_column(String)
    n_trades: Mapped[int] = mapped_column(Integer, default=0)
    winrate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_r: Mapped[float] = mapped_column(Float, default=0.0)
    median_r: Mapped[float] = mapped_column(Float, default=0.0)
    std_r: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_r: Mapped[float] = mapped_column(Float, default=0.0)
    max_consecutive_losers: Mapped[int] = mapped_column(Integer, default=0)
    pnl_r_total: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(
        Float, default=0.0
    )  # Added for dollar aggregation
    std_dev_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    z_score: Mapped[float] = mapped_column(Float, default=0.0)


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    action_type: Mapped[str] = mapped_column(String)  # DISABLE_STRATEGY, TUNE_PARAM
    strategy: Mapped[str] = mapped_column(String)
    regime: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    details_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
