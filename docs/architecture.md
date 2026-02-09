# System Architecture

Comprehensive architecture overview of the Cerberus intraday trading system.

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Data Flows](#data-flows)
- [External Integrations](#external-integrations)
- [Persistence Layer](#persistence-layer)

---

## High-Level Architecture

Cerberus follows a **pipeline architecture** with clear separation between data acquisition, strategy logic, execution, and analytics.

```mermaid
flowchart TB
    subgraph External["External Data Sources"]
        UW[Unusual Whales API<br/>Options Flow Data]
        Alpaca[Alpaca API<br/>Market Data & Trading]
    end
    
    subgraph Core["Core System"]
        Scanner[Scanner<br/>Universe Selection]
        Data[Data Pipeline<br/>Feature Calculation]
        Strategies[Strategy Engine<br/>Signal Generation]
        Execution[Execution Engine<br/>Order Management]
        Position[Position Manager<br/>PnL Tracking]
        Risk[Risk Manager<br/>Pre-Trade Checks]
    end
    
    subgraph Storage["Persistence"]
        DB[(SQLite Database<br/>Trades, Fills, Signals)]
    end
    
    UW -->|Flow Data| Scanner
    Alpaca -->|Market Data<br/>WebSocket| Data
    Scanner -->|Watchlist| Execution
    Data -->|OHLCV + Features| Strategies
    Strategies -->|Signals| Execution
    Execution -->|Risk Check| Risk
    Risk -->|Approved Intents| Execution
    Execution -->|Orders| Alpaca
    Alpaca -->|Fills| Execution
    Execution -->|Fill Events| Position
    Position -->|Trade Complete| DB
    Execution -.->|Signals<br/>Orders<br/>Fills| DB
    
    style External fill:#e1f5ff
    style Core fill:#fff4e1
    style Storage fill:#f0f0f0
```

---

## Core Components

### ExecutionEngine
**Location**: [`src/engine/execution.py`](file:///Users/jacobmcmillan/Empire/Cerberus/src/engine/execution.py)

Central orchestrator managing the main trading loop:
- Receives bars from Alpaca WebSocket
- Routes bars to strategies for signal generation
- Processes signals through risk checks
- Manages watchlist based on scanner results
- Coordinates position and order management

**Key Responsibilities**:
- Bar-by-bar event processing
- Signal routing and validation
- Watchlist management
- End-of-day position flattening

### PositionManager
**Location**: [`src/engine/position_manager.py`](file:///Users/jacobmcmillan/Empire/Cerberus/src/engine/position_manager.py)

Maintains position state and calculates PnL:
- Processes fill events to update positions
- Calculates realized/unrealized PnL
- Evaluates exit conditions (stops, targets, time)
- Tracks MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)

**Key Concept**: Source of truth for all position state (PRD 6.7)

### RiskManager
**Location**: [`src/engine/risk.py`](file:///Users/jacobmcmillan/Empire/Cerberus/src/engine/risk.py)

Pre-trade risk validation:
- Position limit enforcement
- Daily loss limit checks
- Position sizing based on ATR
- Correlation-based exposure limits

**Decision**: Accepts or rejects signals before order submission

### OrderExecutor
**Location**: [`src/engine/orders.py`](file:///Users/jacobmcmillan/Empire/Cerberus/src/engine/orders.py)

Broker API interaction:
- Submits orders to Alpaca
- Cancels orders
- Processes trade updates from WebSocket
- Maps broker order IDs to correlation IDs

### Scanner
**Location**: [`src/scanner/core.py`](file:///Users/jacobmcmillan/Empire/Cerberus/src/scanner/core.py)

Dynamic universe selection:
- Fetches unusual options flow from Unusual Whales
- Applies tradability filters (volume, price, volatility)
- Ranks symbols by flow strength
- Returns top N symbols for watchlist

**Refresh**: Periodic (every N bars, configurable)

### StrategyEngine
**Location**: [`src/engine/strategy_engine.py`](file:///Users/jacobmcmillan/Empire/Cerberus/src/engine/strategy_engine.py)

Strategy coordination:
- Registers strategies at startup
- Routes bars to appropriate strategies based on regime
- Collects signals from all active strategies

---

## Data Flows

### Real-Time Trading Flow

```mermaid
sequenceDiagram
    participant Alpaca as Alpaca<br/>WebSocket
    participant Exec as Execution<br/>Engine
    participant Strat as Strategy<br/>Engine
    participant Risk as Risk<br/>Manager
    participant Order as Order<br/>Executor
    participant Pos as Position<br/>Manager
    participant DB as Database
    
    Alpaca->>Exec: Bar Event
    Exec->>Exec: Update State
    Exec->>Strat: on_bar(symbol, bar)
    Strat->>Strat: Calculate Indicators
    Strat-->>Exec: Signal (optional)
    
    alt Signal Generated
        Exec->>Risk: Validate Signal
        Risk-->>Exec: Approved/Rejected
        Exec->>DB: Persist Signal
        
        alt Approved
            Exec->>Order: Submit Order
            Order->>Alpaca: Place Order
            Alpaca-->>Order: Order Confirmation
            Order-->>Exec: Order Submitted
        end
    end
    
    Alpaca->>Order: Trade Update (Fill)
    Order->>Exec: Fill Event
    Exec->>Pos: Update Position
    Pos-->>Exec: Fill Decision
    Exec->>DB: Persist Fill
    
    alt Position Closed
        Exec->>DB: Persist Trade
    end
```

### Scanner Integration Flow

```mermaid
flowchart LR
    A[Scanner Interval<br/>Reached] --> B{Fetch UW Flow}
    B -->|Success| C[Apply Filters]
    B -->|Failure| D[Log Error<br/>Keep Current Watchlist]
    C --> E[Rank Symbols]
    E --> F[Top N Symbols]
    F --> G[Update Watchlist]
    G --> H[Execution Engine<br/>Processes New Symbols]
    D --> H
    
    style A fill:#e8f4f8
    style G fill:#d4edda
    style D fill:#f8d7da
```

---

## External Integrations

### Alpaca API
**Purpose**: Market data and trade execution

**Endpoints Used**:
- **WebSocket**: Real-time bars and trade updates
- **REST**: Historical data, account info, order management

**Authentication**: API key + secret (environment variables)

**Rate Limits**: 200 requests/minute (REST)

### Unusual Whales API
**Purpose**: Options flow data for scanner

**Endpoints Used**:
- `/api/stock/flow`: Unusual options activity

**Authentication**: API key (environment variable)

**Rate Limits**: Varies by subscription tier

---

## Persistence Layer

### Database Schema

```mermaid
erDiagram
    Trade ||--o{ Fill : contains
    Trade {
        int id PK
        string symbol
        string strategy
        datetime entry_time
        datetime exit_time
        decimal entry_price
        decimal exit_price
        decimal pnl_dollars
        decimal pnl_r_multiple
        string correlation_id
    }
    
    Fill {
        int id PK
        string symbol
        string side
        decimal qty
        decimal price
        datetime timestamp
        string correlation_id
        string broker_order_id
    }
    
    Signal {
        int id PK
        string symbol
        string strategy
        string side
        decimal entry_price
        decimal stop_loss
        decimal take_profit
        string correlation_id
        boolean accepted
        string rejection_reason
        datetime timestamp
    }
    
    Order {
        int id PK
        string symbol
        string side
        decimal qty
        string status
        string broker_order_id
        string correlation_id
        datetime submitted_at
    }
    
    ScannerSnapshot {
        int id PK
        datetime timestamp
        json symbols
        json flow_data
    }
```

### Database Technology

**Engine**: SQLite  
**Location**: `cerberus.db` (root directory)  
**Fault Tolerance**: PRD 11.4 (buffered writes, configurable fail modes)

**Tables**:
- **Trade**: Completed round-trip trades
- **Fill**: Individual fill events from broker
- **Signal**: All signals (accepted + rejected)
- **Order**: Order lifecycle tracking
- **ScannerSnapshot**: Historical watchlist snapshots

---

## Key Design Principles

### 1. Intraday-Only Positions
All positions closed by 16:00 ET (configurable). No overnight holdings.

### 2. Correlation ID Tracking
Every signal → order → fill → trade tracked via `correlation_id` for end-to-end observability (PRD 2.1, 3.2).

### 3. Best-Effort Observability
Database writes and logging are best-effort. Trading decisions never blocked by DB failures (PRD 11.2, 11.4).

### 4. Multi-Axis Regime System
Strategy activation determined by 5 orthogonal axes (not a single BULL/BEAR/CHOP label):
- **Trend**: UP/DOWN/FLAT (SPY cumulative return)
- **Volatility**: LOW/NORMAL/HIGH/SHOCK (realized vol z-score)
- **Liquidity**: GOOD/THIN/STRESSED (dollar volume / range)
- **Risk**: RISK_ON/NEUTRAL/RISK_OFF (VXX momentum)
- **Session**: OPENING/MIDDAY/POWER_HOUR/CLOSE (time-of-day)

Each strategy defines an `ActivationPolicy` specifying which axis states permit trading.

### 5. Source of Truth
PositionManager is the single source of truth for all position state. Broker state reconciled periodically but local state takes precedence for decisions (PRD 6.7).

---

## Related Documentation

- [Order Execution Flow](file:///Users/jacobmcmillan/Empire/Cerberus/docs/order_flow.md) - Detailed signal-to-trade workflow
- [Strategy Development Guide](file:///Users/jacobmcmillan/Empire/Cerberus/docs/strategy_guide.md) - How to create custom strategies
- [README](file:///Users/jacobmcmillan/Empire/Cerberus/README.md) - Setup and getting started
- [Runbook](file:///Users/jacobmcmillan/Empire/Cerberus/docs/runbook.md) - Operational procedures
