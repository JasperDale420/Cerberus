from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.core.settings import get_settings
from src.data.alpaca import AlpacaClient
from src.data.api_client import CentralApiClient

if TYPE_CHECKING:
    from src.agent.bars_provider import JsonlBarsProvider


class UniverseBuilder:
    """
    Constructs the universe of symbols to scan.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        config_path_or_dir: Optional[str] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        central_api_client: Optional[CentralApiClient] = None,
        clock: Optional[Callable[[], datetime]] = None,
        offline_bars_provider: Optional["JsonlBarsProvider"] = None,
    ):
        # Respect caller-provided config (preferred), otherwise load from the same path used by the app.
        self.config = config if isinstance(config, dict) else config_loader.load_config(config_path_or_dir)
        self.logger = logger
        self.alpaca_client = alpaca_client
        self.central_api_client = central_api_client
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
        self.offline_bars_provider = offline_bars_provider
        runtime = get_settings()
        self.use_gateway_data = runtime.use_gateway_data
        self.allow_legacy_failover = bool(runtime.cerberus_failover_to_legacy)

    def _get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Any:
        """Fetch historical bars using selected backend, with optional failover."""
        if self.use_gateway_data and self.central_api_client is not None:
            try:
                return self.central_api_client.get_alpaca_bars(
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                )
            except Exception as e:
                self.logger.warning(
                    "Gateway universe bars fetch failed",
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(e),
                    failover=self.allow_legacy_failover,
                )
                if not self.allow_legacy_failover:
                    raise
        if self.alpaca_client is None:
            return []
        return self.alpaca_client.get_historical_bars(symbol, start, end, timeframe=timeframe)

    def _get_screener_most_actives(self, top: int) -> List[str]:
        """Fetch most actives from selected backend with fallback."""
        if self.use_gateway_data and self.central_api_client is not None:
            try:
                return self.central_api_client.get_alpaca_most_actives(top=top)
            except Exception as e:
                self.logger.warning(
                    "Gateway screener most_actives failed",
                    error=str(e),
                    failover=self.allow_legacy_failover,
                )
                if not self.allow_legacy_failover:
                    raise
        if self.alpaca_client is None:
            return []
        return self.alpaca_client.get_most_actives(top=top)

    def _get_screener_movers(self, top: int) -> Dict[str, List[str]]:
        """Fetch movers from selected backend with fallback."""
        if self.use_gateway_data and self.central_api_client is not None:
            try:
                return self.central_api_client.get_alpaca_movers(top=top)
            except Exception as e:
                self.logger.warning(
                    "Gateway screener movers failed",
                    error=str(e),
                    failover=self.allow_legacy_failover,
                )
                if not self.allow_legacy_failover:
                    raise
        if self.alpaca_client is None:
            return {"gainers": [], "losers": []}
        return self.alpaca_client.get_movers(top=top)

    def _dedupe_preserve_order(self, symbols: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in symbols:
            sym = str(s).strip().upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
        return out

    def _load_symbol_file(self, path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            fallback = None
            if self.offline_bars_provider is not None:
                offline_dir = getattr(self.offline_bars_provider, "bars_dir", None)
                if offline_dir is not None:
                    candidate = Path(offline_dir) / p.name
                    if candidate.exists():
                        fallback = candidate

            if fallback is None:
                self.logger.warning("Universe static file missing", path=str(p))
                return []

            self.logger.info(
                "Universe static file fallback used",
                path=str(p),
                fallback=str(fallback),
            )
            p = fallback
        lines = []
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Allow CSV lines; take first token.
            token = line.split(",")[0].strip()
            if token:
                lines.append(token.upper())
        return lines

    def _get_symbol_volume(self, symbol: str, start: datetime, end: datetime) -> Optional[float]:
        """Get last bar volume for symbol from Alpaca or offline provider."""
        # Try Alpaca first
        if self.alpaca_client is not None or self.central_api_client is not None:
            try:
                bars = self._get_historical_bars(symbol, start, end, timeframe="1Day")
                if isinstance(bars, dict) and "bars" in bars:
                    bars = bars["bars"]
                if isinstance(bars, list) and bars:
                    b = bars[-1]
                    if isinstance(b, dict):
                        v = b.get("v") if b.get("v") is not None else b.get("volume")
                    else:
                        v = getattr(b, "v", None) or getattr(b, "volume", None)
                    if v is not None:
                        return float(v)
            except Exception as e:
                self.logger.warning(
                    "Dynamic universe volume fetch failed (primary source)",
                    symbol=symbol,
                    error=str(e),
                )

        # Fallback to offline bars provider
        if self.offline_bars_provider is not None:
            try:
                # Try daily bars first
                bars = list(self.offline_bars_provider.get_bars(symbol, start, end, timeframe="1Day"))
                if bars:
                    b = bars[-1]
                    v = getattr(b, "volume", None)
                    if v is not None:
                        return float(v)

                # Fallback: aggregate volume from 1Min bars
                bars_1min = list(self.offline_bars_provider.get_bars(symbol, start, end, timeframe="1Min"))
                if bars_1min:
                    total_vol = sum(getattr(b, "volume", 0) or 0 for b in bars_1min)
                    if total_vol > 0:
                        return float(total_vol)
            except Exception as e:
                self.logger.warning(
                    "Dynamic universe volume fetch failed (offline)",
                    symbol=symbol,
                    error=str(e),
                )

        return None

    def build_universe(self) -> List[str]:
        """
        Returns a list of symbols.
        Currently supports explicit symbols; static/dynamic sources can be added behind config.
        """
        universe_config = self.config.get("universe", [])
        universe: List[str] = []

        explicit: List[str] = []
        static_files: List[str] = []
        dyn_cfg: Dict[str, Any] = {}
        if isinstance(universe_config, dict):
            explicit = cast(List[str], universe_config.get("symbols", []))
            static_files = (
                cast(List[str], universe_config.get("static_files", [])) if universe_config.get("static_files") else []
            )
            dyn_cfg = cast(Dict[str, Any], universe_config.get("dynamic", {})) if universe_config.get("dynamic") else {}
        else:
            explicit = cast(List[str], universe_config)

        explicit_added = [str(s).upper() for s in explicit if s]
        universe.extend(explicit_added)

        static_added: List[str] = []
        for fp in static_files:
            static_added.extend(self._load_symbol_file(str(fp)))
        universe.extend(static_added)

        # Dynamic: previous day top volume among candidates (deterministic).
        prev_vol_cfg = dyn_cfg.get("previous_day_top_volume") if isinstance(dyn_cfg, dict) else None
        has_data_source = (
            self.alpaca_client is not None
            or self.central_api_client is not None
            or self.offline_bars_provider is not None
        )
        if isinstance(prev_vol_cfg, dict) and bool(prev_vol_cfg.get("enabled", False)) and has_data_source:
            top_n = int(prev_vol_cfg.get("top_n", 0))
            if top_n > 0:
                candidates = prev_vol_cfg.get("candidates")
                cand_list: List[str] = []
                if isinstance(candidates, list) and candidates:
                    cand_list = [str(x).upper() for x in candidates if x]
                else:
                    cand_list = list(universe)

                from concurrent.futures import ThreadPoolExecutor, as_completed
                from datetime import timedelta

                end = self.clock()
                start = end - timedelta(days=int(prev_vol_cfg.get("lookback_days", 7)))

                vols: List[tuple[str, float]] = []
                sorted_candidates = sorted(set(cand_list))

                # Use ThreadPoolExecutor for parallel volume fetching
                max_workers = min(10, len(sorted_candidates))
                if max_workers > 0:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(self._get_symbol_volume, sym, start, end): sym for sym in sorted_candidates
                        }
                        for future in as_completed(futures):
                            sym = futures[future]
                            try:
                                vol = future.result()
                                if vol is not None and vol > 0:
                                    vols.append((sym, vol))
                            except Exception as e:
                                self.logger.warning(
                                    "Dynamic universe volume fetch failed",
                                    symbol=sym,
                                    error=str(e),
                                )

                vols_sorted = sorted(vols, key=lambda kv: (-kv[1], kv[0]))
                dynamic_added = [sym for sym, _v in vols_sorted[:top_n]]
                universe.extend(dynamic_added)
            else:
                dynamic_added = []
        else:
            dynamic_added = []

        # Dynamic: Alpaca Screener API (most actives + movers)
        # Only runs in LIVE mode - skipped for backtest since API returns current-day data
        is_backtest = self.offline_bars_provider is not None
        screener_cfg = dyn_cfg.get("screener") if isinstance(dyn_cfg, dict) else None
        if (
            isinstance(screener_cfg, dict)
            and bool(screener_cfg.get("enabled", False))
            and (self.alpaca_client is not None or self.central_api_client is not None)
            and not is_backtest  # Skip in backtest - use volume ranking instead
        ):
            screener_added: List[str] = []

            # Get most active stocks by volume
            actives_n = int(screener_cfg.get("most_actives_top_n", 0))
            if actives_n > 0:
                try:
                    actives = self._get_screener_most_actives(top=actives_n)
                    screener_added.extend(actives)
                except Exception as e:
                    self.logger.warning(
                        "Screener most_actives failed",
                        error=str(e),
                    )

            # Get top movers (gainers + losers)
            movers_n = int(screener_cfg.get("movers_top_n", 0))
            if movers_n > 0:
                try:
                    movers = self._get_screener_movers(top=movers_n)
                    screener_added.extend(movers.get("gainers", []))
                    screener_added.extend(movers.get("losers", []))
                except Exception as e:
                    self.logger.warning(
                        "Screener movers failed",
                        error=str(e),
                    )

            # Dedupe and add to dynamic_added
            for sym in screener_added:
                if sym and sym not in dynamic_added:
                    dynamic_added.append(sym)

            universe.extend(screener_added)

        universe = self._dedupe_preserve_order(universe)

        try:
            self.logger.info(
                "Universe built",
                explicit_count=len(set(explicit_added)),
                static_count=len(set(static_added)),
                dynamic_count=len(set(dynamic_added)),
                total=len(universe),
            )
        except Exception:
            pass

        if not universe:
            self.logger.error("Universe is empty; refusing to proceed")
            raise ValueError("Universe is empty; check config/universe.yaml and static files")

        return universe

    def get_universe(self) -> List[str]:
        # Backwards-compatible alias
        return self.build_universe()
