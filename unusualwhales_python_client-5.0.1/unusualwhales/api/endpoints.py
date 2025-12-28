
from .congress import CongressEndpoints
from .contract import ContractEndpoints
from .darkpool import DarkpoolEndpoints
from .earnings import EarningsEndpoints
from .etfs import EtfsEndpoints
from .flow import FlowEndpoints
from .market import MarketEndpoints
from .screener import ScreenerEndpoints
from .seasonality import SeasonalityEndpoints
from .stock import StockEndpoints


class Endpoints:
    @classmethod
    def congress(cls) -> type[CongressEndpoints]:
        return CongressEndpoints

    @classmethod
    def darkpool(cls) -> type[DarkpoolEndpoints]:
        return DarkpoolEndpoints

    @classmethod
    def earnings(cls) -> type[EarningsEndpoints]:
        return EarningsEndpoints

    @classmethod
    def etfs(cls) -> type[EtfsEndpoints]:
        return EtfsEndpoints

    @classmethod
    def market(cls) -> type[MarketEndpoints]:
        return MarketEndpoints

    @classmethod
    def flow(cls) -> type[FlowEndpoints]:
        return FlowEndpoints

    @classmethod
    def contract(cls) -> type[ContractEndpoints]:
        return ContractEndpoints

    @classmethod
    def screener(cls) -> type[ScreenerEndpoints]:
        return ScreenerEndpoints

    @classmethod
    def seasonality(cls) -> type[SeasonalityEndpoints]:
        return SeasonalityEndpoints

    @classmethod
    def stock(cls) -> type[StockEndpoints]:
        return StockEndpoints
