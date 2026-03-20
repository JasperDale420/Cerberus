from src.backtest.fill_models.fixed import FixedSlippageFillModel
from src.backtest.fill_models.protocol import FillModel, FillResult
from src.backtest.fill_models.volume_aware import VolumeAwareFillModel

__all__ = ["FillModel", "FillResult", "FixedSlippageFillModel", "VolumeAwareFillModel", "create_fill_model"]


def create_fill_model(config: dict) -> FillModel:
    """Factory: build FillModel from backtest config."""
    backtest_cfg = config.get("backtest", {})
    model_type = backtest_cfg.get("fill_model", "fixed")
    params = backtest_cfg.get("fill_model_params", {})
    risk_cfg = config.get("risk", {})

    if model_type == "volume_aware":
        from src.backtest.fill_models.volume_aware import VolumeAwareFillModel

        return VolumeAwareFillModel(
            base_slippage_bps=params.get("base_slippage_bps", risk_cfg.get("slippage_bps", 2.0)),
            impact_coefficient=params.get("impact_coefficient", 200.0),
            commission_per_share=params.get("commission_per_share", risk_cfg.get("commission_per_share", 0.001)),
            min_commission=params.get("min_commission", risk_cfg.get("min_commission", 0.0)),
            max_slippage_bps=params.get("max_slippage_bps", 50.0),
        )
    else:
        from src.backtest.fill_models.fixed import FixedSlippageFillModel

        return FixedSlippageFillModel(
            slippage_bps=risk_cfg.get("slippage_bps", 2.0),
            commission_per_share=risk_cfg.get("commission_per_share", 0.001),
            min_commission=risk_cfg.get("min_commission", 0.0),
        )
