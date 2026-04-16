"""Experiment Registry — single source of truth for autoresearch campaigns.

Tracks strategy genealogy from discovery -> refinement -> graduation -> live.
Stores in-sample, holdout, and paper trading metrics for each experiment.
Accumulates learnings across campaigns for knowledge transfer.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from src.analysis.schema import ResearchExperiment, ResearchLearning


class ExperimentRegistry:
    def __init__(self, db_session_factory):
        """Initialize with a SQLAlchemy session factory (callable that returns a Session)."""
        self._session_factory = db_session_factory

    def register_discovery(
        self,
        strategy_name: str,
        campaign_id: str,
        archetype: str,
        commit_hash: str,
        strategy_file_path: Optional[str] = None,
        seed_template: Optional[str] = None,
        is_metrics: Optional[dict] = None,
        holdout_metrics: Optional[dict] = None,
        regime_breakdown: Optional[dict] = None,
        param_stability: Optional[dict] = None,
        config_snapshot: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Register a completed discovery campaign. Returns experiment ID."""
        is_m = is_metrics or {}
        ho_m = holdout_metrics or {}

        # Compute strategy file hash
        file_hash = None
        if strategy_file_path and Path(strategy_file_path).exists():
            file_hash = hashlib.sha256(Path(strategy_file_path).read_bytes()).hexdigest()[:16]

        # Determine status from holdout results
        holdout_pf = ho_m.get("pf", ho_m.get("profit_factor", 0.0))
        status = "holdout_pass" if holdout_pf > 1.0 else "holdout_fail"

        exp = ResearchExperiment(
            strategy_name=strategy_name,
            campaign_id=campaign_id,
            archetype=archetype,
            seed_template=seed_template,
            commit_hash=commit_hash,
            strategy_file_hash=file_hash,
            phase="discovery",
            status=status,
            is_composite_score=is_m.get("composite_score", 0.0),
            is_min_pf=is_m.get("min_pf", 0.0),
            is_avg_pf=is_m.get("avg_pf", 0.0),
            is_sharpe=is_m.get("sharpe", 0.0),
            is_net_pnl=is_m.get("net_pnl", 0.0),
            is_total_trades=is_m.get("total_trades", 0),
            is_max_param_cv=is_m.get("max_param_cv", 0.0),
            holdout_pf=holdout_pf,
            holdout_sharpe=ho_m.get("sharpe", 0.0),
            holdout_net_pnl=ho_m.get("net_pnl", ho_m.get("pnl", 0.0)),
            holdout_total_trades=ho_m.get("total_trades", ho_m.get("trades", 0)),
            holdout_start=ho_m.get("start", ""),
            holdout_end=ho_m.get("end", ""),
            regime_breakdown_json=json.dumps(regime_breakdown) if regime_breakdown else None,
            best_regime=regime_breakdown.get("best", "") if regime_breakdown else None,
            worst_regime=regime_breakdown.get("worst", "") if regime_breakdown else None,
            config_snapshot_json=json.dumps(config_snapshot) if config_snapshot else None,
            param_stability_json=json.dumps(param_stability) if param_stability else None,
            notes=notes,
        )

        with self._session_factory() as session:
            session.add(exp)
            session.commit()
            return exp.id

    def update_status(self, experiment_id: int, new_status: str, **kwargs) -> None:
        """Transition experiment status with optional metric updates."""
        with self._session_factory() as session:
            exp = session.get(ResearchExperiment, experiment_id)
            if exp is None:
                raise ValueError(f"Experiment {experiment_id} not found")
            exp.status = new_status
            exp.updated_at = datetime.utcnow()
            for key, value in kwargs.items():
                if hasattr(exp, key):
                    setattr(exp, key, value)
            session.commit()

    def get_holdout_survivors(self, min_holdout_pf: float = 1.0, min_trades: int = 50) -> list[ResearchExperiment]:
        """Return experiments that passed holdout."""
        with self._session_factory() as session:
            stmt = (
                select(ResearchExperiment)
                .where(
                    ResearchExperiment.status == "holdout_pass",
                    ResearchExperiment.holdout_pf >= min_holdout_pf,
                    ResearchExperiment.holdout_total_trades >= min_trades,
                )
                .order_by(ResearchExperiment.holdout_net_pnl.desc())
            )
            return list(session.scalars(stmt).all())

    def get_refinement_candidates(self) -> list[ResearchExperiment]:
        """Return experiments eligible for refinement."""
        with self._session_factory() as session:
            stmt = (
                select(ResearchExperiment)
                .where(ResearchExperiment.status == "holdout_pass")
                .order_by(ResearchExperiment.holdout_net_pnl.desc())
            )
            return list(session.scalars(stmt).all())

    def get_paper_candidates(self) -> list[ResearchExperiment]:
        """Return experiments eligible for paper trading."""
        with self._session_factory() as session:
            stmt = (
                select(ResearchExperiment)
                .where(ResearchExperiment.status == "refined_pass")
                .order_by(ResearchExperiment.holdout_net_pnl.desc())
            )
            return list(session.scalars(stmt).all())

    def record_learning(self, campaign_id: str, archetype: str, category: str, content: str) -> None:
        """Record a learning from a campaign."""
        with self._session_factory() as session:
            learning = ResearchLearning(
                campaign_id=campaign_id,
                archetype=archetype,
                category=category,
                content=content,
            )
            session.add(learning)
            session.commit()

    def get_learnings(self, archetype: Optional[str] = None, limit: int = 20) -> list[ResearchLearning]:
        """Retrieve learnings, optionally filtered by archetype."""
        with self._session_factory() as session:
            stmt = select(ResearchLearning).order_by(ResearchLearning.created_at.desc())
            if archetype:
                stmt = stmt.where(ResearchLearning.archetype == archetype)
            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def get_genealogy(self, campaign_id: str) -> list[ResearchExperiment]:
        """Trace lineage for a campaign through parent links."""
        with self._session_factory() as session:
            result = []
            current_id = campaign_id
            seen = set()
            while current_id and current_id not in seen:
                seen.add(current_id)
                stmt = select(ResearchExperiment).where(ResearchExperiment.campaign_id == current_id)
                exp = session.scalars(stmt).first()
                if exp:
                    result.append(exp)
                    current_id = exp.parent_campaign_id
                else:
                    break
            return result

    def get_all(self) -> list[ResearchExperiment]:
        """Return all experiments, sorted by creation date."""
        with self._session_factory() as session:
            stmt = select(ResearchExperiment).order_by(ResearchExperiment.created_at.desc())
            return list(session.scalars(stmt).all())

    def export_summary(self) -> list[dict]:
        """Full registry dump for CLI display."""
        experiments = self.get_all()
        return [
            {
                "id": e.id,
                "campaign": e.campaign_id,
                "strategy": e.strategy_name,
                "archetype": e.archetype,
                "phase": e.phase,
                "status": e.status,
                "is_score": e.is_composite_score,
                "is_min_pf": e.is_min_pf,
                "holdout_pf": e.holdout_pf,
                "holdout_pnl": e.holdout_net_pnl,
                "holdout_sharpe": e.holdout_sharpe,
                "holdout_trades": e.holdout_total_trades,
                "paper_days": e.paper_days,
                "paper_pnl": e.paper_net_pnl,
                "commit": e.commit_hash,
                "notes": (e.notes or "")[:60],
            }
            for e in experiments
        ]
