from typing import List, Dict, Any
from datetime import datetime
import yaml
import os
import json
from src.core.logger import StructuredLogger
from src.agent.models import StrategyDailyStats, AgentAction, ActionType
from src.agent.llm import LLMClient
from src.core.config import ConfigLoader

class Agent:
    """
    Offline agent that analyzes strategy performance and adjusts configuration.
    Stage 1: Health & Risk Adjustments (Deterministic)
    Stage 2: Parameter Tuning (LLM Heuristic)
    Stage 3: Code Proposals (LLM Generative)
    """
    def __init__(self, logger: StructuredLogger, config_loader: ConfigLoader, config_path: str = "config/strategies.auto.yaml"):
        self.logger = logger
        self.config_path = config_path
        self.llm_client = LLMClient(config_loader, logger)

    def analyze_performance(self, stats_list: List[StrategyDailyStats]) -> List[AgentAction]:
        """
        Analyzes daily stats and generates actions.
        """
        actions = []
        
        for stats in stats_list:
            # Rule 1: Disable if Expectancy is significantly negative (e.g. < -0.5 R per trade)
            if stats.expectancy < -0.5:
                actions.append(AgentAction(
                    timestamp=datetime.utcnow(),
                    action_type=ActionType.DISABLE_STRATEGY,
                    strategy=stats.strategy,
                    regime=stats.regime,
                    details={"expectancy": stats.expectancy},
                    reason="Expectancy below threshold (-0.5 R)"
                ))
                continue # Skip other checks if disabled

            # Rule 2: Reduce Risk if Drawdown > 20% of allocated R (assuming max_drawdown_r is absolute R drop)
            # Let's say we tolerate 10R drawdown.
            if stats.max_drawdown_r > 10.0:
                actions.append(AgentAction(
                    timestamp=datetime.utcnow(),
                    action_type=ActionType.REDUCE_RISK,
                    strategy=stats.strategy,
                    regime=stats.regime,
                    details={"max_drawdown_r": stats.max_drawdown_r},
                    reason="Drawdown exceeded 10R"
                ))

        return actions

    def tune_parameters(self, stats: StrategyDailyStats, current_config: Dict[str, Any]) -> List[AgentAction]:
        """
        Stage 2: Asks LLM to suggest parameter updates based on stats.
        """
        prompt = f"""
        Analyze the performance of strategy '{stats.strategy}' in regime '{stats.regime}'.
        
        Stats:
        - Win Rate: {stats.winrate:.2f}
        - Avg R: {stats.avg_r:.2f}
        - Max Drawdown R: {stats.max_drawdown_r:.2f}
        - Total PnL R: {stats.total_pnl_r:.2f}
        
        Current Config:
        {json.dumps(current_config, indent=2)}
        
        Suggest parameter changes to improve expectancy or reduce drawdown.
        Return ONLY a JSON object with the new parameter values (subset of config).
        Example: {{"band_sigma": 2.5, "stop_loss_atr": 1.5}}
        If no changes needed, return {{}}.
        """
        
        try:
            response = self.llm_client.complete(prompt, system_prompt="You are a quantitative trading expert.")
            # Clean response (remove markdown code blocks if present)
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            new_params = json.loads(cleaned_response)
            
            if new_params:
                return [AgentAction(
                    timestamp=datetime.utcnow(),
                    action_type=ActionType.TUNE_PARAM,
                    strategy=stats.strategy,
                    regime=stats.regime,
                    details={"new_params": new_params},
                    reason="LLM suggested parameter tuning"
                )]
        except Exception as e:
            self.logger.error("Failed to tune parameters", strategy=stats.strategy, error=str(e))
            
        return []

    def propose_code_changes(self, stats: StrategyDailyStats, strategy_file_path: str) -> List[AgentAction]:
        """
        Stage 3: Generates a new strategy variant based on performance and source code.
        """
        try:
            with open(strategy_file_path, "r") as f:
                source_code = f.read()
                
            prompt = f"""
            The strategy '{stats.strategy}' is underperforming.
            
            Stats:
            - Win Rate: {stats.winrate:.2f}
            - Avg R: {stats.avg_r:.2f}
            - Max Drawdown R: {stats.max_drawdown_r:.2f}
            
            Source Code:
            ```python
            {source_code}
            ```
            
            Propose a new version of this strategy class (e.g., V2) that addresses these issues (e.g., add a filter, change exit logic).
            Return ONLY the full Python code for the new class.
            """
            
            response = self.llm_client.complete(prompt, system_prompt="You are a quantitative developer.")
            # Clean response
            new_code = response.replace("```python", "").replace("```", "").strip()
            
            # Save proposal
            proposal_dir = "src/strategies/proposals"
            os.makedirs(proposal_dir, exist_ok=True)
            proposal_filename = f"{stats.strategy}_v2.py"
            proposal_path = os.path.join(proposal_dir, proposal_filename)
            
            with open(proposal_path, "w") as f:
                f.write(new_code)
                
            return [AgentAction(
                timestamp=datetime.utcnow(),
                action_type=ActionType.CODE_PROPOSAL,
                strategy=stats.strategy,
                regime=stats.regime,
                details={"proposal_path": proposal_path},
                reason="LLM generated code proposal for underperformance"
            )]
            
        except Exception as e:
            self.logger.error("Failed to propose code changes", strategy=stats.strategy, error=str(e))
            
        return []

    def apply_actions(self, actions: List[AgentAction]):
        """
        Applies actions by writing to strategies.auto.yaml.
        """
        if not actions:
            return

        # Load existing auto config or empty
        current_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    current_config = yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.error("Failed to load existing auto config", error=str(e))
        
        # Apply changes
        for action in actions:
            self.logger.info("Applying agent action", action=action)
            
            strat_config = current_config.get(action.strategy, {})
            
            if action.action_type == ActionType.DISABLE_STRATEGY:
                strat_config["enabled"] = False
            elif action.action_type == ActionType.ENABLE_STRATEGY:
                strat_config["enabled"] = True
            elif action.action_type == ActionType.REDUCE_RISK:
                # Halve the risk_per_trade or similar parameter
                # Assuming config structure: strategies: {name: {risk_reward: ..., size: ...}}
                # This depends on how strategies read config. 
                # Let's assume there's a 'risk_factor' or we just disable for now if risk logic isn't granular.
                # Or we set a 'max_risk' override.
                current_risk = strat_config.get("risk_factor", 1.0)
                strat_config["risk_factor"] = current_risk * 0.5
            
            current_config[action.strategy] = strat_config
            action.applied = True

        # Write back
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(current_config, f)
            self.logger.info("Updated auto configuration", path=self.config_path)
        except Exception as e:
            self.logger.error("Failed to write auto config", error=str(e))
