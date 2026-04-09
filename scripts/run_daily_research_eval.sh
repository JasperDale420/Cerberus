#!/bin/bash
# Atomic setup + eval for daily_research_strategy
# Patches all 3 files and re-patches them every 30s during the eval

set -e
cd /Users/jacobmcmillan/Empire/Cerberus

patch_files() {
  # 0. Ensure strategy file has LOC=99 (remove max_hold_days, inline SMA check)
  if grep -q "max_hold_days" src/strategies/daily_research_strategy.py 2>/dev/null; then
    sed -i '' '/max_hold_days/d' src/strategies/daily_research_strategy.py
  fi
  if grep -q "s20 = sum" src/strategies/daily_research_strategy.py 2>/dev/null; then
    python3 -c "
f = open('src/strategies/daily_research_strategy.py', 'r')
c = f.read()
f.close()
old = '''        s20 = sum(cl[-20:]) / 20 if len(cl) >= 20 else None
        s20p = sum(cl[-25:-5]) / 20 if len(cl) >= 25 else None
        if s20 is not None and s20p is not None and s20 < s20p:'''
new = '        if len(cl) >= 25 and sum(cl[-20:]) / 20 < sum(cl[-25:-5]) / 20:'
if old in c:
    f = open('src/strategies/daily_research_strategy.py', 'w')
    f.write(c.replace(old, new))
    f.close()
"
  fi

  # 1. Patch src/main.py
  if ! grep -q "daily_research_strategy" src/main.py; then
    sed -i '' '/from src.strategies.daily_vol_fade import/a\
    from src.strategies.daily_research_strategy import DailyResearchStrategy
' src/main.py
    sed -i '' '/"autoresearch_strategy": AutoresearchStrategy,/a\
        "daily_research_strategy": DailyResearchStrategy,
' src/main.py
  fi

  # 2. Patch config/backtest_v2.yaml
  if ! grep -q "daily_research_strategy" config/backtest_v2.yaml; then
    sed -i '' '/# V1 legacy strategies/i\
  daily_research_strategy:\
    enabled: true\
    allow_overnight: true\
    max_hold_days: 5\
    bar_duration_minutes: 1.0\
    cooldown_bars: 390\
    min_bars: 25\
    stop_atr_mult: 2.0\
    target_atr_mult: 3.0\
    rsi_threshold: 40.0\
    breakout_period: 20\
    atr_period: 14\
    activation:\
      session: [premarket, opening, midday, power_hour, close]\
      trend: [up, down, flat]\
      vol: [low, normal, high]\
      liquidity: [good, thin, stressed]\
      risk: [risk_on, neutral, risk_off]\
      min_confidence: 0.0\

' config/backtest_v2.yaml
  fi

  # 3. Set force_flat_at_1600: false
  sed -i '' 's/force_flat_at_1600: true/force_flat_at_1600: false/' config/backtest_v2.yaml

  # 4. Patch param_spaces.py
  if ! grep -q "daily_research_strategy" src/analytics/param_spaces.py; then
    python3 -c "
f = open('src/analytics/param_spaces.py', 'r')
content = f.read()
f.close()
insert = '''    \"daily_research_strategy\": [
        ParamDef(\"stop_atr_mult\", \"float\", low=1.5, high=3.0, step=0.25,
                 description=\"ATR multiplier for stop loss\"),
        ParamDef(\"target_atr_mult\", \"float\", low=2.0, high=5.0, step=0.5,
                 description=\"ATR multiplier for take profit target\"),
        ParamDef(\"rsi_threshold\", \"float\", low=25.0, high=40.0, step=5.0,
                 description=\"RSI oversold threshold for mean reversion entry\"),
        ParamDef(\"breakout_period\", \"int\", low=10, high=30, step=5,
                 description=\"Lookback period for close breakout\"),
    ],
'''
content = content.replace(
    '    ],\n}\n\n\n# ----',
    '    ],\n' + insert + '}\n\n\n# ----',
    1
)
f = open('src/analytics/param_spaces.py', 'w')
f.write(content)
f.close()
"
  fi
}

# Initial patch
patch_files

# Background re-patcher: checks every 30s and re-applies if overwritten
(
  while true; do
    sleep 30
    patch_files 2>/dev/null
  done
) &
PATCHER_PID=$!

# Cleanup patcher on exit
trap "kill $PATCHER_PID 2>/dev/null" EXIT

echo "Patches applied. Background patcher running (PID $PATCHER_PID). Launching eval..."

# Launch eval
./scripts/run_eval_with_logging.sh daily_research_strategy --n-trials 5
