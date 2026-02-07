# Screener Snapshot Capture - Setup Guide

## Overview

This system captures daily screener data from Alpaca for historical backtest replay.

## Quick Start

**Manual capture:**
```bash
python scripts/capture_screener_snapshot.py
```

**Output:** `data/screener_snapshots/YYYY-MM-DD.jsonl`

---

## Cron Setup (macOS/Linux)

Run daily at **4:05 PM Eastern** (after market close):

```bash
# Edit crontab
crontab -e

# Add this line (adjust path to your Cerberus directory):
5 21 * * 1-5 cd /Users/jacobmcmillan/Empire/Cerberus && /opt/homebrew/Caskroom/miniforge/base/bin/python scripts/capture_screener_snapshot.py >> logs/screener_capture.log 2>&1
```

**Note:** `21:05 UTC` = `4:05 PM ET` (adjust for DST if needed)

---

## Snapshot Format

```json
{
  "timestamp": "2025-12-31T05:59:47.056016+00:00",
  "most_actives": ["PFSA", "OCG", "ZSL", ...],
  "gainers": ["MRNOW", "EKSO", "AEHL", ...],
  "losers": ["RPT", "BNAIW", "EDBLW", ...]
}
```

---

## Future: Backtest Replay

Once you have historical snapshots, you can modify `UniverseBuilder` to load the snapshot for each simulated date instead of calling the live API.
