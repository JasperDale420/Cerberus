# Cerberus Operational Runbook

**Purpose**: This runbook documents common failure scenarios, diagnostic procedures, and recovery actions for the Cerberus trading system.

---

## Quick Reference

| Symptom | Likely Cause | Recovery Action |
|---------|--------------|-----------------|
| No trades executing | Risk limits hit | Check `cerberus.db` positions, review risk config |
| High API errors | Rate limiting | Check logs for 429 errors, reduce polling frequency |
| Stuck positions at EOD | Flatten failure | Manually close via Alpaca console, check logs |
| Database locked | Concurrent access | Kill stale processes, verify single instance |
| Missing bar data | Alpaca API issue | Check Alpaca status page, verify credentials |
| Scanner returns empty | Overly strict filters | Review `config/scanner.yaml` thresholds |

---

## Common Failure Scenarios

### 1. Database Corruption or Lock

**Symptoms**:
- `sqlite3.OperationalError: database is locked`
- Trades not persisting to `cerberus.db`
- Analytics queries timeout

**Diagnosis**:
```bash
# Check for multiple processes accessing the database
lsof cerberus.db

# Check database integrity
sqlite3 cerberus.db "PRAGMA integrity_check;"
```

**Recovery**:
1. Stop all Cerberus processes: `pkill -f "python.*src.main"`
2. Backup the database: `cp cerberus.db cerberus.db.backup.$(date +%Y%m%d_%H%M%S)`
3. Run integrity check (above)
4. If corrupted, restore from backup or rebuild from trade logs
5. Restart with single instance only

**Prevention**:
- Ensure only one Cerberus instance runs at a time
- Use file locks if running multiple services
- Regular backups via cron: `0 0 * * * cp /path/to/cerberus.db /backups/cerberus.db.$(date +\%Y\%m\%d)`

---

### 2. API Rate Limit Exhaustion

**Symptoms**:
- `429 Too Many Requests` errors in logs
- Missing market data / features
- Scanner returns no candidates

**Diagnosis**:
```bash
# Check recent API errors
grep "429" logs/cerberus.log | tail -20

# Count API calls per minute
grep "Alpaca.*fetch" logs/cerberus.log | awk '{print $1}' | uniq -c
```

**Recovery**:
1. Pause the system: send `SIGTERM` to main process
2. Review API call patterns in logs
3. Adjust polling intervals in `config/config.yaml`:
   - Increase `scanner.interval_minutes`
   - Reduce bar fetch frequency
4. Consider caching bar data for backtests
5. Restart after cooldown period (typically 1 minute)

**Prevention**:
- Use `--offline-bars-dir` for backtests to avoid API calls
- Enable Unusual Whales only when needed (`unusual_whales.enabled: false`)
- Monitor API usage via Alpaca dashboard

---

### 3. Stuck Positions (EOD Flatten Failure)

**Symptoms**:
- Positions still open after 16:00 ET
- `flatten_all` errors in logs
- Overnight exposure risk

**Diagnosis**:
```bash
# Check current positions via healthcheck
python -m src.main --healthcheck

# Review flatten attempts in logs
grep "flatten" logs/cerberus.log | tail -20
```

**Recovery**:
1. **URGENT**: Manually close positions via [Alpaca web console](https://app.alpaca.markets)
2. Verify closure: `python -m src.main --healthcheck`
3. Review logs for root cause (API errors, insufficient buying power, etc.)
4. Document incident in `artifacts/incidents/YYYYMMDD_stuck_positions.md`

**Prevention**:
- Monitor EOD flatten logs daily
- Set up alerts for positions held past 16:05 ET
- Test flatten logic in paper mode before live deployment

---

### 4. Strategy Signal Dry-Up (No Trades for Extended Period)

**Symptoms**:
- No trades for multiple days
- Scanner returns candidates but no signals
- Strategies not firing

**Diagnosis**:
```bash
# Check scanner output
tail -200 logs/cerberus.log | grep "Scanner result"

# Verify strategy states
sqlite3 cerberus.db "SELECT strategy, COUNT(*) FROM trades WHERE date(timestamp) = date('now') GROUP BY strategy;"

# Check risk manager state
grep "Risk.*reject" logs/cerberus.log | tail -20
```

**Recovery**:
1. Review risk limits in `config/risk.yaml` - may have hit daily/position caps
2. Check scanner thresholds in `config/scanner.yaml` - may be too strict
3. Verify market regime detection: `grep "Regime" logs/cerberus.log | tail -10`
4. Test strategies in backtest mode to verify signal generation
5. Consider adjusting strategy parameters in `config/strategies.yaml`

**Prevention**:
- Monitor daily trade counts via analytics
- Set up alerts for zero-trade days
- Regular strategy parameter reviews

---

### 5. Scheduler Service Not Starting

**Symptoms**:
- `APScheduler` errors on startup
- No automated EOD analytics running
- Jobs not executing at scheduled times

**Diagnosis**:
```bash
# Check scheduler logs
grep "scheduler" logs/cerberus.log | tail -30

# Verify cron expression syntax
python -c "from apscheduler.triggers.cron import CronTrigger; CronTrigger.from_crontab('0 16 * * 1-5')"
```

**Recovery**:
1. Stop scheduler: `pkill -f "python.*scheduler"`
2. Verify `config/config.yaml` schedule syntax
3. Test scheduler in isolation: `python -m src.main --scheduler --run-once`
4. Check for port conflicts (if using web interface)
5. Restart: `python -m src.main --scheduler &`

**Prevention**:
- Use `systemd` or `supervisord` for production deployment
- Monitor scheduler health via process manager
- Test schedule changes in dev environment first

---

### 6. Missing or Stale Market Data

**Symptoms**:
- `KeyError` on feature access
- "No bars available" warnings
- Strategies skip symbols due to missing data

**Diagnosis**:
```bash
# Check bar fetch errors
grep "fetch.*bars.*fail" logs/cerberus.log | tail -20

# Verify Alpaca connectivity
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account
```

**Recovery**:
1. Verify Alpaca credentials in `.env`
2. Check Alpaca [status page](https://status.alpaca.markets)
3. Restart data pipeline: `python -m src.data.pipeline --backfill --start-date YYYY-MM-DD`
4. Consider using cached/offline data for testing

**Prevention**:
- Implement data staleness checks (already in scanner)
- Cache critical bar data locally
- Monitor Alpaca API status

---

## Healthcheck Verification

Run the healthcheck to verify system state:

```bash
python -m src.main --healthcheck
```

**Expected output**:
```
✓ Database connectivity: OK
✓ Alpaca API: OK
✓ Last bar fetch: 2025-12-28 15:59:00 ET (fresh)
✓ Open positions: 2/10
✓ Available cash: $98,234.56
```

**Troubleshooting healthcheck failures**:
- Database connectivity: Check file permissions on `cerberus.db`
- Alpaca API: Verify credentials, check network connectivity
- Stale bar data: Review API rate limits, check logs for fetch errors

---

## Emergency Procedures

### Manual Position Flatten (Emergency)

If automated flatten fails and positions must be closed immediately:

```bash
# Option 1: Via Alpaca web console (recommended for live)
# Navigate to: https://app.alpaca.markets/paper/account/positions
# Click "Close All Positions"

# Option 2: Via Cerberus CLI (if available)
python -m src.main --flatten-all --force
```

### Kill Switch (Stop All Trading)

```bash
# Stop main process
pkill -f "python.*src.main"

# Verify no positions remain
python -m src.main --healthcheck

# Disable scheduler (if running)
pkill -f "python.*scheduler"
```

---

## Logging and Observability

**Log locations**:
- Runtime logs: `logs/cerberus.log` (JSON format)
- Backtest logs: `logs/backtest_*.log`
- Archived logs: `logs/archive/YYYY-MM-DD/`

**Key log patterns to monitor**:
```bash
# Fatal errors
grep "CRITICAL\|FATAL" logs/cerberus.log

# Risk rejections
grep "Risk.*reject" logs/cerberus.log

# Flatten operations
grep "flatten" logs/cerberus.log

# API errors
grep "429\|500\|503" logs/cerberus.log
```

---

## Contact and Escalation

For issues not covered in this runbook:

1. **Check GitHub Issues**: https://github.com/EmpireTrading/Cerberus/issues
2. **Review SECURITY.md**: For security-related incidents
3. **Consult PRD.md**: For design decisions and architectural context
4. **Test in Paper Mode**: Always test fixes in `--mode paper` before live deployment

---

## Maintenance Tasks

| Task | Frequency | Command | Notes |
|------|-----------|---------|-------|
| Database backup | Daily | `cp cerberus.db backups/` | Automate via cron |
| Log rotation | Weekly | `logrotate -f /path/to/logrotate.conf` | Set up with system |
| Dependency updates | Weekly | `pre-commit autoupdate && pip install -U -r requirements.txt` | Review Dependabot PRs |
| Coverage check | Per PR | `make test` | Enforce 70%+ coverage |
| Security scan | Per PR | `make security` | Bandit + detect-secrets |

---

**Last Updated**: 2025-12-28  
**Version**: 1.0.0
