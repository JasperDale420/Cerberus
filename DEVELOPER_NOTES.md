# Developer Notes

## Recommended Extensions

### VS Code
- **Python**: Recommended for all Python development.
- **Ruff**: Integrated linting and formatting. Very fast.
- **SonarLint**: Connect to SonarQube/SonarCloud for real-time feedback.
- **Mypy**: For type checking highlights.

### JetBrains / PyCharm
- **SonarLint**: Plugin for real-time analysis.
- **Ruff**: Plugin available for better integration.

## Local Workflow Tips

- **Pre-commit**: Always run `pre-commit install` after cloning. This ensures you never commit failing code.
- **Test Coverage**: Run `pytest` locally to see if you broke any existing functionality. The configuration ensures 100% (or high) coverage is visible.
- **Secrets**: If `detect-secrets` blocks a commit, verify it's not a real secret. If it's a false positive, update `.secrets.baseline` using `detect-secrets scan --update .secrets.baseline`.

## Paper-Live Testing Commands

Use the harness to verify system integrity before valid deployment.

### Full Suite Run
To run all checks sequentially (manual):

```bash
# Happy Path
python paper_live_harness.py --scenario happy --duration 2 --inject-signal

# Failure Injection (Expect "PASS (Failures Caught)")
python paper_live_harness.py --scenario failure --duration 2 --inject-signal

# Risk Breach (Expect "PASS (Risk Blocked)")
python paper_live_harness.py --scenario risk --duration 2 --inject-signal
```

**Note**: Ensure `ALPACA_PAPER=true` is set in your `.env`.
