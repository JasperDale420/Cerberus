# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@empire-trading.com** (or your actual security contact)

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

## Trading System Specific Considerations

Given that Cerberus is a trading system that interfaces with real financial APIs:

- **API Key Security**: Never commit API keys, secrets, or credentials. Use environment variables or secure vaults
- **Order Execution**: Report any issues that could cause unintended trades or incorrect order execution immediately
- **Rate Limiting**: Issues that could cause excessive API calls leading to account suspension are high priority
- **Data Integrity**: Report any issues that could corrupt trade history or position tracking

## Security Best Practices for Users

1. **Never run Cerberus in production with default/example credentials**
2. **Enable paper trading mode** (`--mode paper`) until thoroughly tested
3. **Use the `--order-executor noop` flag** for dry-run verification
4. **Keep dependencies updated** via Dependabot
5. **Run security scans** before deploying: `make security`
6. **Monitor logs** for suspicious activity in `logs/` directory

## Security Updates

Security updates will be released as patch versions (e.g., 1.0.1) and announced via:
- GitHub Security Advisories
- CHANGELOG.md with `[SECURITY]` prefix
- Release notes

Thank you for helping keep Cerberus and its users safe!
