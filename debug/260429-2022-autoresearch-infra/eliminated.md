# Eliminated hypotheses — autoresearch infrastructure debug

## Eliminated: heredoc injection via agent commit message
- Hypothesis: agent commit message flowing into the driver's `cat > "$LAST_RESULT_FILE" <<EOFRESULT` could trigger command substitution.
- Test: bash heredoc expansion does not re-parse variable values for `$()` or backticks (single-pass evaluation). Verified with `COMMIT_MSG='hello $(echo INJ)'` → output is literal `hello $(echo INJ)`, not the executed substitution.
- The driver also strips newlines via `tr -d '\n\r'` so a multi-line commit cannot smuggle the EOFRESULT delimiter to terminate the heredoc early.
- Status: **safe**. Heredoc is correctly defensive.
