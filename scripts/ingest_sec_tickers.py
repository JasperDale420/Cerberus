#!/usr/bin/env python3
"""
Script to ingest SEC tickers from https://www.sec.gov/files/company_tickers.json
and store them in the database.
Propagates CIK mapping for fundamental data fetching.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.db import DatabaseDatabase as Database
from src.analysis.schema import SecTicker
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


def ingest_sec_tickers():
    logger = StructuredLogger("ingest_sec_tickers")
    config_loader = ConfigLoader()
    db = Database(config_loader, logger)
    # Drop table to ensure schema update if changed (since we are in restoration/dev mode)
    SecTicker.__table__.drop(bind=db.engine, checkfirst=True)  # type: ignore
    db.init_db()  # Re-create tables

    url = "https://www.sec.gov/files/company_tickers.json"
    logger.info("Fetching SEC tickers", url=url)

    # SEC requires a User-Agent header with email
    headers = {"User-Agent": "EmpireTrading/1.0 (jacob@empire.com)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:  # nosec
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.error("Failed to fetch SEC tickers", error=str(e))
        sys.exit(1)

    # Data format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    tickers_list = list(data.values())
    logger.info("Fetched tickers", count=len(tickers_list))

    with db.get_session() as session:
        count = 0
        for item in tickers_list:
            cik = str(item["cik_str"]).zfill(10)  # Pad to 10 digits
            ticker = item["ticker"].upper()
            title = item["title"]

            # Upsert
            try:
                obj = SecTicker(
                    cik=cik,
                    ticker=ticker,
                    title=title,
                    updated_at=datetime.now(timezone.utc),
                )
                session.merge(obj)
                count += 1
                if count % 1000 == 0:
                    session.commit()
                    logger.info("Processed batch", count=count)
            except Exception as e:
                logger.error(
                    "Failed to merge item", cik=cik, ticker=ticker, error=str(e)
                )
                session.rollback()
                # Continue or break?

        try:
            session.commit()
        except Exception as e:
            logger.error("Final commit failed", error=str(e))

        logger.info("Completed ingestion", total=count)


if __name__ == "__main__":
    ingest_sec_tickers()
