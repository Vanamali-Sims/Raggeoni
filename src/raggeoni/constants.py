"""Frozen Week 1 window and source. Do not change after gold is written."""

from __future__ import annotations

REGIONS: tuple[str, ...] = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")

# Inclusive calendar months. August 2026 is incomplete (today is 2026-08-29).
WINDOW_START_YEAR = 2025
WINDOW_START_MONTH = 8
WINDOW_END_YEAR = 2026
WINDOW_END_MONTH = 7

# Interval-ending timestamps in NEM time (AEST, UTC+10, no DST).
EXPECTED_MIN_TS = "2025-08-01 00:05:00"
EXPECTED_MAX_TS = "2026-08-01 00:00:00"

# 365 days × 288 five-minute intervals (2026 is not a leap year; window includes Feb 2026).
INTERVAL_MINUTES = 5
DAYS_IN_WINDOW = 365
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES
EXPECTED_ROWS_PER_REGION = DAYS_IN_WINDOW * INTERVALS_PER_DAY  # 105_120
EXPECTED_ROWS = EXPECTED_ROWS_PER_REGION * len(REGIONS)  # 525_600

SOURCE_BASE_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand"
SOURCE_PAGE_URL = (
    "https://www.aemo.com.au/energy-systems/electricity/"
    "national-electricity-market-nem/data-nem/aggregated-data"
)

TIMEZONE_NOTE = "NEM time is AEST (UTC+10) year-round; no daylight saving."
GRAIN_NOTE = "5-minute settlement intervals; SETTLEMENTDATE is interval-ending."

USER_AGENT = "raggeoni/0.1 (local research ingest)"
