-- NEM aggregated price and demand. One row = one region × one 5-minute interval.
-- settlement_ts is interval-ending, NEM time (AEST / UTC+10, no DST).

DROP TABLE IF EXISTS price_demand;
DROP TABLE IF EXISTS meta;

CREATE TABLE price_demand (
    settlement_ts TIMESTAMP NOT NULL,
    region VARCHAR NOT NULL,
    rrp DOUBLE NOT NULL,
    total_demand DOUBLE NOT NULL,
    period_type VARCHAR NOT NULL
);

CREATE UNIQUE INDEX price_demand_pk ON price_demand (settlement_ts, region);

CREATE TABLE meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
