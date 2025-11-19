-- Base de datos ya la crea POSTGRES_DB=indices
-- Aquí solo definimos tablas:

CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT    NOT NULL,
    date        DATE    NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    adj_close   DOUBLE PRECISION,
    volume      BIGINT,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS indicators (
    symbol   TEXT NOT NULL,
    date     DATE NOT NULL,
    sma_20   DOUBLE PRECISION,
    sma_50   DOUBLE PRECISION,
    vol_20   DOUBLE PRECISION,
    rsi_14   DOUBLE PRECISION,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS signals (
    symbol          TEXT NOT NULL,
    date            DATE NOT NULL,
    signal_simple   INTEGER,
    signal_ensemble INTEGER,
    model_best      TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS news (
    id           SERIAL PRIMARY KEY,
    symbol       TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT,
    url          TEXT UNIQUE,
    summary      TEXT,
    sentiment    DOUBLE PRECISION
);