CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    prediction_date DATE NOT NULL,
    run_date DATE NOT NULL,
    model_name VARCHAR NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    true_value DOUBLE PRECISION,
    error_abs DOUBLE PRECISION,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_ml_predictions UNIQUE (symbol, prediction_date, model_name, run_date)
);
