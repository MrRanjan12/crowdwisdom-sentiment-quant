"""
Database schema definitions for CrowdWisdomTrading Quant System.
Compatible with both DuckDB and SQLite.
"""

SCHEMA_SQL = """
-- 1. Raw X (Twitter) Posts table
CREATE TABLE IF NOT EXISTS raw_tweets (
    tweet_id VARCHAR PRIMARY KEY,
    timestamp_utc TIMESTAMP NOT NULL,
    asset_symbol VARCHAR NOT NULL,
    text TEXT NOT NULL,
    author_id VARCHAR,
    author_username VARCHAR,
    author_followers INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    retweets INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    source VARCHAR DEFAULT 'apify'
);

CREATE INDEX IF NOT EXISTS idx_tweets_time ON raw_tweets (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_tweets_asset ON raw_tweets (asset_symbol);

-- 2. Raw Historical OHLCV Price bars
CREATE TABLE IF NOT EXISTS raw_prices (
    timestamp_utc TIMESTAMP NOT NULL,
    asset_symbol VARCHAR NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    PRIMARY KEY (timestamp_utc, asset_symbol)
);

CREATE INDEX IF NOT EXISTS idx_prices_time ON raw_prices (timestamp_utc);

-- 3. Ingested News (Tavily/Exa)
CREATE TABLE IF NOT EXISTS raw_news (
    news_id VARCHAR PRIMARY KEY,
    timestamp_utc TIMESTAMP NOT NULL,
    asset_symbol VARCHAR NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR,
    url VARCHAR,
    sentiment_score DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_news_time ON raw_news (timestamp_utc);

-- 4. Aligned Features (Sentiment + Price strictly)
CREATE TABLE IF NOT EXISTS aligned_features (
    timestamp_utc TIMESTAMP NOT NULL,
    asset_symbol VARCHAR NOT NULL,
    close_price DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    post_count INTEGER DEFAULT 0,
    sentiment_score DOUBLE DEFAULT 0.0,
    engagement_weighted_sentiment DOUBLE DEFAULT 0.0,
    sentiment_dispersion DOUBLE DEFAULT 0.0,
    sentiment_momentum DOUBLE DEFAULT 0.0,
    bull_ratio DOUBLE DEFAULT 0.5,
    log_return DOUBLE DEFAULT 0.0,
    realized_volatility DOUBLE DEFAULT 0.0,
    price_sentiment_divergence DOUBLE DEFAULT 0.0,
    rsi_14 DOUBLE DEFAULT 50.0,
    target_h0 DOUBLE,
    target_h1 DOUBLE,
    target_h2 DOUBLE,
    target_h3 DOUBLE,
    target_h4 DOUBLE,
    target_h5 DOUBLE,
    PRIMARY KEY (timestamp_utc, asset_symbol)
);

CREATE INDEX IF NOT EXISTS idx_features_time ON aligned_features (timestamp_utc);

-- 5. Out-of-sample Model Forecasts
CREATE TABLE IF NOT EXISTS model_forecasts (
    forecast_id VARCHAR PRIMARY KEY,
    timestamp_utc TIMESTAMP NOT NULL,
    asset_symbol VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    h0_pred DOUBLE,
    h1_pred DOUBLE,
    h2_pred DOUBLE,
    h3_pred DOUBLE,
    h4_pred DOUBLE,
    h5_pred DOUBLE,
    actual_h0 DOUBLE,
    actual_h1 DOUBLE,
    actual_h2 DOUBLE,
    actual_h3 DOUBLE,
    actual_h4 DOUBLE,
    actual_h5 DOUBLE
);
"""
