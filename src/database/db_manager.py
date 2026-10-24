"""
Database Manager for CrowdWisdomTrading Quant System.
Handles DuckDB and SQLite storage, schema initialization, and transactional queries.
"""

import os
import sqlite3

import pandas as pd
from loguru import logger

from .schema import SCHEMA_SQL


class DatabaseManager:
    """
    Manages persistent storage for raw data and features.
    Defaults to DuckDB, falling back to SQLite if DuckDB is unavailable.
    """
    def __init__(self, db_path: str = "data/crowdwisdom.duckdb", db_type: str = "duckdb"):
        self.db_path = db_path
        self.db_type = db_type.lower()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._duckdb = None
        self._init_backend()
        self.initialize_schema()

    def _init_backend(self):
        if self.db_type == "duckdb":
            try:
                import duckdb
                self._duckdb = duckdb
                logger.info(f"Initialized DuckDB backend at {self.db_path}")
            except ImportError:
                logger.warning("DuckDB not installed. Falling back to SQLite backend.")
                self.db_type = "sqlite"
                if self.db_path.endswith(".duckdb"):
                    self.db_path = self.db_path.replace(".duckdb", ".sqlite")
        else:
            self.db_type = "sqlite"

    def get_connection(self):
        if self.db_type == "duckdb" and self._duckdb is not None:
            return self._duckdb.connect(self.db_path)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def initialize_schema(self):
        """Initializes tables and indexes."""
        conn = self.get_connection()
        try:
            if self.db_type == "duckdb":
                for stmt in SCHEMA_SQL.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
            else:
                cursor = conn.cursor()
                cursor.executescript(SCHEMA_SQL)
                conn.commit()
            logger.info("Database schema initialized successfully.")
        finally:
            conn.close()

    def insert_tweets(self, df_tweets: pd.DataFrame) -> int:
        """Inserts raw tweets DataFrame into raw_tweets table."""
        if df_tweets.empty:
            return 0
        conn = self.get_connection()
        try:
            if self.db_type == "duckdb":
                conn.register("df_temp", df_tweets)
                conn.execute("""
                    INSERT OR REPLACE INTO raw_tweets
                    SELECT * FROM df_temp
                """)
            else:
                df_tweets.to_sql("raw_tweets", conn, if_exists="append", index=False)
            return len(df_tweets)
        finally:
            conn.close()

    def insert_prices(self, df_prices: pd.DataFrame) -> int:
        """Inserts raw OHLCV prices into raw_prices table."""
        if df_prices.empty:
            return 0
        conn = self.get_connection()
        try:
            if self.db_type == "duckdb":
                conn.register("df_temp", df_prices)
                conn.execute("""
                    INSERT OR REPLACE INTO raw_prices
                    SELECT * FROM df_temp
                """)
            else:
                df_prices.to_sql("raw_prices", conn, if_exists="append", index=False)
            return len(df_prices)
        finally:
            conn.close()

    def insert_news(self, df_news: pd.DataFrame) -> int:
        """Inserts news data into raw_news table."""
        if df_news.empty:
            return 0
        conn = self.get_connection()
        try:
            if self.db_type == "duckdb":
                conn.register("df_temp", df_news)
                conn.execute("""
                    INSERT OR REPLACE INTO raw_news
                    SELECT * FROM df_temp
                """)
            else:
                df_news.to_sql("raw_news", conn, if_exists="append", index=False)
            return len(df_news)
        finally:
            conn.close()

    def save_aligned_features(self, df_features: pd.DataFrame) -> int:
        """Saves aligned sentiment and price features."""
        if df_features.empty:
            return 0
        conn = self.get_connection()
        try:
            if self.db_type == "duckdb":
                conn.register("df_temp", df_features)
                conn.execute("""
                    CREATE OR REPLACE TABLE aligned_features AS
                    SELECT * FROM df_temp
                """)
            else:
                df_features.to_sql("aligned_features", conn, if_exists="replace", index=False)
            return len(df_features)
        finally:
            conn.close()

    def fetch_aligned_features(self, asset_symbol: str) -> pd.DataFrame:
        """Fetches aligned features ordered by timestamp."""
        conn = self.get_connection()
        try:
            query = """
                SELECT * FROM aligned_features
                WHERE asset_symbol = ?
                ORDER BY timestamp_utc ASC
            """
            if self.db_type == "duckdb":
                return conn.execute(query, [asset_symbol]).df()
            else:
                return pd.read_sql_query(query, conn, params=[asset_symbol])
        finally:
            conn.close()

    def fetch_latest_news(self, asset_symbol: str, limit: int = 10) -> pd.DataFrame:
        """Fetches latest news for sentiment summary."""
        conn = self.get_connection()
        try:
            query = """
                SELECT * FROM raw_news
                WHERE asset_symbol = ?
                ORDER BY timestamp_utc DESC
                LIMIT ?
            """
            if self.db_type == "duckdb":
                return conn.execute(query, [asset_symbol, limit]).df()
            else:
                return pd.read_sql_query(query, conn, params=[asset_symbol, limit])
        finally:
            conn.close()
