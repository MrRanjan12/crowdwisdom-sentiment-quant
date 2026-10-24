#!/usr/bin/env python3
"""
CrowdWisdomTrading: X (Twitter) Sentiment & Price Prediction Multi-Horizon Agent.
Master End-to-End Orchestrator Pipeline.

Executes:
1. Data Ingestion (Apify X Posts, Historical Prices, Tavily News)
2. Embedded Database Storage (DuckDB / SQLite)
3. Quantitative Feature Engineering (Strictly Sentiment & Price Only)
4. Multi-Horizon Machine Learning & Purged Walk-Forward Validation (n+0 to n+5)
5. Reinforcement Learning Policy Agent
6. Quantitative Backtesting (Sharpe Ratio, Sortino Ratio, Max Drawdown)
7. Visual Matrix Generation
8. 1-Page Sentiment Outlook & Formal Evaluation Report
"""

import os
import sys
import argparse
import yaml
from dotenv import load_dotenv
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.db_manager import DatabaseManager
from src.ingestion.price_loader import PriceLoader
from src.ingestion.apify_scraper import ApifyTwitterScraper
from src.ingestion.tavily_news import TavilyNewsIngester
from src.features.sentiment_engine import SentimentEngine
from src.features.quant_features import QuantFeaturePipeline
from src.models.multi_horizon_forecaster import MultiHorizonForecaster
from src.models.rl_agent import SentimentRLTrader
from src.backtest.engine import BacktestEngine
from src.visualization.matrix_generator import VisualMatrixGenerator
from src.reports.report_generator import QuantReportGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="CrowdWisdomTrading Quant Sentiment Pipeline")
    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset symbol (default: BTC-USD)")
    parser.add_argument("--lookback", type=int, default=180, help="Lookback window in days (default: 180)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Bar timeframe ('1h' or '1d', default: 1h)")
    parser.add_argument("--actor-id", type=str, default="apidojo/tweet-scraper", help="Apify Actor ID")
    parser.add_argument("--db-type", type=str, default="duckdb", help="Database type: 'duckdb' or 'sqlite'")
    parser.add_argument("--skip-rl", action="store_true", help="Skip reinforcement learning agent training")
    return parser.parse_args()

def main():
    # Load environment variables
    load_dotenv()
    args = parse_args()

    # Banner
    logger.info("=" * 75)
    logger.info("   CROWDWISDOMTRADING QUANTITATIVE RESEARCH PIPELINE   ")
    logger.info("   X (Twitter) Sentiment & Price Multi-Horizon Forecaster   ")
    logger.info("=" * 75)
    logger.info(f"Target Asset:     {args.asset}")
    logger.info(f"Time Horizon:     {args.lookback} days ({args.timeframe} bars)")
    logger.info(f"Database Backend: {args.db_type}")
    logger.info(f"Apify Actor:      {args.actor_id}")
    logger.info("-" * 75)

    # 1. Initialize Database
    db_path = "data/crowdwisdom.duckdb" if args.db_type == "duckdb" else "data/crowdwisdom.sqlite"
    db = DatabaseManager(db_path=db_path, db_type=args.db_type)

    # 2. Ingest Historical OHLCV Prices
    logger.info("\n>>> STEP 1: HISTORICAL PRICE INGESTION")
    price_loader = PriceLoader(asset_symbol=args.asset, timeframe=args.timeframe, lookback_days=args.lookback)
    df_prices = price_loader.fetch_prices()
    db.insert_prices(df_prices)
    logger.info(f"Ingested {len(df_prices)} price bars into database.")

    # 3. Ingest X (Twitter) Social Sentiment via Apify
    logger.info("\n>>> STEP 2: X (TWITTER) SOCIAL DATA INGESTION (APIFY)")
    apify_scraper = ApifyTwitterScraper(
        actor_id=args.actor_id,
        asset_symbol=args.asset,
        lookback_days=args.lookback,
        max_posts=5000
    )
    df_tweets = apify_scraper.scrape_posts()
    db.insert_tweets(df_tweets)
    logger.info(f"Ingested {len(df_tweets)} social posts into database.")

    # 4. Ingest Breaking News via Tavily
    logger.info("\n>>> STEP 3: RECENT BREAKING NEWS INGESTION (TAVILY)")
    tavily_ingester = TavilyNewsIngester(asset_symbol=args.asset)
    df_news = tavily_ingester.fetch_recent_news(max_results=10, days_back=7)
    db.insert_news(df_news)
    logger.info(f"Ingested {len(df_news)} news articles into database.")

    # 5. Quantitative Feature Engineering & Alignment (Sentiment & Price Only)
    logger.info("\n>>> STEP 4: QUANTITATIVE FEATURE ENGINEERING & TEMPORAL ALIGNMENT")
    sentiment_engine = SentimentEngine()
    df_sentiment_agg = sentiment_engine.aggregate_hourly_sentiment(df_tweets, freq=args.timeframe)

    quant_pipeline = QuantFeaturePipeline(asset_symbol=args.asset, horizons=[0, 1, 2, 3, 4, 5])
    df_features = quant_pipeline.build_aligned_features(df_prices, df_sentiment_agg)
    db.save_aligned_features(df_features)
    logger.info(f"Engineered and saved {len(df_features)} aligned feature vectors to database.")

    # 6. Multi-Horizon Machine Learning & Purged Walk-Forward CV (n+0 to n+5)
    logger.info("\n>>> STEP 5: MULTI-HORIZON ML MODELING (PURGED WALK-FORWARD CV)")
    forecaster = MultiHorizonForecaster(horizons=[0, 1, 2, 3, 4, 5], n_splits=5, embargo_bars=6)
    df_oos, metrics_summary = forecaster.train_walk_forward(df_features)

    # 7. Reinforcement Learning Policy Agent
    if not args.skip_rl:
        logger.info("\n>>> STEP 6: REINFORCEMENT LEARNING POLICY AGENT")
        rl_agent = SentimentRLTrader(fee_bps=5.0)
        split_idx = int(len(df_features) * 0.7)
        rl_train = df_features.iloc[:split_idx]
        rl_agent.train_policy(rl_train, episodes=8)

    # 8. Quantitative Backtesting & Risk-Adjusted Attribution
    logger.info("\n>>> STEP 7: QUANTITATIVE BACKTESTING & PORTFOLIO ATTRIBUTION")
    backtest_engine = BacktestEngine(initial_capital=100000.0, fee_bps=5.0, risk_free_rate=0.04)
    df_backtest, backtest_summary = backtest_engine.run_multi_horizon_backtest(df_oos)

    # 9. Visual Matrix Generation
    logger.info("\n>>> STEP 8: VISUAL MATRIX OUTPUT GENERATOR")
    visual_gen = VisualMatrixGenerator(output_dir="outputs/plots")
    saved_plots = visual_gen.generate_all_matrices(
        df_backtest=df_backtest,
        metrics_summary=metrics_summary,
        feature_importances=forecaster.feature_importances
    )
    for p in saved_plots:
        logger.info(f"Saved visual matrix plot: {p}")

    # 10. Generate Reports (1-Page Outlook + Formal Evaluation Report)
    logger.info("\n>>> STEP 9: EXECUTIVE REPORTING SUITE")
    reporter = QuantReportGenerator(output_dir="reports")
    outlook_path = reporter.generate_sentiment_outlook_summary(
        asset_symbol=args.asset,
        df_features=df_features,
        df_news=df_news,
        output_file="reports/sentiment_outlook_summary.md"
    )
    eval_report_path = reporter.generate_evaluation_report(
        asset_symbol=args.asset,
        lookback_days=args.lookback,
        metrics_summary=metrics_summary,
        backtest_summary=backtest_summary,
        output_file="reports/model_evaluation_report.md"
    )
    logger.info(f"Generated 1-Page Outlook Summary: {outlook_path}")
    logger.info(f"Generated Formal Model Evaluation Report: {eval_report_path}")

    # Final Summary Banner
    strat = backtest_summary["strategy"]
    bench = backtest_summary["benchmark"]
    logger.info("\n" + "=" * 75)
    logger.info("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY   ")
    logger.info("=" * 75)
    logger.info(f"Strategy Sharpe Ratio:     {strat['sharpe_ratio']}  (Benchmark: {bench['sharpe_ratio']})")
    logger.info(f"Strategy Sortino Ratio:    {strat['sortino_ratio']}  (Benchmark: {bench['sortino_ratio']})")
    logger.info(f"Strategy Maximum Drawdown: {strat['max_drawdown_pct']}%  (Benchmark: {bench['max_drawdown_pct']}%)")
    logger.info(f"Strategy Calmar Ratio:     {strat['calmar_ratio']}  (Benchmark: {bench['calmar_ratio']})")
    logger.info(f"Strategy Win Rate:         {strat['win_rate_pct']}%  (Benchmark: {bench['win_rate_pct']}%)")
    logger.info("-" * 75)
    logger.info(f"Evaluation Report:         {eval_report_path}")
    logger.info(f"1-Page Sentiment Outlook:  {outlook_path}")
    logger.info(f"Visual Matrix Plots:       outputs/plots/")
    logger.info(f"Embedded Database:         {db_path}")
    logger.info("=" * 75)

if __name__ == "__main__":
    main()
