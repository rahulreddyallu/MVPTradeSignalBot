# NIFTY 200 Trading Signal Bot

## Overview

The NIFTY 200 Trading Signal Bot is a comprehensive trading signal generation system designed to analyze NIFTY 200 stocks. It fetches historical price data from Upstox, performs technical analysis using multiple indicators, detects candlestick and chart patterns, generates buy/sell signals with strength ratings, and sends alerts via Telegram.

## System Architecture

### Core Components

1. **Data Retrieval**: Fetches historical OHLCV (Open-High-Low-Close-Volume) data and instrument details from Upstox.
2. **Technical Analysis**: Performs technical analysis using various indicators and pattern recognition methods.
3. **Signal Generation**: Generates buy/sell signals based on the analysis with strength ratings.
4. **Notifications**: Sends alerts via Telegram based on the generated signals.

## File-by-File Analysis

### 1. `compute.py` - Core Analysis Engine

#### Classes

1. **UpstoxClient**
   - Handles API authentication with Upstox.
   - Methods: `authenticate()`, `_refresh_token()`, `get_historical_data()`, `get_instrument_details()`.
   - Error handling and logging for API failures.

2. **TechnicalAnalysis**
   - Core analysis engine calculates over 12 technical indicators.
   - Methods: `_calculate_moving_averages()`, `_calculate_macd()`, `_calculate_supertrend()`, etc.
   - Pattern recognition: `detect_candlestick_patterns()`, `detect_chart_patterns()`.
   - Generates consolidated signals through weighted scoring.

   **Technical Indicators Implemented**:
   - Moving Averages (SMA/EMA)
   - MACD
   - SuperTrend
   - Parabolic SAR
   - Aroon
   - RSI
   - Stochastic Oscillator
   - Rate of Change
   - Bollinger Bands
   - ATR
   - OBV
   - VWAP

3. **TelegramSender**
   - Wrapper for Telegram bot API to send notifications.
   - Methods: `send_message()`.

4. **TradingSignalBot**
   - Orchestrates the entire process.
   - Runs analysis on stocks in the configured list.
   - Formats and sends messages based on signal strength.
   - Methods: `run()`, `_analyze_stock()`.

### 2. `config.py` - Configuration Settings

Contains all configuration parameters:
- API credentials (Upstox and Telegram)
- Stock list from NIFTY 200
- Analysis intervals and lookback periods
- Technical indicator parameters
- Signal thresholds and strength ratings
- Pattern detection settings
- Message template for notifications

### 3. `main.py` - Application Entry Point

Handles program execution flow:
- Initializes connections to Upstox and Telegram.
- Implements scheduling based on market hours.
- Contains the data fetching and analysis pipeline.
- Formats and sends the final reports.
- Error handling and recovery.

**Key Functions**:
- `initialize_upstox()`: Sets up API connection.
- `fetch_ohlcv_data()`: Gets price data.
- `analyze_and_generate_signals()`: Core analysis pipeline.
- `schedule_analysis()`: Runs analysis during market hours.
- `send_telegram_message()`: Notification with retry mechanism.

### 4. `requirements.txt`

Lists all dependencies required for running the bot, including pandas, numpy, pandas-ta, requests, aiogram, and others.

## Clarification on Bot Operation Schedule

### Initial Analysis on Startup

When the bot first launches, it runs an immediate analysis regardless of the time. This is followed by setting up a schedule to run during market hours for subsequent executions.

### Parameters for Immediate Analysis

- **Historical Data Period**: 365 days (from `config.py`).
- **Data Granularity**: Daily candles (`interval="day"`).
- **Date Range**: From today back to 365 days prior.
- **Analysis Types**: Short-term (1D) and long-term (1W) timeframes.

### Parameters for Scheduled Analysis

Identical to the immediate analysis, but the key difference is the timing of the analysis:
- **Runs at Market Open**: 9:15 AM.
- **Hourly During Trading**: 9:00 AM to 3:00 PM.
- **Runs at Market Close**: 3:30 PM.
- **Only on Weekdays**: Monday-Friday.

## Strengths and Implementation Notes

- **Modular Design**: Independent implementation of each indicator and pattern.
- **Configurability**: Easily adjustable parameters in `config.py`.
- **Robust Error Handling**: Extensive logging and error reporting.
- **Signal Aggregation**: Combines signals from multiple indicators to reduce false positives.
- **Scheduled Execution**: Intelligent scheduling around market hours.

The NIFTY 200 Trading Signal Bot demonstrates a professional-level implementation of technical analysis with proper error handling, configuration management, and separation of concerns.

## Running the Bot

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings**: Edit `config.py` with appropriate API credentials and parameters.

3. **Run the Bot**:
   ```bash
   python main.py
   ```

The bot will perform an immediate analysis on startup and then follow the scheduled analysis during market hours.
