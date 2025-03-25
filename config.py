"""
Configuration file for NIFTY 200 Trading Signal Bot
Contains all parameters, credentials and settings
"""

# Upstox API Credentials
UPSTOX_API_KEY = "ad55de1b-c7d1-4adc-b559-3830bf1efd72"
UPSTOX_API_SECRET = "969nyjgapm"
UPSTOX_REDIRECT_URI = "https://localhost"
UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI0TEFGUDkiLCJqdGkiOiI2N2UyMTk4YTk4MzFlNDFlOGE3NmU3ZTciLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQyODcwOTIyLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDI5NDAwMDB9.QlAi6krvFdSUCw3_Lzqgf0rXKmVkbTVu403TkjJhOSk"

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "7209852741:AAEf-_f6TeZK1-_R55yq365iU_54rk95y-c"
TELEGRAM_CHAT_ID = "936205208"
ENABLE_TELEGRAM_ALERTS = True

HISTORICAL_DAYS = 365

# Stock Universe
# List of NIFTY 200 stocks to analyze (instrument IDs from Upstox)
STOCK_LIST = [
    "NSE_EQ|INE009A01021",  # INFOSYS
    "NSE_EQ|INE030A01027",  # TCS
    "NSE_EQ|INE397D01024",  # SBIN
    "NSE_EQ|INE062A01020",  # HDFC BANK
    "NSE_EQ|INE528G01035","HINDUNILVR", "ITC", "LICI", "LT", "SUNPHARMA", "HCLTECH", "KOTAKBANK"  # RELIANCE INDUSTRIES
    # Add more NIFTY 200 stocks here
]

# Data Configuration
INTERVALS = {
    "short_term": "1D",    # Daily for short-term analysis (3-6 months)
    "long_term": "1W"      # Weekly for long-term analysis (>1 year)
}
SHORT_TERM_LOOKBACK = 180  # ~6 months in trading days
LONG_TERM_LOOKBACK = 365   # 1 year

# Technical Indicator Parameters
INDICATORS = {
    # Trend Indicators
    "moving_averages": {
        "ema_short": 9,
        "ema_long": 21,
        "sma_mid": 50,
        "sma_long": 200
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "supertrend": {
        "period": 10,
        "multiplier": 3
    },
    "parabolic_sar": {
        "acceleration_factor": 0.02,
        "max_acceleration_factor": 0.2
    },
    "aroon": {
        "period": 25,
        "uptrend_threshold": 70,
        "downtrend_threshold": 30
    },
    
    # Momentum Indicators
    "rsi": {
        "period": 14,
        "oversold": 30,
        "overbought": 70
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "oversold": 20,
        "overbought": 80
    },
    "roc": {
        "period": 10
    },
    
    # Volatility Indicators
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2
    },
    "atr": {
        "period": 14,
        "multiplier": 1.5
    }
}

# Signal Thresholds
SIGNAL_STRENGTH = {
    "weak": 1,      # One indicator showing buy/sell
    "moderate": 2,  # Two indicators showing buy/sell
    "strong": 3,    # Three or more indicators showing buy/sell
    "very_strong": 5  # Five or more indicators showing buy/sell
}

# Minimum signal strength to generate alert
MINIMUM_SIGNAL_STRENGTH = 3

# Candlestick Pattern Recognition Settings
# True = detect this pattern
CANDLESTICK_PATTERNS = {
    "bullish_engulfing": True,
    "bearish_engulfing": True,
    "doji": True,
    "hammer": True,
    "shooting_star": True,
    "morning_star": True,
    "evening_star": True,
}

# Chart Pattern Recognition Settings
CHART_PATTERNS = {
    "head_and_shoulders": True,
    "inverse_head_and_shoulders": True,
    "double_top": True,
    "double_bottom": True,
    "cup_and_handle": True
}

# How often to run the analysis (in hours)
ANALYSIS_FREQUENCY = 1  # Run every hour

# Message templates
SIGNAL_MESSAGE_TEMPLATE = """
🔔 *TRADING SIGNAL ALERT* 🔔

*Stock:* {stock_name} ({stock_symbol})
*Current Price:* ₹{current_price}
*Signal Type:* {signal_type}
*Timeframe:* {timeframe}
*Strength:* {strength}/5

*Technical Indicators:*
{indicators}

*Patterns Detected:*
{patterns}

*Recommendation:*
{recommendation}

*Generated:* {timestamp}
"""

# Date/Time last updated: 2025-03-24 17:40
