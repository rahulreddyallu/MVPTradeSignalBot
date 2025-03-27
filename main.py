"""
Main script for NIFTY 200 Trading Signal Bot
Handles initialization and execution of the bot
"""

import os
import time
import datetime
import logging
import sys
import traceback
import schedule
import asyncio
import pandas as pd

# Ensure aiogram is installed
try:
    from aiogram import Bot, Dispatcher
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiogram"])
    from aiogram import Bot, Dispatcher

from upstox_client.api_client import ApiClient
from upstox_client.api.market_quote_api import MarketQuoteApi  # Correct import
from upstox_client.api.history_api import HistoryApi
from config import *
import importlib

# Add directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Force reload of compute module
import compute
importlib.reload(compute)

# Import after reload
from compute import TechnicalAnalysis

import re

from config import STOCK_INFO

# Create a reverse mapping from symbol to ISIN
SYMBOL_TO_ISIN = {info["symbol"]: isin for isin, info in STOCK_INFO.items()}

def get_stock_info_by_key(instrument_key):
    """Get stock info from instrument key (e.g., NSE_EQ|INE117A01022)"""
    parts = instrument_key.split('|')
    if len(parts) == 2:
        isin = parts[1]
        if isin in STOCK_INFO:
            return STOCK_INFO[isin]
    # Try direct symbol match as fallback
    if instrument_key in SYMBOL_TO_ISIN:
        isin = SYMBOL_TO_ISIN[instrument_key]
        return STOCK_INFO[isin]
    return {"name": "", "industry": "", "symbol": instrument_key, "series": ""}

def escape_telegram_markdown(text):
    """Escape special characters for Telegram MarkdownV2 formatting."""
    if not text:
        return "N/A"
    # List of all special characters that need to be escaped in MarkdownV2
    special_chars = r'_*[]()~`>#+-=|{}.!'
    escaped_text = re.sub(f'([{re.escape(special_chars)}])', r'\\\1', str(text))
    return escaped_text


# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging
log_filename = f"logs/trading_bot_{datetime.datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Upstox client
def initialize_upstox():
    try:
        api_client = ApiClient()
        api_client.configuration.access_token = UPSTOX_ACCESS_TOKEN
        market_api = MarketQuoteApi(api_client)
        logger.info("✅ Successfully initialized Upstox API client")
        return market_api
    except Exception as e:
        logger.error(f"Error initializing Upstox API client: {e}")
        return None

# Telegram notification function with exponential backoff retry mechanism
async def send_telegram_message(message, retry_attempts=5):
    if ENABLE_TELEGRAM_ALERTS:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        delay = 1  # Initial delay in seconds
        for attempt in range(retry_attempts):
            try:
                # Use MarkdownV2 for formatted messages
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='MarkdownV2')
                break
            except Exception as e:
                if "Too Many Requests" in str(e):
                    retry_after = int(str(e).split("retry after ")[-1].split()[0])
                    logger.error(f"Error sending Telegram message: {e}. Retrying in {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(f"Error sending Telegram message: {e}. Retrying in {delay} seconds.")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
        await bot.session.close()  # Ensure the session is properly closed

def send_startup_notification():
    """Send a startup notification via Telegram"""
    try:
        loop = asyncio.get_event_loop()
        # Escape the entire startup message
        message = escape_telegram_markdown(f"""
🚀 NIFTY 200 Trading Signal Bot Started 🚀

Version: 2.0.0
Started at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analysis Frequency: Every {ANALYSIS_FREQUENCY} hour(s)
Stocks Monitored: {len(STOCK_LIST)} NIFTY 200 stocks
Timeframes Analyzed: 
- Short Term (3-6 months)
- Long Term (>1 year)

New Features:
• Enhanced pattern detection (candlestick & chart patterns)
• Improved signal generation algorithm
• Better risk management with ATR-based stop losses

Bot is now actively monitoring for trading signals.
        """)
        loop.run_until_complete(send_telegram_message(message))
        logger.info("Startup notification sent")
    except Exception as e:
        logger.error(f"Failed to send startup notification: {str(e)}")

def fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day"):
    """
    Fetch historical OHLC data for a given symbol using the Upstox API.
    """
    try:
        # Validate dates
        try:
            datetime.datetime.strptime(start_date, "%Y-%m-%d")
            datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()
        
        # Validate interval
        valid_intervals = ['1minute', '30minute', 'day', 'week', 'month']
        if interval not in valid_intervals:
            logger.error(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
            return pd.DataFrame()
        
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} with {interval} interval")
        
        # Create a HistoryApi instance
        from upstox_client.api.history_api import HistoryApi
        history_api = HistoryApi(market_api.api_client)
        
        # Call the method on the HistoryApi instance
        response = history_api.get_historical_candle_data1(
            instrument_key=symbol,
            interval=interval,
            to_date=end_date,
            from_date=start_date,
            api_version="2.0"
        )
        
        # Extract data from the response - handling HistoricalCandleData object
        if hasattr(response, 'data') and hasattr(response.data, 'candles'):
            candles_data = response.data.candles
            
            # Create DataFrame with proper column names
            df = pd.DataFrame(candles_data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types for all columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp (oldest to newest)
            df.sort_index(inplace=True)
            
            # Check for minimum data points required for pattern detection
            if len(df) < 50:  # Minimum required for reliable chart pattern detection
                logger.warning(f"Retrieved only {len(df)} candles for {symbol}, which may be insufficient for reliable pattern detection")
                
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
        else:
            logger.error(f"No candle data returned for {symbol}")
            if hasattr(response, 'status'):
                logger.error(f"API status: {response.status}")
            if hasattr(response, 'data'):
                logger.error(f"Response data type: {type(response.data)}")
                # List all attributes of the data object
                logger.error(f"Response data attributes: {dir(response.data)}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching historical OHLC data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

async def analyze_and_generate_signals():
    """
    Fetches historical data for symbols in STOCK_LIST, performs technical analysis,
    and generates trading signals with enhanced pattern detection.
    """
    # Import necessary configurations
    from config import SIGNAL_MESSAGE_TEMPLATE, MINIMUM_SIGNAL_STRENGTH
    
    # Log function start with current UTC time
    current_datetime = datetime.datetime.now()
    logger.info(f"Starting analysis at {current_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Current date/time
    current_date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Analysis date: {current_date_str}")
    
    # Calculate date range (based on HISTORICAL_DAYS constant)
    end_date = current_datetime.strftime('%Y-%m-%d')
    start_date = (current_datetime - datetime.timedelta(days=HISTORICAL_DAYS)).strftime('%Y-%m-%d')
    logger.info(f"Analyzing data from {start_date} to {end_date}")

    # Initialize Upstox API client
    market_api = initialize_upstox()
    if not market_api:
        logger.error("Failed to initialize Upstox client")
        return
    
    # Track overall statistics
    successful_analyses = 0
    failed_analyses = 0
    total_signals = 0
    buy_signals = 0
    sell_signals = 0
    
    # Header for daily report
    daily_report = [
        f"📈 TRADING SIGNALS REPORT 📉",
        f"Date: {current_date_str} UTC",
        f"Analyzing {len(STOCK_LIST)} symbols with {HISTORICAL_DAYS} days of historical data",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    # Process each symbol in STOCK_LIST
    for symbol in STOCK_LIST:
        logger.info(f"Processing symbol: {symbol}")
        
        try:
            # Get stock information
            stock_info = get_stock_info_by_key(symbol)
            company_name = stock_info.get("name", "")
            industry = stock_info.get("industry", "")
            trading_symbol = stock_info.get("symbol", symbol.split("|")[-1] if "|" in symbol else symbol)
            
            logger.info(f"Analyzing {company_name} ({trading_symbol}) - {industry}")
            
            # Fetch historical data with daily interval
            data = fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day")
            
            if data.empty:
                logger.error(f"No historical data fetched for {company_name} ({symbol})")
                failed_analyses += 1
                continue
            
            logger.info(f"Analyzing {company_name} ({trading_symbol}) with {len(data)} data points")
            
            # Create a copy with lowercase column names for TechnicalAnalysis
            renamed_data = data.copy()
            renamed_data.columns = [col.lower() for col in renamed_data.columns]
            
            # Perform technical analysis with enhanced pattern detection
            analyzer = TechnicalAnalysis(renamed_data)
            
            # Generate comprehensive signals (indicators + patterns)
            signal_results = analyzer.generate_signals()
            
            # Extract data from results
            indicators_result = signal_results['indicators']
            patterns_result = signal_results['patterns']
            signals = signal_results['individual_signals']
            buy_signals_count = signal_results['buy_signals_count']
            sell_signals_count = signal_results['sell_signals_count']
            
            # Fix for truncated summary in the original code
            signal_summary = f"{'Bullish' if signal_results['signal'] == 'BUY' else 'Bearish' if signal_results['signal'] == 'SELL' else 'Neutral'} signal with {buy_signals_count} buy vs {sell_signals_count} sell signals"
            
            overall_signal = {
                'signal': signal_results['signal'],
                'strength': signal_results['strength'],
                'summary': signal_summary
            }
            
            if signals and overall_signal['strength'] >= MINIMUM_SIGNAL_STRENGTH:
                logger.info(f"Generated signal for {company_name} ({trading_symbol}): {overall_signal['signal']} (Strength: {overall_signal['strength']}/5) - {buy_signals_count} buy vs {sell_signals_count} sell signals")
                total_signals += 1
                
                if overall_signal['signal'] == 'BUY':
                    buy_signals += 1
                elif overall_signal['signal'] == 'SELL':
                    sell_signals += 1
                
                # Add to daily report
                daily_report.append(f"\n{company_name} ({trading_symbol}): {overall_signal['signal']} (Strength: {overall_signal['strength']}/5)")
                daily_report.append(f"{overall_signal['summary']}")
                
                # Format primary indicators for clean template
                primary_indicators = []

                # Check if this is a BUY or SELL signal to show appropriate indicators
                is_buy_signal = overall_signal['signal'] == 'BUY'

                # Moving Averages (EMA Crossover)
                if 'moving_averages' in indicators_result:
                    ma_data = indicators_result['moving_averages']
                    if (is_buy_signal and ma_data.get('signal') == 1) or (not is_buy_signal and ma_data.get('signal') == -1):
                        ema_short = ma_data['values']['ema_short']
                        ema_long = ma_data['values']['ema_long']
                        primary_indicators.append(f"✅ EMA Crossover: {'Bullish' if is_buy_signal else 'Bearish'} ({ema_short:.2f} {('>' if is_buy_signal else '<')} {ema_long:.2f})")

                # RSI
                if 'rsi' in indicators_result:
                    rsi_data = indicators_result['rsi']['values']
                    rsi_value = rsi_data.get('rsi')
                    if rsi_value is not None:
                        if (is_buy_signal and (50 < rsi_value < 70 or rsi_value <= 30)) or (not is_buy_signal and (rsi_value >= 70 or rsi_value < 50)):
                            rsi_status = "Strong momentum" if 50 < rsi_value < 70 else "Overbought" if rsi_value >= 70 else "Oversold" if rsi_value <= 30 else "Neutral"
                            primary_indicators.append(f"✅ RSI: {rsi_value:.2f} ({rsi_status})")

                # Supertrend
                if 'supertrend' in indicators_result:
                    st_data = indicators_result['supertrend']['values']
                    if (is_buy_signal and st_data.get('direction') == 'Bullish') or (not is_buy_signal and st_data.get('direction') == 'Bearish'):
                        primary_indicators.append(f"✅ Supertrend: {st_data.get('direction')}")

                # MACD
                if 'macd' in indicators_result:
                    macd_data = indicators_result['macd']
                    if (is_buy_signal and macd_data.get('signal') == 1) or (not is_buy_signal and macd_data.get('signal') == -1):
                        status = "Positive & expanding" if is_buy_signal else "Negative & expanding"
                        primary_indicators.append(f"✅ MACD: {status}")

                # Bollinger Bands
                if 'bollinger_bands' in indicators_result:
                    bb_data = indicators_result['bollinger_bands']
                    if (is_buy_signal and bb_data.get('signal') == 1) or (not is_buy_signal and bb_data.get('signal') == -1):
                        percent_b = bb_data['values'].get('percent_b')
                        position = "Oversold" if percent_b and percent_b < 0.2 else "Overbought" if percent_b and percent_b > 0.8 else ""
                        if position:
                            primary_indicators.append(f"✅ Bollinger: {position} ({percent_b:.2f})")

                # Stochastic
                if 'stochastic' in indicators_result:
                    stoch_data = indicators_result['stochastic']
                    if (is_buy_signal and stoch_data.get('signal') == 1) or (not is_buy_signal and stoch_data.get('signal') == -1):
                        k_value = stoch_data['values'].get('k')
                        d_value = stoch_data['values'].get('d')
                        if k_value and d_value:
                            primary_indicators.append(f"✅ Stochastic: Crossover ({k_value:.2f})")

                # Parabolic SAR
                if 'parabolic_sar' in indicators_result:
                    psar_data = indicators_result['parabolic_sar']
                    if (is_buy_signal and psar_data.get('signal') == 1) or (not is_buy_signal and psar_data.get('signal') == -1):
                        primary_indicators.append(f"✅ Parabolic SAR: {psar_data['values']['trend']}")

                # Aroon
                if 'aroon' in indicators_result:
                    aroon_data = indicators_result['aroon']['values']
                    if is_buy_signal and aroon_data.get('strong_uptrend'):
                        primary_indicators.append(f"✅ Aroon: Strong Uptrend ({aroon_data.get('aroon_up', 0):.2f})")
                    elif not is_buy_signal and aroon_data.get('strong_downtrend'):
                        primary_indicators.append(f"✅ Aroon: Strong Downtrend ({aroon_data.get('aroon_down', 0):.2f})")

                # Rate of Change
                if 'roc' in indicators_result:
                    roc_data = indicators_result['roc']['values']
                    roc_value = roc_data.get('roc')
                    trend = roc_data.get('trend')
                    if roc_value is not None:
                        if (is_buy_signal and roc_value > 0 and trend == 'Bullish') or (not is_buy_signal and roc_value < 0 and trend == 'Bearish'):
                            primary_indicators.append(f"✅ ROC: {trend} ({roc_value:.2f})")

                # ATR (Average True Range)
                if 'atr' in indicators_result:
                    atr_data = indicators_result['atr']['values']
                    atr_value = atr_data.get('atr')
                    if atr_value:
                        # ATR itself doesn't have bull/bear bias, it's a volatility measure
                        volatility = "High" if atr_value > (data['Close'].iloc[-1] * 0.02) else "Normal" if atr_value > (data['Close'].iloc[-1] * 0.01) else "Low"
                        primary_indicators.append(f"✅ ATR: {atr_value:.2f} ({volatility} volatility)")

                # OBV (On-Balance Volume)
                if 'obv' in indicators_result:
                    obv_data = indicators_result['obv']['values']
                    rising = obv_data.get('rising')
                    if (is_buy_signal and rising) or (not is_buy_signal and not rising):
                        primary_indicators.append(f"✅ OBV: {'Rising' if rising else 'Falling'} Volume")

                # VWAP
                if 'vwap' in indicators_result:
                    vwap_data = indicators_result['vwap']['values']
                    price_to_vwap = vwap_data.get('price_to_vwap')
                    if price_to_vwap:
                        position = price_to_vwap > 1
                        if (is_buy_signal and position) or (not is_buy_signal and not position):
                            status = "Above VWAP" if position else "Below VWAP"
                            primary_indicators.append(f"✅ VWAP: {status} ({price_to_vwap:.2f}x)")

                # Format patterns for clean template
                patterns_text = []

                # Add chart patterns with checkmarks
                for pattern_name, pattern_data in patterns_result.get('chart', {}).items():
                    if (is_buy_signal and pattern_data.get('signal') == 1) or (not is_buy_signal and pattern_data.get('signal') == -1):
                        pattern_name_formatted = pattern_name.replace('_', ' ').title()
                        patterns_text.append(f"✅ {pattern_name_formatted}")

                # Add candlestick patterns with checkmarks
                for pattern_name, pattern_data in patterns_result.get('candlestick', {}).items():
                    if (is_buy_signal and pattern_data.get('signal') == 1) or (not is_buy_signal and pattern_data.get('signal') == -1):
                        pattern_name_formatted = pattern_name.replace('_', ' ').title()
                        patterns_text.append(f"✅ {pattern_name_formatted} (candlestick)")

                # If no patterns found
                if not patterns_text:
                    patterns_text = ["No significant patterns detected"]

                # Get risk management values
                stop_loss = None
                target_price = None
                if 'atr' in indicators_result:
                    if is_buy_signal:
                        stop_loss = indicators_result['atr']['values'].get('buy_stop')
                        if stop_loss:
                            target_price = data['Close'].iloc[-1] + (data['Close'].iloc[-1] - stop_loss) * 2  # 2:1 reward-to-risk ratio
                    else:
                        stop_loss = indicators_result['atr']['values'].get('sell_stop')
                        if stop_loss:
                            target_price = data['Close'].iloc[-1] - (stop_loss - data['Close'].iloc[-1]) * 2  # 2:1 reward-to-risk ratio

                # Aroon Trend Strength
                aroon_value = "N/A"
                if 'aroon' in indicators_result:
                    aroon_data = indicators_result['aroon']['values']
                    if is_buy_signal:
                        aroon_up = aroon_data.get('aroon_up', 0)
                        strength_desc = "Very Strong" if aroon_up > 80 else "Strong" if aroon_up > 70 else "Moderate" if aroon_up > 50 else "Weak"
                        aroon_value = f"{aroon_up:.2f} ({strength_desc})"
                    else:
                        aroon_down = aroon_data.get('aroon_down', 0)
                        strength_desc = "Very Strong" if aroon_down > 80 else "Strong" if aroon_down > 70 else "Moderate" if aroon_down > 50 else "Weak"
                        aroon_value = f"{aroon_down:.2f} ({strength_desc})"

                # Buy/Sell Summary
                total_signals = buy_signals_count + sell_signals_count
                buy_sell_summary = f"{buy_signals_count}/{total_signals} BULLISH SIGNALS | {sell_signals_count}/{total_signals} BEARISH SIGNALS"

                # Format timestamp
                timestamp_short = datetime.datetime.now().strftime("%b-%d %H:%M")

               # Adjust signal_strength to be in range 1-5
               signal_strength = max(1, min(5, overall_signal['strength']))


                # Generate star rating string before formatting
                star_rating = "⭐" * signal_strength
                
                # First format the template with unescaped values
                unescaped_message = SIGNAL_MESSAGE_TEMPLATE.format(
                    stock_name=company_name,
                    stock_symbol=trading_symbol,
                    current_price=f"{data['Close'].iloc[-1]:.2f}",
                    industry=industry,
                    signal_type=overall_signal['signal'],
                    signal_strength=signal_strength,
                    star_rating=star_rating,  # Add this parameter for stars
                    primary_indicators="\n".join(primary_indicators),
                    patterns="\n".join(patterns_text),
                    stop_loss=f"{stop_loss:.2f}" if stop_loss else "N/A",
                    target_price=f"{target_price:.2f}" if target_price else "N/A",
                    trend_strength=aroon_value,
                    buy_sell_summary=buy_sell_summary,
                    timestamp_short=timestamp_short
                )
                
                # Then escape the entire message for MarkdownV2
                message = escape_telegram_markdown(unescaped_message)
                
                # Send via Telegram
                await send_telegram_message(message)
                logger.info(f"Sent signal alert for {company_name} ({trading_symbol})")
                
            elif signals:
                logger.info(f"Signals generated for {company_name} ({trading_symbol}) but strength ({overall_signal['strength']}) below threshold")
                daily_report.append(f"\n{company_name} ({trading_symbol}): {overall_signal['signal']} (Strength: {overall_signal['strength']}/5) - Below threshold")
            else:
                logger.info(f"No signals generated for {company_name} ({trading_symbol})")
                daily_report.append(f"\n{company_name} ({trading_symbol}): No trading signals")
            
            successful_analyses += 1
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            failed_analyses += 1
            daily_report.append(f"\n{symbol}: Error during analysis - {str(e)[:50]}...")
            continue
    
    # Finalize daily report - using plain text instead of markdown
    daily_report.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")  # Safe separator
    daily_report.append("Analysis Summary:")
    daily_report.append(f"• Analyzed: {successful_analyses if successful_analyses is not None else 0} symbols")
    daily_report.append(f"• Failed: {failed_analyses if failed_analyses is not None else 0} symbols") 
    daily_report.append(f"• Total signals generated: {total_signals if total_signals is not None else 0} "
                        f"({buy_signals if buy_signals is not None else 0} BUY, "
                        f"{sell_signals if sell_signals is not None else 0} SELL)")
    daily_report.append(f"• Report time: {current_date_str} UTC")
    
    # Send daily summary report via Telegram
    if successful_analyses > 0:
        escaped_report = escape_telegram_markdown("\n".join(daily_report))
        await send_telegram_message(escaped_report)
    
    # Log summary statistics
    logger.info(f"Analysis completed. Processed {len(STOCK_LIST)} symbols.")
    logger.info(f"Successful analyses: {successful_analyses if successful_analyses is not None else 0}")
    logger.info(f"Failed analyses: {failed_analyses if failed_analyses is not None else 0}")
    logger.info(f"Total signals generated: {total_signals if total_signals is not None else 0} "
                f"({buy_signals if buy_signals is not None else 0} BUY, {sell_signals if sell_signals is not None else 0} SELL)")
    logger.info(f"Analysis completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    
def run_trading_signals():
    """Run the trading signal generation process"""
    start_time = time.time()
    logger.info("Starting trading signal analysis")
    
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(analyze_and_generate_signals())
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Completed trading signal analysis in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error in trading signal analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Send error notification
    try:
        loop = asyncio.get_event_loop()
        # Escape the error message for MarkdownV2 format
        error_message = escape_telegram_markdown(f"""
    ⚠️ ERROR: Trading Signal Bot Failure ⚠️
    
    Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Error: {str(e)}
    
    Please check the logs for more details.
        """)
        loop.run_until_complete(send_telegram_message(error_message))
    except:
        logger.error("Failed to send error notification")

def test_upstox_connection():
    """Test connection to Upstox API"""
    logger.info("Testing Upstox API connection...")
    
    try:
        market_api = initialize_upstox()
        if market_api:
            logger.info("✅ Successfully initialized Upstox API client")
            return True
        else:
            logger.error("❌ Failed to initialize Upstox API client")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Upstox API: {str(e)}")
        return False

def test_telegram_connection():
    """Test connection to Telegram API"""
    logger.info("Testing Telegram API connection...")
    
    try:
        loop = asyncio.get_event_loop()
        # Escape the test message properly for MarkdownV2
        test_message = escape_telegram_markdown("🔍 Test Message - NIFTY 200 Trading Signal Bot connection test successful!")
        result = loop.run_until_complete(send_telegram_message(test_message))
        
        if result:
            logger.info("✅ Successfully sent test message to Telegram")
            return True
        else:
            logger.error("❌ Failed to send test message to Telegram")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Telegram API: {str(e)}")
        return False

def schedule_analysis():
    """Schedule the analysis based on config"""
    # Schedule for specific hours of the day based on market hours
    for hour in range(9, 16):  # 9 AM to 3 PM
        schedule.every().monday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().tuesday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().wednesday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().thursday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().friday.at(f"{hour:02d}:00").do(run_trading_signals)
    
    # Also schedule at market open and close
    schedule.every().monday.at("09:15").do(run_trading_signals)
    schedule.every().tuesday.at("09:15").do(run_trading_signals)
    schedule.every().wednesday.at("09:15").do(run_trading_signals)
    schedule.every().thursday.at("09:15").do(run_trading_signals)
    schedule.every().friday.at("09:15").do(run_trading_signals)
    
    schedule.every().monday.at("15:30").do(run_trading_signals)
    schedule.every().tuesday.at("15:30").do(run_trading_signals)
    schedule.every().wednesday.at("15:30").do(run_trading_signals)
    schedule.every().thursday.at("15:30").do(run_trading_signals)
    schedule.every().friday.at("15:30").do(run_trading_signals)
    
    logger.info(f"Analysis scheduled during market hours (9:00 AM - 3:30 PM) on weekdays")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    """Main function to run the Trading Signal Bot"""
    logger.info("=" * 50)
    logger.info("NIFTY 200 Trading Signal Bot - Starting Up")
    logger.info("Enhanced with Pattern Detection - Version 2.0")
    logger.info("=" * 50)
    
    # Test connections
    upstox_connected = test_upstox_connection()
    telegram_connected = test_telegram_connection()
    
    if not upstox_connected:
        logger.error("Cannot proceed without Upstox API connection")
        return
    
    if not telegram_connected:
        logger.warning("Telegram connection failed, proceeding without notifications")
    
    # Send startup notification
    send_startup_notification()
    
    # Run immediately on startup
    run_trading_signals()
    
    # Schedule future runs
    try:
        schedule_analysis()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
