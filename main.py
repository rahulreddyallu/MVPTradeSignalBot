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
from compute import *

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
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
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
        message = f"""
🚀 *NIFTY 200 Trading Signal Bot Started* 🚀

*Version:* 1.0.0
*Started at:* {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
*Analysis Frequency:* Every {ANALYSIS_FREQUENCY} hour(s)
*Stocks Monitored:* {len(STOCK_LIST)} NIFTY 200 stocks
*Timeframes Analyzed:* 
- Short Term (3-6 months)
- Long Term (>1 year)

Bot is now actively monitoring for trading signals.
        """
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
    and generates trading signals.
    """
    # Import necessary configurations
    from config import SIGNAL_MESSAGE_TEMPLATE, MINIMUM_SIGNAL_STRENGTH
    
    # Log function start with current UTC time
    current_datetime = datetime.datetime.now()
    logger.info(f"Starting analysis at {current_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Current date/time
    current_date_str = "2025-03-26 02:25:45"
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
    
    # Header for daily report
    daily_report = [
        f"📈 TRADING SIGNALS REPORT 📉",
        f"Date: {current_date_str} UTC",
        f"Analyzing {len(STOCK_LIST)} symbols with {HISTORICAL_DAYS} days of historical data",
        "-" * 40
    ]

    # Process each symbol in STOCK_LIST
    for symbol in STOCK_LIST:
        logger.info(f"Processing symbol: {symbol}")
        
        try:
            # Fetch historical data with daily interval
            data = fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day")
            
            if data.empty:
                logger.error(f"No historical data fetched for {symbol}")
                failed_analyses += 1
                continue
            
            logger.info(f"Analyzing {symbol} with {len(data)} data points")
            
            # Create a copy with lowercase column names for TechnicalAnalysis
            renamed_data = data.copy()
            renamed_data.columns = [col.lower() for col in renamed_data.columns]
            
            # Perform technical analysis
            analyzer = TechnicalAnalysis(renamed_data)
            
            # Generate signals using comprehensive analysis (indicators + patterns)
            signal_results = analyzer.generate_signals()
            
            # Extract data from results
            indicators_result = signal_results['indicators']
            patterns_result = signal_results['patterns']
            signals = signal_results['individual_signals']
            overall_signal = {
                'signal': signal_results['signal'],
                'strength': signal_results['strength'],
                'summary': f"{'Bullish' if signal_results['signal'] == 'BUY' else 'Bearish' if signal_results['signal'] == 'SELL' else 'Neutral'} signal with {signal_results['buy_signals_count' if signal_results['signal'] == 'BUY' else 'sell_signals_count']} indicators confirming"
            }
            
            if signals and overall_signal['strength'] >= MINIMUM_SIGNAL_STRENGTH:
                logger.info(f"Generated {len(signals)} signals for {symbol} with overall strength {overall_signal['strength']}/5")
                total_signals += len(signals)
                
                # Add to daily report
                daily_report.append(f"\n{symbol}: {overall_signal['signal']} (Strength: {overall_signal['strength']}/5)")
                daily_report.append(f"{overall_signal['summary']}")
                
                # Format indicators text for message template
                indicators_text = []
                
                # 1. Moving Averages
                if 'moving_averages' in indicators_result:
                    ma_data = indicators_result['moving_averages']['values']
                    ema_short = ma_data['ema_short']
                    ema_long = ma_data['ema_long']
                    sma_mid = ma_data['sma_mid']
                    sma_long = ma_data['sma_long']
                    
                    indicators_text.append(f"• Moving Averages:")
                    indicators_text.append(f"  - EMA: Short {ema_short} | Long {ema_long}")
                    if sma_mid:
                        indicators_text.append(f"  - SMA: Mid {sma_mid} | Long {sma_long}")
                
                # 2. RSI
                if 'rsi' in indicators_result:
                    rsi_data = indicators_result['rsi']['values']
                    rsi_value = rsi_data['rsi']
                    if rsi_value:
                        indicators_text.append(f"• RSI: {rsi_value:.2f} (OB:{rsi_data['overbought_threshold']}/OS:{rsi_data['oversold_threshold']})")
                
                # 3. MACD
                if 'macd' in indicators_result:
                    macd_data = indicators_result['macd']['values']
                    macd_line = macd_data['macd_line']
                    signal_line = macd_data['signal_line']
                    hist = macd_data['histogram']
                    indicators_text.append(f"• MACD: {macd_line:.4f} | Signal: {signal_line:.4f} | Hist: {hist:.4f}")
                
                # 4. Supertrend
                if 'supertrend' in indicators_result:
                    st_data = indicators_result['supertrend']['values']
                    st_value = st_data.get('supertrend')
                    direction = st_data['direction']
                    indicators_text.append(f"• Supertrend: {direction}" + (f" | Value: {st_value:.2f}" if st_value else ""))
                
                # 5. Bollinger Bands
                if 'bollinger_bands' in indicators_result:
                    bb_data = indicators_result['bollinger_bands']['values']
                    middle = bb_data['middle']
                    upper = bb_data['upper']
                    lower = bb_data['lower']
                    percent_b = bb_data.get('percent_b')
                    bandwidth = bb_data.get('bandwidth')
                    
                    indicators_text.append(f"• Bollinger Bands:")
                    indicators_text.append(f"  - Bands: Upper {upper:.2f} | Middle {middle:.2f} | Lower {lower:.2f}")
                    if percent_b is not None:
                        indicators_text.append(f"  - Position: %B {percent_b:.2f}" + (f" | Bandwidth: {bandwidth:.2f}" if bandwidth else ""))
                
                # 6. Stochastic
                if 'stochastic' in indicators_result:
                    stoch_data = indicators_result['stochastic']['values']
                    k_value = stoch_data.get('k')
                    d_value = stoch_data.get('d')
                    if k_value and d_value:
                        indicators_text.append(f"• Stochastic: %K {k_value:.2f} | %D {d_value:.2f}")
                
                # 7. Parabolic SAR
                if 'parabolic_sar' in indicators_result:
                    psar_data = indicators_result['parabolic_sar']['values']
                    sar_value = psar_data.get('sar')
                    trend = psar_data['trend']
                    if sar_value:
                        indicators_text.append(f"• Parabolic SAR: {sar_value:.2f} | Trend: {trend}")
                
                # 8. Aroon
                if 'aroon' in indicators_result:
                    aroon_data = indicators_result['aroon']['values']
                    aroon_up = aroon_data.get('aroon_up')
                    aroon_down = aroon_data.get('aroon_down')
                    strong_uptrend = aroon_data.get('strong_uptrend')
                    strong_downtrend = aroon_data.get('strong_downtrend')
                    
                    if aroon_up is not None and aroon_down is not None:
                        trend_str = "Strong Uptrend" if strong_uptrend else "Strong Downtrend" if strong_downtrend else "Neutral"
                        indicators_text.append(f"• Aroon: Up {aroon_up:.2f} | Down {aroon_down:.2f} | {trend_str}")
                
                # 9. Rate of Change
                if 'roc' in indicators_result:
                    roc_data = indicators_result['roc']['values']
                    roc_value = roc_data.get('roc')
                    trend = roc_data.get('trend')
                    if roc_value is not None:
                        indicators_text.append(f"• Rate of Change: {roc_value:.2f} | Trend: {trend}")
                
                # 10. ATR
                if 'atr' in indicators_result:
                    atr_data = indicators_result['atr']['values']
                    atr_value = atr_data.get('atr')
                    buy_stop = atr_data.get('buy_stop')
                    sell_stop = atr_data.get('sell_stop')
                    
                    indicators_text.append(f"• ATR: {atr_value:.2f}")
                    if overall_signal['signal'] == 'BUY' and buy_stop:
                        indicators_text.append(f"  - Suggested Stop Loss: {buy_stop:.2f}")
                    elif overall_signal['signal'] == 'SELL' and sell_stop:
                        indicators_text.append(f"  - Suggested Stop Loss: {sell_stop:.2f}")
                
                # 11. OBV
                if 'obv' in indicators_result:
                    obv_data = indicators_result['obv']['values']
                    rising = obv_data.get('rising')
                    indicators_text.append(f"• On-Balance Volume: {'Rising' if rising else 'Falling'}")
                
                # 12. VWAP
                if 'vwap' in indicators_result:
                    vwap_data = indicators_result['vwap']['values']
                    vwap_value = vwap_data.get('vwap')
                    price_to_vwap = vwap_data.get('price_to_vwap')
                    if vwap_value:
                        position = "Above VWAP" if price_to_vwap > 1 else "Below VWAP"
                        indicators_text.append(f"• VWAP: {vwap_value:.2f} | Price: {position} ({price_to_vwap:.2f}x)")
                
                # Format patterns text using detected patterns
                patterns_text = []
                
                # Add candlestick patterns
                if patterns_result['candlestick']:
                    patterns_text.append("• Candlestick Patterns:")
                    for pattern_name, pattern_data in patterns_result['candlestick'].items():
                        signal_type = "🟢 BUY" if pattern_data['signal'] == 1 else "🔴 SELL"
                        patterns_text.append(f"  - {pattern_name.replace('_', ' ').title()}: {signal_type}")
                
                # Add chart patterns
                if patterns_result['chart']:
                    patterns_text.append("• Chart Patterns:")
                    for pattern_name, pattern_data in patterns_result['chart'].items():
                        signal_type = "🟢 BUY" if pattern_data['signal'] == 1 else "🔴 SELL"
                        patterns_text.append(f"  - {pattern_name.replace('_', ' ').title()}: {signal_type}")
                
                # If no patterns detected
                if not patterns_text:
                    patterns_text = ["No specific chart patterns detected"]
                
                # Generate recommendation based on overall signal
                if overall_signal['signal'] == 'BUY':
                    recommendation = f"Consider LONG position. {overall_signal['summary']}."
                    if 'atr' in indicators_result:
                        stop_loss = indicators_result['atr']['values'].get('buy_stop')
                        if stop_loss:
                            recommendation += f" Set stop loss at {stop_loss:.2f}"
                elif overall_signal['signal'] == 'SELL':
                    recommendation = f"Consider SHORT position. {overall_signal['summary']}."
                    if 'atr' in indicators_result:
                        stop_loss = indicators_result['atr']['values'].get('sell_stop')
                        if stop_loss:
                            recommendation += f" Set stop loss at {stop_loss:.2f}"
                else:
                    recommendation = "No clear signal. Consider staying out of the market."
                
                # Format message using template
                message = SIGNAL_MESSAGE_TEMPLATE.format(
                    stock_name="",  # Would need company name lookup
                    stock_symbol=symbol,
                    current_price=f"{data['Close'].iloc[-1]:.2f}",
                    signal_type=overall_signal['signal'],
                    timeframe="Daily",
                    strength=overall_signal['strength'],
                    indicators="\n".join(indicators_text),
                    patterns="\n".join(patterns_text),
                    recommendation=recommendation,
                    timestamp=current_date_str
                )
                
                # Send via Telegram
                await send_telegram_message(message)
                logger.info(f"Sent signal alert for {symbol}")
                
            elif signals:
                logger.info(f"Signals generated for {symbol} but strength ({overall_signal['strength']}) below threshold")
                daily_report.append(f"\n{symbol}: {overall_signal['signal']} (Strength: {overall_signal['strength']}/5) - Below threshold")
            else:
                logger.info(f"No signals generated for {symbol}")
                daily_report.append(f"\n{symbol}: No trading signals")
            
            successful_analyses += 1
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            failed_analyses += 1
            daily_report.append(f"\n{symbol}: Error during analysis - {str(e)[:50]}...")
            continue
    
    # Finalize daily report
    daily_report.append("\n" + "-" * 40)
    daily_report.append(f"Analysis Summary:")
    daily_report.append(f"• Analyzed: {successful_analyses} symbols")
    daily_report.append(f"• Failed: {failed_analyses} symbols") 
    daily_report.append(f"• Total signals generated: {total_signals}")
    daily_report.append(f"• Report time: {current_date_str} UTC")
    
    # Send daily summary report via Telegram
    if successful_analyses > 0:
        await send_telegram_message("\n".join(daily_report))
    
    # Log summary statistics
    logger.info(f"Analysis completed. Processed {len(STOCK_LIST)} symbols.")
    logger.info(f"Successful analyses: {successful_analyses}")
    logger.info(f"Failed analyses: {failed_analyses}")
    logger.info(f"Total signals generated: {total_signals}")
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
            loop.run_until_complete(send_telegram_message(f"""
⚠️ *ERROR: Trading Signal Bot Failure* ⚠️

*Time:* {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
*Error:* {str(e)}

Please check the logs for more details.
            """))
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
        result = loop.run_until_complete(send_telegram_message("🔍 *Test Message* - NIFTY 200 Trading Signal Bot connection test successful!"))
        
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
