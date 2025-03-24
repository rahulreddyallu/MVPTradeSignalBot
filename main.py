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
from upstox_client.models.ohlc import Ohlc as OHLCInterval
try:
    from aiogram import Bot, Dispatcher
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiogram"])
    from aiogram import Bot, Dispatcher
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
        # Directly use the access token
        access_token = UPSTOX_ACCESS_TOKEN  # Ensure this is set in your config
        upstox_client = Upstox(UPSTOX_API_KEY, access_token)
        return upstox_client
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

# Telegram notification function with retry mechanism
async def send_telegram_message(message, retry_attempts=5):
    if ENABLE_TELEGRAM_ALERTS:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
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
                    logger.error(f"Error sending Telegram message: {e}")
                    break
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

def fetch_ohlcv_data(symbol, start_date, end_date):
    try:
        # Initialize Upstox client
        upstox_client = initialize_upstox()
        if not upstox_client:
            raise Exception("Failed to initialize Upstox client")
        
        # Fetch OHLCV data
        ohlcv_data = upstox_client.get_ohlc(symbol, OHLCInterval.Day_1, start_date, end_date)
        df = pd.DataFrame(ohlcv_data)
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        return pd.DataFrame()

async def analyze_and_generate_signals():
    start_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORICAL_DAYS)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Example for NIFTY 200 stocks
    symbols = STOCK_LIST  # Replace with actual NIFTY 200 symbols

    for symbol in symbols:
        data = fetch_ohlcv_data(symbol, start_date, end_date)
        if data.empty:
            logger.error(f"No data fetched for {symbol}")
            continue

        data['EMA_SHORT'] = calculate_ema(data['Close'], EMA_SHORT)
        data['EMA_LONG'] = calculate_ema(data['Close'], EMA_LONG)
        data['RSI'] = calculate_rsi(data['Close'], RSI_PERIOD)
        data['MACD'], data['MACD_SIGNAL'] = calculate_macd(data['Close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        data['BB_UPPER'], data['BB_LOWER'] = calculate_bollinger_bands(data['Close'], BB_PERIOD, BB_STDDEV)
        data['SUPERTREND'] = calculate_supertrend(data, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
        data['ADX'] = calculate_adx(data, ADX_PERIOD)
        data['VWAP'] = calculate_vwap(data)

        signals = generate_signals(data)

        for signal in signals:
            message = f"{signal}\nSymbol: {symbol}\nPrice: {data['Close'].iloc[-1]}"
            await send_telegram_message(message)

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
        upstox_client = initialize_upstox()
        if upstox_client:
            logger.info("✅ Successfully authenticated with Upstox API")
            return True
        else:
            logger.error("❌ Failed to authenticate with Upstox API")
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
