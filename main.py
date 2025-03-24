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
from compute import TradingSignalBot, UpstoxClient, TelegramSender
import config

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

def send_startup_notification():
    """Send a startup notification via Telegram"""
    try:
        telegram = TelegramSender(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        message = f"""
🚀 *NIFTY 200 Trading Signal Bot Started* 🚀

*Version:* 1.0.0
*Started at:* {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
*Analysis Frequency:* Every {config.ANALYSIS_FREQUENCY} hour(s)
*Stocks Monitored:* {len(config.STOCK_LIST)} NIFTY 200 stocks
*Timeframes Analyzed:* 
- Short Term (3-6 months)
- Long Term (>1 year)

Bot is now actively monitoring for trading signals.
        """
        telegram.send_message(message)
        logger.info("Startup notification sent")
    except Exception as e:
        logger.error(f"Failed to send startup notification: {str(e)}")

def run_trading_signals():
    """Run the trading signal generation process"""
    start_time = time.time()
    logger.info("Starting trading signal analysis")
    
    try:
        # Initialize the bot
        bot = TradingSignalBot()
        
        # Run the analysis
        bot.run()
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Completed trading signal analysis in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error in trading signal analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Send error notification
        try:
            telegram = TelegramSender(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
            error_message = f"""
⚠️ *ERROR: Trading Signal Bot Failure* ⚠️

*Time:* {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
*Error:* {str(e)}

Please check the logs for more details.
            """
            telegram.send_message(error_message)
        except:
            logger.error("Failed to send error notification")

def test_upstox_connection():
    """Test connection to Upstox API"""
    logger.info("Testing Upstox API connection...")
    
    try:
        client = UpstoxClient()
        if client.authenticate():
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
        telegram = TelegramSender(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        result = telegram.send_message("🔍 *Test Message* - NIFTY 200 Trading Signal Bot connection test successful!")
        
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
    import schedule
    
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
    
    # Alternatively, if you want to run at a fixed interval regardless of market hours:
    # schedule.every(config.ANALYSIS_FREQUENCY).hours.do(run_trading_signals)
    
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