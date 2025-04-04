"""
Core logic for NIFTY 200 Trading Signal Bot
Contains technical analysis, pattern recognition, and signal generation
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import datetime
import re
import time
import json
import logging
import telegram
from telegram.ext import Updater
import requests
from upstox_client.api_client import ApiClient
from upstox_client.api.login_api import LoginApi
from upstox_client.api.market_quote_api import MarketQuoteApi
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def escape_telegram_markdown(text):
    """Escape special characters for Telegram MarkdownV2 formatting."""
    if text is None:
        return "N/A"
    
    # Characters that need escaping in MarkdownV2
    special_chars = '_*[]()~`>#+-=|{}.!'
    
    # Escape each special character with a backslash
    result = ""
    for char in str(text):
        if char in special_chars:
            result += f"\\{char}"
        else:
            result += char
    
    return result

class UpstoxClient:
    def __init__(self):
        """Initialize Upstox client with API credentials"""
        self.api_key = config.UPSTOX_API_KEY
        self.api_secret = config.UPSTOX_API_SECRET
        self.redirect_uri = config.UPSTOX_REDIRECT_URI
        self.code = config.UPSTOX_CODE
        self.access_token = None
        self.client = None
        
    def authenticate(self):
        """Authenticate with Upstox API"""
        try:
            # Initialize API client
            api_client = ApiClient()
            login_api = LoginApi(api_client)
            
            # Assuming you have the client_id and other necessary details
            client_id = self.api_key
            redirect_uri = self.redirect_uri
            api_version = "v2"
            
            # This will just return the authorization URL
            # In a real application, you'd need to handle the OAuth flow manually
            auth_url = login_api.authorize(client_id, redirect_uri, api_version)
            logger.info(f"Authorization URL: {auth_url}")
            
            # Assuming you have the access token after the OAuth flow
            api_client.configuration.access_token = self.code  # The UPSTOX_CODE is actually the access token
            self.client = MarketQuoteApi(api_client)
            
            # Test the connection
            try:
                profile = self.client.get_profile()
                logger.info(f"Authentication successful for user: {profile['data']['user_name']}")
                return True
            except Exception as e:
                logger.error(f"Failed to validate access token: {str(e)}")
                # If the access token is invalid or expired, try to refresh it
                return self._refresh_token()
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def _refresh_token(self):
        """Refresh the access token if needed"""
        try:
            # Generate and set access token using authorization code
            url = "https://api.upstox.com/v2/login/authorization/token"
            headers = {
                'accept': 'application/json',
                'Api-Version': '2.0',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = {
                'code': self.code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            response = requests.post(url, headers=headers, data=data)
            response_data = json.loads(response.text)
            
            if 'access_token' not in response_data:
                logger.error(f"Authentication failed: {response.text}")
                return False
            
            # Store access token
            self.access_token = response_data['access_token']
            
            # Create client with access token
            api_client = ApiClient()
            api_client.configuration.access_token = self.access_token
            self.client = MarketQuoteApi(api_client)
            
            logger.info("Authentication refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            return False
    
    def get_historical_data(self, instrument_key, interval, from_date, to_date):
        """
        Get historical OHLCV data from Upstox
        
        Args:
            instrument_key: Instrument identifier
            interval: Time interval (1D, 1W, etc.)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert dates to epoch
            from_epoch = int(time.mktime(datetime.datetime.strptime(from_date, "%Y-%m-%d").timetuple()))
            to_epoch = int(time.mktime(datetime.datetime.strptime(to_date, "%Y-%m-%d").timetuple()))
            
            # Make API request
            historical_data = self.client.historical_candle_data(
                instrument_key=instrument_key,
                interval=interval,
                to_date=to_epoch,
                from_date=from_epoch
            )
            
            # Extract candle data
            candles = historical_data['data']['candles']
            
            # Create DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None

    def get_instrument_details(self, instrument_key):
        """Get instrument details from Upstox"""
        try:
            # Get market quote for the instrument
            market_quote = self.client.get_market_quote_full(instrument_key)
            
            # Extract basic instrument details from response
            instrument_details = {
                'name': market_quote['data']['company_name'],
                'tradingsymbol': market_quote['data']['symbol'],
                'exchange': market_quote['data']['exchange'],
                'last_price': market_quote['data']['last_price']
            }
            
            return instrument_details
            
        except Exception as e:
            logger.error(f"Error getting instrument details: {str(e)}")
            return None


class TechnicalAnalysis:
    def __init__(self, df):
        """
        Initialize Technical Analysis with OHLCV DataFrame
        
        Args:
            df: DataFrame with OHLCV data (index=timestamp, columns=[open, high, low, close, volume])
        """
        # Ensure column names are lowercase
        self.df = df.copy()
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.df.columns:
                logging.error(f"Missing required column: {col}")
                raise ValueError(f"DataFrame must contain {col} column")
        
        # Initialize results dictionaries and signals list
        self.indicators_result = {}
        self.patterns_result = {'candlestick': {}, 'chart': {}}
        self.signals = []
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators using pandas-ta"""
        try:
            self._calculate_moving_averages()
        except Exception as e:
            logger.warning(f"Error calculating Moving Averages: {e}")
            
        try:
            self._calculate_macd()
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            
        try:
            self._calculate_supertrend()
        except Exception as e:
            logger.warning(f"Error calculating SuperTrend: {e}")
            
        try:
            self._calculate_parabolic_sar()
        except Exception as e:
            logger.warning(f"Error calculating Parabolic SAR: {e}")
            
        try:
            self._calculate_aroon()
        except Exception as e:
            logger.warning(f"Error calculating Aroon: {e}")
            
        try:
            self._calculate_rsi()
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            
        try:
            self._calculate_stochastic()
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
            
        try:
            self._calculate_rate_of_change()
        except Exception as e:
            logger.warning(f"Error calculating ROC: {e}")
            
        try:
            self._calculate_bollinger_bands()
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            
        try:
            self._calculate_atr()
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            
        try:
            self._calculate_obv()
        except Exception as e:
            logger.warning(f"Error calculating OBV: {e}")
            
        try:
            self._calculate_vwap()
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
        
        return self.indicators_result
    
    def _calculate_moving_averages(self):
        """Calculate Simple and Exponential Moving Averages"""
        params = config.INDICATORS["moving_averages"]
        
        # Calculate SMAs
        self.df['sma_mid'] = ta.sma(self.df['close'], length=params['sma_mid'])
        self.df['sma_long'] = ta.sma(self.df['close'], length=params['sma_long'])
        
        # Calculate EMAs
        self.df['ema_short'] = ta.ema(self.df['close'], length=params['ema_short'])
        self.df['ema_long'] = ta.ema(self.df['close'], length=params['ema_long'])
        
        # Generate signals
        self.df['ema_crossover'] = 0
        self.df.loc[self.df['ema_short'] > self.df['ema_long'], 'ema_crossover'] = 1
        self.df.loc[self.df['ema_short'] < self.df['ema_long'], 'ema_crossover'] = -1
        
        # Detect crossovers
        self.df['ema_buy_signal'] = ((self.df['ema_crossover'].shift(1) == -1) & 
                                     (self.df['ema_crossover'] == 1)).astype(int)
        self.df['ema_sell_signal'] = ((self.df['ema_crossover'].shift(1) == 1) & 
                                      (self.df['ema_crossover'] == -1)).astype(int)
        
        # Save to results
        current_ema_signal = 0
        if self.df['ema_buy_signal'].iloc[-1] == 1:
            current_ema_signal = 1
        elif self.df['ema_sell_signal'].iloc[-1] == 1:
            current_ema_signal = -1
        
        self.indicators_result['moving_averages'] = {
            'signal': current_ema_signal,
            'values': {
                'ema_short': round(self.df['ema_short'].iloc[-1], 2),
                'ema_long': round(self.df['ema_long'].iloc[-1], 2),
                'sma_mid': round(self.df['sma_mid'].iloc[-1], 2) if not pd.isna(self.df['sma_mid'].iloc[-1]) else None,
                'sma_long': round(self.df['sma_long'].iloc[-1], 2) if not pd.isna(self.df['sma_long'].iloc[-1]) else None
            }
        }
        
        # Add to signals list
        if current_ema_signal != 0:
            self.signals.append({
                'indicator': 'EMA Crossover',
                'signal': 'BUY' if current_ema_signal == 1 else 'SELL',
                'strength': 3
            })
    
    def _calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence) using pandas-ta"""
        params = config.INDICATORS["macd"]
        
        # Calculate MACD with pandas-ta
        macd = ta.macd(
            self.df['close'], 
            fast=params['fast_period'], 
            slow=params['slow_period'], 
            signal=params['signal_period']
        )
        
        # Add MACD components to dataframe
        # Handle possible different naming conventions
        macd_columns = macd.columns.tolist()
        
        # Find the appropriate columns
        macd_line_col = next((col for col in macd_columns if col.startswith('MACD_')), None)
        signal_line_col = next((col for col in macd_columns if col.startswith('MACDs_')), None)
        histogram_col = next((col for col in macd_columns if col.startswith('MACDh_')), None)
        
        # Fallback to positional access if pattern matching fails
        if not all([macd_line_col, signal_line_col, histogram_col]) and len(macd_columns) >= 3:
            macd_line_col = macd_columns[0]
            signal_line_col = macd_columns[1]
            histogram_col = macd_columns[2]
            
        if not all([macd_line_col, signal_line_col, histogram_col]):
            logger.warning(f"Could not find MACD columns in: {macd_columns}")
            raise ValueError("Could not determine MACD column names")
        
        self.df['macd_line'] = macd[macd_line_col]
        self.df['signal_line'] = macd[signal_line_col]
        self.df['macd_histogram'] = macd[histogram_col]
        
        # Generate signals
        self.df['macd_crossover'] = 0
        self.df.loc[self.df['macd_line'] > self.df['signal_line'], 'macd_crossover'] = 1
        self.df.loc[self.df['macd_line'] < self.df['signal_line'], 'macd_crossover'] = -1
        
        # Detect crossovers
        self.df['macd_buy_signal'] = ((self.df['macd_crossover'].shift(1) == -1) & 
                                      (self.df['macd_crossover'] == 1)).astype(int)
        self.df['macd_sell_signal'] = ((self.df['macd_crossover'].shift(1) == 1) & 
                                       (self.df['macd_crossover'] == -1)).astype(int)
        
        # Save to results
        current_macd_signal = 0
        if self.df['macd_buy_signal'].iloc[-1] == 1:
            current_macd_signal = 1
        elif self.df['macd_sell_signal'].iloc[-1] == 1:
            current_macd_signal = -1
        
        self.indicators_result['macd'] = {
            'signal': current_macd_signal,
            'values': {
                'macd_line': round(self.df['macd_line'].iloc[-1], 4),
                'signal_line': round(self.df['signal_line'].iloc[-1], 4),
                'histogram': round(self.df['macd_histogram'].iloc[-1], 4)
            }
        }
        
        # Add to signals list
        if current_macd_signal != 0:
            self.signals.append({
                'indicator': 'MACD Crossover',
                'signal': 'BUY' if current_macd_signal == 1 else 'SELL',
                'strength': 3
            })
    
    def _calculate_supertrend(self):
        """Calculate Supertrend indicator using pandas-ta"""
        params = config.INDICATORS["supertrend"]
        
        try:
            # Calculate Supertrend using pandas-ta
            supertrend = ta.supertrend(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                length=params['period'],
                multiplier=params['multiplier']
            )
            
            # Print column names for debugging
            column_names = supertrend.columns.tolist()
            logger.info(f"SuperTrend columns: {column_names}")
            
            # Find the appropriate columns by positional reference (more reliable)
            if len(column_names) >= 2:
                # Typically, the first column is the SuperTrend value and the second is the direction
                supert_column = column_names[0]
                direction_column = column_names[1]
                
                logger.info(f"Using SuperTrend columns: {supert_column} and {direction_column}")
                
                # Extract SuperTrend components using the determined column names
                self.df['supertrend'] = supertrend[supert_column]
                self.df['supertrend_direction'] = supertrend[direction_column]
                
                # Generate signals
                # SuperTrend direction: 1 is bullish (uptrend), -1 is bearish (downtrend)
                # Note: The way signals are generated may depend on how direction is encoded in your version 
                
                # Check the actual values to determine convention
                direction_values = supertrend[direction_column].unique()
                logger.info(f"SuperTrend direction values: {direction_values}")
                
                # Try to determine the convention
                if 1 in direction_values and -1 in direction_values:
                    # Standard convention: 1 = bullish, -1 = bearish
                    self.df['supertrend_buy_signal'] = ((self.df['supertrend_direction'].shift(1) == -1) & 
                                                     (self.df['supertrend_direction'] == 1)).astype(int)
                    self.df['supertrend_sell_signal'] = ((self.df['supertrend_direction'].shift(1) == 1) & 
                                                      (self.df['supertrend_direction'] == -1)).astype(int)
                    
                    is_bullish = self.df['supertrend_direction'].iloc[-1] == 1
                
                elif True in direction_values and False in direction_values:
                    # Alternative convention: True = bullish, False = bearish
                    self.df['supertrend_buy_signal'] = ((self.df['supertrend_direction'].shift(1) == False) & 
                                                     (self.df['supertrend_direction'] == True)).astype(int)
                    self.df['supertrend_sell_signal'] = ((self.df['supertrend_direction'].shift(1) == True) & 
                                                      (self.df['supertrend_direction'] == False)).astype(int)
                    
                    is_bullish = self.df['supertrend_direction'].iloc[-1] == True
                
                else:
                    # If we can't determine the convention, assume standard
                    logger.warning(f"Unknown SuperTrend direction convention: {direction_values}")
                    self.df['supertrend_buy_signal'] = 0
                    self.df['supertrend_sell_signal'] = 0
                    is_bullish = self.df['close'].iloc[-1] > self.df['supertrend'].iloc[-1]
                
                # Save to results
                current_supertrend_signal = 0
                if self.df['supertrend_buy_signal'].iloc[-1] == 1:
                    current_supertrend_signal = 1
                elif self.df['supertrend_sell_signal'].iloc[-1] == 1:
                    current_supertrend_signal = -1
                
                self.indicators_result['supertrend'] = {
                    'signal': current_supertrend_signal,
                    'values': {
                        'supertrend': round(self.df['supertrend'].iloc[-1], 2) if not pd.isna(self.df['supertrend'].iloc[-1]) else None,
                        'direction': 'Bullish' if is_bullish else 'Bearish'
                    }
                }
                
                # Add to signals list
                if current_supertrend_signal != 0:
                    self.signals.append({
                        'indicator': 'Supertrend',
                        'signal': 'BUY' if current_supertrend_signal == 1 else 'SELL',
                        'strength': 4
                    })
            else:
                raise ValueError(f"SuperTrend calculation returned insufficient columns: {column_names}")
                    
        except Exception as e:
            logger.warning(f"Error calculating SuperTrend: {e}. Skipping this indicator.")
            # Return empty results to prevent further errors
            self.indicators_result['supertrend'] = {
                'signal': 0,
                'values': {
                    'supertrend': None,
                    'direction': 'Unknown'
                }
            }

    def _calculate_parabolic_sar(self):
        """Calculate Parabolic SAR indicator using pandas-ta"""
        params = config.INDICATORS["parabolic_sar"]
        
        try:
            # Calculate PSAR using pandas-ta
            psar = ta.psar(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                af=params['acceleration_factor'],
                max_af=params['max_acceleration_factor']
            )
            
            # Get actual column names
            psar_columns = psar.columns.tolist()
            
            # Find appropriate columns by pattern
            psar_long_col = None
            psar_short_col = None
            
            for col in psar_columns:
                if 'PSARl_' in col:
                    psar_long_col = col
                elif 'PSARs_' in col:
                    psar_short_col = col
            
            # Fallback to positional columns if needed
            if psar_long_col is None and len(psar_columns) >= 2:
                psar_long_col = psar_columns[0]
                psar_short_col = psar_columns[1]
            
            # Check if we found the columns
            if psar_long_col is None or psar_short_col is None:
                logger.warning(f"Could not find PSAR columns in: {psar_columns}")
                raise ValueError(f"PSAR columns not found in: {psar_columns}")
                
            # Add PSAR components to dataframe
            self.df['psar'] = psar[psar_long_col]  # PSARl for long positions
            self.df['psar_short'] = psar[psar_short_col]  # PSARs for short positions
            
            # Determine trend direction
            self.df['bull'] = self.df['close'] > self.df['psar']
            
            # Generate signals
            self.df['psar_buy_signal'] = ((self.df['bull'].shift(1) == False) & 
                                       (self.df['bull'] == True)).astype(int)
            self.df['psar_sell_signal'] = ((self.df['bull'].shift(1) == True) & 
                                        (self.df['bull'] == False)).astype(int)
            
            # Save to results
            current_psar_signal = 0
            if self.df['psar_buy_signal'].iloc[-1] == 1:
                current_psar_signal = 1
            elif self.df['psar_sell_signal'].iloc[-1] == 1:
                current_psar_signal = -1
            
            # Get current PSAR value (either long or short depending on trend)
            current_psar = self.df['psar'].iloc[-1] if not pd.isna(self.df['psar'].iloc[-1]) else self.df['psar_short'].iloc[-1]
            
            self.indicators_result['parabolic_sar'] = {
                'signal': current_psar_signal,
                'values': {
                    'sar': round(current_psar, 2) if not pd.isna(current_psar) else None,
                    'trend': 'Bullish' if self.df['bull'].iloc[-1] else 'Bearish'
                }
            }
            
            # Add to signals list
            if current_psar_signal != 0:
                self.signals.append({
                    'indicator': 'Parabolic SAR',
                    'signal': 'BUY' if current_psar_signal == 1 else 'SELL',
                    'strength': 3
                })
        except Exception as e:
            logger.warning(f"Error calculating Parabolic SAR: {e}. Skipping this indicator.")
            self.indicators_result['parabolic_sar'] = {
                'signal': 0,
                'values': {
                    'sar': None,
                    'trend': 'Unknown'
                }
            }

    def _calculate_aroon(self):
        """Calculate Aroon indicator using pandas-ta"""
        params = config.INDICATORS["aroon"]
        
        try:
            # Calculate Aroon using pandas-ta
            aroon = ta.aroon(
                high=self.df['high'],
                low=self.df['low'],
                length=params['period']
            )
            
            # Get column names
            aroon_columns = aroon.columns.tolist()
            
            # Find appropriate columns
            aroon_up_col = next((col for col in aroon_columns if 'AROONU_' in col), None)
            aroon_down_col = next((col for col in aroon_columns if 'AROOND_' in col), None)
            
            # Fallback to positional if needed
            if not all([aroon_up_col, aroon_down_col]) and len(aroon_columns) >= 2:
                aroon_up_col = aroon_columns[0]
                aroon_down_col = aroon_columns[1]
                
            if not all([aroon_up_col, aroon_down_col]):
                logger.warning(f"Could not find Aroon columns in: {aroon_columns}")
                raise ValueError("Could not determine Aroon column names")
            
            # Add Aroon components to dataframe
            self.df['aroon_up'] = aroon[aroon_up_col]
            self.df['aroon_down'] = aroon[aroon_down_col]
            
            # Generate signals
            self.df['aroon_crossover'] = 0
            self.df.loc[self.df['aroon_up'] > self.df['aroon_down'], 'aroon_crossover'] = 1
            self.df.loc[self.df['aroon_up'] < self.df['aroon_down'], 'aroon_crossover'] = -1
            
            # Detect crossovers
            self.df['aroon_buy_signal'] = ((self.df['aroon_crossover'].shift(1) == -1) & 
                                         (self.df['aroon_crossover'] == 1)).astype(int)
            self.df['aroon_sell_signal'] = ((self.df['aroon_crossover'].shift(1) == 1) & 
                                          (self.df['aroon_crossover'] == -1)).astype(int)
            
            # Strong trend signals
            self.df['aroon_strong_uptrend'] = ((self.df['aroon_up'] > params['uptrend_threshold']) & 
                                             (self.df['aroon_down'] < params['downtrend_threshold'])).astype(int)
            self.df['aroon_strong_downtrend'] = ((self.df['aroon_down'] > params['uptrend_threshold']) & 
                                               (self.df['aroon_up'] < params['downtrend_threshold'])).astype(int)
            
            # Save to results
            current_aroon_signal = 0
            if self.df['aroon_buy_signal'].iloc[-1] == 1:
                current_aroon_signal = 1
            elif self.df['aroon_sell_signal'].iloc[-1] == 1:
                current_aroon_signal = -1
            
            # Check for strong trends
            is_strong_uptrend = self.df['aroon_strong_uptrend'].iloc[-1] == 1
            is_strong_downtrend = self.df['aroon_strong_downtrend'].iloc[-1] == 1
            
            if is_strong_uptrend:
                current_aroon_signal = 1
            elif is_strong_downtrend:
                current_aroon_signal = -1
            
            self.indicators_result['aroon'] = {
                'signal': current_aroon_signal,
                'values': {
                    'aroon_up': round(self.df['aroon_up'].iloc[-1], 2) if not pd.isna(self.df['aroon_up'].iloc[-1]) else None,
                    'aroon_down': round(self.df['aroon_down'].iloc[-1], 2) if not pd.isna(self.df['aroon_down'].iloc[-1]) else None,
                    'strong_uptrend': is_strong_uptrend,
                    'strong_downtrend': is_strong_downtrend
                }
            }
            
            # Add to signals list
            if current_aroon_signal != 0:
                strength = 3  # Default strength
                if is_strong_uptrend or is_strong_downtrend:
                    strength = 4  # Higher strength for strong trends
                    
                self.signals.append({
                    'indicator': 'Aroon',
                    'signal': 'BUY' if current_aroon_signal == 1 else 'SELL',
                    'strength': strength
                })
        except Exception as e:
            logger.warning(f"Error calculating Aroon: {e}. Skipping this indicator.")
            self.indicators_result['aroon'] = {
                'signal': 0,
                'values': {
                    'aroon_up': None,
                    'aroon_down': None,
                    'strong_uptrend': False,
                    'strong_downtrend': False
                }
            }

    def _calculate_rsi(self):
        """Calculate Relative Strength Index using pandas-ta"""
        params = config.INDICATORS["rsi"]
        
        # Calculate RSI using pandas-ta
        self.df['rsi'] = ta.rsi(
            close=self.df['close'],
            length=params['period']
        )
        
        # Generate signals
        self.df['rsi_buy_signal'] = (self.df['rsi'] < params['oversold']).astype(int)
        self.df['rsi_sell_signal'] = (self.df['rsi'] > params['overbought']).astype(int)
        
        # Save to results
        current_rsi_signal = 0
        if self.df['rsi_buy_signal'].iloc[-1] == 1:
            current_rsi_signal = 1
        elif self.df['rsi_sell_signal'].iloc[-1] == 1:
            current_rsi_signal = -1
        
        self.indicators_result['rsi'] = {
            'signal': current_rsi_signal,
            'values': {
                'rsi': round(self.df['rsi'].iloc[-1], 2) if not pd.isna(self.df['rsi'].iloc[-1]) else None,
                'oversold_threshold': params['oversold'],
                'overbought_threshold': params['overbought']
            }
        }
        
        # Add to signals list
        if current_rsi_signal != 0:
            self.signals.append({
                'indicator': 'RSI',
                'signal': 'BUY' if current_rsi_signal == 1 else 'SELL',
                'strength': 2
            })

    def _calculate_stochastic(self):
        """Calculate Stochastic Oscillator using pandas-ta"""
        params = config.INDICATORS["stochastic"]
        
        try:
            # Calculate Stochastic using pandas-ta
            stoch = ta.stoch(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                k=params['k_period'],
                d=params['d_period']
            )
            
            # Get column names
            stoch_columns = stoch.columns.tolist()
            
            # Find appropriate columns
            k_col = next((col for col in stoch_columns if 'STOCHk_' in col), None)
            d_col = next((col for col in stoch_columns if 'STOCHd_' in col), None)
            
            # Fallback to positional if needed
            if not all([k_col, d_col]) and len(stoch_columns) >= 2:
                k_col = stoch_columns[0]
                d_col = stoch_columns[1]
                
            if not all([k_col, d_col]):
                logger.warning(f"Could not find Stochastic columns in: {stoch_columns}")
                raise ValueError("Could not determine Stochastic column names")
            
            # Add Stochastic components to dataframe
            self.df['stoch_k'] = stoch[k_col]
            self.df['stoch_d'] = stoch[d_col]
            
            # Generate signals
            self.df['stoch_signal'] = 0
            
            # Buy signal: K crosses above D in oversold region
            buy_condition = ((self.df['stoch_k'] > self.df['stoch_d']) &  # K above D
                             (self.df['stoch_k'].shift(1) <= self.df['stoch_d'].shift(1)) &  # Crossover
                             (self.df['stoch_k'] < params['oversold'] + 10))  # In oversold region
            
            # Sell signal: K crosses below D in overbought region
            sell_condition = ((self.df['stoch_k'] < self.df['stoch_d']) &  # K below D
                              (self.df['stoch_k'].shift(1) >= self.df['stoch_d'].shift(1)) &  # Crossover
                              (self.df['stoch_k'] > params['overbought'] - 10))  # In overbought region
            
            self.df.loc[buy_condition, 'stoch_signal'] = 1
            self.df.loc[sell_condition, 'stoch_signal'] = -1
            
            # Save to results
            current_stoch_signal = self.df['stoch_signal'].iloc[-1]
            
            self.indicators_result['stochastic'] = {
                'signal': current_stoch_signal,
                'values': {
                    'k': round(self.df['stoch_k'].iloc[-1], 2) if not pd.isna(self.df['stoch_k'].iloc[-1]) else None,
                    'd': round(self.df['stoch_d'].iloc[-1], 2) if not pd.isna(self.df['stoch_d'].iloc[-1]) else None,
                    'oversold': params['oversold'],
                    'overbought': params['overbought']
                }
            }
            
            # Add to signals list
            if current_stoch_signal != 0:
                self.signals.append({
                    'indicator': 'Stochastic',
                    'signal': 'BUY' if current_stoch_signal == 1 else 'SELL',
                    'strength': 2
                })
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}. Skipping this indicator.")
            self.indicators_result['stochastic'] = {
                'signal': 0,
                'values': {
                    'k': None,
                    'd': None,
                    'oversold': params['oversold'],
                    'overbought': params['overbought']
                }
            }

    def _calculate_rate_of_change(self):
        """Calculate Rate of Change (ROC) using pandas-ta"""
        params = config.INDICATORS["roc"]
        
        # Calculate ROC using pandas-ta
        self.df['roc'] = ta.roc(
            close=self.df['close'],
            length=params['period']
        )
        
        # Generate signals
        self.df['roc_crossover'] = 0
        self.df.loc[self.df['roc'] > 0, 'roc_crossover'] = 1
        self.df.loc[self.df['roc'] < 0, 'roc_crossover'] = -1
        
        # Detect crossovers
        self.df['roc_buy_signal'] = ((self.df['roc_crossover'].shift(1) == -1) & 
                                    (self.df['roc_crossover'] == 1)).astype(int)
        self.df['roc_sell_signal'] = ((self.df['roc_crossover'].shift(1) == 1) & 
                                     (self.df['roc_crossover'] == -1)).astype(int)
        
        # Save to results
        current_roc_signal = 0
        if self.df['roc_buy_signal'].iloc[-1] == 1:
            current_roc_signal = 1
        elif self.df['roc_sell_signal'].iloc[-1] == 1:
            current_roc_signal = -1
        
        self.indicators_result['roc'] = {
            'signal': current_roc_signal,
            'values': {
                'roc': round(self.df['roc'].iloc[-1], 2) if not pd.isna(self.df['roc'].iloc[-1]) else None,
                'trend': 'Bullish' if self.df['roc'].iloc[-1] > 0 else 'Bearish'
            }
        }
        
        # Add to signals list
        if current_roc_signal != 0:
            self.signals.append({
                'indicator': 'Rate of Change',
                'signal': 'BUY' if current_roc_signal == 1 else 'SELL',
                'strength': 2
            })

    def _calculate_bollinger_bands(self):
        """Calculate Bollinger Bands using pandas-ta"""
        params = config.INDICATORS["bollinger_bands"]
        
        try:
            # Calculate Bollinger Bands using pandas-ta
            bbands = ta.bbands(
                close=self.df['close'],
                length=params['period'],
                std=params['std_dev']
            )
            
            # Get actual column names from the result
            bbands_columns = bbands.columns.tolist()
            
            # Find the appropriate columns by pattern matching
            lower_band_col = None
            middle_band_col = None
            upper_band_col = None
            
            # Look for columns by pattern
            for col in bbands_columns:
                if 'BBL_' in col:
                    lower_band_col = col
                elif 'BBM_' in col:
                    middle_band_col = col
                elif 'BBU_' in col:
                    upper_band_col = col
            
            # Fallback - use positional columns if pattern matching fails
            if lower_band_col is None and len(bbands_columns) >= 3:
                lower_band_col = bbands_columns[0]
                middle_band_col = bbands_columns[1]
                upper_band_col = bbands_columns[2]
                
            # Check if we found the columns
            if lower_band_col is None or middle_band_col is None or upper_band_col is None:
                logger.warning(f"Could not find Bollinger Bands columns in: {bbands_columns}")
                raise ValueError("Could not determine Bollinger Bands column names")
                
            # Add Bollinger Bands components to dataframe
            self.df['bb_lower'] = bbands[lower_band_col]
            self.df['bb_middle'] = bbands[middle_band_col]
            self.df['bb_upper'] = bbands[upper_band_col]
            
            # Calculate %B (position within bands)
            self.df['bb_pct_b'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
            
            # Generate signals - check for RSI confirmation
            if 'rsi' not in self.df.columns:
                self._calculate_rsi()
                
            self.df['bb_buy_signal'] = ((self.df['close'] <= self.df['bb_lower']) & 
                                      (self.df['rsi'] < 30)).astype(int)
            self.df['bb_sell_signal'] = ((self.df['close'] >= self.df['bb_upper']) & 
                                       (self.df['rsi'] > 70)).astype(int)
            
            # Save to results
            current_bb_signal = 0
            if self.df['bb_buy_signal'].iloc[-1] == 1:
                current_bb_signal = 1
            elif self.df['bb_sell_signal'].iloc[-1] == 1:
                current_bb_signal = -1
            
            # Calculate Bollinger Bandwidth (indicator of volatility)
            bb_bandwidth = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
            
            self.indicators_result['bollinger_bands'] = {
                'signal': current_bb_signal,
                'values': {
                    'middle': round(self.df['bb_middle'].iloc[-1], 2) if not pd.isna(self.df['bb_middle'].iloc[-1]) else None,
                    'upper': round(self.df['bb_upper'].iloc[-1], 2) if not pd.isna(self.df['bb_upper'].iloc[-1]) else None,
                    'lower': round(self.df['bb_lower'].iloc[-1], 2) if not pd.isna(self.df['bb_lower'].iloc[-1]) else None,
                    'percent_b': round(self.df['bb_pct_b'].iloc[-1], 2) if not pd.isna(self.df['bb_pct_b'].iloc[-1]) else None,
                    'bandwidth': round(bb_bandwidth.iloc[-1], 2) if not pd.isna(bb_bandwidth.iloc[-1]) else None
                }
            }
            
            # Add to signals list
            if current_bb_signal != 0:
                self.signals.append({
                    'indicator': 'Bollinger Bands',
                    'signal': 'BUY' if current_bb_signal == 1 else 'SELL',
                    'strength': 3
                })
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}. Skipping this indicator.")
            self.indicators_result['bollinger_bands'] = {
                'signal': 0,
                'values': {
                    'middle': None,
                    'upper': None,
                    'lower': None,
                    'percent_b': None,
                    'bandwidth': None
                }
            }

    def _calculate_atr(self):
        """Calculate Average True Range (ATR) using pandas-ta"""
        params = config.INDICATORS["atr"]
        
        try:
            # Calculate ATR using pandas-ta
            atr_value = ta.atr(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                length=params['period']
            )
            
            self.df['atr'] = atr_value
            
            # Calculate stop loss levels
            self.df['atr_buy_stop'] = self.df['close'] - (self.df['atr'] * params['multiplier'])
            self.df['atr_sell_stop'] = self.df['close'] + (self.df['atr'] * params['multiplier'])
            
            # Save to results
            self.indicators_result['atr'] = {
                'signal': 0,  # ATR doesn't generate buy/sell signals directly
                'values': {
                    'atr': round(self.df['atr'].iloc[-1], 2),
                    'buy_stop': round(self.df['atr_buy_stop'].iloc[-1], 2),
                    'sell_stop': round(self.df['atr_sell_stop'].iloc[-1], 2)
                }
            }
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}. Skipping this indicator.")
            self.indicators_result['atr'] = {
                'signal': 0,
                'values': {
                    'atr': None,
                    'buy_stop': None,
                    'sell_stop': None
                }
            }

    def _calculate_obv(self):
        """Calculate On-Balance Volume (OBV) using pandas-ta"""
        # Calculate OBV using pandas-ta
        self.df['obv'] = ta.obv(self.df['close'], self.df['volume'])
        
        # Calculate OBV moving average
        self.df['obv_ma'] = self.df['obv'].rolling(14).mean()
        
        # Generate signals
        self.df['obv_signal'] = 0
        
        # Bullish divergence: Price making new lows but OBV making higher lows
        price_lower_low = (self.df['close'] < self.df['close'].shift(1)) & (self.df['close'].shift(1) < self.df['close'].shift(2))
        obv_higher_low = (self.df['obv'] > self.df['obv'].shift(1)) & (self.df['obv'].shift(1) > self.df['obv'].shift(2))
        bullish_divergence = price_lower_low & obv_higher_low
        
        # Bearish divergence: Price making new highs but OBV making lower highs
        price_higher_high = (self.df['close'] > self.df['close'].shift(1)) & (self.df['close'].shift(1) > self.df['close'].shift(2))
        obv_lower_high = (self.df['obv'] < self.df['obv'].shift(1)) & (self.df['obv'].shift(1) < self.df['obv'].shift(2))
        bearish_divergence = price_higher_high & obv_lower_high
        
        self.df.loc[bullish_divergence, 'obv_signal'] = 1
        self.df.loc[bearish_divergence, 'obv_signal'] = -1
        
        # Save to results
        current_obv_signal = self.df['obv_signal'].iloc[-1]
        
        rising_obv = self.df['obv'].iloc[-1] > self.df['obv'].iloc[-5]  # Check if OBV is rising
        
        self.indicators_result['obv'] = {
            'signal': current_obv_signal,
            'values': {
                'obv': int(self.df['obv'].iloc[-1]),
                'obv_ma': int(self.df['obv_ma'].iloc[-1]) if not pd.isna(self.df['obv_ma'].iloc[-1]) else None,
                'rising': rising_obv
            }
        }
        
        # Add to signals list
        if current_obv_signal != 0:
            self.signals.append({
                'indicator': 'On-Balance Volume',
                'signal': 'BUY' if current_obv_signal == 1 else 'SELL',
                'strength': 2
            })

    def _calculate_vwap(self):
        """Calculate Volume Weighted Average Price (VWAP) using pandas-ta"""
        # Calculate typical price
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        volume_price = typical_price * self.df['volume']
        
        # Calculate cumulative values
        cumulative_volume_price = volume_price.cumsum()
        cumulative_volume = self.df['volume'].cumsum()
        
        # Calculate VWAP
        self.df['vwap'] = cumulative_volume_price / cumulative_volume
        
        # Generate signals
        self.df['vwap_signal'] = 0
        
        # Buy: Price crossing above VWAP with high volume
        high_volume = self.df['volume'] > self.df['volume'].rolling(20).mean() * 1.5
        price_cross_above_vwap = (self.df['close'] > self.df['vwap']) & (self.df['close'].shift(1) < self.df['vwap'].shift(1))
        buy_signal = price_cross_above_vwap & high_volume
        
        # Sell: Price crossing below VWAP with high volume
        price_cross_below_vwap = (self.df['close'] < self.df['vwap']) & (self.df['close'].shift(1) > self.df['vwap'].shift(1))
        sell_signal = price_cross_below_vwap & high_volume
        
        self.df.loc[buy_signal, 'vwap_signal'] = 1
        self.df.loc[sell_signal, 'vwap_signal'] = -1
        
        # Save to results
        current_vwap_signal = self.df['vwap_signal'].iloc[-1]
        
        self.indicators_result['vwap'] = {
            'signal': current_vwap_signal,
            'values': {
                'vwap': round(self.df['vwap'].iloc[-1], 2),
                'price_to_vwap': round(self.df['close'].iloc[-1] / self.df['vwap'].iloc[-1], 2)
            }
        }
        
        # Add to signals list
        if current_vwap_signal != 0:
            self.signals.append({
                'indicator': 'VWAP',
                'signal': 'BUY' if current_vwap_signal == 1 else 'SELL',
                'strength': 3
            })

    def detect_candlestick_patterns(self):
        """Detect candlestick patterns in the data"""
        patterns = {}
        
        # Only detect enabled patterns
        candlestick_config = config.CANDLESTICK_PATTERNS
        
        # Bullish Engulfing
        if candlestick_config.get('bullish_engulfing', False):
            bullish_engulfing = (
                (self.df['close'].shift(1) < self.df['open'].shift(1)) &  # Previous candle is bearish
                (self.df['close'] > self.df['open']) &  # Current candle is bullish
                (self.df['open'] < self.df['close'].shift(1)) &  # Current open below previous close
                (self.df['close'] > self.df['open'].shift(1))  # Current close above previous open
            )
            if bullish_engulfing.iloc[-1]:
                patterns['bullish_engulfing'] = {
                    'signal': 1,  # Buy signal
                    'strength': 3
                }
        
        # Bearish Engulfing
        if candlestick_config.get('bearish_engulfing', False):
            bearish_engulfing = (
                (self.df['close'].shift(1) > self.df['open'].shift(1)) &  # Previous candle is bullish
                (self.df['close'] < self.df['open']) &  # Current candle is bearish
                (self.df['open'] > self.df['close'].shift(1)) &  # Current open above previous close
                (self.df['close'] < self.df['open'].shift(1))  # Current close below previous open
            )
            if bearish_engulfing.iloc[-1]:
                patterns['bearish_engulfing'] = {
                    'signal': -1,  # Sell signal
                    'strength': 3
                }
        
        # Doji
        if candlestick_config.get('doji', False):
            doji = abs(self.df['close'] - self.df['open']) <= (0.05 * (self.df['high'] - self.df['low']))
            if doji.iloc[-1]:
                # Determine if bullish or bearish based on trend
                if self.df['close'].iloc[-1] > self.df['close'].iloc[-5]:
                    patterns['doji'] = {
                        'signal': -1,  # Sell signal (potential reversal in uptrend)
                        'strength': 2
                    }
                else:
                    patterns['doji'] = {
                        'signal': 1,  # Buy signal (potential reversal in downtrend)
                        'strength': 2
                    }
        
        # Hammer
        if candlestick_config.get('hammer', False):
            hammer = (
                (self.df['high'] - self.df['low'] > 3 * (self.df['open'] - self.df['close'])) &  # Long shadow
                (self.df['close'] - self.df['low'] > 0.6 * (self.df['high'] - self.df['low'])) &  # Close near high
                (self.df['open'] - self.df['low'] > 0.6 * (self.df['high'] - self.df['low']))  # Open near high
            )
            if hammer.iloc[-1] and self.df['close'].iloc[-2] < self.df['close'].iloc[-3]:
                patterns['hammer'] = {
                    'signal': 1,  # Buy signal
                    'strength': 3
                }
        
        # Shooting Star
        if candlestick_config.get('shooting_star', False):
            shooting_star = (
                (self.df['high'] - self.df['low'] > 3 * (self.df['open'] - self.df['close'])) &  # Long shadow
                (self.df['high'] - self.df['close'] > 0.6 * (self.df['high'] - self.df['low'])) &  # Close near low
                (self.df['high'] - self.df['open'] > 0.6 * (self.df['high'] - self.df['low']))  # Open near low
            )
            if shooting_star.iloc[-1] and self.df['close'].iloc[-2] > self.df['close'].iloc[-3]:
                patterns['shooting_star'] = {
                    'signal': -1,  # Sell signal
                    'strength': 3
                }
        
        # Morning Star
        if candlestick_config.get('morning_star', False):
            morning_star = (
                (self.df['close'].shift(2) < self.df['open'].shift(2)) &  # First day bearish
                (abs(self.df['close'].shift(1) - self.df['open'].shift(1)) <
                 0.3 * (self.df['high'].shift(1) - self.df['low'].shift(1))) &  # Second day small body
                (self.df['close'] > self.df['open']) &  # Third day bullish
                (self.df['close'] > (self.df['close'].shift(2) + self.df['open'].shift(2)) / 2)  # Close above midpoint of first day
            )
            if morning_star.iloc[-1]:
                patterns['morning_star'] = {
                    'signal': 1,  # Buy signal
                    'strength': 4
                }
        
        # Evening Star
        if candlestick_config.get('evening_star', False):
            evening_star = (
                (self.df['close'].shift(2) > self.df['open'].shift(2)) &  # First day bullish
                (abs(self.df['close'].shift(1) - self.df['open'].shift(1)) <
                 0.3 * (self.df['high'].shift(1) - self.df['low'].shift(1))) &  # Second day small body
                (self.df['close'] < self.df['open']) &  # Third day bearish
                (self.df['close'] < (self.df['close'].shift(2) + self.df['open'].shift(2)) / 2)  # Close below midpoint of first day
            )
            if evening_star.iloc[-1]:
                patterns['evening_star'] = {
                    'signal': -1,  # Sell signal
                    'strength': 4
                }
        
        self.patterns_result['candlestick'] = patterns
        
        # Add patterns to signals list
        for pattern_name, pattern_data in patterns.items():
            self.signals.append({
                'indicator': f"Candlestick: {pattern_name.replace('_', ' ').title()}",
                'signal': 'BUY' if pattern_data['signal'] == 1 else 'SELL',
                'strength': pattern_data['strength']
            })
        
        return patterns

    def detect_chart_patterns(self):
        """Detect chart patterns in the data"""
        patterns = {}
        
        # Only detect enabled patterns
        chart_config = config.CHART_PATTERNS
        
        # Function to detect local maxima/minima
        def detect_peaks(data, window=10):
            peaks = []
            for i in range(window, len(data) - window):
                if data.iloc[i] == max(data.iloc[i-window:i+window+1]):
                    peaks.append((i, data.iloc[i]))
            return peaks
        
        def detect_troughs(data, window=10):
            troughs = []
            for i in range(window, len(data) - window):
                if data.iloc[i] == min(data.iloc[i-window:i+window+1]):
                    troughs.append((i, data.iloc[i]))
            return troughs
        
        # Head and Shoulders
        if chart_config.get('head_and_shoulders', False):
            # Find recent peaks
            peaks = detect_peaks(self.df['high'], window=10)
            
            # We need at least 3 recent peaks for H&S pattern
            if len(peaks) >= 3:
                # Get the last 3 peaks
                last_peaks = sorted(peaks[-3:])
                
                # Check if the middle peak is higher than the other two (head is higher than shoulders)
                if last_peaks[1][1] > last_peaks[0][1] and last_peaks[1][1] > last_peaks[2][1]:
                    # Check if the shoulders are at approximately the same level (within 5%)
                    if abs(last_peaks[0][1] - last_peaks[2][1]) / last_peaks[0][1] < 0.05:
                        # Check if price recently broke the neckline
                        # The neckline is the line connecting the lows after the first and second peaks
                        troughs = detect_troughs(self.df['low'], window=5)
                        if len(troughs) >= 2:
                            trough1 = troughs[-2][1]
                            trough2 = troughs[-1][1]
                            neckline = trough1
                            
                            # Check if the price recently broke below the neckline
                            if self.df['close'].iloc[-1] < neckline:
                                patterns['head_and_shoulders'] = {
                                    'signal': -1,  # Sell signal
                                    'strength': 4
                                }
        
        # Inverse Head and Shoulders
        if chart_config.get('inverse_head_and_shoulders', False):
            # Find recent troughs
            troughs = detect_troughs(self.df['low'], window=10)
            
            # We need at least 3 recent troughs for inverse H&S pattern
            if len(troughs) >= 3:
                # Get the last 3 troughs
                last_troughs = sorted(troughs[-3:])
                
                # Check if the middle trough is lower than the other two (head is lower than shoulders)
                if last_troughs[1][1] < last_troughs[0][1] and last_troughs[1][1] < last_troughs[2][1]:
                    # Check if the shoulders are at approximately the same level (within 5%)
                    if abs(last_troughs[0][1] - last_troughs[2][1]) / last_troughs[0][1] < 0.05:
                        # Check if price recently broke the neckline
                        # The neckline is the line connecting the highs after the first and second troughs
                        peaks = detect_peaks(self.df['high'], window=5)
                        if len(peaks) >= 2:
                            peak1 = peaks[-2][1]
                            peak2 = peaks[-1][1]
                            neckline = peak1
                            
                            # Check if the price recently broke above the neckline
                            if self.df['close'].iloc[-1] > neckline:
                                patterns['inverse_head_and_shoulders'] = {
                                    'signal': 1,  # Buy signal
                                    'strength': 4
                                }
        
        # Double Top
        if chart_config.get('double_top', False):
            # Find recent peaks
            peaks = detect_peaks(self.df['high'], window=10)
            
            # We need at least 2 recent peaks for double top pattern
            if len(peaks) >= 2:
                # Get the last 2 peaks
                peak1 = peaks[-2][1]
                peak2 = peaks[-1][1]
                
                # Check if the peaks are at approximately the same level (within 3%)
                if abs(peak1 - peak2) / peak1 < 0.03:
                    # Find the trough between the peaks
                    troughs = detect_troughs(self.df['low'], window=5)
                    if troughs:
                        neckline = troughs[-1][1]
                        
                        # Check if the price recently broke below the neckline
                        if self.df['close'].iloc[-1] < neckline:
                            patterns['double_top'] = {
                                'signal': -1,  # Sell signal
                                'strength': 3
                            }
        
        # Double Bottom
        if chart_config.get('double_bottom', False):
            # Find recent troughs
            troughs = detect_troughs(self.df['low'], window=10)
            
            # We need at least 2 recent troughs for double bottom pattern
            if len(troughs) >= 2:
                # Get the last 2 troughs
                trough1 = troughs[-2][1]
                trough2 = troughs[-1][1]
                
                # Check if the troughs are at approximately the same level (within 3%)
                if abs(trough1 - trough2) / trough1 < 0.03:
                    # Find the peak between the troughs
                    peaks = detect_peaks(self.df['high'], window=5)
                    if peaks:
                        neckline = peaks[-1][1]
                        
                        # Check if the price recently broke above the neckline
                        if self.df['close'].iloc[-1] > neckline:
                            patterns['double_bottom'] = {
                                'signal': 1,  # Buy signal
                                'strength': 3
                            }
        
        # Cup and Handle
        if chart_config.get('cup_and_handle', False):
            # This pattern is complex and may need a more sophisticated approach
            # Here's a simplified version
            
            if len(self.df) >= 100:  # Need enough data for this pattern
                # Look for a rounded bottom (cup) followed by a smaller pullback (handle)
                
                # Check for a rounded bottom in the first part
                mid_point = len(self.df) - 30
                cup_start = mid_point - 50
                cup_end = mid_point
                
                if cup_start >= 0:  # Ensure we have enough data
                    start_price = self.df['close'].iloc[cup_start]
                    end_price = self.df['close'].iloc[cup_end]
                    
                    # Price should be similar at start and end of cup
                    if abs(start_price - end_price) / start_price < 0.05:
                        # Check for a dip in the middle
                        middle_min = self.df['low'].iloc[cup_start:cup_end].min()
                        if middle_min < 0.9 * start_price:
                            # Look for a smaller pullback (handle)
                            handle_start = cup_end
                            handle_end = len(self.df) - 1
                            
                            handle_low = self.df['low'].iloc[handle_start:handle_end].min()
                            
                            # Handle should not go as low as the cup
                            if handle_low > middle_min:
                                # Check if price recently broke the cup level
                                if self.df['close'].iloc[-1] > start_price:
                                    patterns['cup_and_handle'] = {
                                        'signal': 1,  # Buy signal
                                        'strength': 4
                                    }
        
        self.patterns_result['chart'] = patterns
        
        # Add patterns to signals list
        for pattern_name, pattern_data in patterns.items():
            self.signals.append({
                'indicator': f"Chart Pattern: {pattern_name.replace('_', ' ').title()}",
                'signal': 'BUY' if pattern_data['signal'] == 1 else 'SELL',
                'strength': pattern_data['strength']
            })
        
        return patterns

    def generate_signals(self):
        """Generate trading signals based on all indicators and patterns"""
        # Clear existing signals list before recalculating
        self.signals = []
        
        # Get signals from all indicators
        self.calculate_all_indicators()
        
        # Get signals from patterns
        self.detect_candlestick_patterns()
        self.detect_chart_patterns()
        
        # Calculate overall signal strength
        buy_signals = [s for s in self.signals if s['signal'] == 'BUY']
        sell_signals = [s for s in self.signals if s['signal'] == 'SELL']
        
        buy_strength = sum(s['strength'] for s in buy_signals)
        sell_strength = sum(s['strength'] for s in sell_signals)
        
        # Determine overall signal
        overall_signal = 'NEUTRAL'
        signal_strength = 0
        
        if buy_strength > sell_strength and buy_strength >= config.MINIMUM_SIGNAL_STRENGTH:
            overall_signal = 'BUY'
            signal_strength = min(5, int(buy_strength / 3))  # Scale to 1-5
        elif sell_strength > buy_strength and sell_strength >= config.MINIMUM_SIGNAL_STRENGTH:
            overall_signal = 'SELL'
            signal_strength = min(5, int(sell_strength / 3))  # Scale to 1-5
        
        return {
            'signal': overall_signal,
            'strength': signal_strength,
            'indicators': self.indicators_result,
            'patterns': self.patterns_result,
            'individual_signals': self.signals,
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals),
            'current_price': self.df['close'].iloc[-1],
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_overall_signal(self):
        """
        Determine overall trading signal based on all indicator signals.
        
        Returns:
        --------
        dict with overall signal information
        """
        if not self.signals:
            # Calculate indicators and patterns if not already done
            self.generate_signals()
            
        # Count buy and sell signals
        buy_signals = [s for s in self.signals if s['signal'] == 'BUY']
        sell_signals = [s for s in self.signals if s['signal'] == 'SELL']
        
        # Calculate weighted signal strength
        buy_strength = sum(s['strength'] for s in buy_signals)
        sell_strength = sum(s['strength'] for s in sell_signals)
        
        # Determine overall signal
        if buy_strength > sell_strength:
            signal_type = 'BUY'
            strength = min(5, max(1, int((buy_strength - sell_strength) / 2)))
            summary = f"Bullish signal with {len(buy_signals)} indicators confirming"
        elif sell_strength > buy_strength:
            signal_type = 'SELL'
            strength = min(5, max(1, int((sell_strength - buy_strength) / 2)))
            summary = f"Bearish signal with {len(sell_signals)} indicators confirming"
        else:
            signal_type = 'NEUTRAL'
            strength = 0
            summary = "Mixed signals, no clear direction"
        
        return {
            'signal': signal_type,
            'strength': strength,
            'summary': summary,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_indicators_summary(self):
        """Get a formatted string summary of all indicators"""
        summary = []
        
        # Ensure we have calculated all indicators
        if not self.indicators_result:
            self.calculate_all_indicators()
        
        # Moving Averages
        if 'moving_averages' in self.indicators_result:
            ma = self.indicators_result['moving_averages']
            summary.append(f"Moving Averages: {'Bullish' if ma['signal'] == 1 else 'Bearish' if ma['signal'] == -1 else 'Neutral'}")
            summary.append(f"  EMA(9): {ma['values']['ema_short']}, EMA(21): {ma['values']['ema_long']}")
            
        # RSI
        if 'rsi' in self.indicators_result:
            rsi = self.indicators_result['rsi']['values']['rsi']
            summary.append(f"RSI: {rsi:.2f} - {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}")
            
        # MACD
        if 'macd' in self.indicators_result:
            macd = self.indicators_result['macd']
            summary.append(f"MACD: {'Bullish' if macd['signal'] == 1 else 'Bearish' if macd['signal'] == -1 else 'Neutral'}")
            
        # Supertrend
        if 'supertrend' in self.indicators_result:
            st = self.indicators_result['supertrend']
            summary.append(f"Supertrend: {st['values']['direction']}")
            
        # Pattern Detection 
        if self.patterns_result:
            patterns_found = []
            
            # Candlestick patterns
            for pattern, data in self.patterns_result['candlestick'].items():
                signal = 'Bullish' if data['signal'] == 1 else 'Bearish'
                patterns_found.append(f"{pattern.replace('_', ' ').title()} ({signal})")
                
            # Chart patterns 
            for pattern, data in self.patterns_result['chart'].items():
                signal = 'Bullish' if data['signal'] == 1 else 'Bearish'
                patterns_found.append(f"{pattern.replace('_', ' ').title()} ({signal})")
            
            if patterns_found:
                summary.append(f"Patterns Detected: {', '.join(patterns_found)}")
        
        return '\n'.join(summary)

class TelegramSender:
    def __init__(self, token, chat_id):
        """Initialize Telegram sender with bot token and chat ID"""
        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=token)
    
    def send_message(self, text):
        """Send message to Telegram chat with MarkdownV2 formatting"""
        try:
            self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode='MarkdownV2'  # Change from 'Markdown' to 'MarkdownV2'
            )
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False


class TradingSignalBot:
    def __init__(self):
        """Initialize Trading Signal Bot"""
        self.upstox_client = UpstoxClient()
        self.telegram = TelegramSender(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    
    def run(self):
        """Run the trading signal bot for all stocks in the list"""
        # Authenticate with Upstox
        if not self.upstox_client.authenticate():
            logger.error("Failed to authenticate with Upstox API")
            return
        
        # Get current date
        today = datetime.datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Process each stock
        for instrument_key in config.STOCK_LIST:
            try:
                # Get instrument details
                instrument_details = self.upstox_client.get_instrument_details(instrument_key)
                if not instrument_details:
                    logger.error(f"Failed to get details for instrument: {instrument_key}")
                    continue
                
                stock_name = instrument_details['name']
                stock_symbol = instrument_details['tradingsymbol']
                
                logger.info(f"Analyzing {stock_name} ({stock_symbol})")
                
                # Process short term signals
                short_term_from_date = (datetime.datetime.datetime.now() - datetime.timedelta(days=config.SHORT_TERM_LOOKBACK)).strftime("%Y-%m-%d")
                short_term_signals = self._analyze_stock(
                    instrument_key, 
                    stock_name, 
                    stock_symbol, 
                    config.INTERVALS["short_term"], 
                    short_term_from_date, 
                    today,
                    "Short Term (3-6 months)"
                )
                
                # Process long term signals
                long_term_from_date = (datetime.datetime.datetime.now() - datetime.timedelta(days=config.LONG_TERM_LOOKBACK)).strftime("%Y-%m-%d")
                long_term_signals = self._analyze_stock(
                    instrument_key, 
                    stock_name, 
                    stock_symbol, 
                    config.INTERVALS["long_term"], 
                    long_term_from_date, 
                    today,
                    "Long Term (> 1 year)"
                )
                
                # Sleep to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {instrument_key}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _analyze_stock(self, instrument_key, stock_name, stock_symbol, interval, from_date, to_date, timeframe):
        """Analyze a single stock and send signals if needed"""
        # Get historical data
        df = self.upstox_client.get_historical_data(instrument_key, interval, from_date, to_date)
        if df is None or len(df) < 50:  # Need at least 50 candles for analysis
            logger.warning(f"Insufficient data for {stock_symbol} with interval {interval}")
            return None
        
        # Perform technical analysis
        analysis = TechnicalAnalysis(df)
        signals = analysis.generate_signals()
        
        # Only send message if there's a BUY or SELL signal
        if signals['signal'] != 'NEUTRAL':
            message = self._format_signal_message(
                stock_name,
                stock_symbol,
                signals,
                timeframe
            )
            
            self.telegram.send_message(message)
            logger.info(f"Sent {signals['signal']} signal for {stock_symbol} ({timeframe}) - {signals['buy_signals_count']} buy vs {signals['sell_signals_count']} sell signals")
        
        return signals

    def _format_signal_message(self, stock_name, stock_symbol, signals, timeframe):
        """Format the signal message for Telegram with MarkdownV2 escaping"""
        
        # Get industry information (if available, or add a placeholder)
        industry = "Services"  # Replace with actual logic to get industry
        
        # Get signal strength (1-5 stars)
        signal_strength = min(5, max(1, signals.get('strength', 0)))
        
        # Format primary indicators section
        primary_indicators = []
        
        # EMA
        if 'moving_averages' in signals.get('indicators', {}):
            ma_data = signals['indicators']['moving_averages']
            if ma_data['signal'] == 1:
                ema_short = escape_telegram_markdown(f"{ma_data['values']['ema_short']:.2f}")
                ema_long = escape_telegram_markdown(f"{ma_data['values']['ema_long']:.2f}")
                primary_indicators.append(f"✅ EMA Crossover: Bullish \\({ema_short} > {ema_long}\\)")
        
        # RSI
        if 'rsi' in signals.get('indicators', {}):
            rsi_data = signals['indicators']['rsi']['values']
            rsi_value = rsi_data.get('rsi', 0)
            if rsi_value:
                rsi_status = "Strong momentum" if 50 < rsi_value < 70 else "Overbought" if rsi_value >= 70 else "Oversold" if rsi_value <= 30 else "Neutral"
                rsi_value_escaped = escape_telegram_markdown(f"{rsi_value:.2f}")
                primary_indicators.append(f"✅ RSI: {rsi_value_escaped} \\({escape_telegram_markdown(rsi_status)}\\)")
        
        # Supertrend
        if 'supertrend' in signals.get('indicators', {}):
            st_data = signals['indicators']['supertrend']
            if st_data['values'].get('direction') == 'Bullish':
                primary_indicators.append(f"✅ Supertrend: Bullish")
        
        # MACD
        if 'macd' in signals.get('indicators', {}):
            macd_data = signals['indicators']['macd']
            if macd_data['signal'] == 1:
                primary_indicators.append(f"✅ MACD: Positive & expanding")
        
        # Format patterns section
        patterns_text = []
        for pattern_type in ['chart', 'candlestick']:
            for pattern_name, pattern_data in signals.get('patterns', {}).get(pattern_type, {}).items():
                if pattern_data.get('signal') == 1:  # Only include bullish patterns for BUY signals
                    pattern_name_formatted = escape_telegram_markdown(pattern_name.replace('_', ' ').title())
                    suffix = " \\(candlestick\\)" if pattern_type == "candlestick" else ""
                    patterns_text.append(f"✅ {pattern_name_formatted}{suffix}")
        
        # If no patterns found
        if not patterns_text:
            patterns_text = ["No significant patterns detected"]
        
        # Get risk management values
        stop_loss = signals.get('indicators', {}).get('atr', {}).get('values', {}).get('buy_stop', 0)
        current_price = signals.get('current_price', 0)
        target_price = current_price + (current_price - stop_loss) * 2  # 2:1 reward-to-risk ratio
        
        # Get trend strength
        trend_strength = ""
        if 'aroon' in signals.get('indicators', {}):
            aroon_data = signals['indicators']['aroon']['values']
            aroon_up = aroon_data.get('aroon_up', 0)
            strength_desc = "Very Strong" if aroon_up > 80 else "Strong" if aroon_up > 70 else "Moderate" if aroon_up > 50 else "Weak"
            aroon_up_escaped = escape_telegram_markdown(f"{aroon_up:.2f}")
            trend_strength = f"*AROON TREND STRENGTH:* {aroon_up_escaped} \\({escape_telegram_markdown(strength_desc)}\\)"
        
        # Build buy/sell summary
        buy_signals_count = signals.get('buy_signals_count', 0)
        sell_signals_count = signals.get('sell_signals_count', 0)
        total_signals = buy_signals_count + sell_signals_count
        buy_sell_summary = f"{escape_telegram_markdown(str(buy_signals_count))}/{escape_telegram_markdown(str(total_signals))} BULLISH SIGNALS \\| {escape_telegram_markdown(str(sell_signals_count))}/{escape_telegram_markdown(str(total_signals))} BEARISH SIGNALS"
        
        # Format timestamp to shorter version
        try:
            timestamp_obj = datetime.datetime.strptime(signals.get('timestamp', ""), "%Y-%m-%d %H:%M:%S")
            timestamp_short = escape_telegram_markdown(timestamp_obj.strftime("%b-%d %H:%M"))
        except:
            timestamp_short = escape_telegram_markdown(datetime.datetime.now().strftime("%b-%d %H:%M"))
        
        # Escape values for MarkdownV2
        safe_stock_name = escape_telegram_markdown(stock_name)
        safe_stock_symbol = escape_telegram_markdown(stock_symbol)
        safe_current_price = escape_telegram_markdown(f"₹{current_price:.2f}")
        safe_industry = escape_telegram_markdown(industry)
        safe_signal_type = escape_telegram_markdown(signals.get('signal', "NEUTRAL"))
        safe_stop_loss = escape_telegram_markdown(f"₹{stop_loss:.2f}")
        safe_target_price = escape_telegram_markdown(f"₹{target_price:.2f}")
        
        # Format the message using template
        message = config.SIGNAL_MESSAGE_TEMPLATE.format(
            stock_name=safe_stock_name,
            stock_symbol=safe_stock_symbol,
            current_price=safe_current_price,
            industry=safe_industry,
            signal_type=safe_signal_type,
            signal_strength=signal_strength,
            primary_indicators="\n".join(primary_indicators),
            patterns="\n".join(patterns_text),
            stop_loss=safe_stop_loss,
            target_price=safe_target_price,
            trend_strength=trend_strength,
            buy_sell_summary=buy_sell_summary,
            timestamp_short=timestamp_short
        )
        
        return message
