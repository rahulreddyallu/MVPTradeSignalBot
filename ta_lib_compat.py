"""
Ta-lib compatibility layer for Google Colab
Implements ta-lib functions using pandas-ta
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

# This class mimics the ta-lib API using pandas-ta
class TALibCompat:
    @staticmethod
    def RSI(close, timeperiod=14):
        """Calculate RSI using pandas-ta"""
        return pd.Series(ta.rsi(close, length=timeperiod))
    
    @staticmethod
    def EMA(close, timeperiod):
        """Calculate EMA using pandas-ta"""
        return pd.Series(ta.ema(close, length=timeperiod))
    
    @staticmethod
    def SMA(close, timeperiod):
        """Calculate SMA using pandas-ta"""
        return pd.Series(ta.sma(close, length=timeperiod))
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calculate MACD using pandas-ta"""
        macd = ta.macd(close, fast=fastperiod, slow=slowperiod, signal=signalperiod)
        # Return in same format as ta-lib
        return macd["MACD_12_26_9"], macd["MACDs_12_26_9"], macd["MACDh_12_26_9"]
    
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
        """Calculate Bollinger Bands using pandas-ta"""
        bbands = ta.bbands(close, length=timeperiod, std=nbdevup)
        return bbands["BBU_20_2.0"], bbands["BBM_20_2.0"], bbands["BBL_20_2.0"]
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
        """Calculate Stochastic Oscillator using pandas-ta"""
        stoch = ta.stoch(high, low, close, k=fastk_period, d=slowd_period, smooth_k=slowk_period)
        return stoch["STOCHk_14_3_3"], stoch["STOCHd_14_3_3"]
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """Calculate ATR using pandas-ta"""
        return pd.Series(ta.atr(high, low, close, length=timeperiod))
    
    @staticmethod
    def ROC(close, timeperiod=10):
        """Calculate Rate of Change using pandas-ta"""
        return pd.Series(ta.roc(close, length=timeperiod))
    
    @staticmethod
    def AROON(high, low, timeperiod=25):
        """Calculate Aroon using pandas-ta"""
        aroon = ta.aroon(high, low, length=timeperiod)
        return aroon["AROOND_25"], aroon["AROONU_25"]