class TechnicalIndicators:
    """Calculate and interpret technical indicators for trading signals"""
    
    def __init__(self, df, params=None):
        """
        Initialize Technical Analysis with OHLCV DataFrame
        
        Args:
            df: DataFrame with OHLCV data (index=timestamp, columns=[open, high, low, close, volume])
            params: Optional TradingParameters instance for indicator parameters
        """
        # Ensure column names are lowercase
        self.df = df.copy()
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if 'volume' in missing_columns:
            # If volume is missing, create placeholder volume data
            self.df['volume'] = 0
            missing_columns.remove('volume')
        
        if missing_columns:
            raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")
        
        # Set parameters from config or use defaults
        self.params = params or TradingParameters()
        
        # Initialize results dictionary and signals list
        self.indicators = {}
        self.signals = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self):
        """Calculate all technical indicators"""
        indicator_methods = {
            # Trend indicators
            "moving_averages": self.calculate_moving_averages,
            "supertrend": self.calculate_supertrend,
            "parabolic_sar": self.calculate_parabolic_sar,
            "adx": self.calculate_adx,
            "alligator": self.calculate_alligator,
            "cpr": self.calculate_cpr,
            "aroon": self.calculate_aroon,
            
            # Momentum indicators
            "macd": self.calculate_macd,
            "rsi": self.calculate_rsi,
            "stochastic": self.calculate_stochastic,
            "roc": self.calculate_rate_of_change,
            "stochastic_rsi": self.calculate_stochastic_rsi,
            "williams_r": self.calculate_williams_r,
            
            # Volatility indicators
            "bollinger_bands": self.calculate_bollinger_bands,
            "atr": self.calculate_atr,
            "atr_bands": self.calculate_atr_bands,
            
            # Volume indicators
            "obv": self.calculate_obv,
            "vwap": self.calculate_vwap,
            "volume_profile": self.calculate_volume_profile,
            
            # Support/Resistance indicators
            "support_resistance": self.calculate_support_resistance,
            "fibonacci_retracement": self.calculate_fibonacci_retracement
        }
        
        # Calculate each indicator and handle errors
        for indicator_name, calculation_method in indicator_methods.items():
            try:
                calculation_method()
                self.logger.debug(f"Successfully calculated {indicator_name}")
            except Exception as e:
                self.logger.warning(f"Error calculating {indicator_name}: {str(e)}")
                # Add error entry to indicators
                self.indicators[indicator_name] = {
                    'signal': 0,
                    'error': str(e)
                }
        
        return self.indicators
    
    def calculate_moving_averages(self):
        """Calculate Simple and Exponential Moving Averages"""
        # Get parameters for moving averages
        sma_periods = self.params.get_indicator_param('sma_periods')
        ema_periods = self.params.get_indicator_param('ema_periods')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate SMAs for different periods
        for period in sma_periods:
            new_cols[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
        
        # Calculate EMAs for different periods
        for period in ema_periods:
            new_cols[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        
        # Generate crossover signals
        # EMA crossover (typically 9 & 21)
        short_ema = f'ema_{ema_periods[0]}'  # Typically 9
        long_ema = f'ema_{ema_periods[1]}'   # Typically 21
        
        # Add crossover signals
        new_cols['ema_crossover'] = 0
        
        # We need to add these to the DataFrame temporarily to do comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
            
        # Now we can do the comparisons
        mask_above = temp_df[short_ema] > temp_df[long_ema]
        mask_below = temp_df[short_ema] < temp_df[long_ema]
        
        new_cols['ema_crossover'] = pd.Series(0, index=self.df.index)
        new_cols['ema_crossover'].loc[mask_above] = 1
        new_cols['ema_crossover'].loc[mask_below] = -1
        
        # Detect crossovers (signals) - temporary assignments to calculate
        new_cols['ema_buy_signal'] = (
            (temp_df['ema_crossover'].shift(1) == -1) & 
            (temp_df['ema_crossover'] == 1)
        ).astype(int)
        
        new_cols['ema_sell_signal'] = (
            (temp_df['ema_crossover'].shift(1) == 1) & 
            (temp_df['ema_crossover'] == -1)
        ).astype(int)
        
        # Golden Cross / Death Cross (SMA 50 and 200)
        mid_term = f'sma_{sma_periods[3]}'  # Typically 50
        long_term = f'sma_{sma_periods[4]}' # Typically 200
        
        new_cols['golden_cross'] = (
            (temp_df[mid_term].shift(1) <= temp_df[long_term].shift(1)) & 
            (temp_df[mid_term] > temp_df[long_term])
        ).astype(int)
        
        new_cols['death_cross'] = (
            (temp_df[mid_term].shift(1) >= temp_df[long_term].shift(1)) & 
            (temp_df[mid_term] < temp_df[long_term])
        ).astype(int)
        
        # Determine overall trend direction
        new_cols['uptrend'] = (
            (temp_df['close'] > temp_df[mid_term]) & 
            (temp_df[mid_term] > temp_df[long_term])
        ).astype(int)
        
        new_cols['downtrend'] = (
            (temp_df['close'] < temp_df[mid_term]) & 
            (temp_df[mid_term] < temp_df[long_term])
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        
        # Check for fresh EMA crossover
        if self.df['ema_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_strength = 2
        elif self.df['ema_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_strength = 2
        
        # Check for golden/death cross (stronger signals)
        if self.df['golden_cross'].iloc[-1] == 1:
            current_signal = 1
            signal_strength = 3
        elif self.df['death_cross'].iloc[-1] == 1:
            current_signal = -1
            signal_strength = 3
        
        # Check price position relative to key MAs for trend confirmation
        price_above_short = self.df['close'].iloc[-1] > self.df[short_ema].iloc[-1]
        price_above_mid = self.df['close'].iloc[-1] > self.df[mid_term].iloc[-1]
        price_above_long = self.df['close'].iloc[-1] > self.df[long_term].iloc[-1]
        
        # Save to indicators dictionary
        self.indicators['moving_averages'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'ema_short': round(self.df[short_ema].iloc[-1], 2) if not pd.isna(self.df[short_ema].iloc[-1]) else None,
                'ema_long': round(self.df[long_ema].iloc[-1], 2) if not pd.isna(self.df[long_ema].iloc[-1]) else None,
                'sma_mid': round(self.df[mid_term].iloc[-1], 2) if not pd.isna(self.df[mid_term].iloc[-1]) else None,
                'sma_long': round(self.df[long_term].iloc[-1], 2) if not pd.isna(self.df[long_term].iloc[-1]) else None,
                'price_above_ema_short': price_above_short,
                'price_above_sma_mid': price_above_mid,
                'price_above_sma_long': price_above_long,
                'golden_cross': self.df['golden_cross'].iloc[-1] == 1,
                'death_cross': self.df['death_cross'].iloc[-1] == 1,
                'uptrend': self.df['uptrend'].iloc[-1] == 1,
                'downtrend': self.df['downtrend'].iloc[-1] == 1
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            signal_name = "EMA Crossover"
            if self.df['golden_cross'].iloc[-1] == 1:
                signal_name = "Golden Cross"
            elif self.df['death_cross'].iloc[-1] == 1:
                signal_name = "Death Cross"
                
            self.signals.append({
                'indicator': 'Moving Averages',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_name
            })
    
    def calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Get MACD parameters
        fast_period = self.params.get_indicator_param('macd_fast')
        slow_period = self.params.get_indicator_param('macd_slow')
        signal_period = self.params.get_indicator_param('macd_signal')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate MACD components
        fast_ema = self.df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.df['close'].ewm(span=slow_period, adjust=False).mean()
        
        new_cols['macd_line'] = fast_ema - slow_ema
        new_cols['signal_line'] = new_cols['macd_line'].ewm(span=signal_period, adjust=False).mean()
        new_cols['macd_histogram'] = new_cols['macd_line'] - new_cols['signal_line']
        
        # Generate signals based on crossovers
        new_cols['macd_crossover'] = pd.Series(0, index=self.df.index)
        
        # We need to add these to the DataFrame temporarily to do comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
            
        # Set crossover values
        above_mask = temp_df['macd_line'] > temp_df['signal_line']
        below_mask = temp_df['macd_line'] < temp_df['signal_line']
        
        new_cols['macd_crossover'].loc[above_mask] = 1
        new_cols['macd_crossover'].loc[below_mask] = -1
        
        # Detect crossovers
        new_cols['macd_buy_signal'] = (
            (temp_df['macd_crossover'].shift(1) == -1) & 
            (temp_df['macd_crossover'] == 1)
        ).astype(int)
        
        new_cols['macd_sell_signal'] = (
            (temp_df['macd_crossover'].shift(1) == 1) & 
            (temp_df['macd_crossover'] == -1)
        ).astype(int)
        
        # Detect histogram direction changes
        new_cols['hist_direction'] = np.sign(
            temp_df['macd_histogram'] - temp_df['macd_histogram'].shift(1)
        )
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Bullish divergence: Price makes lower lows but MACD makes higher lows
        # Bearish divergence: Price makes higher highs but MACD makes lower highs
        # (Complex calculation implemented in a simplified way)
        bullish_divergence = False
        bearish_divergence = False
        
        # Generate signal
        current_signal = 0
        signal_strength = this_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['macd_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            this_strength = 2
        elif self.df['macd_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            this_strength = 2
        
        # Strengthen signal if MACD line is above/below zero line
        if current_signal == 1 and self.df['macd_line'].iloc[-1] > 0:
            signal_type += " Above Zero"
            this_strength += 1
        elif current_signal == -1 and self.df['macd_line'].iloc[-1] < 0:
            signal_type += " Below Zero"
            this_strength += 1
            
        # Check for favorable histogram direction
        histogram_favorable = False
        if (current_signal == 1 and self.df['hist_direction'].iloc[-1] > 0) or \
           (current_signal == -1 and self.df['hist_direction'].iloc[-1] < 0):
            signal_type += " with Expanding Histogram"
            histogram_favorable = True
            this_strength += 1
        
        signal_strength = this_strength
        
        self.indicators['macd'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'macd_line': round(self.df['macd_line'].iloc[-1], 4),
                'signal_line': round(self.df['signal_line'].iloc[-1], 4),
                'histogram': round(self.df['macd_histogram'].iloc[-1], 4),
                'hist_direction': 'Increasing' if self.df['hist_direction'].iloc[-1] > 0 else 
                                 'Decreasing' if self.df['hist_direction'].iloc[-1] < 0 else 'Unchanged',
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'MACD',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_rsi(self):
        """Calculate Relative Strength Index (RSI)"""
        # Get RSI parameters
        period = self.params.get_indicator_param('rsi_period')
        oversold = self.params.get_indicator_param('rsi_oversold')
        overbought = self.params.get_indicator_param('rsi_overbought')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate price changes
        delta = self.df['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over the specified period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        new_cols['rsi'] = 100 - (100 / (1 + rs))
        
        # We need to add this to the DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        temp_df['rsi'] = new_cols['rsi']
        
        # Generate signals
        new_cols['rsi_oversold'] = temp_df['rsi'] < oversold
        new_cols['rsi_overbought'] = temp_df['rsi'] > overbought
        
        # Detect crosses above oversold and below overbought
        new_cols['rsi_buy_signal'] = (
            (temp_df['rsi'] > oversold) & 
            (temp_df['rsi'].shift(1) <= oversold)
        ).astype(int)
        
        new_cols['rsi_sell_signal'] = (
            (temp_df['rsi'] < overbought) & 
            (temp_df['rsi'].shift(1) >= overbought)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for oversold/overbought conditions
        if self.df['rsi_oversold'].iloc[-1]:
            current_signal = 1
            signal_type = "Oversold"
            signal_strength = 1
        elif self.df['rsi_overbought'].iloc[-1]:
            current_signal = -1
            signal_type = "Overbought"
            signal_strength = 1
        
        # Check for crosses (stronger signals)
        if self.df['rsi_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Cross from Oversold"
            signal_strength = 2
        elif self.df['rsi_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Cross from Overbought"
            signal_strength = 2
        
        self.indicators['rsi'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'rsi': round(self.df['rsi'].iloc[-1], 2) if not pd.isna(self.df['rsi'].iloc[-1]) else None,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['rsi_oversold'].iloc[-1],
                'is_overbought': self.df['rsi_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'RSI',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_stochastic(self):
        """Calculate Stochastic Oscillator"""
        # Get Stochastic parameters
        k_period = self.params.get_indicator_param('stoch_k_period')
        d_period = self.params.get_indicator_param('stoch_d_period')
        slowing = self.params.get_indicator_param('stoch_slowing')
        oversold = self.params.get_indicator_param('stoch_oversold')
        overbought = self.params.get_indicator_param('stoch_overbought')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate %K (The current close in relation to the range over k_period)
        lowest_low = self.df['low'].rolling(window=k_period).min()
        highest_high = self.df['high'].rolling(window=k_period).max()
        
        new_cols['stoch_k_raw'] = 100 * ((self.df['close'] - lowest_low) / 
                                     (highest_high - lowest_low))
        
        # Apply slowing for %K
        new_cols['stoch_k'] = new_cols['stoch_k_raw'].rolling(window=slowing).mean()
        
        # Calculate %D (Simple moving average of %K)
        new_cols['stoch_d'] = new_cols['stoch_k'].rolling(window=d_period).mean()
        
        # We need to add these to the DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Generate signals
        new_cols['stoch_oversold'] = temp_df['stoch_k'] < oversold
        new_cols['stoch_overbought'] = temp_df['stoch_k'] > overbought
        
        # Detect K crossing above D in oversold region
        new_cols['stoch_buy_signal'] = (
            (temp_df['stoch_k'] > temp_df['stoch_d']) & 
            (temp_df['stoch_k'].shift(1) <= temp_df['stoch_d'].shift(1)) &
            (temp_df['stoch_k'] < oversold + 5)
        ).astype(int)
        
        # Detect K crossing below D in overbought region
        new_cols['stoch_sell_signal'] = (
            (temp_df['stoch_k'] < temp_df['stoch_d']) & 
            (temp_df['stoch_k'].shift(1) >= temp_df['stoch_d'].shift(1)) &
            (temp_df['stoch_k'] > overbought - 5)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers in oversold/overbought regions
        if self.df['stoch_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover from Oversold"
            signal_strength = 2
        elif self.df['stoch_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover from Overbought"
            signal_strength = 2
        
        # Also check for extreme oversold/overbought conditions
        elif self.df['stoch_oversold'].iloc[-1] and self.df['stoch_k'].iloc[-1] < oversold - 10:
            current_signal = 1
            signal_type = "Extremely Oversold"
            signal_strength = 1
        elif self.df['stoch_overbought'].iloc[-1] and self.df['stoch_k'].iloc[-1] > overbought + 10:
            current_signal = -1
            signal_type = "Extremely Overbought"
            signal_strength = 1
        
        self.indicators['stochastic'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'stoch_k': round(self.df['stoch_k'].iloc[-1], 2) if not pd.isna(self.df['stoch_k'].iloc[-1]) else None,
                'stoch_d': round(self.df['stoch_d'].iloc[-1], 2) if not pd.isna(self.df['stoch_d'].iloc[-1]) else None,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['stoch_oversold'].iloc[-1],
                'is_overbought': self.df['stoch_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Stochastic',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        # Get Bollinger Bands parameters
        period = self.params.get_indicator_param('bb_period')
        std_dev = self.params.get_indicator_param('bb_std_dev')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate middle band (SMA)
        new_cols['bb_middle'] = self.df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        new_cols['bb_std'] = self.df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        new_cols['bb_upper'] = new_cols['bb_middle'] + (std_dev * new_cols['bb_std'])
        new_cols['bb_lower'] = new_cols['bb_middle'] - (std_dev * new_cols['bb_std'])
        
        # Calculate %B (position within bands)
        new_cols['bb_pct_b'] = (self.df['close'] - new_cols['bb_lower']) / (new_cols['bb_upper'] - new_cols['bb_lower'])
        
        # Calculate bandwidth
        new_cols['bb_bandwidth'] = (new_cols['bb_upper'] - new_cols['bb_lower']) / new_cols['bb_middle']
        
        # Generate signals - price touching or breaking bands
        new_cols['bb_touch_upper'] = self.df['high'] >= new_cols['bb_upper']
        new_cols['bb_touch_lower'] = self.df['low'] <= new_cols['bb_lower']
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for price at bands
        if self.df['bb_touch_lower'].iloc[-1]:
            current_signal = 1
            signal_type = "Price at Lower Band"
            signal_strength = 1
        elif self.df['bb_touch_upper'].iloc[-1]:
            current_signal = -1
            signal_type = "Price at Upper Band"
            signal_strength = 1
        
        # Check for Bollinger Band squeeze (narrowing bandwidth)
        # Get average bandwidth over last 20 periods
        avg_bandwidth = self.df['bb_bandwidth'].iloc[-20:].mean()
        current_bandwidth = self.df['bb_bandwidth'].iloc[-1]
        
        is_squeeze = current_bandwidth < (avg_bandwidth * 0.8)
        
        # Check for %B signals
        if self.df['bb_pct_b'].iloc[-1] < 0:
            current_signal = 1
            signal_type = "Price Below Lower Band"
            signal_strength = 2
        elif self.df['bb_pct_b'].iloc[-1] > 1:
            current_signal = -1
            signal_type = "Price Above Upper Band"
            signal_strength = 2
        
        # Improve signal with RSI confirmation
        if 'rsi' in self.df.columns:
            if current_signal == 1 and self.df['rsi'].iloc[-1] < 40:
                signal_type += " with RSI Confirmation"
                signal_strength += 1
            elif current_signal == -1 and self.df['rsi'].iloc[-1] > 60:
                signal_type += " with RSI Confirmation"
                signal_strength += 1
        
        self.indicators['bollinger_bands'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'middle': round(self.df['bb_middle'].iloc[-1], 2) if not pd.isna(self.df['bb_middle'].iloc[-1]) else None,
                'upper': round(self.df['bb_upper'].iloc[-1], 2) if not pd.isna(self.df['bb_upper'].iloc[-1]) else None,
                'lower': round(self.df['bb_lower'].iloc[-1], 2) if not pd.isna(self.df['bb_lower'].iloc[-1]) else None,
                'percent_b': round(self.df['bb_pct_b'].iloc[-1], 2) if not pd.isna(self.df['bb_pct_b'].iloc[-1]) else None,
                'bandwidth': round(self.df['bb_bandwidth'].iloc[-1], 3) if not pd.isna(self.df['bb_bandwidth'].iloc[-1]) else None,
                'is_squeeze': is_squeeze,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Bollinger Bands',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_atr(self):
        """Calculate Average True Range (ATR)"""
        # Get ATR parameters
        period = self.params.get_indicator_param('atr_period')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate True Range
        high_low = self.df['high'] - self.df['low']
        high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
        low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
        
        # Take the maximum of the three
        new_cols['tr'] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate ATR (simple moving average of TR for first 'period' values, then smoothed)
        new_cols['atr'] = new_cols['tr'].rolling(window=period).mean()
        
        # Calculate ATR percentage (relative to price)
        new_cols['atr_pct'] = 100 * new_cols['atr'] / self.df['close']
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # ATR doesn't generate signals directly, but provides volatility information
        atr_value = self.df['atr'].iloc[-1]
        atr_pct = self.df['atr_pct'].iloc[-1]
        
        # Determine if volatility is high (above average)
        avg_atr_pct = self.df['atr_pct'].rolling(window=20).mean().iloc[-1]
        high_volatility = atr_pct > (avg_atr_pct * 1.2)
        
        # Save values
        self.indicators['atr'] = {
            'signal': 0,  # ATR doesn't generate buy/sell signals directly
            'signal_strength': 0,
            'values': {
                'atr': round(atr_value, 2),
                'atr_pct': round(atr_pct, 2),
                'high_volatility': high_volatility,
            }
        }
    
    def calculate_atr_bands(self):
        """Calculate ATR Bands (similar to Keltner Channels)"""
        # Ensure ATR is calculated
        if 'atr' not in self.df.columns:
            self.calculate_atr()
            
        # Get parameters
        multiplier_upper = 2.0  # Default value if not in params
        multiplier_lower = 2.0  # Default value if not in params
        
        try:
            atr_bands_params = self.params.get_indicator_param('atr_bands')
            if atr_bands_params:
                multiplier_upper = atr_bands_params.get('multiplier_upper', 2.0)
                multiplier_lower = atr_bands_params.get('multiplier_lower', 2.0)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate ATR Bands
        new_cols['atr_band_middle'] = self.df['close'].rolling(window=20).mean()  # 20-period SMA
        new_cols['atr_band_upper'] = new_cols['atr_band_middle'] + (multiplier_upper * self.df['atr'])
        new_cols['atr_band_lower'] = new_cols['atr_band_middle'] - (multiplier_lower * self.df['atr'])
        
        # Generate signals
        new_cols['atr_band_touch_upper'] = self.df['high'] >= new_cols['atr_band_upper']
        new_cols['atr_band_touch_lower'] = self.df['low'] <= new_cols['atr_band_lower']
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Check for band touches in recent periods
        lookback = 5
        upper_touches = self.df['atr_band_touch_upper'].iloc[-lookback:].sum()
        lower_touches = self.df['atr_band_touch_lower'].iloc[-lookback:].sum()
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for recent touches
        if lower_touches > 0:
            current_signal = 1
            signal_type = "Price at Lower Band"
            signal_strength = 1
        elif upper_touches > 0:
            current_signal = -1
            signal_type = "Price at Upper Band"
            signal_strength = 1
        
        # Check if price is breaking out of bands
        if self.df['close'].iloc[-1] < self.df['atr_band_lower'].iloc[-1]:
            current_signal = 1
            signal_type = "Price Below Lower Band"
            signal_strength = 2
        elif self.df['close'].iloc[-1] > self.df['atr_band_upper'].iloc[-1]:
            current_signal = -1
            signal_type = "Price Above Upper Band"
            signal_strength = 2
        
        self.indicators['atr_bands'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'middle': round(self.df['atr_band_middle'].iloc[-1], 2),
                'upper': round(self.df['atr_band_upper'].iloc[-1], 2),
                'lower': round(self.df['atr_band_lower'].iloc[-1], 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'ATR Bands',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_adx(self):
        """Calculate Average Directional Index (ADX)"""
        # Get ADX parameters
        period = self.params.get_indicator_param('adx_period')
        threshold = self.params.get_indicator_param('adx_threshold')
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate True Range if not already calculated
        if 'tr' not in self.df.columns:
            high_low = self.df['high'] - self.df['low']
            high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
            low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
            new_cols['tr'] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate Directional Movement
        new_cols['up_move'] = self.df['high'] - self.df['high'].shift(1)
        new_cols['down_move'] = self.df['low'].shift(1) - self.df['low']
        
        # Calculate Positive (DM+) and Negative (DM-) Directional Movement
        dm_plus = np.where(
            (new_cols['up_move'] > new_cols['down_move']) & (new_cols['up_move'] > 0),
            new_cols['up_move'],
            0
        )
        
        dm_minus = np.where(
            (new_cols['down_move'] > new_cols['up_move']) & (new_cols['down_move'] > 0),
            new_cols['down_move'],
            0
        )
        
        new_cols['dm_plus'] = pd.Series(dm_plus, index=self.df.index)
        new_cols['dm_minus'] = pd.Series(dm_minus, index=self.df.index)
        
        # Calculate Smoothed Directional Movement and True Range
        tr = new_cols['tr'] if 'tr' not in self.df.columns else self.df['tr']
        new_cols['tr_period'] = tr.rolling(window=period).sum()
        new_cols['dm_plus_period'] = new_cols['dm_plus'].rolling(window=period).sum()
        new_cols['dm_minus_period'] = new_cols['dm_minus'].rolling(window=period).sum()
        
        # Calculate Directional Indicators (DI+ and DI-)
        new_cols['di_plus'] = 100 * new_cols['dm_plus_period'] / new_cols['tr_period']
        new_cols['di_minus'] = 100 * new_cols['dm_minus_period'] / new_cols['tr_period']
        
        # Calculate Directional Index (DX)
        new_cols['dx'] = 100 * abs(new_cols['di_plus'] - new_cols['di_minus']) / (new_cols['di_plus'] + new_cols['di_minus'])
        
        # Calculate ADX (Average of DX)
        new_cols['adx'] = new_cols['dx'].rolling(window=period).mean()
        
        # We need to add these to the DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        temp_df_cols = {}
        for key, val in new_cols.items():
            temp_df[key] = val
            temp_df_cols[key] = val
        
        # Generate signals - Strong trend when ADX is above threshold
        new_cols['adx_strong_trend'] = temp_df['adx'] > threshold
        
        # Direction of trend based on DI+ vs DI-
        new_cols['adx_trend_direction'] = np.where(
            temp_df['di_plus'] > temp_df['di_minus'],
            1,  # Bullish
            -1  # Bearish
        )
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check if we have a strong trend
        adx_value = self.df['adx'].iloc[-1]
        is_strong_trend = adx_value > threshold
        
        # Check the direction of the trend
        trend_direction = self.df['adx_trend_direction'].iloc[-1]
        
        # Check if DI+ crosses above DI- (buy signal)
        di_crossover_buy = (self.df['di_plus'].iloc[-1] > self.df['di_minus'].iloc[-1]) and \
                           (self.df['di_plus'].iloc[-2] <= self.df['di_minus'].iloc[-2])
        
        # Check if DI- crosses above DI+ (sell signal)
        di_crossover_sell = (self.df['di_minus'].iloc[-1] > self.df['di_plus'].iloc[-1]) and \
                            (self.df['di_minus'].iloc[-2] <= self.df['di_plus'].iloc[-2])
        
        # Generate signals
        if di_crossover_buy:
            current_signal = 1
            signal_type = "DI+ crossed above DI-"
            signal_strength = 2
        elif di_crossover_sell:
            current_signal = -1
            signal_type = "DI- crossed above DI+"
            signal_strength = 2
        elif is_strong_trend:
            if trend_direction == 1:
                current_signal = 1
                signal_type = "Strong Bullish Trend"
                signal_strength = 2
            else:
                current_signal = -1
                signal_type = "Strong Bearish Trend"
                signal_strength = 2
        
        self.indicators['adx'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'adx': round(adx_value, 2),
                'di_plus': round(self.df['di_plus'].iloc[-1], 2),
                'di_minus': round(self.df['di_minus'].iloc[-1], 2),
                'trend_strength': 'Strong' if adx_value > threshold else 'Weak',
                'trend_direction': 'Bullish' if trend_direction == 1 else 'Bearish',
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'ADX',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_stochastic_rsi(self):
        """Calculate Stochastic RSI"""
        # Ensure RSI is calculated
        if 'rsi' not in self.df.columns:
            self.calculate_rsi()
            
        # Get parameters
        period = 14  # Default if not specified
        smooth_k = 3
        smooth_d = 3
        oversold = 20
        overbought = 80
        
        try:
            stoch_rsi_params = self.params.get_indicator_param('stochastic_rsi')
            if stoch_rsi_params:
                period = stoch_rsi_params.get('period', 14)
                smooth_k = stoch_rsi_params.get('smooth_k', 3)
                smooth_d = stoch_rsi_params.get('smooth_d', 3)
                oversold = stoch_rsi_params.get('oversold', 20)
                overbought = stoch_rsi_params.get('overbought', 80)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate Stochastic RSI
        # First find the lowest low and highest high of RSI within the period
        lowest_rsi = self.df['rsi'].rolling(window=period).min()
        highest_rsi = self.df['rsi'].rolling(window=period).max()
        
        # Calculate raw K (current RSI relative to its range)
        denominator = highest_rsi - lowest_rsi
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1, denominator)
        stoch_rsi_k_raw = 100 * (self.df['rsi'] - lowest_rsi) / denominator
        
        new_cols['stoch_rsi_k_raw'] = pd.Series(stoch_rsi_k_raw, index=self.df.index)
        
        # Smooth K
        new_cols['stoch_rsi_k'] = new_cols['stoch_rsi_k_raw'].rolling(window=smooth_k).mean()
        
        # Calculate D (moving average of K)
        new_cols['stoch_rsi_d'] = new_cols['stoch_rsi_k'].rolling(window=smooth_d).mean()
        
        # We need to add these to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Generate signals
        new_cols['stoch_rsi_oversold'] = temp_df['stoch_rsi_k'] < oversold
        new_cols['stoch_rsi_overbought'] = temp_df['stoch_rsi_k'] > overbought
        
        # Detect K crossing above D from oversold region
        new_cols['stoch_rsi_buy_signal'] = (
            (temp_df['stoch_rsi_k'] > temp_df['stoch_rsi_d']) & 
            (temp_df['stoch_rsi_k'].shift(1) <= temp_df['stoch_rsi_d'].shift(1)) &
            (temp_df['stoch_rsi_k'] < 30)
        ).astype(int)
        
        # Detect K crossing below D from overbought region
        new_cols['stoch_rsi_sell_signal'] = (
            (temp_df['stoch_rsi_k'] < temp_df['stoch_rsi_d']) & 
            (temp_df['stoch_rsi_k'].shift(1) >= temp_df['stoch_rsi_d'].shift(1)) &
            (temp_df['stoch_rsi_k'] > 70)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers in oversold/overbought regions
        if self.df['stoch_rsi_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover from Oversold"
            signal_strength = 3
        elif self.df['stoch_rsi_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover from Overbought"
            signal_strength = 3
        
        # Also check for extreme oversold/overbought conditions
        elif self.df['stoch_rsi_oversold'].iloc[-1] and self.df['stoch_rsi_k'].iloc[-1] < 10:
            current_signal = 1
            signal_type = "Extremely Oversold"
            signal_strength = 2
        elif self.df['stoch_rsi_overbought'].iloc[-1] and self.df['stoch_rsi_k'].iloc[-1] > 90:
            current_signal = -1
            signal_type = "Extremely Overbought"
            signal_strength = 2
        
        self.indicators['stochastic_rsi'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'k': round(self.df['stoch_rsi_k'].iloc[-1], 2) if not pd.isna(self.df['stoch_rsi_k'].iloc[-1]) else None,
                'd': round(self.df['stoch_rsi_d'].iloc[-1], 2) if not pd.isna(self.df['stoch_rsi_d'].iloc[-1]) else None,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['stoch_rsi_oversold'].iloc[-1] if not pd.isna(self.df['stoch_rsi_oversold'].iloc[-1]) else False,
                'is_overbought': self.df['stoch_rsi_overbought'].iloc[-1] if not pd.isna(self.df['stoch_rsi_overbought'].iloc[-1]) else False,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Stochastic RSI',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_williams_r(self):
        """Calculate Williams %R"""
        # Get parameters
        period = 14  # Default if not specified
        oversold = -80
        overbought = -20
        
        try:
            williams_params = self.params.get_indicator_param('williams_r')
            if williams_params:
                period = williams_params.get('period', 14)
                oversold = williams_params.get('oversold', -80)
                overbought = williams_params.get('overbought', -20)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate highest high and lowest low over period
        highest_high = self.df['high'].rolling(window=period).max()
        lowest_low = self.df['low'].rolling(window=period).min()
        
        # Calculate Williams %R: ((highest_high - close) / (highest_high - lowest_low)) * -100
        denominator = highest_high - lowest_low
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1, denominator)
        williams_r = ((highest_high - self.df['close']) / denominator) * -100
        
        new_cols['williams_r'] = pd.Series(williams_r, index=self.df.index)
        
        # We need to add this to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        temp_df['williams_r'] = new_cols['williams_r']
        
        # Generate signals
        new_cols['williams_oversold'] = temp_df['williams_r'] <= oversold
        new_cols['williams_overbought'] = temp_df['williams_r'] >= overbought
        
        # Detect crosses from oversold/overbought regions
        new_cols['williams_buy_signal'] = (
            (temp_df['williams_r'] > oversold) & 
            (temp_df['williams_r'].shift(1) <= oversold)
        ).astype(int)
        
        new_cols['williams_sell_signal'] = (
            (temp_df['williams_r'] < overbought) & 
            (temp_df['williams_r'].shift(1) >= overbought)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crosses from oversold/overbought regions
        if self.df['williams_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Cross from Oversold"
            signal_strength = 2
        elif self.df['williams_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Cross from Overbought"
            signal_strength = 2
        
        # Check for extreme conditions
        elif self.df['williams_oversold'].iloc[-1] and self.df['williams_r'].iloc[-1] < oversold - 10:
            current_signal = 1
            signal_type = "Extremely Oversold"
            signal_strength = 1
        elif self.df['williams_overbought'].iloc[-1] and self.df['williams_r'].iloc[-1] > overbought + 10:
            current_signal = -1
            signal_type = "Extremely Overbought"
            signal_strength = 1
        
        self.indicators['williams_r'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'williams_r': round(self.df['williams_r'].iloc[-1], 2),
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['williams_oversold'].iloc[-1],
                'is_overbought': self.df['williams_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Williams %R',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_obv(self):
        """Calculate On-Balance Volume (OBV)"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['obv'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate OBV
        obv = np.zeros(len(self.df))
        obv[0] = self.df['volume'].iloc[0]
        
        # Calculate OBV sequentially
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + self.df['volume'].iloc[i]
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - self.df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        new_cols['obv'] = pd.Series(obv, index=self.df.index)
        
        # Calculate OBV moving average
        new_cols['obv_ma'] = new_cols['obv'].rolling(window=20).mean()
        
        # We need to add these to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Generate signals - Signal based on OBV vs its moving average
        new_cols['obv_signal'] = np.zeros(len(self.df))
        
        # Set signal values based on comparisons
        above_mask = temp_df['obv'] > temp_df['obv_ma']
        below_mask = temp_df['obv'] < temp_df['obv_ma']
        
        new_cols['obv_signal'] = pd.Series(0, index=self.df.index)
        new_cols['obv_signal'].loc[above_mask] = 1
        new_cols['obv_signal'].loc[below_mask] = -1
        
        # Detect crossovers
        new_cols['obv_buy_signal'] = (
            (temp_df['obv_signal'] == 1) & 
            (temp_df['obv_signal'].shift(1) == -1)
        ).astype(int)
        
        new_cols['obv_sell_signal'] = (
            (temp_df['obv_signal'] == -1) & 
            (temp_df['obv_signal'].shift(1) == 1)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Check for divergence (simplified calculation)
        # Bullish divergence: Price making lower lows but OBV making higher lows
        # Bearish divergence: Price making higher highs but OBV making lower highs
        
        # Get last 20 periods for local analysis
        last_n = 20
        bullish_div = False
        bearish_div = False
        
        if len(self.df) > last_n:
            subset = self.df.iloc[-last_n:]
            
            # Get local min/max for price and OBV
            price_min_idx = subset['close'].idxmin()
            price_max_idx = subset['close'].idxmax()
            obv_min_idx = subset['obv'].idxmin()
            obv_max_idx = subset['obv'].idxmax()
            
            # Check if the timing of min/max values shows divergence
            bullish_div = price_min_idx > obv_min_idx and self.df['close'].iloc[-1] < self.df['close'].iloc[-10].mean()
            bearish_div = price_max_idx > obv_max_idx and self.df['close'].iloc[-1] > self.df['close'].iloc[-10].mean()
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['obv_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            signal_strength = 2
        elif self.df['obv_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            signal_strength = 2
        
        # Check for divergence (stronger signals)
        elif bullish_div:
            current_signal = 1
            signal_type = "Bullish Divergence"
            signal_strength = 3
        elif bearish_div:
            current_signal = -1
            signal_type = "Bearish Divergence"
            signal_strength = 3
        
        # Also check if OBV is trending
        obv_rising = self.df['obv'].iloc[-1] > self.df['obv'].iloc[-5]
        
        self.indicators['obv'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'obv': int(self.df['obv'].iloc[-1]),
                'obv_ma': int(self.df['obv_ma'].iloc[-1]) if not pd.isna(self.df['obv_ma'].iloc[-1]) else None,
                'rising': obv_rising,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'On-Balance Volume',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_vwap(self):
        """Calculate Volume Weighted Average Price (VWAP)"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['vwap'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Get parameters
        reset_period = 'day'  # Default if not specified
        
        try:
            vwap_params = self.params.get_indicator_param('vwap')
            if vwap_params:
                reset_period = vwap_params.get('reset_period', 'day')
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate typical price
        new_cols['typical_price'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Calculate volume * typical price
        new_cols['vol_tp'] = new_cols['typical_price'] * self.df['volume']
        
        # Reset cumulative values at the start of each period
        if reset_period == 'day':
            # Check if index includes date information
            if pd.api.types.is_datetime64_any_dtype(self.df.index):
                # Create date groups
                date_groups = self.df.index.date
                
                # Initialize VWAP column with default values
                new_cols['vwap'] = pd.Series(0.0, index=self.df.index)

                # Compute VWAP separately for each date group
                for date in pd.unique(date_groups):
                    # Get indices for this date
                    mask = [d == date for d in date_groups]
                    
                    # Calculate cumulative values
                    cum_vol_tp = new_cols['vol_tp'][mask].cumsum()
                    cum_vol = self.df['volume'][mask].cumsum()
                    
                    # Calculate VWAP where volume is non-zero
                    vwap_vals = cum_vol_tp / cum_vol
                    vwap_vals = vwap_vals.replace([np.inf, -np.inf], np.nan)
                    
                    # Assign VWAP values to the appropriate rows
                    indices = self.df.index[mask]
                    for i, idx in enumerate(indices):
                        new_cols['vwap'][idx] = vwap_vals.iloc[i]
            else:
                # If we don't have datetime index, just calculate cumulative values
                cum_vol_tp = new_cols['vol_tp'].cumsum()
                cum_vol = self.df['volume'].cumsum()
                
                # Calculate VWAP
                new_cols['vwap'] = cum_vol_tp / cum_vol
        
        # We need to add these to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Generate signals - based on price relative to VWAP
        new_cols['price_above_vwap'] = temp_df['close'] > temp_df['vwap']
        
        # Detect crosses
        new_cols['vwap_cross_above'] = (
            (temp_df['close'] > temp_df['vwap']) & 
            (temp_df['close'].shift(1) <= temp_df['vwap'].shift(1))
        ).astype(int)
        
        new_cols['vwap_cross_below'] = (
            (temp_df['close'] < temp_df['vwap']) & 
            (temp_df['close'].shift(1) >= temp_df['vwap'].shift(1))
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crosses
        if self.df['vwap_cross_above'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Cross Above VWAP"
            signal_strength = 2
        elif self.df['vwap_cross_below'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Cross Below VWAP"
            signal_strength = 2
        
        # Check price position relative to VWAP for confirmation
        elif self.df['price_above_vwap'].iloc[-1] and self.df['close'].iloc[-1] > self.df['close'].iloc[-5:].mean():
            current_signal = 1
            signal_type = "Price Above VWAP"
            signal_strength = 1
        elif not self.df['price_above_vwap'].iloc[-1] and self.df['close'].iloc[-1] < self.df['close'].iloc[-5:].mean():
            current_signal = -1
            signal_type = "Price Below VWAP"
            signal_strength = 1
        
        # Calculate price distance from VWAP (as percentage)
        if not pd.isna(self.df['vwap'].iloc[-1]) and self.df['vwap'].iloc[-1] != 0:
            vwap_distance = abs((self.df['close'].iloc[-1] - self.df['vwap'].iloc[-1]) / self.df['vwap'].iloc[-1]) * 100
        else:
            vwap_distance = None
        
        self.indicators['vwap'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'vwap': round(self.df['vwap'].iloc[-1], 2) if not pd.isna(self.df['vwap'].iloc[-1]) else None,
                'price_above_vwap': self.df['price_above_vwap'].iloc[-1],
                'distance_pct': round(vwap_distance, 2) if vwap_distance is not None else None,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'VWAP',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_supertrend(self):
        """Calculate Supertrend indicator"""
        # Ensure ATR is calculated
        if 'atr' not in self.df.columns:
            self.calculate_atr()
            
        # Get parameters
        period = 10
        multiplier = 3.0
        
        try:
            supertrend_params = self.params.get_indicator_param('supertrend')
            if supertrend_params:
                period = supertrend_params.get('period', 10)
                multiplier = supertrend_params.get('multiplier', 3.0)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate basic bands
        hl2 = (self.df['high'] + self.df['low']) / 2
        new_cols['basic_upper_band'] = hl2 + (multiplier * self.df['atr'])
        new_cols['basic_lower_band'] = hl2 - (multiplier * self.df['atr'])
        
        # Initialize final band arrays
        final_upper = np.zeros(len(self.df))
        final_lower = np.zeros(len(self.df))
        supertrend = np.zeros(len(self.df))
        
        # Calculate SuperTrend sequentially
        for i in range(1, len(self.df)):
            # Upper band
            if new_cols['basic_upper_band'].iloc[i] < final_upper[i-1] or self.df['close'].iloc[i-1] > final_upper[i-1]:
                final_upper[i] = new_cols['basic_upper_band'].iloc[i]
            else:
                final_upper[i] = final_upper[i-1]
                
            # Lower band
            if new_cols['basic_lower_band'].iloc[i] > final_lower[i-1] or self.df['close'].iloc[i-1] < final_lower[i-1]:
                final_lower[i] = new_cols['basic_lower_band'].iloc[i]
            else:
                final_lower[i] = final_lower[i-1]
                
            # SuperTrend
            if supertrend[i-1] == final_upper[i-1] and self.df['close'].iloc[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
            elif supertrend[i-1] == final_upper[i-1] and self.df['close'].iloc[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
            elif supertrend[i-1] == final_lower[i-1] and self.df['close'].iloc[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
            elif supertrend[i-1] == final_lower[i-1] and self.df['close'].iloc[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = supertrend[i-1]
        
        new_cols['supertrend_upper'] = pd.Series(final_upper, index=self.df.index)
        new_cols['supertrend_lower'] = pd.Series(final_lower, index=self.df.index)
        new_cols['supertrend'] = pd.Series(supertrend, index=self.df.index)
        
        # Determine the trend direction
        new_cols['supertrend_direction'] = np.where(
            self.df['close'] > new_cols['supertrend'],
            1,  # Uptrend
            -1  # Downtrend
        )
        
        # We need to add these to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Detect changes in trend direction
        new_cols['supertrend_buy_signal'] = (
            (temp_df['supertrend_direction'] == 1) & 
            (temp_df['supertrend_direction'].shift(1) == -1)
        ).astype(int)
        
        new_cols['supertrend_sell_signal'] = (
            (temp_df['supertrend_direction'] == -1) & 
            (temp_df['supertrend_direction'].shift(1) == 1)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for trend changes
        if self.df['supertrend_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Trend Change"
            signal_strength = 3
        elif self.df['supertrend_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Trend Change"
            signal_strength = 3
        
        # If no recent signal, check for ongoing trend
        elif self.df['supertrend_direction'].iloc[-1] == 1 and not np.any(self.df['supertrend_direction'].iloc[-5:-1] == -1):
            current_signal = 1
            signal_type = "Strong Uptrend"
            signal_strength = 2
        elif self.df['supertrend_direction'].iloc[-1] == -1 and not np.any(self.df['supertrend_direction'].iloc[-5:-1] == 1):
            current_signal = -1
            signal_type = "Strong Downtrend"
            signal_strength = 2
        
        self.indicators['supertrend'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'supertrend': round(self.df['supertrend'].iloc[-1], 2),
                'direction': 'Uptrend' if self.df['supertrend_direction'].iloc[-1] == 1 else 'Downtrend',
                'upper_band': round(self.df['supertrend_upper'].iloc[-1], 2),
                'lower_band': round(self.df['supertrend_lower'].iloc[-1], 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'SuperTrend',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_aroon(self):
        """Calculate Aroon Indicator"""
        # Get parameters
        period = 14
        uptrend_threshold = 70
        downtrend_threshold = 30
        
        try:
            aroon_params = self.params.get_indicator_param('aroon')
            if aroon_params:
                period = aroon_params.get('period', 14)
                uptrend_threshold = aroon_params.get('uptrend_threshold', 70)
                downtrend_threshold = aroon_params.get('downtrend_threshold', 30)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate Aroon Up and Down
        # Aroon Up = 100 * (period - distance from high) / period
        # Aroon Down = 100 * (period - distance from low) / period
        
        # Get rolling window max/min indices
        high_indices = self.df['high'].rolling(window=period).apply(
            lambda x: x.argmax(), raw=True).fillna(0).astype(int)
        low_indices = self.df['low'].rolling(window=period).apply(
            lambda x: x.argmin(), raw=True).fillna(0).astype(int)
        
        # Create np arrays for indices and Aroon values
        indices = np.arange(len(self.df))
        
        # Calculate periods since max/min
        periods_since_high = indices - high_indices
        periods_since_low = indices - low_indices
        
        # Calculate Aroon values
        aroon_up = 100 * (period - periods_since_high) / period
        aroon_down = 100 * (period - periods_since_low) / period
        
        new_cols['aroon_up'] = pd.Series(aroon_up, index=self.df.index)
        new_cols['aroon_down'] = pd.Series(aroon_down, index=self.df.index)
        
        # Calculate Aroon Oscillator (Up - Down)
        new_cols['aroon_oscillator'] = new_cols['aroon_up'] - new_cols['aroon_down']
        
        # We need to add these to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Generate signals - Aroon Up crossing above threshold (uptrend)
        new_cols['aroon_uptrend'] = (
            (temp_df['aroon_up'] > uptrend_threshold) & 
            (temp_df['aroon_down'] < 50)
        ).astype(int)
        
        # Aroon Down crossing above threshold (downtrend)
        new_cols['aroon_downtrend'] = (
            (temp_df['aroon_down'] > uptrend_threshold) & 
            (temp_df['aroon_up'] < 50)
        ).astype(int)
        
        # Aroon Crossovers
        new_cols['aroon_bullish_cross'] = (
            (temp_df['aroon_up'] > temp_df['aroon_down']) & 
            (temp_df['aroon_up'].shift(1) <= temp_df['aroon_down'].shift(1))
        ).astype(int)
        
        new_cols['aroon_bearish_cross'] = (
            (temp_df['aroon_up'] < temp_df['aroon_down']) & 
            (temp_df['aroon_up'].shift(1) >= temp_df['aroon_down'].shift(1))
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for fresh crossovers
        if self.df['aroon_bullish_cross'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            signal_strength = 2
        elif self.df['aroon_bearish_cross'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            signal_strength = 2
        
        # Check for strong trends
        elif self.df['aroon_uptrend'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Strong Uptrend"
            signal_strength = 3
        elif self.df['aroon_downtrend'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Strong Downtrend"
            signal_strength = 3
        
        self.indicators['aroon'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'aroon_up': round(self.df['aroon_up'].iloc[-1], 2),
                'aroon_down': round(self.df['aroon_down'].iloc[-1], 2),
                'oscillator': round(self.df['aroon_oscillator'].iloc[-1], 2),
                'uptrend': self.df['aroon_uptrend'].iloc[-1] == 1,
                'downtrend': self.df['aroon_downtrend'].iloc[-1] == 1,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Aroon',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_rate_of_change(self):
        """Calculate Rate of Change (ROC)"""
        # Get parameters
        period = 10
        
        try:
            roc_params = self.params.get_indicator_param('roc')
            if roc_params:
                period = roc_params.get('period', 10)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Calculate ROC: ((close_current - close_n_periods_ago) / close_n_periods_ago) * 100
        new_cols['roc'] = ((self.df['close'] - self.df['close'].shift(period)) / 
                            self.df['close'].shift(period)) * 100
        
        # We need to add this to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        temp_df['roc'] = new_cols['roc']
        
        # Calculate ROC moving average for trend
        new_cols['roc_ma'] = temp_df['roc'].rolling(window=10).mean()
        
        # Generate signals - ROC crosses above/below zero
        new_cols['roc_positive'] = temp_df['roc'] > 0
        new_cols['roc_negative'] = temp_df['roc'] < 0
        
        new_cols['roc_cross_above_zero'] = (
            (temp_df['roc'] > 0) & 
            (temp_df['roc'].shift(1) <= 0)
        ).astype(int)
        
        new_cols['roc_cross_below_zero'] = (
            (temp_df['roc'] < 0) & 
            (temp_df['roc'].shift(1) >= 0)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for zero line crossovers
        if self.df['roc_cross_above_zero'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Zero Line Cross"
            signal_strength = 2
        elif self.df['roc_cross_below_zero'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Zero Line Cross"
            signal_strength = 2
        
        # Check for extreme ROC values
        elif self.df['roc'].iloc[-1] > 10:
            current_signal = -1  # Often indicates overbought
            signal_type = "Extremely High ROC (Potential Reversal)"
            signal_strength = 1
        elif self.df['roc'].iloc[-1] < -10:
            current_signal = 1  # Often indicates oversold
            signal_type = "Extremely Low ROC (Potential Reversal)"
            signal_strength = 1
        
        # Check for trend direction based on ROC and its MA
        elif self.df['roc'].iloc[-1] > self.df['roc_ma'].iloc[-1] and self.df['roc_positive'].iloc[-1]:
            current_signal = 1
            signal_type = "Accelerating Positive Momentum"
            signal_strength = 1
        elif self.df['roc'].iloc[-1] < self.df['roc_ma'].iloc[-1] and self.df['roc_negative'].iloc[-1]:
            current_signal = -1
            signal_type = "Accelerating Negative Momentum"
            signal_strength = 1
        
        self.indicators['roc'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'roc': round(self.df['roc'].iloc[-1], 2) if not pd.isna(self.df['roc'].iloc[-1]) else None,
                'roc_ma': round(self.df['roc_ma'].iloc[-1], 2) if not pd.isna(self.df['roc_ma'].iloc[-1]) else None,
                'positive': self.df['roc_positive'].iloc[-1],
                'accelerating': self.df['roc'].iloc[-1] > self.df['roc_ma'].iloc[-1] if not pd.isna(self.df['roc_ma'].iloc[-1]) else None,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Rate of Change',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_parabolic_sar(self):
        """Calculate Parabolic SAR"""
        # Get parameters
        af_start = 0.02
        af_increment = 0.02
        af_max = 0.2
        
        try:
            psar_params = self.params.get_indicator_param('parabolic_sar')
            if psar_params:
                af_start = psar_params.get('acceleration_factor', 0.02)
                af_max = psar_params.get('max_acceleration_factor', 0.2)
        except:
            pass
        
        # Initialize new columns dictionary
        new_cols = {}
        
        # Initialize arrays
        psar = np.zeros(len(self.df))
        psar_direction = np.zeros(len(self.df))  # 1 for uptrend, -1 for downtrend
        extreme_point = np.zeros(len(self.df))
        acceleration_factor = np.zeros(len(self.df))
        
        # Initialize values
        # Use the first two values to determine initial trend
        if len(self.df) > 1:
            if self.df['close'].iloc[1] > self.df['close'].iloc[0]:
                # Start in uptrend
                psar_direction[0] = 1
                psar[0] = self.df['low'].iloc[0]
                extreme_point[0] = self.df['high'].iloc[0]
            else:
                # Start in downtrend
                psar_direction[0] = -1
                psar[0] = self.df['high'].iloc[0]
                extreme_point[0] = self.df['low'].iloc[0]
                
            acceleration_factor[0] = af_start
        
        # Calculate Parabolic SAR sequentially
        for i in range(1, len(self.df)):
            # Previous values
            prev_psar = psar[i-1]
            prev_direction = psar_direction[i-1]
            prev_ep = extreme_point[i-1]
            prev_af = acceleration_factor[i-1]
            
            # Current high and low
            high = self.df['high'].iloc[i]
            low = self.df['low'].iloc[i]
            
            # Calculate current SAR value
            if prev_direction == 1:
                # Uptrend
                psar[i] = prev_psar + prev_af * (prev_ep - prev_psar)
                # Ensure SAR is below the previous two lows
                if i > 1:
                    psar[i] = min(psar[i], self.df['low'].iloc[i-1], self.df['low'].iloc[i-2])
                    
                # Check for trend reversal
                if psar[i] > low:
                    # Trend reversal
                    psar_direction[i] = -1
                    psar[i] = prev_ep
                    extreme_point[i] = low
                    acceleration_factor[i] = af_start
                else:
                    # Trend continues
                    psar_direction[i] = 1
                    # Update extreme point and acceleration factor if new high
                    if high > prev_ep:
                        extreme_point[i] = high
                        acceleration_factor[i] = min(prev_af + af_increment, af_max)
                    else:
                        extreme_point[i] = prev_ep
                        acceleration_factor[i] = prev_af
            else:
                # Downtrend
                psar[i] = prev_psar - prev_af * (prev_psar - prev_ep)
                # Ensure SAR is above the previous two highs
                if i > 1:
                    psar[i] = max(psar[i], self.df['high'].iloc[i-1], self.df['high'].iloc[i-2])
                    
                # Check for trend reversal
                if psar[i] < high:
                    # Trend reversal
                    psar_direction[i] = 1
                    psar[i] = prev_ep
                    extreme_point[i] = high
                    acceleration_factor[i] = af_start
                else:
                    # Trend continues
                    psar_direction[i] = -1
                    # Update extreme point and acceleration factor if new low
                    if low < prev_ep:
                        extreme_point[i] = low
                        acceleration_factor[i] = min(prev_af + af_increment, af_max)
                    else:
                        extreme_point[i] = prev_ep
                        acceleration_factor[i] = prev_af
        
        new_cols['psar'] = pd.Series(psar, index=self.df.index)
        new_cols['psar_direction'] = pd.Series(psar_direction, index=self.df.index)
        
        # We need to add these to DataFrame temporarily for comparisons
        temp_df = self.df.copy()
        for key, val in new_cols.items():
            temp_df[key] = val
        
        # Detect trend changes
        new_cols['psar_buy_signal'] = (
            (temp_df['psar_direction'] == 1) & 
            (temp_df['psar_direction'].shift(1) == -1)
        ).astype(int)
        
        new_cols['psar_sell_signal'] = (
            (temp_df['psar_direction'] == -1) & 
            (temp_df['psar_direction'].shift(1) == 1)
        ).astype(int)
        
        # Add all new columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for trend changes
        if self.df['psar_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Trend Change"
            signal_strength = 3
        elif self.df['psar_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Trend Change"
            signal_strength = 3
        
        # If no recent signal, check for ongoing trend
        elif self.df['psar_direction'].iloc[-1] == 1 and np.all(self.df['psar_direction'].iloc[-3:] == 1):
            current_signal = 1
            signal_type = "Strong Uptrend"
            signal_strength = 1
        elif self.df['psar_direction'].iloc[-1] == -1 and np.all(self.df['psar_direction'].iloc[-3:] == -1):
            current_signal = -1
            signal_type = "Strong Downtrend"
            signal_strength = 1
        
        self.indicators['parabolic_sar'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'psar': round(self.df['psar'].iloc[-1], 2),
                'direction': 'Uptrend' if self.df['psar_direction'].iloc[-1] == 1 else 'Downtrend',
                'distance_from_price': abs(round(self.df['close'].iloc[-1] - self.df['psar'].iloc[-1], 2)),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Parabolic SAR',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_volume_profile(self):
        """Calculate Volume Profile"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['volume_profile'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Get recent data for volume profile (last N periods)
        period = 20
        recent_data = self.df.iloc[-period:].copy()
        
        # If not enough data, use all available data
        if len(recent_data) < 10:
            recent_data = self.df.copy()
        
        # Create price bins
        price_range = recent_data['high'].max() - recent_data['low'].min()
        bin_size = price_range / 10  # Divide into 10 bins
        
        # Create bin edges
        price_bins = [recent_data['low'].min() + (i * bin_size) for i in range(11)]
        
        # Assign prices to bins
        # To avoid SettingWithCopyWarning, create an explicit copy
        recent_data = recent_data.copy()
        recent_data.loc[:, 'price_bin'] = pd.cut(recent_data['typical_price'], bins=price_bins, labels=False)
        
        # Group by price bin and sum volume
        volume_profile = recent_data.groupby('price_bin')['volume'].sum()
        
        # Find the bin with the highest volume (POC - Point of Control)
        poc_bin = volume_profile.idxmax()
        poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
        
        # Find significant support/resistance levels (bins with high volume)
        threshold = volume_profile.mean() * 1.5
        high_volume_bins = volume_profile[volume_profile > threshold].index.tolist()
        
        # Convert bins to price levels
        high_volume_levels = [(price_bins[bin_idx] + price_bins[bin_idx + 1]) / 2 for bin_idx in high_volume_bins]
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check if current price is approaching a high volume level
        current_price = self.df['close'].iloc[-1]
        
        # Find closest high volume level
        if high_volume_levels:
            closest_level = min(high_volume_levels, key=lambda x: abs(x - current_price))
            distance_pct = abs((closest_level - current_price) / current_price) * 100
            
            # If price is very close to a high volume level, it might act as support/resistance
            if distance_pct < 1.0:
                # Determine if it's likely support or resistance
                if closest_level < current_price and self.df['close'].iloc[-1] > self.df['close'].iloc[-2]:
                    current_signal = 1
                    signal_type = "Bouncing from Volume Support"
                    signal_strength = 1
                elif closest_level > current_price and self.df['close'].iloc[-1] < self.df['close'].iloc[-2]:
                    current_signal = -1
                    signal_type = "Rejected at Volume Resistance"
                    signal_strength = 1
        
        self.indicators['volume_profile'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'poc_price': round(poc_price, 2),
                'high_volume_levels': [round(level, 2) for level in high_volume_levels],
                'nearest_level': round(closest_level, 2) if high_volume_levels else None,
                'dist_pct_to_nearest': round(distance_pct, 2) if high_volume_levels else None,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Volume Profile',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_support_resistance(self):
        """Calculate Support and Resistance levels"""
        # Get parameters
        pivot_period = 5
        pivot_threshold = 0.03
        
        try:
            sr_params = self.params.get_indicator_param('support_resistance')
            if sr_params:
                pivot_period = sr_params.get('pivot_period', 5)
                pivot_threshold = sr_params.get('pivot_threshold', 0.03)
        except:
            pass
        
        # Find potential pivot points (local highs and lows)
        highs = []
        lows = []
        
        # We'll use a rolling window approach to find local highs and lows
        for i in range(pivot_period, len(self.df) - pivot_period):
            # Get window
            window = self.df.iloc[i - pivot_period:i + pivot_period + 1]
            
            # Check if middle point is a high or low pivot
            middle_high = window['high'].iloc[pivot_period]
            middle_low = window['low'].iloc[pivot_period]
            
            # Check if it's a local high
            if middle_high >= window['high'].max():
                highs.append((self.df.index[i], middle_high))
                
            # Check if it's a local low
            if middle_low <= window['low'].min():
                lows.append((self.df.index[i], middle_low))
        
        # Function to cluster levels that are close to each other
        def cluster_levels(levels, threshold):
            if not levels:
                return []
                
            # Sort levels by price
            levels.sort(key=lambda x: x[1])
            
            # Initialize clusters
            clusters = [[levels[0]]]
            
            # Group levels into clusters
            for level in levels[1:]:
                price = level[1]
                last_cluster_price = clusters[-1][-1][1]
                
                # If this level is close to the last cluster, add it there
                if abs(price - last_cluster_price) / last_cluster_price <= threshold:
                    clusters[-1].append(level)
                else:
                    # Start a new cluster
                    clusters.append([level])
            
            # Calculate average price for each cluster
            cluster_levels = []
            for cluster in clusters:
                avg_price = sum(level[1] for level in cluster) / len(cluster)
                latest_date = max(level[0] for level in cluster)
                cluster_levels.append((latest_date, avg_price))
                
            return cluster_levels
        
        # Cluster the levels
        resistance_levels = cluster_levels(highs, pivot_threshold)
        support_levels = cluster_levels(lows, pivot_threshold)
        
        # Sort levels by price
        resistance_levels.sort(key=lambda x: x[1])
        support_levels.sort(key=lambda x: x[1])
        
        # Find the most recent levels (focus on recent market structure)
        recent_resistance = []
        recent_support = []
        
        # We'll consider only the most recent levels that are close to current price
        current_price = self.df['close'].iloc[-1]
        price_threshold = current_price * 0.1  # 10% range
        
        for date, price in resistance_levels:
            if current_price - price_threshold <= price <= current_price + price_threshold:
                recent_resistance.append(price)
                
        for date, price in support_levels:
            if current_price - price_threshold <= price <= current_price + price_threshold:
                recent_support.append(price)
        
        # Find closest levels
        closest_resistance = None
        closest_support = None
        
        if recent_resistance:
            closest_resistance = min(recent_resistance, key=lambda x: abs(x - current_price) if x > current_price else float('inf'))
            
        if recent_support:
            closest_support = min(recent_support, key=lambda x: abs(x - current_price) if x < current_price else float('inf'))
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check if price is near support or resistance
        if closest_support is not None:
            support_distance = abs((current_price - closest_support) / current_price)
            
            if support_distance < 0.02:  # Within 2% of support
                current_signal = 1
                signal_type = "Near Support Level"
                signal_strength = 2
        
        if closest_resistance is not None:
            resistance_distance = abs((closest_resistance - current_price) / current_price)
            
            if resistance_distance < 0.02:  # Within 2% of resistance
                current_signal = -1
                signal_type = "Near Resistance Level"
                signal_strength = 2
        
        # If price broke through resistance or bounced off support recently
        if len(self.df) > 2:
            prev_price = self.df['close'].iloc[-2]
            
            if closest_resistance is not None and prev_price < closest_resistance < current_price:
                current_signal = 1
                signal_type = "Breakout Above Resistance"
                signal_strength = 3
            elif closest_support is not None and prev_price > closest_support > current_price:
                current_signal = -1
                signal_type = "Breakdown Below Support"
                signal_strength = 3
        
        self.indicators['support_resistance'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'closest_resistance': round(closest_resistance, 2) if closest_resistance is not None else None,
                'closest_support': round(closest_support, 2) if closest_support is not None else None,
                'all_resistances': [round(r, 2) for r in recent_resistance],
                'all_supports': [round(s, 2) for s in recent_support],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Support/Resistance',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_fibonacci_retracement(self):
        """Calculate Fibonacci Retracement levels"""
        # Get parameters
        lookback = 100
        
        try:
            fib_params = self.params.get_indicator_param('fibonacci_retracement')
            if fib_params:
                lookback = fib_params.get('lookback', 100)
        except:
            pass
        
        # Get subset of data for analysis
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
        
        subset = self.df.iloc[-lookback:]
        
        # Find significant high and low
        high_idx = subset['high'].idxmax()
        low_idx = subset['low'].idxmin()
        
        # Determine if we're in an uptrend or downtrend
        if high_idx > low_idx:
            # Uptrend - retracement from high to low
            trend = 'uptrend'
            high_val = subset.loc[high_idx, 'high']
            low_val = subset.loc[low_idx, 'low']
        else:
            # Downtrend - retracement from low to high
            trend = 'downtrend'
            high_val = subset.loc[low_idx, 'low']
            low_val = subset.loc[high_idx, 'high']
        
        # Calculate Fibonacci levels
        diff = abs(high_val - low_val)
        
        # Standard Fibonacci ratios
        fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        
        # Calculate the actual price levels
        if trend == 'uptrend':
            fib_levels = [high_val - (ratio * diff) for ratio in fib_ratios]
        else:
            fib_levels = [low_val + (ratio * diff) for ratio in fib_ratios]
        
        # Create dictionary of level values
        fib_dict = {f'fib_{int(ratio * 1000)}': level for ratio, level in zip(fib_ratios, fib_levels)}
        
        # Find closest Fibonacci level to current price
        current_price = self.df['close'].iloc[-1]
        closest_level = min(fib_levels, key=lambda x: abs(x - current_price))
        closest_ratio = fib_ratios[fib_levels.index(closest_level)]
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check if price is near a Fibonacci level
        price_distance = abs((current_price - closest_level) / current_price)
        
        if price_distance < 0.01:  # Within 1% of a Fibonacci level
            # Determine if this is likely support or resistance
            if (trend == 'uptrend' and self.df['close'].iloc[-1] < self.df['close'].iloc[-2]) or \
               (trend == 'downtrend' and self.df['close'].iloc[-1] > self.df['close'].iloc[-2]):
                # Price might be finding support at this level
                current_signal = 1
                signal_type = f"Potential Support at {int(closest_ratio * 100)}% Fib Level"
                signal_strength = 2
            else:
                # Price might be finding resistance at this level
                current_signal = -1
                signal_type = f"Potential Resistance at {int(closest_ratio * 100)}% Fib Level"
                signal_strength = 2
        
        self.indicators['fibonacci_retracement'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'trend': trend,
                'levels': {f'{int(ratio * 100)}%': round(level, 2) for ratio, level in zip(fib_ratios, fib_levels)},
                'closest_level': round(closest_level, 2),
                'closest_ratio': f'{int(closest_ratio * 100)}%',
                'distance_pct': round(price_distance * 100, 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Fibonacci Retracement',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def get_signals(self):
        """Get all trading signals generated by indicators"""
        return self.signals
    
    def get_overall_signal(self):
        """Calculate overall signal based on all indicators"""
        if not self.signals:
            return {
                'signal': 0,
                'strength': 0,
                'description': 'No trading signals detected'
            }
        
        # Count buy and sell signals with their strengths
        buy_signals = [s for s in self.signals if s['signal'] == 'BUY']
        sell_signals = [s for s in self.signals if s['signal'] == 'SELL']
        
        buy_strength = sum(s['strength'] for s in buy_signals)
        sell_strength = sum(s['strength'] for s in sell_signals)
        
        # Determine overall signal
        if buy_strength > sell_strength:
            overall_strength = min(5, max(1, round((buy_strength - sell_strength) / 2)))
            return {
                'signal': 1,
                'strength': overall_strength,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'description': f'BUY - {len(buy_signals)} signals for vs {len(sell_signals)} against'
            }
        elif sell_strength > buy_strength:
            overall_strength = min(5, max(1, round((sell_strength - buy_strength) / 2)))
            return {
                'signal': -1,
                'strength': overall_strength,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'description': f'SELL - {len(sell_signals)} signals for vs {len(buy_signals)} against'
            }
        else:
            return {
                'signal': 0,
                'strength': 0,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'description': 'NEUTRAL - No clear direction'
            }
        



class ChartPatterns:
    """Detect complex chart patterns in price data"""
    
    def __init__(self, df, params=None):
        """
        Initialize chart pattern detector with OHLCV DataFrame
        
        Args:
            df: DataFrame with OHLCV data (index=timestamp, columns=[open, high, low, close, volume])
            params: Optional TradingParameters instance for pattern parameters
        """
        # Create a copy of the DataFrame to avoid modifying the original
        self.df = df.copy()
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Set parameters from config or use defaults
        self.params = params or TradingParameters()
        
        # Initialize patterns and signals lists
        self.patterns = []
        self.signals = []
        self.logger = logging.getLogger(__name__)
        
        # Minimum data length required for pattern detection
        self.min_data_length = 40
        
        # Ensure we have enough data
        if len(df) < self.min_data_length:
            self.logger.warning(f"Not enough data for chart pattern detection. Need at least {self.min_data_length} bars.")
    
    def detect_all_patterns(self):
        """Run all chart pattern detection algorithms and collect results"""
        if len(self.df) < self.min_data_length:
            return []
            
        # Find swing points first as they're used by many patterns
        self.find_swing_points()
        
        # Run all detection methods
        detection_methods = [
            self.detect_head_and_shoulders,
            self.detect_double_patterns,
            self.detect_triple_patterns,
            self.detect_wedges,
            self.detect_rectangle,
            self.detect_flags,
            self.detect_cup_and_handle,
            self.detect_rounding_patterns
        ]
        
        for method in detection_methods:
            try:
                method()
            except Exception as e:
                self.logger.error(f"Error detecting pattern with {method.__name__}: {str(e)}")
        
        return self.patterns
    
    def find_swing_points(self, window_size=None):
        """
        Find swing highs and lows in the price data
        
        Args:
            window_size: Number of bars to use when detecting swing points
                         If None, will use a dynamic size based on volatility
        """
        # If window size not specified, calculate dynamic window based on volatility
        if window_size is None:
            if 'atr' not in self.df.columns:
                # Calculate ATR if not already present
                atr_period = 14
                high_low = self.df['high'] - self.df['low']
                high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
                low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
                tr = pd.DataFrame({'hl': high_low, 'hcp': high_close_prev, 'lcp': low_close_prev}).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate average price
                avg_price = self.df['close'].mean()
                
                # Calculate ATR as percentage of price
                atr_pct = (atr / avg_price).mean() * 100
                
                # Adjust window size based on volatility
                if atr_pct > 2.0:
                    window_size = 3  # High volatility, use smaller window
                elif atr_pct > 1.0:
                    window_size = 5  # Medium volatility
                else:
                    window_size = 7  # Low volatility, use larger window
            else:
                # If ATR is already calculated, use it
                atr_pct = (self.df['atr'] / self.df['close']).mean() * 100
                if atr_pct > 2.0:
                    window_size = 3
                elif atr_pct > 1.0:
                    window_size = 5
                else:
                    window_size = 7
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
            
        half_window = window_size // 2
        
        # Initialize columns for swing points
        swing_highs = np.zeros(len(self.df))
        swing_lows = np.zeros(len(self.df))
        
        # Find swing points
        for i in range(half_window, len(self.df) - half_window):
            # Get window around current point
            window = self.df.iloc[i - half_window:i + half_window + 1]
            
            # Check if current point is a swing high
            if self.df['high'].iloc[i] == window['high'].max():
                swing_highs[i] = 1
                
            # Check if current point is a swing low
            if self.df['low'].iloc[i] == window['low'].min():
                swing_lows[i] = 1
        
        # Save swing points to DataFrame
        self.df['swing_high'] = swing_highs
        self.df['swing_low'] = swing_lows
        
        # Return indices of swing points
        swing_high_indices = np.where(swing_highs == 1)[0]
        swing_low_indices = np.where(swing_lows == 1)[0]
        
        return swing_high_indices, swing_low_indices
    
    def detect_head_and_shoulders(self):
        """Detect head and shoulders and inverse head and shoulders patterns"""
        # Get parameters
        try:
            head_tolerance = self.params.get_chart_pattern_param('head_and_shoulders')['head_tolerance']
            shoulder_tolerance = self.params.get_chart_pattern_param('head_and_shoulders')['shoulder_tolerance']
        except:
            head_tolerance = 0.03
            shoulder_tolerance = 0.05
        
        # Ensure swing points are calculated
        if 'swing_high' not in self.df.columns or 'swing_low' not in self.df.columns:
            self.find_swing_points()
        
        # Get swing point indices
        swing_high_indices = np.where(self.df['swing_high'] == 1)[0]
        swing_low_indices = np.where(self.df['swing_low'] == 1)[0]
        
        # We need at least 5 swing points for a valid pattern
        if len(swing_high_indices) < 3 or len(swing_low_indices) < 2:
            return
        
        # Regular Head and Shoulders (Bearish)
        for i in range(len(swing_high_indices) - 2):
            # Get three consecutive swing highs
            idx1 = swing_high_indices[i]
            idx2 = swing_high_indices[i + 1]
            idx3 = swing_high_indices[i + 2]
            
            # Get prices at these points
            price1 = self.df['high'].iloc[idx1]
            price2 = self.df['high'].iloc[idx2]
            price3 = self.df['high'].iloc[idx3]
            
            # Check if middle high (head) is higher than the shoulders
            if price2 > price1 and price2 > price3:
                # Check if shoulders are roughly at the same level
                shoulder_diff = abs(price1 - price3) / price1
                if shoulder_diff <= shoulder_tolerance:
                    # Find a neckline connecting the lows between shoulders and head
                    # Look for two swing lows between the shoulders
                    between_lows = [idx for idx in swing_low_indices if idx1 < idx < idx3]
                    
                    if len(between_lows) >= 2:
                        # Get the first and last lows between shoulders
                        low1_idx = between_lows[0]
                        low2_idx = between_lows[-1]
                        
                        # Get prices at these points
                        low1_price = self.df['low'].iloc[low1_idx]
                        low2_price = self.df['low'].iloc[low2_idx]
                        
                        # Calculate neckline
                        x1, y1 = low1_idx, low1_price
                        x2, y2 = low2_idx, low2_price
                        
                        # Check for neckline break
                        neckline_broken = False
                        if idx3 < len(self.df) - 1:
                            # Check a few bars after the right shoulder
                            for j in range(1, min(10, len(self.df) - idx3 - 1)):
                                check_idx = idx3 + j
                                # Calculate neckline level at this point
                                if x2 == x1:  # Avoid division by zero
                                    neckline_at_j = y1
                                else:
                                    slope = (y2 - y1) / (x2 - x1)
                                    neckline_at_j = y1 + slope * (check_idx - x1)
                                    
                                # Check if price has broken the neckline
                                if self.df['close'].iloc[check_idx] < neckline_at_j:
                                    neckline_broken = True
                                    break
                        
                        # Calculate pattern size for signal strength
                        pattern_size = (price2 - min(low1_price, low2_price)) / min(low1_price, low2_price) * 100
                        signal_strength = 2
                        if pattern_size > 5:
                            signal_strength = 3
                        if pattern_size > 10:
                            signal_strength = 4
                        
                        # Add to patterns list
                        pattern = {
                            'type': 'Head and Shoulders',
                            'direction': 'bearish',
                            'start_idx': idx1,
                            'end_idx': idx3,
                            'signal': -1,
                            'confirmation': neckline_broken,
                            'strength': signal_strength,
                            'sizes': {
                                'left_shoulder': price1,
                                'head': price2,
                                'right_shoulder': price3,
                                'neckline': (low1_price + low2_price) / 2
                            }
                        }
                        
                        self.patterns.append(pattern)
                        
                        # Add to signals if confirmed
                        if neckline_broken:
                            self.signals.append({
                                'pattern': 'Head and Shoulders',
                                'signal': 'SELL',
                                'strength': signal_strength,
                                'start_idx': idx1,
                                'end_idx': idx3
                            })
        
        # Inverse Head and Shoulders (Bullish)
        for i in range(len(swing_low_indices) - 2):
            # Get three consecutive swing lows
            idx1 = swing_low_indices[i]
            idx2 = swing_low_indices[i + 1]
            idx3 = swing_low_indices[i + 2]
            
            # Get prices at these points
            price1 = self.df['low'].iloc[idx1]
            price2 = self.df['low'].iloc[idx2]
            price3 = self.df['low'].iloc[idx3]
            
            # Check if middle low (head) is lower than the shoulders
            if price2 < price1 and price2 < price3:
                # Check if shoulders are roughly at the same level
                shoulder_diff = abs(price1 - price3) / price1
                if shoulder_diff <= shoulder_tolerance:
                    # Find a neckline connecting the highs between shoulders and head
                    # Look for two swing highs between the shoulders
                    between_highs = [idx for idx in swing_high_indices if idx1 < idx < idx3]
                    
                    if len(between_highs) >= 2:
                        # Get the first and last highs between shoulders
                        high1_idx = between_highs[0]
                        high2_idx = between_highs[-1]
                        
                        # Get prices at these points
                        high1_price = self.df['high'].iloc[high1_idx]
                        high2_price = self.df['high'].iloc[high2_idx]
                        
                        # Calculate neckline
                        x1, y1 = high1_idx, high1_price
                        x2, y2 = high2_idx, high2_price
                        
                        # Check for neckline break
                        neckline_broken = False
                        if idx3 < len(self.df) - 1:
                            # Check a few bars after the right shoulder
                            for j in range(1, min(10, len(self.df) - idx3 - 1)):
                                check_idx = idx3 + j
                                # Calculate neckline level at this point
                                if x2 == x1:  # Avoid division by zero
                                    neckline_at_j = y1
                                else:
                                    slope = (y2 - y1) / (x2 - x1)
                                    neckline_at_j = y1 + slope * (check_idx - x1)
                                    
                                # Check if price has broken the neckline
                                if self.df['close'].iloc[check_idx] > neckline_at_j:
                                    neckline_broken = True
                                    break
                        
                        # Calculate pattern size for signal strength
                        pattern_size = (max(high1_price, high2_price) - price2) / price2 * 100
                        signal_strength = 2
                        if pattern_size > 5:
                            signal_strength = 3
                        if pattern_size > 10:
                            signal_strength = 4
                        
                        # Add to patterns list
                        pattern = {
                            'type': 'Inverse Head and Shoulders',
                            'direction': 'bullish',
                            'start_idx': idx1,
                            'end_idx': idx3,
                            'signal': 1,
                            'confirmation': neckline_broken,
                            'strength': signal_strength,
                            'sizes': {
                                'left_shoulder': price1,
                                'head': price2,
                                'right_shoulder': price3,
                                'neckline': (high1_price + high2_price) / 2
                            }
                        }
                        
                        self.patterns.append(pattern)
                        
                        # Add to signals if confirmed
                        if neckline_broken:
                            self.signals.append({
                                'pattern': 'Inverse Head and Shoulders',
                                'signal': 'BUY',
                                'strength': signal_strength,
                                'start_idx': idx1,
                                'end_idx': idx3
                            })
    
    def detect_double_patterns(self):
        """Detect double top and double bottom patterns"""
        # Get parameters
        try:
            tolerance = self.params.get_chart_pattern_param('double_pattern')['tolerance']
            lookback = self.params.get_chart_pattern_param('double_pattern')['lookback']
        except:
            tolerance = 0.03
            lookback = 50
        
        # Ensure swing points are calculated
        if 'swing_high' not in self.df.columns or 'swing_low' not in self.df.columns:
            self.find_swing_points()
        
        # Get last N bars for analysis
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Get swing point indices
        swing_high_indices = np.where(recent_df['swing_high'] == 1)[0]
        swing_low_indices = np.where(recent_df['swing_low'] == 1)[0]
        
        # We need at least 2 swing points for a valid pattern
        if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
            return
        
        # Double Top (Bearish)
        for i in range(len(swing_high_indices) - 1):
            # Get two consecutive swing highs
            idx1 = swing_high_indices[i]
            idx2 = swing_high_indices[i + 1]
            
            # Ensure they're separated by at least a few bars
            if idx2 - idx1 < 5:
                continue
                
            # Get prices at these points
            price1 = recent_df['high'].iloc[idx1]
            price2 = recent_df['high'].iloc[idx2]
            
            # Check if tops are roughly at the same level
            price_diff = abs(price1 - price2) / price1
            if price_diff <= tolerance:
                # Find the lowest point between the two tops
                between = recent_df.iloc[idx1:idx2]
                valley_idx = between['low'].idxmin() - recent_df.index[0]
                valley_price = between['low'].min()
                
                # Check for neckline break
                neckline_broken = False
                if idx2 < len(recent_df) - 1:
                    # Check a few bars after the second top
                    for j in range(1, min(10, len(recent_df) - idx2 - 1)):
                        check_idx = idx2 + j
                        # Check if price has broken below the valley
                        if recent_df['close'].iloc[check_idx] < valley_price:
                            neckline_broken = True
                            break
                
                # Calculate pattern size for signal strength
                pattern_size = (price1 - valley_price) / valley_price * 100
                signal_strength = 2
                if pattern_size > 5:
                    signal_strength = 3
                if pattern_size > 10:
                    signal_strength = 4
                
                # Add to patterns list
                pattern = {
                    'type': 'Double Top',
                    'direction': 'bearish',
                    'start_idx': recent_df.index[0] + idx1,
                    'end_idx': recent_df.index[0] + idx2,
                    'signal': -1,
                    'confirmation': neckline_broken,
                    'strength': signal_strength,
                    'sizes': {
                        'top1': price1,
                        'top2': price2,
                        'valley': valley_price
                    }
                }
                
                self.patterns.append(pattern)
                
                # Add to signals if confirmed
                if neckline_broken:
                    self.signals.append({
                        'pattern': 'Double Top',
                        'signal': 'SELL',
                        'strength': signal_strength,
                        'start_idx': recent_df.index[0] + idx1,
                        'end_idx': recent_df.index[0] + idx2
                    })
        
        # Double Bottom (Bullish)
        for i in range(len(swing_low_indices) - 1):
            # Get two consecutive swing lows
            idx1 = swing_low_indices[i]
            idx2 = swing_low_indices[i + 1]
            
            # Ensure they're separated by at least a few bars
            if idx2 - idx1 < 5:
                continue
                
            # Get prices at these points
            price1 = recent_df['low'].iloc[idx1]
            price2 = recent_df['low'].iloc[idx2]
            
            # Check if bottoms are roughly at the same level
            price_diff = abs(price1 - price2) / price1
            if price_diff <= tolerance:
                # Find the highest point between the two bottoms
                between = recent_df.iloc[idx1:idx2]
                peak_idx = between['high'].idxmax() - recent_df.index[0]
                peak_price = between['high'].max()
                
                # Check for neckline break
                neckline_broken = False
                if idx2 < len(recent_df) - 1:
                    # Check a few bars after the second bottom
                    for j in range(1, min(10, len(recent_df) - idx2 - 1)):
                        check_idx = idx2 + j
                        # Check if price has broken above the peak
                        if recent_df['close'].iloc[check_idx] > peak_price:
                            neckline_broken = True
                            break
                
                # Calculate pattern size for signal strength
                pattern_size = (peak_price - price1) / price1 * 100
                signal_strength = 2
                if pattern_size > 5:
                    signal_strength = 3
                if pattern_size > 10:
                    signal_strength = 4
                
                # Add to patterns list
                pattern = {
                    'type': 'Double Bottom',
                    'direction': 'bullish',
                    'start_idx': recent_df.index[0] + idx1,
                    'end_idx': recent_df.index[0] + idx2,
                    'signal': 1,
                    'confirmation': neckline_broken,
                    'strength': signal_strength,
                    'sizes': {
                        'bottom1': price1,
                        'bottom2': price2,
                        'peak': peak_price
                    }
                }
                
                self.patterns.append(pattern)
                
                # Add to signals if confirmed
                if neckline_broken:
                    self.signals.append({
                        'pattern': 'Double Bottom',
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'start_idx': recent_df.index[0] + idx1,
                        'end_idx': recent_df.index[0] + idx2
                    })
    
    def detect_triple_patterns(self):
        """Detect triple top and triple bottom patterns"""
        # Get parameters
        try:
            tolerance = self.params.get_chart_pattern_param('triple_pattern')['tolerance']
            lookback = self.params.get_chart_pattern_param('triple_pattern')['lookback']
        except:
            tolerance = 0.03
            lookback = 100
        
        # Ensure swing points are calculated
        if 'swing_high' not in self.df.columns or 'swing_low' not in self.df.columns:
            self.find_swing_points()
        
        # Get last N bars for analysis
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Get swing point indices
        swing_high_indices = np.where(recent_df['swing_high'] == 1)[0]
        swing_low_indices = np.where(recent_df['swing_low'] == 1)[0]
        
        # We need at least 3 swing points for a valid pattern
        if len(swing_high_indices) < 3 or len(swing_low_indices) < 3:
            return
        
        # Triple Top (Bearish)
        for i in range(len(swing_high_indices) - 2):
            # Get three consecutive swing highs
            idx1 = swing_high_indices[i]
            idx2 = swing_high_indices[i + 1]
            idx3 = swing_high_indices[i + 2]
            
            # Ensure they're separated by at least a few bars
            if idx2 - idx1 < 5 or idx3 - idx2 < 5:
                continue
                
            # Get prices at these points
            price1 = recent_df['high'].iloc[idx1]
            price2 = recent_df['high'].iloc[idx2]
            price3 = recent_df['high'].iloc[idx3]
            
            # Check if tops are roughly at the same level
            price_diff1 = abs(price1 - price2) / price1
            price_diff2 = abs(price2 - price3) / price2
            price_diff3 = abs(price1 - price3) / price1
            
            if price_diff1 <= tolerance and price_diff2 <= tolerance and price_diff3 <= tolerance:
                # Find the lowest points between the tops
                between1 = recent_df.iloc[idx1:idx2]
                between2 = recent_df.iloc[idx2:idx3]
                
                valley1_price = between1['low'].min()
                valley2_price = between2['low'].min()
                
                # Use the higher of the two valleys as neckline
                neckline = max(valley1_price, valley2_price)
                
                # Check for neckline break
                neckline_broken = False
                if idx3 < len(recent_df) - 1:
                    # Check a few bars after the third top
                    for j in range(1, min(10, len(recent_df) - idx3 - 1)):
                        check_idx = idx3 + j
                        # Check if price has broken below the neckline
                        if recent_df['close'].iloc[check_idx] < neckline:
                            neckline_broken = True
                            break
                
                # Calculate pattern size for signal strength
                pattern_size = (price1 - neckline) / neckline * 100
                signal_strength = 3
                if pattern_size > 5:
                    signal_strength = 4
                if pattern_size > 10:
                    signal_strength = 5
                
                # Add to patterns list
                pattern = {
                    'type': 'Triple Top',
                    'direction': 'bearish',
                    'start_idx': recent_df.index[0] + idx1,
                    'end_idx': recent_df.index[0] + idx3,
                    'signal': -1,
                    'confirmation': neckline_broken,
                    'strength': signal_strength,
                    'sizes': {
                        'top1': price1,
                        'top2': price2,
                        'top3': price3,
                        'neckline': neckline
                    }
                }
                
                self.patterns.append(pattern)
                
                # Add to signals if confirmed
                if neckline_broken:
                    self.signals.append({
                        'pattern': 'Triple Top',
                        'signal': 'SELL',
                        'strength': signal_strength,
                        'start_idx': recent_df.index[0] + idx1,
                        'end_idx': recent_df.index[0] + idx3
                    })
        
        # Triple Bottom (Bullish)
        for i in range(len(swing_low_indices) - 2):
            # Get three consecutive swing lows
            idx1 = swing_low_indices[i]
            idx2 = swing_low_indices[i + 1]
            idx3 = swing_low_indices[i + 2]
            
            # Ensure they're separated by at least a few bars
            if idx2 - idx1 < 5 or idx3 - idx2 < 5:
                continue
                
            # Get prices at these points
            price1 = recent_df['low'].iloc[idx1]
            price2 = recent_df['low'].iloc[idx2]
            price3 = recent_df['low'].iloc[idx3]
            
            # Check if bottoms are roughly at the same level
            price_diff1 = abs(price1 - price2) / price1
            price_diff2 = abs(price2 - price3) / price2
            price_diff3 = abs(price1 - price3) / price1
            
            if price_diff1 <= tolerance and price_diff2 <= tolerance and price_diff3 <= tolerance:
                # Find the highest points between the bottoms
                between1 = recent_df.iloc[idx1:idx2]
                between2 = recent_df.iloc[idx2:idx3]
                
                peak1_price = between1['high'].max()
                peak2_price = between2['high'].max()
                
                # Use the lower of the two peaks as neckline
                neckline = min(peak1_price, peak2_price)
                
                # Check for neckline break
                neckline_broken = False
                if idx3 < len(recent_df) - 1:
                    # Check a few bars after the third bottom
                    for j in range(1, min(10, len(recent_df) - idx3 - 1)):
                        check_idx = idx3 + j
                        # Check if price has broken above the neckline
                        if recent_df['close'].iloc[check_idx] > neckline:
                            neckline_broken = True
                            break
                
                # Calculate pattern size for signal strength
                pattern_size = (neckline - price1) / price1 * 100
                signal_strength = 3
                if pattern_size > 5:
                    signal_strength = 4
                if pattern_size > 10:
                    signal_strength = 5
                
                # Add to patterns list
                pattern = {
                    'type': 'Triple Bottom',
                    'direction': 'bullish',
                    'start_idx': recent_df.index[0] + idx1,
                    'end_idx': recent_df.index[0] + idx3,
                    'signal': 1,
                    'confirmation': neckline_broken,
                    'strength': signal_strength,
                    'sizes': {
                        'bottom1': price1,
                        'bottom2': price2,
                        'bottom3': price3,
                        'neckline': neckline
                    }
                }
                
                self.patterns.append(pattern)
                
                # Add to signals if confirmed
                if neckline_broken:
                    self.signals.append({
                        'pattern': 'Triple Bottom',
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'start_idx': recent_df.index[0] + idx1,
                        'end_idx': recent_df.index[0] + idx3
                    })
    
    def detect_wedges(self):
        """Detect rising and falling wedge patterns"""
        # Ensure swing points are calculated
        if 'swing_high' not in self.df.columns or 'swing_low' not in self.df.columns:
            self.find_swing_points()
        
        # Get last 60 bars for analysis
        lookback = 60
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Get swing point indices
        swing_high_indices = np.where(recent_df['swing_high'] == 1)[0]
        swing_low_indices = np.where(recent_df['swing_low'] == 1)[0]
        
        # We need at least 3 swing points of each kind for a valid pattern
        if len(swing_high_indices) < 3 or len(swing_low_indices) < 3:
            return
        
        # Function to check if points form a line with increasing/decreasing slope
        def check_trendline(points, increasing=True):
            # Check if we have at least 3 points
            if len(points) < 3:
                return False
                
            # Calculate slopes between consecutive points
            slopes = []
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                if x2 == x1:  # Avoid division by zero
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)
            
            # Check if all slopes have the same sign and are increasing/decreasing
            if increasing:
                return all(s > 0 for s in slopes) and slopes[-1] > slopes[0]
            else:
                return all(s < 0 for s in slopes) and slopes[-1] < slopes[0]
        
        # Get high and low points
        high_points = [(idx, recent_df['high'].iloc[idx]) for idx in swing_high_indices]
        low_points = [(idx, recent_df['low'].iloc[idx]) for idx in swing_low_indices]
        
        # Check for Rising Wedge (Bearish)
        # Upper trendline with decreasing slope, lower trendline with decreasing slope but steeper
        upper_decreasing = check_trendline(high_points, increasing=False)
        lower_decreasing = check_trendline(low_points, increasing=False)
        
        if upper_decreasing and lower_decreasing:
            # Calculate slopes
            upper_slope = (high_points[-1][1] - high_points[0][1]) / (high_points[-1][0] - high_points[0][0])
            lower_slope = (low_points[-1][1] - low_points[0][1]) / (low_points[-1][0] - low_points[0][0])
            
            # Check if lower trendline is steeper
            if lower_slope < upper_slope:
                # Check if wedge is narrowing
                upper_y_end = high_points[-1][1]
                lower_y_end = low_points[-1][1]
                
                start_width = abs(high_points[0][1] - low_points[0][1])
                end_width = abs(upper_y_end - lower_y_end)
                
                if end_width < start_width:
                    # Check for break below lower trendline
                    breakout = False
                    last_idx = max(high_points[-1][0], low_points[-1][0])
                    
                    if last_idx < len(recent_df) - 1:
                        # Calculate lower trendline level at current point
                        x1, y1 = low_points[0]
                        x2, y2 = low_points[-1]
                        
                        current_idx = len(recent_df) - 1
                        if x2 != x1:  # Avoid division by zero
                            slope = (y2 - y1) / (x2 - x1)
                            trendline_level = y1 + slope * (current_idx - x1)
                            
                            # Check if price has broken below trendline
                            if recent_df['close'].iloc[-1] < trendline_level:
                                breakout = True
                    
                    # Calculate pattern size for signal strength
                    pattern_size = (start_width / recent_df['close'].iloc[0]) * 100
                    signal_strength = 3
                    if pattern_size > 5:
                        signal_strength = 4
                    
                    # Add to patterns list
                    pattern = {
                        'type': 'Rising Wedge',
                        'direction': 'bearish',
                        'start_idx': recent_df.index[0] + min(high_points[0][0], low_points[0][0]),
                        'end_idx': recent_df.index[0] + max(high_points[-1][0], low_points[-1][0]),
                        'signal': -1,
                        'confirmation': breakout,
                        'strength': signal_strength
                    }
                    
                    self.patterns.append(pattern)
                    
                    # Add to signals if confirmed
                    if breakout:
                        self.signals.append({
                            'pattern': 'Rising Wedge',
                            'signal': 'SELL',
                            'strength': signal_strength,
                            'start_idx': recent_df.index[0] + min(high_points[0][0], low_points[0][0]),
                            'end_idx': recent_df.index[0] + max(high_points[-1][0], low_points[-1][0])
                        })
        
        # Check for Falling Wedge (Bullish)
        # Upper trendline with increasing slope, lower trendline with increasing slope but steeper
        upper_increasing = check_trendline(high_points, increasing=True)
        lower_increasing = check_trendline(low_points, increasing=True)
        
        if upper_increasing and lower_increasing:
            # Calculate slopes
            upper_slope = (high_points[-1][1] - high_points[0][1]) / (high_points[-1][0] - high_points[0][0])
            lower_slope = (low_points[-1][1] - low_points[0][1]) / (low_points[-1][0] - low_points[0][0])
            
            # Check if upper trendline is steeper
            if upper_slope > lower_slope:
                # Check if wedge is narrowing
                upper_y_end = high_points[-1][1]
                lower_y_end = low_points[-1][1]
                
                start_width = abs(high_points[0][1] - low_points[0][1])
                end_width = abs(upper_y_end - lower_y_end)
                
                if end_width < start_width:
                    # Check for break above upper trendline
                    breakout = False
                    last_idx = max(high_points[-1][0], low_points[-1][0])
                    
                    if last_idx < len(recent_df) - 1:
                        # Calculate upper trendline level at current point
                        x1, y1 = high_points[0]
                        x2, y2 = high_points[-1]
                        
                        current_idx = len(recent_df) - 1
                        if x2 != x1:  # Avoid division by zero
                            slope = (y2 - y1) / (x2 - x1)
                            trendline_level = y1 + slope * (current_idx - x1)
                            
                            # Check if price has broken above trendline
                            if recent_df['close'].iloc[-1] > trendline_level:
                                breakout = True
                    
                    # Calculate pattern size for signal strength
                    pattern_size = (start_width / recent_df['close'].iloc[0]) * 100
                    signal_strength = 3
                    if pattern_size > 5:
                        signal_strength = 4
                    
                    # Add to patterns list
                    pattern = {
                        'type': 'Falling Wedge',
                        'direction': 'bullish',
                        'start_idx': recent_df.index[0] + min(high_points[0][0], low_points[0][0]),
                        'end_idx': recent_df.index[0] + max(high_points[-1][0], low_points[-1][0]),
                        'signal': 1,
                        'confirmation': breakout,
                        'strength': signal_strength
                    }
                    
                    self.patterns.append(pattern)
                    
                    # Add to signals if confirmed
                    if breakout:
                        self.signals.append({
                            'pattern': 'Falling Wedge',
                            'signal': 'BUY',
                            'strength': signal_strength,
                            'start_idx': recent_df.index[0] + min(high_points[0][0], low_points[0][0]),
                            'end_idx': recent_df.index[0] + max(high_points[-1][0], low_points[-1][0])
                        })
    
    def detect_rectangle(self):
        """Detect rectangle (range) patterns"""
        # Get parameters
        try:
            lookback = self.params.get_chart_pattern_param('rectangle')['lookback']
            tolerance = self.params.get_chart_pattern_param('rectangle')['tolerance']
            min_touches = self.params.get_chart_pattern_param('rectangle')['min_touches']
        except:
            lookback = 60
            tolerance = 0.05
            min_touches = 4
        
        # Ensure swing points are calculated
        if 'swing_high' not in self.df.columns or 'swing_low' not in self.df.columns:
            self.find_swing_points()
        
        # Get last N bars for analysis
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Get swing point indices
        swing_high_indices = np.where(recent_df['swing_high'] == 1)[0]
        swing_low_indices = np.where(recent_df['swing_low'] == 1)[0]
        
        # Function to cluster levels that are close to each other
        def cluster_levels(levels, tolerance):
            if not levels:
                return []
                
            # Sort levels by price
            sorted_levels = sorted(levels)
            
            # Initialize clusters
            clusters = [[sorted_levels[0]]]
            
            # Group levels into clusters
            for level in sorted_levels[1:]:
                last_cluster_avg = sum(clusters[-1]) / len(clusters[-1])
                
                # If this level is close to the last cluster, add it there
                if abs(level - last_cluster_avg) / last_cluster_avg <= tolerance:
                    clusters[-1].append(level)
                else:
                    # Start a new cluster
                    clusters.append([level])
                    
            # Calculate average price for each cluster
            cluster_levels = [sum(cluster) / len(cluster) for cluster in clusters]
            
            return cluster_levels
        
        # Get high and low values
        highs = [recent_df['high'].iloc[idx] for idx in swing_high_indices]
        lows = [recent_df['low'].iloc[idx] for idx in swing_low_indices]
        
        # Cluster similar price levels
        resistance_levels = cluster_levels(highs, tolerance)
        support_levels = cluster_levels(lows, tolerance)
        
        # Check for rectangle pattern
        for resistance in resistance_levels:
            for support in support_levels:
                # Calculate price difference
                price_range = resistance - support
                mid_price = (resistance + support) / 2
                
                # Check if resistance and support are roughly parallel and non-trivial
                if price_range / mid_price > 0.02:  # At least 2% range
                    # Count touches of resistance and support
                    resistance_touches = sum(1 for h in highs if abs(h - resistance) / resistance <= tolerance)
                    support_touches = sum(1 for l in lows if abs(l - support) / support <= tolerance)
                    
                    # Check if we have enough touches
                    if resistance_touches + support_touches >= min_touches:
                        # Find earliest and latest touches
                        all_touches = [(idx, 'r') for idx in swing_high_indices if abs(recent_df['high'].iloc[idx] - resistance) / resistance <= tolerance]
                        all_touches += [(idx, 's') for idx in swing_low_indices if abs(recent_df['low'].iloc[idx] - support) / support <= tolerance]
                        
                        all_touches.sort(key=lambda x: x[0])
                        
                        if len(all_touches) >= 2:
                            start_idx = all_touches[0][0]
                            end_idx = all_touches[-1][0]
                            
                            # Check if last touch was a support or resistance
                            last_touch_type = all_touches[-1][1]
                            
                            # Check for breakout
                            breakout = 0  # 0 for none, 1 for bullish, -1 for bearish
                            
                            if end_idx < len(recent_df) - 1:
                                current_close = recent_df['close'].iloc[-1]
                                
                                if current_close > resistance * 1.01:  # 1% above resistance
                                    breakout = 1
                                elif current_close < support * 0.99:  # 1% below support
                                    breakout = -1
                            
                            # Calculate pattern size for signal strength
                            pattern_size = (price_range / mid_price) * 100
                            signal_strength = 2
                            if pattern_size > 5:
                                signal_strength = 3
                            
                            # Determine signal direction based on breakout or last touch
                            if breakout != 0:
                                signal = breakout
                            else:
                                # If last touch was resistance, expect downward move
                                if last_touch_type == 'r':
                                    signal = -1
                                else:
                                    signal = 1
                            
                            # Add to patterns list
                            pattern = {
                                'type': 'Rectangle',
                                'direction': 'bullish' if signal > 0 else 'bearish',
                                'start_idx': recent_df.index[0] + start_idx,
                                'end_idx': recent_df.index[0] + end_idx,
                                'signal': signal,
                                'confirmation': breakout != 0,
                                'strength': signal_strength,
                                'sizes': {
                                    'resistance': resistance,
                                    'support': support,
                                    'range_pct': pattern_size
                                }
                            }
                            
                            self.patterns.append(pattern)
                            
                            # Add to signals if confirmed
                            if breakout != 0:
                                self.signals.append({
                                    'pattern': 'Rectangle Breakout',
                                    'signal': 'BUY' if breakout > 0 else 'SELL',
                                    'strength': signal_strength,
                                    'start_idx': recent_df.index[0] + start_idx,
                                    'end_idx': recent_df.index[0] + end_idx
                                })
    
    def detect_flags(self):
        """Detect flag and pennant patterns"""
        # Get parameters
        try:
            lookback = self.params.get_chart_pattern_param('flag')['lookback']
            pole_threshold = self.params.get_chart_pattern_param('flag')['pole_threshold']
            consolidation_threshold = self.params.get_chart_pattern_param('flag')['consolidation_threshold']
            max_bars = self.params.get_chart_pattern_param('flag')['max_bars']
        except:
            lookback = 30
            pole_threshold = 0.15
            consolidation_threshold = 0.05
            max_bars = 15
        
        # Get last N bars for analysis
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Check for sharp moves that could be flag poles
        for i in range(lookback - max_bars):
            # Check a window of up to 5 bars for sharp move
            max_window = min(5, lookback - i)
            
            # Calculate price change over the window
            start_price = recent_df['close'].iloc[i]
            end_idx = i + max_window - 1
            end_price = recent_df['close'].iloc[end_idx]
            
            price_change = (end_price - start_price) / start_price
            
            # Check if we have a sharp move that could be a flag pole
            if abs(price_change) >= pole_threshold:
                # We found a potential pole, now check for consolidation after it
                consolidation_start = end_idx + 1
                
                # Make sure we have enough bars remaining
                if consolidation_start >= lookback - 3:
                    continue
                
                # Check if subsequent bars form a consolidation
                consolidation_end = min(consolidation_start + max_bars, lookback - 1)
                
                # Get high and low of consolidation period
                consolidation_high = recent_df['high'].iloc[consolidation_start:consolidation_end].max()
                consolidation_low = recent_df['low'].iloc[consolidation_start:consolidation_end].min()
                
                # Calculate consolidation range as percentage
                mid_price = (consolidation_high + consolidation_low) / 2
                consolidation_range = (consolidation_high - consolidation_low) / mid_price
                
                # Check if consolidation is narrow enough
                if consolidation_range <= consolidation_threshold:
                    # Calculate slope of consolidation
                    first_close = recent_df['close'].iloc[consolidation_start]
                    last_close = recent_df['close'].iloc[consolidation_end]
                    
                    consolidation_slope = (last_close - first_close) / (consolidation_end - consolidation_start)
                    
                    # Determine if this is a bull or bear flag
                    if price_change > 0:
                        flag_type = 'Bull Flag'
                        expected_signal = 1  # Bullish
                    else:
                        flag_type = 'Bear Flag'
                        expected_signal = -1  # Bearish
                    
                    # Check for breakout
                    breakout = False
                    if consolidation_end < lookback - 1:
                        current_price = recent_df['close'].iloc[-1]
                        
                        if expected_signal > 0 and current_price > consolidation_high:
                            breakout = True
                        elif expected_signal < 0 and current_price < consolidation_low:
                            breakout = True
                    
                    # Calculate pattern size for signal strength
                    pattern_size = abs(price_change) * 100
                    signal_strength = 3
                    if pattern_size > 10:
                        signal_strength = 4
                    
                    # Add to patterns list
                    pattern = {
                        'type': flag_type,
                        'direction': 'bullish' if expected_signal > 0 else 'bearish',
                        'start_idx': recent_df.index[0] + i,
                        'end_idx': recent_df.index[0] + consolidation_end,
                        'signal': expected_signal,
                        'confirmation': breakout,
                        'strength': signal_strength,
                        'sizes': {
                            'pole_change_pct': price_change * 100,
                            'consolidation_range_pct': consolidation_range * 100
                        }
                    }
                    
                    self.patterns.append(pattern)
                    
                    # Add to signals if confirmed
                    if breakout:
                        self.signals.append({
                            'pattern': flag_type,
                            'signal': 'BUY' if expected_signal > 0 else 'SELL',
                            'strength': signal_strength,
                            'start_idx': recent_df.index[0] + i,
                            'end_idx': recent_df.index[0] + consolidation_end
                        })
    
    def detect_cup_and_handle(self):
        """Detect cup and handle patterns"""
        # Get parameters
        try:
            depth_threshold = self.params.get_chart_pattern_param('cup_and_handle')['depth_threshold']
            volume_confirmation = self.params.get_chart_pattern_param('cup_and_handle')['volume_confirmation']
        except:
            depth_threshold = 0.15
            volume_confirmation = True
        
        # Ensure swing points are calculated
        if 'swing_high' not in self.df.columns or 'swing_low' not in self.df.columns:
            self.find_swing_points()
        
        # We need at least 50 bars for cup and handle
        lookback = 100
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Get swing point indices
        swing_high_indices = np.where(recent_df['swing_high'] == 1)[0]
        swing_low_indices = np.where(recent_df['swing_low'] == 1)[0]
        
        # We need at least a few swing points
        if len(swing_high_indices) < 2 or len(swing_low_indices) < 1:
            return
        
        # Iterate through pairs of highs that could form the cup rim
        for i in range(len(swing_high_indices) - 1):
            left_idx = swing_high_indices[i]
            
            # Look for right rim within reasonable distance
            for j in range(i + 1, len(swing_high_indices)):
                right_idx = swing_high_indices[j]
                
                # Cup should be at least 15 bars wide
                if right_idx - left_idx < 15:
                    continue
                
                # Get prices at left and right rim
                left_price = recent_df['high'].iloc[left_idx]
                right_price = recent_df['high'].iloc[right_idx]
                
                # Rims should be at similar price levels
                price_diff = abs(right_price - left_price) / left_price
                if price_diff > 0.05:  # 5% tolerance
                    continue
                
                # Find the lowest point between the rims (the bottom of the cup)
                cup_slice = recent_df.iloc[left_idx:right_idx]
                bottom_idx = cup_slice['low'].idxmin() - recent_df.index[0]
                bottom_price = cup_slice['low'].min()
                
                # Calculate cup depth as percentage
                cup_depth = (left_price - bottom_price) / left_price
                
                # Cup should have significant depth
                if cup_depth < depth_threshold:
                    continue
                
                # Check cup shape - should be rounded
                # Divide the cup into segments and check if it follows a U shape
                cup_width = right_idx - left_idx
                segment_count = 5
                segment_size = cup_width // segment_count
                
                # Skip if cup is too narrow for segments
                if segment_size == 0:
                    continue
                
                # Calculate average prices for segments
                segment_prices = []
                for k in range(segment_count):
                    start = left_idx + k * segment_size
                    end = start + segment_size
                    if end > right_idx:
                        end = right_idx
                    segment_avg = recent_df['low'].iloc[start:end].mean()
                    segment_prices.append(segment_avg)
                
                # Check if middle segments are lower (U shape)
                u_shape = (segment_prices[0] > segment_prices[1] > segment_prices[2] and
                           segment_prices[2] < segment_prices[3] < segment_prices[4])
                
                if not u_shape:
                    continue
                
                # Now look for a handle after the right rim
                handle_start = right_idx
                
                # Handle should be within 15 bars of right rim
                handle_end = min(handle_start + 15, len(recent_df) - 1)
                
                # Find the lowest point in the handle
                handle_bottom = recent_df['low'].iloc[handle_start:handle_end].min()
                
                # Handle should retrace 30-50% of the cup
                handle_retrace = (right_price - handle_bottom) / (right_price - bottom_price)
                
                if handle_retrace < 0.3 or handle_retrace > 0.5:
                    continue
                
                # Check for breakout above the cup rim
                breakout = False
                if handle_end < len(recent_df) - 1:
                    rim_level = (left_price + right_price) / 2
                    breakout_price = recent_df['close'].iloc[-1]
                    
                    if breakout_price > rim_level:
                        breakout = True
                
                # Check volume pattern if required
                volume_pattern_good = True
                if volume_confirmation and 'volume' in recent_df.columns:
                    # Volume should be higher on breakout
                    if breakout:
                        avg_volume = recent_df['volume'].iloc[handle_start:handle_end].mean()
                        breakout_volume = recent_df['volume'].iloc[-5:].mean()
                        
                        if breakout_volume <= avg_volume:
                            volume_pattern_good = False
                
                # If we got here, we found a cup and handle
                signal_strength = 4  # Cup and handle is a strong pattern
                
                # Add to patterns list
                pattern = {
                    'type': 'Cup and Handle',
                    'direction': 'bullish',
                    'start_idx': recent_df.index[0] + left_idx,
                    'end_idx': recent_df.index[0] + handle_end,
                    'signal': 1,
                    'confirmation': breakout and volume_pattern_good,
                    'strength': signal_strength,
                    'sizes': {
                        'cup_depth_pct': cup_depth * 100,
                        'handle_retrace_pct': handle_retrace * 100
                    }
                }
                
                self.patterns.append(pattern)
                
                # Add to signals if confirmed
                if breakout and volume_pattern_good:
                    self.signals.append({
                        'pattern': 'Cup and Handle',
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'start_idx': recent_df.index[0] + left_idx,
                        'end_idx': recent_df.index[0] + handle_end
                    })
    
    def detect_rounding_patterns(self):
        """Detect rounding bottom and rounding top patterns"""
        # Get parameters
        try:
            curve_smoothness = self.params.get_chart_pattern_param('rounding_patterns')['curve_smoothness']
            min_points = self.params.get_chart_pattern_param('rounding_patterns')['min_points']
        except:
            curve_smoothness = 0.7
            min_points = 10
        
        # We need at least 40 bars for rounding patterns
        lookback = 60
        data_len = len(self.df)
        if data_len < lookback:
            lookback = data_len
            
        recent_df = self.df.iloc[-lookback:].copy()
        
        # Function to check how well points fit a curve (simplified)
        def is_curved(prices, increasing=True):
            # Create x coordinates
            x = np.arange(len(prices))
            
            # Fit quadratic curve
            coeffs = np.polyfit(x, prices, 2)
            
            # Check direction of curve (coefficient of x^2 term)
            if (increasing and coeffs[0] > 0) or (not increasing and coeffs[0] < 0):
                # Generate fitted values
                fitted = np.polyval(coeffs, x)
                
                # Calculate R-squared to measure fit
                ss_total = np.sum((prices - np.mean(prices))**2)
                ss_residual = np.sum((prices - fitted)**2)
                
                r_squared = 1 - (ss_residual / ss_total)
                
                return r_squared >= curve_smoothness
            
            return False
        
        # Check for Rounding Bottom (Bullish)
        for window_size in range(min_points, lookback // 2):
            # Slide a window through the data
            for i in range(lookback - window_size):
                # Get subset of data
                subset = recent_df.iloc[i:i+window_size]
                
                # Get the lowest prices
                prices = subset['low'].values
                
                # Check if prices form a curve
                if is_curved(prices, increasing=True):
                    # Check for upward breakout
                    breakout = False
                    if i + window_size < len(recent_df) - 1:
                        pattern_high = prices.max()
                        current_price = recent_df['close'].iloc[-1]
                        
                        if current_price > pattern_high:
                            breakout = True
                    
                    # Calculate pattern size
                    pattern_size = (prices.max() - prices.min()) / prices.min() * 100
                    signal_strength = 3
                    if pattern_size > 5:
                        signal_strength = 4
                    
                    # Add to patterns list
                    pattern = {
                        'type': 'Rounding Bottom',
                        'direction': 'bullish',
                        'start_idx': recent_df.index[0] + i,
                        'end_idx': recent_df.index[0] + i + window_size - 1,
                        'signal': 1,
                        'confirmation': breakout,
                        'strength': signal_strength,
                        'sizes': {
                            'depth_pct': pattern_size
                        }
                    }
                    
                    self.patterns.append(pattern)
                    
                    # Add to signals if confirmed
                    if breakout:
                        self.signals.append({
                            'pattern': 'Rounding Bottom',
                            'signal': 'BUY',
                            'strength': signal_strength,
                            'start_idx': recent_df.index[0] + i,
                            'end_idx': recent_df.index[0] + i + window_size - 1
                        })
        
        # Check for Rounding Top (Bearish)
        for window_size in range(min_points, lookback // 2):
            # Slide a window through the data
            for i in range(lookback - window_size):
                # Get subset of data
                subset = recent_df.iloc[i:i+window_size]
                
                # Get the highest prices
                prices = subset['high'].values
                
                # Check if prices form a curve
                if is_curved(prices, increasing=False):
                    # Check for downward breakout
                    breakout = False
                    if i + window_size < len(recent_df) - 1:
                        pattern_low = subset['low'].min()
                        current_price = recent_df['close'].iloc[-1]
                        
                        if current_price < pattern_low:
                            breakout = True
                    
                    # Calculate pattern size
                    pattern_size = (prices.max() - prices.min()) / prices.min() * 100
                    signal_strength = 3
                    if pattern_size > 5:
                        signal_strength = 4
                    
                    # Add to patterns list
                    pattern = {
                        'type': 'Rounding Top',
                        'direction': 'bearish',
                        'start_idx': recent_df.index[0] + i,
                        'end_idx': recent_df.index[0] + i + window_size - 1,
                        'signal': -1,
                        'confirmation': breakout,
                        'strength': signal_strength,
                        'sizes': {
                            'height_pct': pattern_size
                        }
                    }
                    
                    self.patterns.append(pattern)
                    
                    # Add to signals if confirmed
                    if breakout:
                        self.signals.append({
                            'pattern': 'Rounding Top',
                            'signal': 'SELL',
                            'strength': signal_strength,
                            'start_idx': recent_df.index[0] + i,
                            'end_idx': recent_df.index[0] + i + window_size - 1
                        })
    
    def get_signals(self):
        """Get all trading signals generated by chart patterns"""
        return self.signals
    
    def get_latest_patterns(self):
        """Get all detected patterns from most recent analysis"""
        return self.patterns
    
    def get_pattern_signals(self):
        """Get the overall signal from chart patterns"""
        if not self.signals:
            return {
                'signal': 0,
                'strength': 0,
                'description': 'No chart patterns detected'
            }
        
        # Count buy and sell signals with their strengths
        buy_signals = [s for s in self.signals if s['signal'] == 'BUY']
        sell_signals = [s for s in self.signals if s['signal'] == 'SELL']
        
        buy_strength = sum(s['strength'] for s in buy_signals)
        sell_strength = sum(s['strength'] for s in sell_signals)
        
        # Determine overall signal
        if buy_strength > sell_strength:
            overall_strength = min(5, max(1, round((buy_strength - sell_strength) / 2)))
            return {
                'signal': 1,
                'strength': overall_strength,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'description': f'BUY - {len(buy_signals)} bullish patterns vs {len(sell_signals)} bearish'
            }
        elif sell_strength > buy_strength:
            overall_strength = min(5, max(1, round((sell_strength - buy_strength) / 2)))
            return {
                'signal': -1,
                'strength': overall_strength,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'description': f'SELL - {len(sell_signals)} bearish patterns vs {len(buy_signals)} bullish'
            }
        else:
            return {
                'signal': 0,
                'strength': 0,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'description': 'NEUTRAL - Conflicting chart patterns'
            }
        






class CandlestickPatterns:
    """Complete candlestick pattern detection with precise validation criteria"""
    
    def __init__(self, df, params=None):
        """
        Initialize candlestick pattern detector with OHLCV DataFrame
        
        Args:
            df: DataFrame with OHLCV data (index=timestamp, columns=[open, high, low, close, volume])
            params: Optional TradingParameters instance for pattern parameters
        """
        # Create a copy of the DataFrame to avoid modifying the original
        self.df = df.copy()
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Ensure we have OHLC data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")
        
        # Set parameters from config or use defaults
        self.params = params or TradingParameters()
        
        # Initialize patterns and signals lists
        self.patterns = []
        self.signals = []
        self.logger = logging.getLogger(__name__)
        
        # Pre-calculate candle properties
        self._calculate_candle_dimensions()
        self._calculate_trend_context()
    
    def _calculate_candle_dimensions(self):
        """Calculate key dimensions and properties of each candle"""
        # Create new columns dictionary
        new_cols = {}
        
        # Body size
        new_cols['body_size'] = abs(self.df['close'] - self.df['open'])
        
        # Range (high to low)
        new_cols['range'] = self.df['high'] - self.df['low']
        
        # Body percentage (body size as percentage of range)
        new_cols['body_pct'] = new_cols['body_size'] / new_cols['range']
        new_cols['body_pct'] = new_cols['body_pct'].replace([np.inf, np.nan], 0)
        
        # Upper shadow
        new_cols['upper_shadow'] = np.where(
            self.df['close'] >= self.df['open'],
            self.df['high'] - self.df['close'],  # If bullish
            self.df['high'] - self.df['open']    # If bearish
        )
        
        # Lower shadow
        new_cols['lower_shadow'] = np.where(
            self.df['close'] >= self.df['open'],
            self.df['open'] - self.df['low'],   # If bullish
            self.df['close'] - self.df['low']   # If bearish
        )
        
        # Shadow percentages
        new_cols['upper_shadow_pct'] = new_cols['upper_shadow'] / new_cols['range']
        new_cols['lower_shadow_pct'] = new_cols['lower_shadow'] / new_cols['range']
        
        # Replace NaN and Inf values
        for col in ['upper_shadow_pct', 'lower_shadow_pct']:
            new_cols[col] = new_cols[col].replace([np.inf, np.nan], 0)
        
        # Bullish/Bearish flag
        new_cols['bullish'] = self.df['close'] > self.df['open']
        new_cols['bearish'] = self.df['close'] < self.df['open']
        
        # Relative body size (compared to recent candles)
        avg_body = self.df['body_size'].rolling(window=10).mean().shift(1)
        new_cols['rel_body_size'] = new_cols['body_size'] / avg_body
        new_cols['rel_body_size'] = new_cols['rel_body_size'].replace([np.inf, np.nan], 1)
        
        # Price change percentage
        new_cols['price_change_pct'] = (self.df['close'] - self.df['close'].shift(1)) / self.df['close'].shift(1) * 100
        
        # Add all columns to DataFrame at once
        for col_name, col_data in new_cols.items():
            self.df[col_name] = col_data
    
    def _calculate_trend_context(self):
        """Calculate trend context for pattern interpretation"""
        # Short term trend (10 periods)
        self.df['short_trend'] = np.where(
            self.df['close'] > self.df['close'].shift(10),
            1,  # Uptrend
            np.where(
                self.df['close'] < self.df['close'].shift(10),
                -1,  # Downtrend
                0   # Sideways
            )
        )
        
        # Medium term trend (20 periods)
        self.df['medium_trend'] = np.where(
            self.df['close'] > self.df['close'].shift(20),
            1,  # Uptrend
            np.where(
                self.df['close'] < self.df['close'].shift(20),
                -1,  # Downtrend
                0   # Sideways
            )
        )
    
    def detect_marubozu(self):
        """
        Detect Marubozu candlestick pattern
        
        Marubozu has a very long body with little or no shadows
        """
        # Get parameters
        try:
            shadow_threshold = self.params.get_candlestick_pattern_param('marubozu')['shadow_threshold']
            body_pct = self.params.get_candlestick_pattern_param('marubozu')['body_pct']
        except:
            shadow_threshold = 0.05
            body_pct = 0.95
        
        # Create mask for pattern criteria
        marubozu_mask = (
            (self.df['body_pct'] >= body_pct) &
            (self.df['upper_shadow_pct'] <= shadow_threshold) &
            (self.df['lower_shadow_pct'] <= shadow_threshold) &
            (self.df['rel_body_size'] >= 1.2)  # Body must be larger than average
        )
        
        bullish_marubozu = marubozu_mask & self.df['bullish']
        bearish_marubozu = marubozu_mask & self.df['bearish']
        
        # Save patterns with their indices
        for i in np.where(bullish_marubozu)[0]:
            if i >= len(self.df) - 1:  # Skip the last candle
                continue
                
            signal_strength = 3
            
            # Context adjustment
            if self.df['short_trend'].iloc[i] == -1:  # In downtrend, stronger bullish signal
                signal_strength += 1
            elif self.df['short_trend'].iloc[i] == 1:  # In uptrend, continuation
                signal_strength = 2
            
            self.patterns.append({
                'type': 'Bullish Marubozu',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': 1,
                'strength': signal_strength
            })
            
            self.signals.append({
                'pattern': 'Bullish Marubozu',
                'signal': 'BUY',
                'strength': signal_strength,
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
            })
        
        for i in np.where(bearish_marubozu)[0]:
            if i >= len(self.df) - 1:  # Skip the last candle
                continue
                
            signal_strength = 3
            
            # Context adjustment
            if self.df['short_trend'].iloc[i] == 1:  # In uptrend, stronger bearish signal
                signal_strength += 1
            elif self.df['short_trend'].iloc[i] == -1:  # In downtrend, continuation
                signal_strength = 2
            
            self.patterns.append({
                'type': 'Bearish Marubozu',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': -1,
                'strength': signal_strength
            })
            
            self.signals.append({
                'pattern': 'Bearish Marubozu',
                'signal': 'SELL',
                'strength': signal_strength,
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
            })
    
    def detect_doji(self):
        """
        Detect Doji candlestick pattern
        
        Doji has a very small body with shadows on both sides
        """
        # Get parameters
        try:
            body_threshold = self.params.get_candlestick_pattern_param('doji')['body_threshold']
        except:
            body_threshold = 0.1
        
        # Create mask for pattern criteria
        doji_mask = (
            (self.df['body_pct'] <= body_threshold) &
            (self.df['range'] > 0)  # Ensure there is some range (not a flat candle)
        )
        
        # Long-legged doji has significant shadows on both sides
        long_legged_mask = doji_mask & (
            (self.df['upper_shadow_pct'] >= 0.25) &
            (self.df['lower_shadow_pct'] >= 0.25)
        )
        
        # Dragonfly doji has a long lower shadow and minimal upper shadow
        dragonfly_mask = doji_mask & (
            (self.df['lower_shadow_pct'] >= 0.65) &
            (self.df['upper_shadow_pct'] <= 0.1)
        )
        
        # Gravestone doji has a long upper shadow and minimal lower shadow
        gravestone_mask = doji_mask & (
            (self.df['upper_shadow_pct'] >= 0.65) &
            (self.df['lower_shadow_pct'] <= 0.1)
        )
        
        # Regular doji (not any of the special types)
        regular_doji_mask = doji_mask & ~(long_legged_mask | dragonfly_mask | gravestone_mask)
        
        # Save patterns with their indices
        for mask, doji_type, base_signal in [
            (regular_doji_mask, 'Doji', 0),
            (long_legged_mask, 'Long-Legged Doji', 0),
            (dragonfly_mask, 'Dragonfly Doji', 1),  # Bullish
            (gravestone_mask, 'Gravestone Doji', -1)  # Bearish
        ]:
            for i in np.where(mask)[0]:
                if i >= len(self.df) - 1:  # Skip the last candle
                    continue
                
                # Determine signal based on doji type and context
                signal = base_signal
                signal_strength = 1  # Default strength
                
                # Context adjustment
                if base_signal == 0:  # Regular and Long-Legged Doji
                    if self.df['short_trend'].iloc[i] == 1:  # In uptrend
                        signal = -1  # Potential reversal
                        signal_strength = 2 if doji_type == 'Long-Legged Doji' else 1
                    elif self.df['short_trend'].iloc[i] == -1:  # In downtrend
                        signal = 1  # Potential reversal
                        signal_strength = 2 if doji_type == 'Long-Legged Doji' else 1
                elif base_signal == 1:  # Dragonfly Doji
                    signal_strength = 2
                    if self.df['short_trend'].iloc[i] == -1:  # In downtrend
                        signal_strength = 3  # Stronger reversal signal
                elif base_signal == -1:  # Gravestone Doji
                    signal_strength = 2
                    if self.df['short_trend'].iloc[i] == 1:  # In uptrend
                        signal_strength = 3  # Stronger reversal signal
                
                self.patterns.append({
                    'type': doji_type,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                    'signal': signal,
                    'strength': signal_strength
                })
                
                if signal != 0:
                    self.signals.append({
                        'pattern': doji_type,
                        'signal': 'BUY' if signal > 0 else 'SELL',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
    
    def detect_spinning_tops(self):
        """
        Detect Spinning Top candlestick pattern
        
        Spinning top has a small body with long shadows on both sides
        """
        # Get parameters
        try:
            body_threshold = self.params.get_candlestick_pattern_param('spinning_top')['body_threshold']
            shadow_threshold = self.params.get_candlestick_pattern_param('spinning_top')['shadow_threshold']
        except:
            body_threshold = 0.25
            shadow_threshold = 0.35
        
        # Create mask for pattern criteria
        spinning_top_mask = (
            (self.df['body_pct'] <= body_threshold) &  # Small body
            (self.df['body_pct'] > 0.1) &  # Larger than doji
            (self.df['upper_shadow_pct'] >= shadow_threshold) &  # Long upper shadow
            (self.df['lower_shadow_pct'] >= shadow_threshold) &  # Long lower shadow
            (self.df['range'] > 0)  # Ensure there is some range
        )
        
        # Save patterns with their indices
        for i in np.where(spinning_top_mask)[0]:
            if i >= len(self.df) - 1:  # Skip the last candle
                continue
                
            # Determine signal based on context
            signal = 0  # Default neutral
            signal_strength = 1
            
            # In strong trend, spinning tops suggest indecision
            if abs(self.df['short_trend'].iloc[i]) == 1:
                signal = -self.df['short_trend'].iloc[i]  # Opposite of trend
                signal_strength = 1
            
            self.patterns.append({
                'type': 'Spinning Top',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': signal,
                'strength': signal_strength
            })
            
            if signal != 0:
                self.signals.append({
                    'pattern': 'Spinning Top',
                    'signal': 'BUY' if signal > 0 else 'SELL',
                    'strength': signal_strength,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                })
    
    def detect_paper_umbrella(self):
        """
        Detect Paper Umbrella patterns (Hammer and Hanging Man)
        
        Paper umbrella has a small body at the top with little/no upper shadow
        and a long lower shadow (at least 2x body)
        """
        # Get parameters for hammer and hanging man
        try:
            lower_shadow_ratio = self.params.get_candlestick_pattern_param('hammer')['lower_shadow_ratio']
            upper_shadow_threshold = self.params.get_candlestick_pattern_param('hammer')['upper_shadow_threshold']
        except:
            lower_shadow_ratio = 2.0
            upper_shadow_threshold = 0.1
        
        # Create mask for paper umbrella criteria
        umbrella_mask = (
            (self.df['lower_shadow'] >= lower_shadow_ratio * self.df['body_size']) &  # Long lower shadow
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &  # Little/no upper shadow
            (self.df['body_pct'] <= 0.3) &  # Body is small part of range
            (self.df['range'] > 0)  # Ensure there is some range
        )
        
        # Save umbrella patterns for later categorization as hammer or hanging man
        self.df['is_umbrella'] = umbrella_mask
    
    def detect_hammer(self):
        """
        Detect Hammer candlestick pattern
        
        Hammer is a paper umbrella in a downtrend
        """
        if 'is_umbrella' not in self.df.columns:
            self.detect_paper_umbrella()
        
        # Hammer appears in a downtrend
        hammer_mask = (
            self.df['is_umbrella'] & 
            (self.df['short_trend'] == -1)
        )
        
        # Add color distinction
        bullish_hammer = hammer_mask & self.df['bullish']
        bearish_hammer = hammer_mask & self.df['bearish']
        
        # Save patterns with their indices
        for mask, hammer_type, signal_adj in [
            (bullish_hammer, 'Bullish Hammer', 1),
            (bearish_hammer, 'Hammer', 0.5)  # Bearish hammer is still bullish but less so
        ]:
            for i in np.where(mask)[0]:
                if i >= len(self.df) - 1:  # Skip the last candle
                    continue
                
                signal = 1  # Hammer is bullish
                signal_strength = 3 * signal_adj
                
                # More bullish if the price continues up the next day
                if i < len(self.df) - 2 and self.df['bullish'].iloc[i+1]:
                    signal_strength += 1
                
                self.patterns.append({
                    'type': hammer_type,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                    'signal': signal,
                    'strength': signal_strength
                })
                
                self.signals.append({
                    'pattern': hammer_type,
                    'signal': 'BUY',
                    'strength': signal_strength,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                })
    
    def detect_hanging_man(self):
        """
        Detect Hanging Man candlestick pattern
        
        Hanging Man is a paper umbrella in an uptrend
        """
        if 'is_umbrella' not in self.df.columns:
            self.detect_paper_umbrella()
        
        # Hanging Man appears in an uptrend
        hanging_man_mask = (
            self.df['is_umbrella'] & 
            (self.df['short_trend'] == 1)
        )
        
        # Add color distinction
        bullish_hanging_man = hanging_man_mask & self.df['bullish']
        bearish_hanging_man = hanging_man_mask & self.df['bearish']
        
        # Save patterns with their indices
        for mask, hanging_type, signal_adj in [
            (bearish_hanging_man, 'Bearish Hanging Man', 1),
            (bullish_hanging_man, 'Hanging Man', 0.5)  # Bullish hanging man is still bearish but less so
        ]:
            for i in np.where(mask)[0]:
                if i >= len(self.df) - 1:  # Skip the last candle
                    continue
                
                signal = -1  # Hanging Man is bearish
                signal_strength = 3 * signal_adj
                
                # More bearish if the price continues down the next day
                if i < len(self.df) - 2 and self.df['bearish'].iloc[i+1]:
                    signal_strength += 1
                
                self.patterns.append({
                    'type': hanging_type,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                    'signal': signal,
                    'strength': signal_strength
                })
                
                self.signals.append({
                    'pattern': hanging_type,
                    'signal': 'SELL',
                    'strength': signal_strength,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                })
    
    def detect_shooting_star(self):
        """
        Detect Shooting Star candlestick pattern
        
        Shooting Star has a small body at the bottom with little/no lower shadow
        and a long upper shadow (at least 2x body)
        """
        # Create mask for shooting star criteria
        shooting_star_mask = (
            (self.df['upper_shadow'] >= 2.0 * self.df['body_size']) &  # Long upper shadow
            (self.df['lower_shadow_pct'] <= 0.1) &  # Little/no lower shadow
            (self.df['body_pct'] <= 0.3) &  # Body is small part of range
            (self.df['range'] > 0) &  # Ensure there is some range
            (self.df['short_trend'] == 1)  # Appears in uptrend
        )
        
        # Add color distinction
        bullish_star = shooting_star_mask & self.df['bullish']
        bearish_star = shooting_star_mask & self.df['bearish']
        
        # Save patterns with their indices
        for mask, star_type, signal_adj in [
            (bearish_star, 'Bearish Shooting Star', 1),
            (bullish_star, 'Shooting Star', 0.5)  # Bullish shooting star is still bearish but less so
        ]:
            for i in np.where(mask)[0]:
                if i >= len(self.df) - 1:  # Skip the last candle
                    continue
                
                signal = -1  # Shooting Star is bearish
                signal_strength = 3 * signal_adj
                
                # More bearish if the price continues down the next day
                if i < len(self.df) - 2 and self.df['bearish'].iloc[i+1]:
                    signal_strength += 1
                
                self.patterns.append({
                    'type': star_type,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                    'signal': signal,
                    'strength': signal_strength
                })
                
                self.signals.append({
                    'pattern': star_type,
                    'signal': 'SELL',
                    'strength': signal_strength,
                    'idx': i,
                    'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                })
    
    def detect_engulfing(self):
        """
        Detect Bullish and Bearish Engulfing patterns
        
        Engulfing pattern is a two-candle pattern where the second candle completely
        engulfs the body of the first candle, and they have opposite colors
        """
        # Get parameters
        try:
            body_size_factor = self.params.get_candlestick_pattern_param('engulfing')['body_size_factor']
        except:
            body_size_factor = 1.1
        
        # Bullish Engulfing
        for i in range(1, len(self.df) - 1):
            # Check if current candle is bullish and previous is bearish
            if self.df['bullish'].iloc[i] and self.df['bearish'].iloc[i-1]:
                # Check if current candle body engulfs previous candle body
                if (self.df['open'].iloc[i] <= self.df['close'].iloc[i-1] and
                    self.df['close'].iloc[i] >= self.df['open'].iloc[i-1] and
                    self.df['body_size'].iloc[i] >= body_size_factor * self.df['body_size'].iloc[i-1]):
                    
                    signal = 1  # Bullish Engulfing is bullish
                    signal_strength = 3
                    
                    # More significant in a downtrend
                    if self.df['short_trend'].iloc[i] == -1:
                        signal_strength += 1
                    
                    # More significant if body is much larger
                    if self.df['body_size'].iloc[i] >= 2 * self.df['body_size'].iloc[i-1]:
                        signal_strength += 1
                    
                    self.patterns.append({
                        'type': 'Bullish Engulfing',
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                        'signal': signal,
                        'strength': signal_strength
                    })
                    
                    self.signals.append({
                        'pattern': 'Bullish Engulfing',
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
        
        # Bearish Engulfing
        for i in range(1, len(self.df) - 1):
            # Check if current candle is bearish and previous is bullish
            if self.df['bearish'].iloc[i] and self.df['bullish'].iloc[i-1]:
                # Check if current candle body engulfs previous candle body
                if (self.df['open'].iloc[i] >= self.df['close'].iloc[i-1] and
                    self.df['close'].iloc[i] <= self.df['open'].iloc[i-1] and
                    self.df['body_size'].iloc[i] >= body_size_factor * self.df['body_size'].iloc[i-1]):
                    
                    signal = -1  # Bearish Engulfing is bearish
                    signal_strength = 3
                    
                    # More significant in an uptrend
                    if self.df['short_trend'].iloc[i] == 1:
                        signal_strength += 1
                    
                    # More significant if body is much larger
                    if self.df['body_size'].iloc[i] >= 2 * self.df['body_size'].iloc[i-1]:
                        signal_strength += 1
                    
                    self.patterns.append({
                        'type': 'Bearish Engulfing',
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                        'signal': signal,
                        'strength': signal_strength
                    })
                    
                    self.signals.append({
                        'pattern': 'Bearish Engulfing',
                        'signal': 'SELL',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
    
    def detect_harami(self):
        """
        Detect Bullish and Bearish Harami patterns
        
        Harami pattern is a two-candle pattern where the second candle is completely
        contained within the body of the first candle, and they have opposite colors
        """
        # Get parameters
        try:
            body_size_ratio = self.params.get_candlestick_pattern_param('harami')['body_size_ratio']
        except:
            body_size_ratio = 0.6
        
        # Bullish Harami
        for i in range(1, len(self.df) - 1):
            # Check if current candle is bullish and previous is bearish
            if self.df['bullish'].iloc[i] and self.df['bearish'].iloc[i-1]:
                # Check if current candle body is inside previous candle body
                if (self.df['open'].iloc[i] > self.df['close'].iloc[i-1] and
                    self.df['close'].iloc[i] < self.df['open'].iloc[i-1] and
                    self.df['body_size'].iloc[i] <= body_size_ratio * self.df['body_size'].iloc[i-1]):
                    
                    signal = 1  # Bullish Harami is bullish
                    signal_strength = 2
                    
                    # More significant in a downtrend
                    if self.df['short_trend'].iloc[i] == -1:
                        signal_strength += 1
                    
                    self.patterns.append({
                        'type': 'Bullish Harami',
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                        'signal': signal,
                        'strength': signal_strength
                    })
                    
                    self.signals.append({
                        'pattern': 'Bullish Harami',
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
        
        # Bearish Harami
        for i in range(1, len(self.df) - 1):
            # Check if current candle is bearish and previous is bullish
            if self.df['bearish'].iloc[i] and self.df['bullish'].iloc[i-1]:
                # Check if current candle body is inside previous candle body
                if (self.df['open'].iloc[i] < self.df['close'].iloc[i-1] and
                    self.df['close'].iloc[i] > self.df['open'].iloc[i-1] and
                    self.df['body_size'].iloc[i] <= body_size_ratio * self.df['body_size'].iloc[i-1]):
                    
                    signal = -1  # Bearish Harami is bearish
                    signal_strength = 2
                    
                    # More significant in an uptrend
                    if self.df['short_trend'].iloc[i] == 1:
                        signal_strength += 1
                    
                    self.patterns.append({
                        'type': 'Bearish Harami',
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                        'signal': signal,
                        'strength': signal_strength
                    })
                    
                    self.signals.append({
                        'pattern': 'Bearish Harami',
                        'signal': 'SELL',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
    
    def detect_piercing_pattern(self):
        """
        Detect Piercing Pattern
        
        Piercing Pattern is a two-candle bullish reversal pattern with a long bearish
        candle followed by a bullish candle that closes more than halfway up the first candle
        """
        for i in range(1, len(self.df) - 1):
            # Check if current candle is bullish and previous is bearish
            if self.df['bullish'].iloc[i] and self.df['bearish'].iloc[i-1]:
                # Check if current candle opens below previous low and closes above midpoint
                prev_midpoint = (self.df['open'].iloc[i-1] + self.df['close'].iloc[i-1]) / 2
                
                if (self.df['open'].iloc[i] <= self.df['low'].iloc[i-1] and
                    self.df['close'].iloc[i] > prev_midpoint and
                    self.df['close'].iloc[i] < self.df['open'].iloc[i-1]):
                    
                    signal = 1  # Piercing Pattern is bullish
                    signal_strength = 3
                    
                    # More significant in a downtrend
                    if self.df['short_trend'].iloc[i] == -1:
                        signal_strength += 1
                    
                    # More significant if it closes higher up the previous candle
                    penetration = (self.df['close'].iloc[i] - self.df['close'].iloc[i-1]) / (self.df['open'].iloc[i-1] - self.df['close'].iloc[i-1])
                    if penetration > 0.75:
                        signal_strength += 1
                    
                    self.patterns.append({
                        'type': 'Piercing Pattern',
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                        'signal': signal,
                        'strength': signal_strength
                    })
                    
                    self.signals.append({
                        'pattern': 'Piercing Pattern',
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
    
    def detect_dark_cloud_cover(self):
        """
        Detect Dark Cloud Cover
        
        Dark Cloud Cover is a two-candle bearish reversal pattern with a long bullish
        candle followed by a bearish candle that opens above the high and closes more
        than halfway down the first candle
        """
        for i in range(1, len(self.df) - 1):
            # Check if current candle is bearish and previous is bullish
            if self.df['bearish'].iloc[i] and self.df['bullish'].iloc[i-1]:
                # Check if current candle opens above previous high and closes below midpoint
                prev_midpoint = (self.df['open'].iloc[i-1] + self.df['close'].iloc[i-1]) / 2
                
                if (self.df['open'].iloc[i] >= self.df['high'].iloc[i-1] and
                    self.df['close'].iloc[i] < prev_midpoint and
                    self.df['close'].iloc[i] > self.df['open'].iloc[i-1]):
                    
                    signal = -1  # Dark Cloud Cover is bearish
                    signal_strength = 3
                    
                    # More significant in an uptrend
                    if self.df['short_trend'].iloc[i] == 1:
                        signal_strength += 1
                    
                    # More significant if it closes lower down the previous candle
                    penetration = (self.df['open'].iloc[i] - self.df['close'].iloc[i]) / (self.df['close'].iloc[i-1] - self.df['open'].iloc[i-1])
                    if penetration > 0.75:
                        signal_strength += 1
                    
                    self.patterns.append({
                        'type': 'Dark Cloud Cover',
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                        'signal': signal,
                        'strength': signal_strength
                    })
                    
                    self.signals.append({
                        'pattern': 'Dark Cloud Cover',
                        'signal': 'SELL',
                        'strength': signal_strength,
                        'idx': i,
                        'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
                    })
    
    def detect_morning_star(self):
        """
        Detect Morning Star pattern
        
        Morning Star is a three-candle bullish reversal pattern with a long bearish candle,
        followed by a small body (star), followed by a bullish candle closing well into the first candle
        """
        # Get parameters
        try:
            body_size_threshold = self.params.get_candlestick_pattern_param('morning_star')['body_size_threshold']
            body_size_factor = self.params.get_candlestick_pattern_param('morning_star')['body_size_factor']
        except:
            body_size_threshold = 0.3
            body_size_factor = 0.6
        
        for i in range(2, len(self.df) - 1):
            # First candle: Bearish with significant body
            if not self.df['bearish'].iloc[i-2] or self.df['body_pct'].iloc[i-2] <= body_size_threshold:
                continue
                
            # Second candle: Small body (star)
            if self.df['body_pct'].iloc[i-1] > body_size_threshold:
                continue
                
            # Gap down between first and second candle
            if self.df['high'].iloc[i-1] >= self.df['close'].iloc[i-2]:
                continue
                
            # Third candle: Bullish closing well into the first candle
            if not self.df['bullish'].iloc[i]:
                continue
                
            # Check penetration into first candle
            first_candle_range = self.df['open'].iloc[i-2] - self.df['close'].iloc[i-2]
            penetration = (self.df['close'].iloc[i] - self.df['open'].iloc[i]) / first_candle_range
            
            if penetration < body_size_factor:
                continue
            
            signal = 1  # Morning Star is bullish
            signal_strength = 4
            
            # More significant in a downtrend
            if self.df['short_trend'].iloc[i] == -1:
                signal_strength += 1
            
            self.patterns.append({
                'type': 'Morning Star',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': signal,
                'strength': signal_strength
            })
            
            self.signals.append({
                'pattern': 'Morning Star',
                'signal': 'BUY',
                'strength': signal_strength,
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
            })
    
    def detect_evening_star(self):
        """
        Detect Evening Star pattern
        
        Evening Star is a three-candle bearish reversal pattern with a long bullish candle,
        followed by a small body (star), followed by a bearish candle closing well into the first candle
        """
        # Get parameters
        try:
            body_size_threshold = self.params.get_candlestick_pattern_param('evening_star')['body_size_threshold']
            body_size_factor = self.params.get_candlestick_pattern_param('evening_star')['body_size_factor']
        except:
            body_size_threshold = 0.3
            body_size_factor = 0.6
        
        for i in range(2, len(self.df) - 1):
            # First candle: Bullish with significant body
            if not self.df['bullish'].iloc[i-2] or self.df['body_pct'].iloc[i-2] <= body_size_threshold:
                continue
                
            # Second candle: Small body (star)
            if self.df['body_pct'].iloc[i-1] > body_size_threshold:
                continue
                
            # Gap up between first and second candle
            if self.df['low'].iloc[i-1] <= self.df['close'].iloc[i-2]:
                continue
                
            # Third candle: Bearish closing well into the first candle
            if not self.df['bearish'].iloc[i]:
                continue
                
            # Check penetration into first candle
            first_candle_range = self.df['close'].iloc[i-2] - self.df['open'].iloc[i-2]
            penetration = (self.df['open'].iloc[i] - self.df['close'].iloc[i]) / first_candle_range
            
            if penetration < body_size_factor:
                continue
            
            signal = -1  # Evening Star is bearish
            signal_strength = 4
            
            # More significant in an uptrend
            if self.df['short_trend'].iloc[i] == 1:
                signal_strength += 1
            
            self.patterns.append({
                'type': 'Evening Star',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': signal,
                'strength': signal_strength
            })
            
            self.signals.append({
                'pattern': 'Evening Star',
                'signal': 'SELL',
                'strength': signal_strength,
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
            })
    
    def detect_three_white_soldiers(self):
        """
        Detect Three White Soldiers pattern
        
        Three White Soldiers is a bullish reversal pattern consisting of three consecutive
        long-bodied bullish candles, each opening within the previous candle's body
        and closing higher than the previous candle
        """
        # Get parameters
        try:
            trend_threshold = self.params.get_candlestick_pattern_param('three_white_soldiers')['trend_threshold']
        except:
            trend_threshold = 0.01
        
        for i in range(2, len(self.df) - 1):
            # Check for three consecutive bullish candles
            if not (self.df['bullish'].iloc[i] and self.df['bullish'].iloc[i-1] and self.df['bullish'].iloc[i-2]):
                continue
            
            # Each candle should have a decent body
            if not (self.df['body_pct'].iloc[i] > 0.5 and 
                    self.df['body_pct'].iloc[i-1] > 0.5 and 
                    self.df['body_pct'].iloc[i-2] > 0.5):
                continue
            
            # Each candle should open within the previous candle's body
            if not (self.df['open'].iloc[i-1] > self.df['open'].iloc[i-2] and 
                    self.df['open'].iloc[i-1] < self.df['close'].iloc[i-2] and
                    self.df['open'].iloc[i] > self.df['open'].iloc[i-1] and 
                    self.df['open'].iloc[i] < self.df['close'].iloc[i-1]):
                continue
            
            # Each candle should close higher than the previous
            if not (self.df['close'].iloc[i-1] > self.df['close'].iloc[i-2] and 
                    self.df['close'].iloc[i] > self.df['close'].iloc[i-1]):
                continue
            
            # The trend should be consistently up
            price_change1 = (self.df['close'].iloc[i-1] - self.df['open'].iloc[i-2]) / self.df['open'].iloc[i-2]
            price_change2 = (self.df['close'].iloc[i] - self.df['open'].iloc[i-1]) / self.df['open'].iloc[i-1]
            
            if price_change1 < trend_threshold or price_change2 < trend_threshold:
                continue
            
            signal = 1  # Three White Soldiers is bullish
            signal_strength = 4
            
            # More significant in a downtrend (reversal)
            if self.df['medium_trend'].iloc[i] == -1:
                signal_strength += 1
            
            self.patterns.append({
                'type': 'Three White Soldiers',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': signal,
                'strength': signal_strength
            })
            
            self.signals.append({
                'pattern': 'Three White Soldiers',
                'signal': 'BUY',
                'strength': signal_strength,
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
            })
    
    def detect_three_black_crows(self):
        """
        Detect Three Black Crows pattern
        
        Three Black Crows is a bearish reversal pattern consisting of three consecutive
        long-bodied bearish candles, each opening within the previous candle's body
        and closing lower than the previous candle
        """
        for i in range(2, len(self.df) - 1):
            # Check for three consecutive bearish candles
            if not (self.df['bearish'].iloc[i] and self.df['bearish'].iloc[i-1] and self.df['bearish'].iloc[i-2]):
                continue
            
            # Each candle should have a decent body
            if not (self.df['body_pct'].iloc[i] > 0.5 and 
                    self.df['body_pct'].iloc[i-1] > 0.5 and 
                    self.df['body_pct'].iloc[i-2] > 0.5):
                continue
            
            # Each candle should open within the previous candle's body
            if not (self.df['open'].iloc[i-1] < self.df['open'].iloc[i-2] and 
                    self.df['open'].iloc[i-1] > self.df['close'].iloc[i-2] and
                    self.df['open'].iloc[i] < self.df['open'].iloc[i-1] and 
                    self.df['open'].iloc[i] > self.df['close'].iloc[i-1]):
                continue
            
            # Each candle should close lower than the previous
            if not (self.df['close'].iloc[i-1] < self.df['close'].iloc[i-2] and 
                    self.df['close'].iloc[i] < self.df['close'].iloc[i-1]):
                continue
            
            signal = -1  # Three Black Crows is bearish
            signal_strength = 4
            
            # More significant in an uptrend (reversal)
            if self.df['medium_trend'].iloc[i] == 1:
                signal_strength += 1
            
            self.patterns.append({
                'type': 'Three Black Crows',
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i]),
                'signal': signal,
                'strength': signal_strength
            })
            
            self.signals.append({
                'pattern': 'Three Black Crows',
                'signal': 'SELL',
                'strength': signal_strength,
                'idx': i,
                'date': self.df.index[i] if hasattr(self.df.index, 'date') else str(self.df.index[i])
            })
    
    def detect_all_patterns(self):
        """Run all candlestick pattern detection methods"""
        # Reset patterns and signals
        self.patterns = []
        self.signals = []
        
        # Run all detection methods
        detection_methods = [
            self.detect_marubozu,
            self.detect_doji,
            self.detect_spinning_tops,
            self.detect_hammer,
            self.detect_hanging_man,
            self.detect_shooting_star,
            self.detect_engulfing,
            self.detect_harami,
            self.detect_piercing_pattern,
            self.detect_dark_cloud_cover,
            self.detect_morning_star,
            self.detect_evening_star,
            self.detect_three_white_soldiers,
            self.detect_three_black_crows
        ]
        
        for method in detection_methods:
            try:
                method()
            except Exception as e:
                self.logger.error(f"Error detecting pattern with {method.__name__}: {str(e)}")
        
        return self.patterns
    
    def get_latest_patterns(self):
        """Get patterns detected in the most recent candle"""
        if not self.patterns:
            self.detect_all_patterns()
            
        latest_idx = len(self.df) - 2  # Skip the last candle, it might be incomplete
        
        return [p for p in self.patterns if p['idx'] == latest_idx]
    
    def get_pattern_signals(self):
        """Get the overall signal from candlestick patterns"""
        if not self.signals:
            self.detect_all_patterns()
            
        # Filter signals for the last 3 candles
        latest_idx = len(self.df) - 2  # Skip the last candle, it might be incomplete
        recent_signals = [s for s in self.signals if s['idx'] >= latest_idx - 2]
        
        if not recent_signals:
            return {
                'signal': 0,
                'strength': 0,
                'description': 'No recent candlestick patterns detected'
            }
        
        # Count buy and sell signals with their strengths
        buy_signals = [s for s in recent_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in recent_signals if s['signal'] == 'SELL']
        
        buy_strength = sum(s['strength'] for s in buy_signals)
        sell_strength = sum(s['strength'] for s in sell_signals)
        
        # Determine overall signal
        if buy_strength > sell_strength:
            overall_strength = min(5, max(1, round((buy_strength - sell_strength) / 2)))
            return {
                'signal': 1,
                'strength': overall_strength,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'patterns': [s['pattern'] for s in recent_signals],
                'description': f'BUY - {len(buy_signals)} bullish patterns vs {len(sell_signals)} bearish'
            }
        elif sell_strength > buy_strength:
            overall_strength = min(5, max(1, round((sell_strength - buy_strength) / 2)))
            return {
                'signal': -1,
                'strength': overall_strength,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'patterns': [s['pattern'] for s in recent_signals],
                'description': f'SELL - {len(sell_signals)} bearish patterns vs {len(buy_signals)} bullish'
            }
        else:
            return {
                'signal': 0,
                'strength': 0,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'patterns': [s['pattern'] for s in recent_signals],
                'description': 'NEUTRAL - Conflicting candlestick patterns'
            }