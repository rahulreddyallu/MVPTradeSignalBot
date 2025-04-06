"""
Backtesting Engine for NIFTY 200 Trading Signal Bot
Provides comprehensive testing capabilities for trading strategies
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from datetime import timedelta
import logging
import json
import sys
import traceback
from tqdm import tqdm
import pathlib
from copy import deepcopy
import itertools  # For groupby functionality in walk-forward testing

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import bot components
from compute import TechnicalAnalysis, UpstoxClient
import config

# Setup logging
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/backtest_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Core backtesting engine to simulate trading strategy performance
    """
    
    def __init__(self, initial_capital=100000, commission=0.0, slippage=0.0):
        """
        Initialize the backtesting engine with parameters
        
        Args:
            initial_capital: Starting capital in INR
            commission: Trading commission as percentage (0.01 = 1%)
            slippage: Average slippage estimate as percentage
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Trading data
        self.positions = {}  # Current open positions
        self.trades = []     # Completed trades
        self.equity_curve = []  # Daily equity values
        self.equity_dates = []  # Dates for equity curve
        
        # Current datetime in simulation
        self.current_date = None
        
        logger.info(f"Backtest engine initialized with ₹{initial_capital:,.2f} capital")
    
    def run_backtest(self, df, symbol, name=None, industry=None):
        """
        Run backtest on a single instrument
        
        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            symbol: Symbol or instrument identifier
            name: Full name of the instrument (optional)
            industry: Industry sector (optional)
            
        Returns:
            Dict with backtest results
        """
        # Reset state for new backtest
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.equity_dates = [df.index[0]]
        
        # Store stock info
        stock_name = name if name else symbol
        stock_industry = industry if industry else "Unknown"
        
        # Ensure DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        df_columns = [col.lower() for col in df.columns]
        
        # Create a copy with lowercase column names
        data = df.copy()
        data.columns = df_columns
        
        # Validate data has required columns
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column {col} in data")
                return None
        
        # We need at least 100 periods of data for proper indicator calculation
        if len(data) < 100:
            logger.error(f"Insufficient data for {symbol} - need at least 100 periods, got {len(data)}")
            return None
            
        logger.info(f"Starting backtest for {stock_name} ({symbol}) with {len(data)} data points")
        
        # Simulate each trading day (skip first 100 days for indicator warmup)
        for i in tqdm(range(100, len(data)), desc=f"Backtesting {symbol}"):
            # Set current date
            self.current_date = data.index[i]
            
            # Get data up to current bar (this is what would be available in real-time)
            current_data = data.iloc[:i+1].copy()
            
            # Get current price
            current_price = current_data['close'].iloc[-1]
            
            # Run technical analysis
            analyzer = TechnicalAnalysis(current_data)
            signal_results = analyzer.generate_signals()
            
            # Check for signal and execute trades
            self._process_signals(signal_results, current_price, symbol, stock_name)
            
            # Update stops for existing positions
            self._manage_positions(current_price, symbol, current_data)
            
            # Record equity at end of this bar
            total_equity = self.current_capital + self._calculate_positions_value({symbol: current_price})
            self.equity_curve.append(total_equity)
            self.equity_dates.append(self.current_date)
        
        # Close any remaining positions at the end of simulation
        self._close_all_positions({symbol: data['close'].iloc[-1]})
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed for {stock_name}")
        logger.info(f"Final equity: ₹{self.equity_curve[-1]:,.2f} | Return: {metrics['total_return']:.2f}%")
        logger.info(f"Total trades: {metrics['total_trades']} | Win rate: {metrics['win_rate']:.2f}%")
        
        # Return results
        return {
            'symbol': symbol,
            'name': stock_name,
            'industry': stock_industry,
            'equity_curve': self.equity_curve,
            'equity_dates': self.equity_dates,
            'trades': self.trades,
            'metrics': metrics
        }
        
    def _process_signals(self, signal_results, current_price, symbol, stock_name):
        """Process trading signals and execute orders"""
        
        signal = signal_results['signal']  # BUY, SELL, or NEUTRAL
        strength = signal_results['strength']  # 1-5
        
        # Only consider signals with minimum strength (from config)
        if strength < config.MINIMUM_SIGNAL_STRENGTH:
            return
        
        # Get ATR value for stop loss calculation
        atr = None
        if 'atr' in signal_results['indicators'] and 'values' in signal_results['indicators']['atr']:
            atr_data = signal_results['indicators']['atr']['values']
            atr = atr_data.get('atr')
        
        # Process BUY signal
        if signal == 'BUY' and symbol not in self.positions:
            # Calculate position size using percentage risk approach (1% risk per trade)
            risk_amount = self.current_capital * 0.01
            
            # Calculate stop loss (ATR-based or percentage based)
            if atr:
                stop_loss = current_price - (atr * 1.5)  # 1.5 ATR below entry
            else:
                stop_loss = current_price * 0.95  # 5% below entry
            
            # Calculate position size based on risk and stop loss
            risk_per_share = current_price - stop_loss
            
            if risk_per_share <= 0:
                logger.warning(f"Invalid risk per share: {risk_per_share}, skipping trade for {symbol}")
                return
                
            shares = int(risk_amount / risk_per_share)
            
            if shares <= 0:
                logger.warning(f"Invalid position size: {shares}, skipping trade for {symbol}")
                return
                
            # Calculate cost with commission
            cost = (shares * current_price) * (1 + self.commission)
            
            # Check if we have enough capital
            if cost > self.current_capital:
                logger.warning(f"Insufficient capital for {symbol} - Need ₹{cost:,.2f}, have ₹{self.current_capital:,.2f}")
                return
            
            # Execute the buy order
            self.positions[symbol] = {
                'entry_price': current_price,
                'entry_date': self.current_date,
                'shares': shares,
                'stop_loss': stop_loss,
                'target': current_price + (risk_per_share * 2),  # 2:1 reward/risk ratio
                'strategy': 'SIGNAL'
            }
            
            # Deduct cost from capital
            self.current_capital -= cost
            
            logger.debug(f"BUY {shares} shares of {symbol} at ₹{current_price:,.2f} | Stop: ₹{stop_loss:,.2f}")
        
        # Process SELL signal for existing position
        elif signal == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate proceeds after commission
            proceeds = (position['shares'] * current_price) * (1 - self.commission)
            
            # Calculate P&L
            entry_value = position['shares'] * position['entry_price']
            exit_value = position['shares'] * current_price
            profit = exit_value - entry_value
            profit_pct = (profit / entry_value) * 100
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'name': stock_name,
                'entry_date': position['entry_date'],
                'exit_date': self.current_date,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'shares': position['shares'],
                'profit': profit,
                'profit_pct': profit_pct,
                'exit_reason': 'SELL Signal',
                'duration': (self.current_date - position['entry_date']).days
            })
            
            # Add proceeds to capital
            self.current_capital += proceeds
            
            # Remove position
            del self.positions[symbol]
            
            logger.debug(f"SELL {symbol} at ₹{current_price:,.2f} | Profit: ₹{profit:,.2f} ({profit_pct:.2f}%)")
    
    def _manage_positions(self, current_price, symbol, data):
        """Manage existing positions (check stops, trailing stops, etc.)"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            # Calculate proceeds after commission
            proceeds = (position['shares'] * current_price) * (1 - self.commission)
            
            # Calculate P&L
            entry_value = position['shares'] * position['entry_price']
            exit_value = position['shares'] * current_price
            profit = exit_value - entry_value
            profit_pct = (profit / entry_value) * 100
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'name': symbol,  # We may not have the name here
                'entry_date': position['entry_date'],
                'exit_date': self.current_date,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'shares': position['shares'],
                'profit': profit,
                'profit_pct': profit_pct,
                'exit_reason': 'Stop Loss',
                'duration': (self.current_date - position['entry_date']).days
            })
            
            # Add proceeds to capital
            self.current_capital += proceeds
            
            # Remove position
            del self.positions[symbol]
            
            logger.debug(f"STOP LOSS {symbol} at ₹{current_price:,.2f} | Loss: ₹{profit:,.2f} ({profit_pct:.2f}%)")
            
        # Check take profit
        elif current_price >= position['target']:
            # Calculate proceeds after commission
            proceeds = (position['shares'] * current_price) * (1 - self.commission)
            
            # Calculate P&L
            entry_value = position['shares'] * position['entry_price']
            exit_value = position['shares'] * current_price
            profit = exit_value - entry_value
            profit_pct = (profit / entry_value) * 100
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'name': symbol,  # We may not have the name here
                'entry_date': position['entry_date'],
                'exit_date': self.current_date,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'shares': position['shares'],
                'profit': profit,
                'profit_pct': profit_pct,
                'exit_reason': 'Take Profit',
                'duration': (self.current_date - position['entry_date']).days
            })
            
            # Add proceeds to capital
            self.current_capital += proceeds
            
            # Remove position
            del self.positions[symbol]
            
            logger.debug(f"TAKE PROFIT {symbol} at ₹{current_price:,.2f} | Profit: ₹{profit:,.2f} ({profit_pct:.2f}%)")
        
        # Update trailing stop if applicable
        # Assuming we trail the stop once we're in profit by 1 ATR
        else:
            entry_price = position['entry_price']
            original_stop = position['stop_loss']
            
            # If we're in profit by at least 1 ATR, update stop to breakeven
            if current_price > entry_price * 1.02:  # 2% profit
                new_stop = max(original_stop, entry_price * 0.995)  # Break-even minus small buffer
                position['stop_loss'] = new_stop
    
    def _close_all_positions(self, last_prices):
        """Close all positions at the end of simulation"""
        for symbol, position in list(self.positions.items()):
            if symbol in last_prices:
                close_price = last_prices[symbol]
                
                # Calculate proceeds after commission
                proceeds = (position['shares'] * close_price) * (1 - self.commission)
                
                # Calculate P&L
                entry_value = position['shares'] * position['entry_price']
                exit_value = position['shares'] * close_price
                profit = exit_value - entry_value
                profit_pct = (profit / entry_value) * 100
                
                # Record trade
                self.trades.append({
                    'symbol': symbol,
                    'name': symbol,  # We may not have the name here
                    'entry_date': position['entry_date'],
                    'exit_date': self.current_date,
                    'entry_price': position['entry_price'],
                    'exit_price': close_price,
                    'shares': position['shares'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': 'End of Simulation',
                    'duration': (self.current_date - position['entry_date']).days
                })
                
                # Add proceeds to capital
                self.current_capital += proceeds
                
                # Log the close
                logger.debug(f"CLOSE {symbol} at ₹{close_price:,.2f} | Profit: ₹{profit:,.2f} ({profit_pct:.2f}%)")
                
                # Remove position
                del self.positions[symbol]
    
    def _calculate_positions_value(self, current_prices):
        """Calculate the current value of all open positions"""
        value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                value += position['shares'] * current_prices[symbol]
        
        return value
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from backtest results"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'avg_trade_pct': 0,
                'avg_trade_duration': 0
            }
        
        # Calculate returns
        total_return = ((self.equity_curve[-1] / self.equity_curve[0]) - 1) * 100
        
        # Calculate annualized return
        start_date = self.equity_dates[0]
        end_date = self.equity_dates[-1]
        trading_days = (end_date - start_date).days
        
        if trading_days > 0:
            annual_factor = 365 / trading_days
            annualized_return = ((1 + total_return/100) ** annual_factor - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate max drawdown
        peak = self.equity_curve[0]
        drawdown = 0
        max_drawdown = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak * 100
            drawdown = dd
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate trade statistics
        if not self.trades:
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'avg_trade_pct': 0,
                'avg_trade_duration': 0,
                'expectancy': 0
            }
        
        # Calculate trade statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Win rate (percentage of profitable trades)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        # Average profit percentages
        avg_win_pct = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        avg_trade_pct = np.mean([t['profit_pct'] for t in self.trades]) if self.trades else 0
        
        # Calculate profit factor (gross profits / gross losses)
        gross_profit = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade duration
        avg_trade_duration = np.mean([t['duration'] for t in self.trades]) if self.trades else 0
        
        # Calculate expectancy: (Win% * Avg Win) - (Loss% * Avg Loss)
        win_rate_decimal = win_count / total_trades if total_trades > 0 else 0
        loss_rate_decimal = 1 - win_rate_decimal
        
        expectancy = (win_rate_decimal * avg_win_pct) - (loss_rate_decimal * abs(avg_loss_pct))
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_trade_pct': avg_trade_pct,
            'avg_trade_duration': avg_trade_duration,
            'expectancy': expectancy
        }
    
    def plot_equity_curve(self, title=None, benchmark_data=None):
        """
        Plot equity curve from backtest results
        
        Args:
            title: Chart title
            benchmark_data: Optional benchmark data as DataFrame with dates and values
            
        Returns:
            Matplotlib figure object
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            logger.warning("Not enough equity data to plot curve")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to pandas Series for easier plotting
        equity_series = pd.Series(self.equity_curve, index=self.equity_dates)
        
        # Plot equity curve
        ax.plot(equity_series.index, equity_series.values, label='Strategy', color='#1f77b4', linewidth=2)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            # Align benchmark to our date range
            benchmark = benchmark_data.loc[
                (benchmark_data.index >= equity_series.index[0]) & 
                (benchmark_data.index <= equity_series.index[-1])
            ]
            
            if not benchmark.empty:
                # Normalize benchmark to same starting value
                benchmark_norm = benchmark['close'] * (equity_series.iloc[0] / benchmark['close'].iloc[0])
                ax.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark', color='#ff7f0e', 
                      linewidth=1.5, linestyle='--', alpha=0.8)
        
        # Format the plot
        ax.set_title(title if title else 'Equity Curve', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity (₹)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        
        # Add annotations with key metrics
        metrics = self._calculate_performance_metrics()
        
        annotation_text = (
            f"Return: {metrics['total_return']:.2f}%\n"
            f"Ann. Return: {metrics['annualized_return']:.2f}%\n"
            f"Max DD: {metrics['max_drawdown']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']:.2f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}"
        )
        
        # Add text box with metrics
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.97, annotation_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        return fig
    
    def plot_monthly_returns(self):
        """
        Plot monthly returns heatmap
        
        Returns:
            Matplotlib figure object
        """
        if not self.equity_curve or len(self.equity_curve) < 30:
            logger.warning("Not enough equity data to plot monthly returns")
            return None
        
        # Convert equity curve to pandas Series
        equity_series = pd.Series(self.equity_curve, index=self.equity_dates)
        
        # Calculate daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Group by year and month
        monthly_returns = pd.DataFrame({
            'year': daily_returns.index.year,
            'month': daily_returns.index.month,
            'return': daily_returns.values
        })
        
        # Calculate cumulative monthly return
        monthly_returns = monthly_returns.groupby(['year', 'month'])['return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Create a pivot table for heatmap
        heatmap_data = monthly_returns.pivot_table(
            index='year', 
            columns='month', 
            values='return'
        )
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, len(heatmap_data) * 0.6 + 2))
        
        # Create heatmap
        sns.heatmap(
            heatmap_data * 100,  # Convert to percentage
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Monthly Return (%)'}
        )
        
        ax.set_title('Monthly Returns (%)', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_drawdown_chart(self):
        """
        Plot drawdown chart
        
        Returns:
            Matplotlib figure object
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            logger.warning("Not enough equity data to plot drawdown chart")
            return None
        
        # Convert equity curve to pandas Series
        equity_series = pd.Series(self.equity_curve, index=self.equity_dates)
        
        # Calculate drawdown series
        rolling_max = equity_series.cummax()
        drawdown_series = (equity_series - rolling_max) / rolling_max * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot drawdown
        ax.fill_between(drawdown_series.index, drawdown_series.values, 0, color='#d62728', alpha=0.3)
        ax.plot(drawdown_series.index, drawdown_series.values, color='#d62728', linewidth=1)
        
        # Format plot
        ax.set_title('Drawdown Chart', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        
        # Add annotation with max drawdown
        max_dd = drawdown_series.min()
        ax.axhline(y=max_dd, color='k', linestyle='--', alpha=0.5)
        ax.text(drawdown_series.index[0], max_dd * 1.1, f'Max DD: {max_dd:.2f}%', fontsize=10)
        
        plt.tight_layout()
        
        return fig


class PortfolioBacktester:
    """Enhanced backtester for portfolio-level analysis"""
    
    def __init__(self, initial_capital=1000000, commission=0.0, slippage=0.0):
        """Initialize portfolio backtester"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Portfolio tracking
        self.positions = {}      # Current open positions by symbol
        self.trades = []         # Historical trades
        self.equity_curve = []   # Daily portfolio value
        self.equity_dates = []   # Dates for equity curve
        self.symbols_data = {}   # Price data by symbol
        self.allocation_history = []  # History of capital allocations
        
        # Current date in simulation
        self.current_date = None
        
        logger.info(f"Portfolio backtester initialized with ₹{initial_capital:,.2f} capital")
    
    def run_portfolio_backtest(self, data_dict, start_date=None, end_date=None, 
                               rebalance_freq='monthly', max_positions=10, allocation_method='equal'):
        """
        Run portfolio backtest across multiple symbols
        
        Args:
            data_dict: Dict mapping symbols to DataFrames with OHLCV data
            start_date: Start date for backtest (will use earliest common date if None)
            end_date: End date for backtest (will use latest common date if None)
            rebalance_freq: Portfolio rebalancing frequency ('daily', 'weekly', 'monthly')
            max_positions: Maximum number of concurrent positions
            allocation_method: How to allocate capital ('equal', 'risk_parity', 'fixed')
            
        Returns:
            Dict with portfolio backtest results
        """
        # Reset state for new backtest
        self.current_capital = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.allocation_history = []
        
        # Store data by symbol
        self.symbols_data = data_dict
        
        # Find common date range if not specified
        if start_date is None or end_date is None:
            common_dates = set.intersection(*[set(df.index) for df in data_dict.values()])
            if not common_dates:
                logger.error("No common dates found across symbols")
                return None
                
            all_common_dates = sorted(common_dates)
            
            if start_date is None:
                start_date = all_common_dates[0]
            if end_date is None:
                end_date = all_common_dates[-1]
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        logger.info(f"Running portfolio backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Total symbols: {len(data_dict)}")
        
        # Create a unified date range using business days
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        self.equity_dates = date_range
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(date_range, rebalance_freq)
        logger.info(f"Rebalancing frequency: {rebalance_freq}, total rebalances: {len(rebalance_dates)}")
        
        # Track current prices across all symbols
        current_prices = {}
        
        # Progress bar
        with tqdm(total=len(date_range), desc="Running Portfolio Backtest") as pbar:
            # Step through each trading day
            for current_date in date_range:
                self.current_date = current_date
                
                # Get current prices for all symbols
                for symbol, df in self.symbols_data.items():
                    # Find closest available price if exact date not available
                    if current_date in df.index:
                        current_prices[symbol] = df.loc[current_date, 'close']
                    else:
                        # Find closest previous date
                        prev_dates = df.index[df.index < current_date]
                        if len(prev_dates) > 0:
                            closest_date = prev_dates[-1]
                            current_prices[symbol] = df.loc[closest_date, 'close']
                
                # Check if it's a rebalance date
                if current_date in rebalance_dates:
                    # Generate signals for all symbols
                    signals = {}
                    
                    for symbol, df in self.symbols_data.items():
                        # Get data up to current date
                        df_subset = df[df.index <= current_date].copy()
                        
                        # Ensure we have enough data for technical analysis
                        if len(df_subset) >= 100:
                            # Prepare data with lowercase column names
                            df_lower = df_subset.copy()
                            df_lower.columns = [col.lower() for col in df_lower.columns]
                            
                            # Run technical analysis
                            analyzer = TechnicalAnalysis(df_lower)
                            signal_results = analyzer.generate_signals()
                            
                            signals[symbol] = signal_results
                            
                    # Rebalance portfolio based on signals
                    self._rebalance_portfolio(signals, current_prices, max_positions, allocation_method)
                
                # Daily position management (stop loss, take profit)
                self._manage_portfolio_positions(current_prices)
                
                # Record daily equity
                portfolio_value = self.cash + self._calculate_positions_value(current_prices)
                self.equity_curve.append(portfolio_value)
                
                # Update progress bar
                pbar.update(1)
        
        # Close any open positions at the end
        self._close_all_positions(current_prices)
        
        # Calculate portfolio performance metrics
        metrics = self._calculate_performance_metrics()
        
        logger.info(f"Portfolio backtest completed")
        logger.info(f"Final equity: ₹{self.equity_curve[-1]:,.2f} | Return: {metrics['total_return']:.2f}%")
        logger.info(f"Total trades: {metrics['total_trades']} | Win rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Return results
        return {
            'equity_curve': self.equity_curve,
            'equity_dates': self.equity_dates,
            'trades': self.trades,
            'metrics': metrics,
            'allocation_history': self.allocation_history
        }
    
    def _get_rebalance_dates(self, date_range, frequency):
        """Determine rebalance dates based on frequency"""
        if frequency == 'daily':
            return date_range
        
        elif frequency == 'weekly':
            # Get dates that are Mondays (weekday=0)
            return [d for d in date_range if d.weekday() == 0]
        
        elif frequency == 'monthly':
            # Get first trading day of each month
            rebalance_dates = []
            current_month = None
            
            for date in date_range:
                if date.month != current_month:
                    rebalance_dates.append(date)
                    current_month = date.month
            
            return rebalance_dates
        
        else:
            logger.warning(f"Unknown rebalance frequency: {frequency}, defaulting to monthly")
            return self._get_rebalance_dates(date_range, 'monthly')
    
    def _rebalance_portfolio(self, signals, current_prices, max_positions, allocation_method):
        """
        Rebalance portfolio based on signals
        
        Args:
            signals: Dict mapping symbols to signal results
            current_prices: Dict mapping symbols to current prices
            max_positions: Maximum number of positions to hold
            allocation_method: How to allocate capital
        """
        # Record rebalance event
        rebalance_record = {
            'date': self.current_date,
            'portfolio_value': self.cash + self._calculate_positions_value(current_prices),
            'cash': self.cash,
            'actions': []
        }
        
        # First, close positions we want to exit based on signals
        for symbol in list(self.positions.keys()):
            if symbol in signals and signals[symbol]['signal'] == 'SELL':
                if symbol in current_prices:
                    self._close_position(symbol, current_prices[symbol], 'Rebalance - Sell Signal')
                    rebalance_record['actions'].append({
                        'symbol': symbol,
                        'action': 'CLOSE',
                        'reason': 'Sell Signal'
                    })
        
        # Next, find new positions to enter
        # Filter to only BUY signals with sufficient strength
        buy_candidates = []
        
        for symbol, signal_result in signals.items():
            if (signal_result['signal'] == 'BUY' and 
                signal_result['strength'] >= config.MINIMUM_SIGNAL_STRENGTH and
                symbol not in self.positions):
                
                buy_candidates.append({
                    'symbol': symbol,
                    'strength': signal_result['strength'],
                    'indicators': signal_result['indicators'],
                    'signal': signal_result
                })
        
        # Sort candidates by signal strength (highest first)
        buy_candidates.sort(key=lambda x: x['strength'], reverse=True)
        
        # Limit to max_positions (considering existing positions)
        available_slots = max_positions - len(self.positions)
        buy_candidates = buy_candidates[:available_slots]
        
        # If we have new positions to enter, calculate allocation
        if buy_candidates:
            # Calculate allocation amount
            if allocation_method == 'equal':
                # Equal position sizing based on available cash and positions
                total_positions = len(self.positions) + len(buy_candidates)
                target_position_value = (self.cash + self._calculate_positions_value(current_prices)) / total_positions
                
                # Calculate amounts to allocate to new positions
                for candidate in buy_candidates:
                    symbol = candidate['symbol']
                    
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        
                        # Calculate maximum affordable shares with this allocation
                        allocation = min(target_position_value, self.cash)
                        shares = int(allocation / current_price)
                        
                        # If we can buy at least 1 share, execute the order
                        if shares > 0:
                            cost = shares * current_price * (1 + self.commission)
                            
                            if cost <= self.cash:
                                # Get ATR if available
                                atr = None
                                if ('atr' in candidate['indicators'] and 
                                    'values' in candidate['indicators']['atr'] and
                                    'atr' in candidate['indicators']['atr']['values']):
                                    atr = candidate['indicators']['atr']['values']['atr']
                                
                                # Calculate stop loss
                                if atr:
                                    stop_loss = current_price - (atr * 1.5)  # 1.5 ATR below entry
                                else:
                                    stop_loss = current_price * 0.95  # 5% below entry
                                
                                # Execute buy order
                                self.positions[symbol] = {
                                    'entry_price': current_price,
                                    'entry_date': self.current_date,
                                    'shares': shares,
                                    'stop_loss': stop_loss,
                                    'target': current_price + (current_price - stop_loss) * 2,  # 2:1 reward/risk
                                    'strategy': 'SIGNAL'
                                }
                                
                                # Deduct cost from cash
                                self.cash -= cost
                                
                                logger.debug(f"BUY {shares} shares of {symbol} at ₹{current_price:.2f} | Stop: ₹{stop_loss:.2f}")
                                
                                rebalance_record['actions'].append({
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares,
                                    'price': current_price,
                                    'cost': cost,
                                    'strength': candidate['strength']
                                })
            
            elif allocation_method == 'risk_parity':
                # Allocate based on risk (inverse volatility)
                # Get volatilities
                volatilities = {}
                for candidate in buy_candidates:
                    symbol = candidate['symbol']
                    if symbol in self.symbols_data:
                        # Get recent data
                        df = self.symbols_data[symbol]
                        df_recent = df[df.index <= self.current_date].tail(60)
                        
                        if len(df_recent) > 30:  # Need enough data
                            # Calculate historical volatility
                            returns = df_recent['close'].pct_change().dropna()
                            volatility = returns.std() * (252 ** 0.5)  # Annualized
                            volatilities[symbol] = volatility
                
                if volatilities:
                    # Calculate inverse volatility weights
                    inv_vol = {s: 1/v for s, v in volatilities.items()}
                    total_inv_vol = sum(inv_vol.values())
                    weights = {s: v/total_inv_vol for s, v in inv_vol.items()}
                    
                    portfolio_value = self.cash + self._calculate_positions_value(current_prices)
                    
                    # Calculate allocation for each buy candidate
                    for candidate in buy_candidates:
                        symbol = candidate['symbol']
                        
                        if symbol in weights and symbol in current_prices:
                            weight = weights[symbol]
                            allocation = portfolio_value * weight
                            current_price = current_prices[symbol]
                            
                            # Calculate shares
                            shares = int(allocation / current_price)
                            
                            if shares > 0:
                                cost = shares * current_price * (1 + self.commission)
                                
                                if cost <= self.cash:
                                    # Same stop loss calculation as above
                                    atr = None
                                    if ('atr' in candidate['indicators'] and 
                                        'values' in candidate['indicators']['atr'] and
                                        'atr' in candidate['indicators']['atr']['values']):
                                        atr = candidate['indicators']['atr']['values']['atr']
                                    
                                    if atr:
                                        stop_loss = current_price - (atr * 1.5)
                                    else:
                                        stop_loss = current_price * 0.95
                                    
                                    # Execute buy order
                                    self.positions[symbol] = {
                                        'entry_price': current_price,
                                        'entry_date': self.current_date,
                                        'shares': shares,
                                        'stop_loss': stop_loss,
                                        'target': current_price + (current_price - stop_loss) * 2,
                                        'strategy': 'SIGNAL'
                                    }
                                    
                                    # Deduct cost from cash
                                    self.cash -= cost
                                    
                                    logger.debug(f"BUY {shares} shares of {symbol} at ₹{current_price:.2f} | Stop: ₹{stop_loss:.2f}")
                                    
                                    rebalance_record['actions'].append({
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'shares': shares,
                                        'price': current_price,
                                        'cost': cost,
                                        'strength': candidate['strength'],
                                        'weight': weight
                                    })
            else:
                # Default to equal allocation if method not recognized
                logger.warning(f"Unknown allocation method: {allocation_method}, defaulting to equal")
                # Re-run with equal allocation
                self._rebalance_portfolio(signals, current_prices, max_positions, 'equal')
        
        # If we closed positions but didn't add new ones, rebalance existing positions
        elif len(rebalance_record['actions']) > 0 and len(self.positions) > 0:
            # For now, we'll just leave existing positions as is
            # A more sophisticated approach could adjust position sizes here
            pass
        
        # Add rebalance record to history
        self.allocation_history.append(rebalance_record)
    
    def _close_position(self, symbol, price, reason):
        """Close a position and record the trade"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Calculate proceeds after commission
        proceeds = position['shares'] * price * (1 - self.commission)
        
        # Calculate P&L
        entry_value = position['shares'] * position['entry_price']
        exit_value = position['shares'] * price
        profit = exit_value - entry_value
        profit_pct = (profit / entry_value) * 100
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'name': symbol,  # We may not always have the name
            'entry_date': position['entry_date'],
            'exit_date': self.current_date,
            'entry_price': position['entry_price'],
            'exit_price': price,
            'shares': position['shares'],
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': reason,
            'duration': (self.current_date - position['entry_date']).days
        })
        
        # Add proceeds to cash
        self.cash += proceeds
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"{reason}: {symbol} at ₹{price:.2f} | P&L: ₹{profit:.2f} ({profit_pct:.2f}%)")
    
    def _manage_portfolio_positions(self, current_prices):
        """Check stops and targets for all positions"""
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                position = self.positions[symbol]
                current_price = current_prices[symbol]
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    self._close_position(symbol, current_price, 'Stop Loss')
                
                # Check take profit
                elif current_price >= position['target']:
                    self._close_position(symbol, current_price, 'Take Profit')
                    
                # Update trailing stops if applicable
                elif current_price > position['entry_price'] * 1.02:  # 2% profit
                    new_stop = max(position['stop_loss'], position['entry_price'] * 0.995)
                    position['stop_loss'] = new_stop
    
    def _close_all_positions(self, current_prices):
        """Close all positions at the end of simulation"""
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                self._close_position(symbol, current_prices[symbol], 'End of Simulation')
    
    def _calculate_positions_value(self, current_prices):
        """Calculate current value of all open positions"""
        value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                value += position['shares'] * current_prices[symbol]
        return value
    
    def _calculate_performance_metrics(self):
        """Calculate portfolio performance metrics"""
        # Similar to BacktestEngine's method, but for portfolio
        # Calculate returns
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'avg_trade_pct': 0,
                'avg_trade_duration': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0
            }
        
        # Convert to pandas Series for calculations
        equity_series = pd.Series(self.equity_curve, index=self.equity_dates)
        
        # Calculate returns
        returns_series = equity_series.pct_change().fillna(0)
        
        # Calculate metrics
        total_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100
        
        # Calculate annualized return
        days_count = (self.equity_dates[-1] - self.equity_dates[0]).days
        if days_count > 0:
            years = days_count / 365
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate max drawdown
        rolling_max = equity_series.cummax()
        drawdown = ((equity_series - rolling_max) / rolling_max) * 100
        max_drawdown = drawdown.min()
        
        # Calculate trade statistics
        if not self.trades:
            # Calculate Sharpe and Sortino using return series
            daily_returns = returns_series.values
            risk_free_rate = 0.04 / 252  # Assuming 4% annual risk-free rate
            
            excess_returns = daily_returns - risk_free_rate
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            downside_returns = excess_returns.copy()
            downside_returns[downside_returns > 0] = 0
            sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(252) if np.std(downside_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': abs(max_drawdown),
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'avg_trade_pct': 0,
                'avg_trade_duration': 0,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
        
        # Calculate trade statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win_pct = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        avg_trade_pct = np.mean([t['profit_pct'] for t in self.trades]) if self.trades else 0
        
        # Calculate profit factor
        gross_profit = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade duration
        avg_trade_duration = np.mean([t['duration'] for t in self.trades]) if self.trades else 0
        
        # Calculate Sharpe and Sortino Ratios
        daily_returns = returns_series.values
        risk_free_rate = 0.04 / 252  # Assuming 4% annual risk-free rate
        
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        downside_returns = excess_returns.copy()
        downside_returns[downside_returns > 0] = 0
        sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(252) if np.std(downside_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_trade_pct': avg_trade_pct,
            'avg_trade_duration': avg_trade_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def plot_equity_curve(self, benchmark_data=None, title=None):
        """Plot portfolio equity curve with optional benchmark comparison"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            logger.warning("Not enough equity data to plot curve")
            return None
        
        # Convert to pandas Series
        equity_series = pd.Series(self.equity_curve, index=self.equity_dates)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot equity curve
        ax.plot(equity_series.index, equity_series.values, label='Portfolio', linewidth=2)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            # Align benchmark to our date range
            benchmark = benchmark_data.loc[
                (benchmark_data.index >= equity_series.index[0]) & 
                (benchmark_data.index <= equity_series.index[-1])
            ]
            
            if not benchmark.empty:
                # Normalize benchmark to same starting value
                benchmark_norm = benchmark['close'] * (equity_series.iloc[0] / benchmark['close'].iloc[0])
                ax.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark', 
                      linewidth=1.5, linestyle='--', alpha=0.8)
        
        # Format plot
        ax.set_title(title if title else 'Portfolio Performance', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format dates
        fig.autofmt_xdate()
        
        # Add metrics annotation
        metrics = self._calculate_performance_metrics()
        
        annotation_text = (
            f"Return: {metrics['total_return']:.2f}%\n"
            f"Ann. Return: {metrics['annualized_return']:.2f}%\n"
            f"Max DD: {metrics['max_drawdown']:.2f}%\n"
            f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
            f"Win Rate: {metrics['win_rate']:.2f}%\n"
            f"Trades: {metrics['total_trades']}"
        )
        
        # Add text box with metrics
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.97, annotation_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        return fig
    
    def plot_allocations(self):
        """Plot portfolio allocations over time"""
        if not self.allocation_history or len(self.allocation_history) < 2:
            logger.warning("Not enough allocation history to plot")
            return None
        
        # Extract dates and data
        dates = [record['date'] for record in self.allocation_history]
        
        # Get all unique symbols
        all_symbols = set()
        for record in self.allocation_history:
            for action in record['actions']:
                if action['action'] == 'BUY':
                    all_symbols.add(action['symbol'])
        
        # Create DataFrame with allocations
        allocation_df = pd.DataFrame(index=dates, columns=list(all_symbols))
        
        # Fill in allocation data
        for i, record in enumerate(self.allocation_history):
            # Clear previous allocations
            if i > 0:
                allocation_df.iloc[i] = allocation_df.iloc[i-1]
                
            # Add new allocations and remove exits
            for action in record['actions']:
                if action['action'] == 'BUY':
                    allocation_df.loc[record['date'], action['symbol']] = action['cost']
                elif action['action'] == 'CLOSE':
                    allocation_df.loc[record['date'], action['symbol']] = 0
        
        # Fill missing values
        allocation_df = allocation_df.fillna(0)
        
        # Convert to percentages of total
        for idx in allocation_df.index:
            total = allocation_df.loc[idx].sum()
            if total > 0:
                allocation_df.loc[idx] = allocation_df.loc[idx] / total * 100
        
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_symbols)))
        
        # Plot stacked area chart
        allocation_df.plot.area(ax=ax, stacked=True, color=colors, alpha=0.7)
        
        # Format
        ax.set_title('Portfolio Allocation Over Time', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Allocation (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Format dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig


class BacktestRunner:
    """Main class for running backtests with the trading strategy"""
    
    def __init__(self):
        """Initialize the backtest runner"""
        self.upstox_client = None
        self.results_dir = "backtest_results"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def initialize_upstox(self):
        """Initialize Upstox client for data access"""
        try:
            # Create the Upstox client instance
            self.upstox_client = UpstoxClient()
            
            # Check if we have a static access token in config
            if hasattr(config, 'UPSTOX_ACCESS_TOKEN'):
                # Different possible attribute/method names for setting the token
                if hasattr(self.upstox_client, 'set_access_token'):
                    self.upstox_client.set_access_token(config.UPSTOX_ACCESS_TOKEN)
                elif hasattr(self.upstox_client, 'set_token'):
                    self.upstox_client.set_token(config.UPSTOX_ACCESS_TOKEN)
                elif hasattr(self.upstox_client, 'access_token'):
                    self.upstox_client.access_token = config.UPSTOX_ACCESS_TOKEN
                elif hasattr(self.upstox_client, 'token'):
                    self.upstox_client.token = config.UPSTOX_ACCESS_TOKEN
                else:
                    # If none of the above work, try this fallback approach
                    # which assumes the client has an auth_token attribute
                    self.upstox_client.auth_token = config.UPSTOX_ACCESS_TOKEN
                    
                logger.info("Upstox client initialized with static access token")
                return True
            else:
                logger.error("UPSTOX_ACCESS_TOKEN not found in config")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize Upstox client: {str(e)}")
            return False
        
    def get_benchmark_data(self, benchmark='NSE_INDEX|Nifty 50', start_date=None, end_date=None):
        """Get benchmark data for comparison"""
        if self.upstox_client is None and not self.initialize_upstox():
            logger.error("Failed to initialize Upstox client")
            return None
        
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching benchmark data for {benchmark} from {start_date} to {end_date}")
        
        try:
            benchmark_data = self.upstox_client.get_historical_data(
                benchmark, 'day', start_date, end_date
            )
            
            if benchmark_data is None or len(benchmark_data) < 10:
                logger.warning(f"Unable to fetch sufficient benchmark data")
                return None
                
            logger.info(f"Successfully fetched {len(benchmark_data)} data points for benchmark")
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {str(e)}")
            return None
    
    def fetch_data_for_symbols(self, symbols, start_date=None, end_date=None, interval='day'):
        """Fetch historical data for multiple symbols"""
        if self.upstox_client is None and not self.initialize_upstox():
            logger.error("Failed to initialize Upstox client")
            return {}
        
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
        data_dict = {}
        
        for symbol in tqdm(symbols, desc="Fetching Data"):
            try:
                df = self.upstox_client.get_historical_data(
                    symbol, interval, start_date, end_date
                )
                
                if df is not None and len(df) >= 100:  # Need at least 100 data points
                    data_dict[symbol] = df
                    logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Successfully fetched data for {len(data_dict)}/{len(symbols)} symbols")
        return data_dict
    
    def run_single_backtest(self, symbol, start_date=None, end_date=None, 
                           initial_capital=100000, commission=0.0015, plot=True):
        """
        Run backtest on a single symbol
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital amount
            commission: Commission rate (e.g., 0.0015 for 0.15%)
            plot: Whether to generate and save plots
            
        Returns:
            Dict with backtest results
        """
        # Initialize data if needed
        if self.upstox_client is None and not self.initialize_upstox():
            logger.error("Failed to initialize Upstox client")
            return None
        
        logger.info(f"Running backtest for {symbol}")
        
        # Get stock information if available
        stock_name = symbol
        industry = "Unknown"
        
        # Check if we have stock info in config
        if "|" in symbol:
            isin = symbol.split("|")[1]
            if hasattr(config, 'STOCK_INFO') and isin in config.STOCK_INFO:
                stock_info = config.STOCK_INFO[isin]
                stock_name = stock_info.get('name', symbol)
                industry = stock_info.get('industry', "Unknown")
        
        # Fetch historical data
        df = self.upstox_client.get_historical_data(
            symbol, 'day', start_date, end_date
        )
        
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for {symbol}, cannot run backtest")
            return None
        
        # Create output directory for this symbol
        symbol_safe = symbol.replace('|', '_').replace(' ', '_')
        output_dir = os.path.join(self.results_dir, symbol_safe)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize and run backtest engine
        engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
        results = engine.run_backtest(df, symbol, name=stock_name, industry=industry)
        
        if results is None:
            logger.error(f"Backtest failed for {symbol}")
            return None
        
        # Save results to JSON
        self._save_results(results, os.path.join(output_dir, 'backtest_results.json'))
        
        if plot:
            try:
                # Get benchmark data if possible
                benchmark_df = self.get_benchmark_data()
                
                # Plot and save equity curve
                fig = engine.plot_equity_curve(f"Equity Curve: {stock_name}", benchmark_df)
                if fig:
                    fig.savefig(os.path.join(output_dir, 'equity_curve.png'))
                    plt.close(fig)
                
                # Plot and save monthly returns
                fig = engine.plot_monthly_returns()
                if fig:
                    fig.savefig(os.path.join(output_dir, 'monthly_returns.png'))
                    plt.close(fig)
                
                # Plot and save drawdown chart
                fig = engine.plot_drawdown_chart()
                if fig:
                    fig.savefig(os.path.join(output_dir, 'drawdown.png'))
                    plt.close(fig)
                    
            except Exception as e:
                logger.error(f"Error generating plots: {str(e)}")
        
        return results
    
    def run_portfolio_backtest(self, symbols=None, start_date=None, end_date=None, 
                              initial_capital=1000000, max_positions=10, rebalance_freq='monthly',
                              allocation_method='equal', commission=0.0015, plot=True):
        """
        Run portfolio backtest on multiple symbols
        
        Args:
            symbols: List of symbols to include in portfolio (defaults to config.STOCK_LIST)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial portfolio capital
            max_positions: Maximum number of simultaneous positions
            rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
            allocation_method: How to allocate capital ('equal', 'risk_parity')
            commission: Commission rate
            plot: Whether to generate and save plots
            
        Returns:
            Dict with portfolio backtest results
        """
        # Use default symbols list if none provided
        if symbols is None:
            if hasattr(config, 'STOCK_LIST'):
                symbols = config.STOCK_LIST[:20]  # Limit to first 20 by default
            else:
                logger.error("No symbols list provided and no default list found in config")
                return None
        
        # Initialize data if needed
        if self.upstox_client is None and not self.initialize_upstox():
            logger.error("Failed to initialize Upstox client")
            return None
        
        logger.info(f"Running portfolio backtest with {len(symbols)} symbols")
        
        # Fetch data for all symbols
        data_dict = self.fetch_data_for_symbols(symbols, start_date, end_date)
        
        if not data_dict:
            logger.error("No data fetched, cannot run portfolio backtest")
            return None
            
        # Create output directory for portfolio results
        output_dir = os.path.join(self.results_dir, 'portfolio')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize and run portfolio backtester
        backtester = PortfolioBacktester(initial_capital=initial_capital, commission=commission)
        results = backtester.run_portfolio_backtest(
            data_dict, 
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            max_positions=max_positions,
            allocation_method=allocation_method
        )
        
        if results is None:
            logger.error("Portfolio backtest failed")
            return None
        
        # Save results to JSON
        self._save_results(results, os.path.join(output_dir, 'portfolio_results.json'))
        
        if plot:
            try:
                # Get benchmark data if possible
                benchmark_df = self.get_benchmark_data()
                
                # Plot and save equity curve
                fig = backtester.plot_equity_curve(benchmark_df, 'Portfolio Performance')
                if fig:
                    fig.savefig(os.path.join(output_dir, 'portfolio_equity_curve.png'))
                    plt.close(fig)
                
                # Plot and save allocations
                fig = backtester.plot_allocations()
                if fig:
                    fig.savefig(os.path.join(output_dir, 'portfolio_allocations.png'))
                    plt.close(fig)
                    
            except Exception as e:
                logger.error(f"Error generating portfolio plots: {str(e)}")
        
        return results
    
    def run_walk_forward_test(self, symbol, in_sample_days=120, out_sample_days=30, 
                             start_date=None, end_date=None, initial_capital=100000, 
                             commission=0.0015, plot=True):
        """
        Run walk-forward test on a symbol
        
        Args:
            symbol: Symbol to test
            in_sample_days: Number of days for in-sample period
            out_sample_days: Number of days for out-of-sample period
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital
            commission: Commission rate
            plot: Whether to generate plots
            
        Returns:
            Dict with walk-forward test results
        """
        # Initialize data if needed
        if self.upstox_client is None and not self.initialize_upstox():
            logger.error("Failed to initialize Upstox client")
            return None
        
        logger.info(f"Running walk-forward test for {symbol}")
        
        # Get stock information if available
        stock_name = symbol
        
        # Check if we have stock info in config
        if "|" in symbol:
            isin = symbol.split("|")[1]
            if hasattr(config, 'STOCK_INFO') and isin in config.STOCK_INFO:
                stock_info = config.STOCK_INFO[isin]
                stock_name = stock_info.get('name', symbol)
        
        # Ensure we have enough data for walk-forward testing
        if start_date is None:
            # Need extra history for walk-forward testing
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365*2)).strftime('%Y-%m-%d')
            
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Fetch historical data
        df = self.upstox_client.get_historical_data(
            symbol, 'day', start_date, end_date
        )
        
        if df is None or len(df) < (in_sample_days + out_sample_days * 2):
            logger.error(f"Insufficient data for {symbol}, cannot run walk-forward test")
            return None
        
        # Create output directory for this symbol
        symbol_safe = symbol.replace('|', '_').replace(' ', '_')
        output_dir = os.path.join(self.results_dir, 'walk_forward', symbol_safe)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create walk-forward periods
        periods = []
        total_days = len(df)
        max_periods = (total_days - in_sample_days) // out_sample_days
        
        if max_periods < 1:
            logger.error(f"Not enough data for walk-forward testing of {symbol}")
            return None
            
        logger.info(f"Creating {max_periods} walk-forward periods")
        
        for i in range(max_periods):
            in_sample_start = i * out_sample_days
            in_sample_end = in_sample_start + in_sample_days
            out_sample_start = in_sample_end
            out_sample_end = min(out_sample_start + out_sample_days, total_days)
            
            if out_sample_end - out_sample_start < out_sample_days // 2:
                # Skip if out-sample period is too short
                continue
                
            periods.append({
                'in_sample': {
                    'start': in_sample_start,
                    'end': in_sample_end,
                    'data': df.iloc[in_sample_start:in_sample_end]
                },
                'out_sample': {
                    'start': out_sample_start,
                    'end': out_sample_end,
                    'data': df.iloc[out_sample_start:out_sample_end]
                }
            })
            
        logger.info(f"Created {len(periods)} valid walk-forward periods")
        
        # Run backtests for each period
        period_results = []
        
        for i, period in enumerate(periods):
            logger.info(f"Testing period {i+1}/{len(periods)}")
            
            # Use in-sample data for parameter optimization (for future enhancement)
            # For now, we'll just use default parameters
            
            # Run backtest on out-of-sample data
            engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
            results = engine.run_backtest(
                period['out_sample']['data'], 
                symbol, 
                name=f"{stock_name} - Period {i+1}"
            )
            
            if results is not None:
                period_results.append({
                    'period': i + 1,
                    'in_sample_start': period['in_sample']['data'].index[0],
                    'in_sample_end': period['in_sample']['data'].index[-1],
                    'out_sample_start': period['out_sample']['data'].index[0],
                    'out_sample_end': period['out_sample']['data'].index[-1],
                    'metrics': results['metrics'],
                    'equity_curve': results['equity_curve']
                })
        
        if not period_results:
            logger.error(f"No valid results from walk-forward test for {symbol}")
            return None
            
        # Calculate aggregated metrics
        wf_metrics = self._calculate_walk_forward_metrics(period_results)
        
        # Combine all results
        walk_forward_results = {
            'symbol': symbol,
            'name': stock_name,
            'in_sample_days': in_sample_days,
            'out_sample_days': out_sample_days,
            'periods': period_results,
            'metrics': wf_metrics
        }
        
        # Save results
        self._save_results(walk_forward_results, os.path.join(output_dir, 'walk_forward_results.json'))
        
        if plot:
            try:
                # Plot period returns
                self._plot_walk_forward_returns(period_results, symbol, output_dir)
                
                # Plot metrics consistency
                for metric in ['total_return', 'win_rate', 'profit_factor']:
                    self._plot_walk_forward_metric(period_results, metric, symbol, output_dir)
                
            except Exception as e:
                logger.error(f"Error generating walk-forward plots: {str(e)}")
                
        return walk_forward_results
    
    def _calculate_walk_forward_metrics(self, period_results):
        """Calculate aggregated metrics from walk-forward periods"""
        if not period_results:
            return {}
            
        # Extract metrics from all periods
        returns = [p['metrics']['total_return'] for p in period_results]
        win_rates = [p['metrics']['win_rate'] for p in period_results]
        profit_factors = [p['metrics'].get('profit_factor', 0) for p in period_results]
        
        # Calculate statistics
        metrics = {
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'min_return': min(returns),
            'max_return': max(returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_profit_factor': np.mean([pf for pf in profit_factors if pf != float('inf')]),
            'profitable_periods': sum(1 for r in returns if r > 0),
            'total_periods': len(returns),
            'robustness': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        }
        
        # Calculate consecutive wins/losses
        profitable = [r > 0 for r in returns]
        max_consecutive_wins = max(sum(1 for _ in group) for key, group in itertools.groupby(profitable) if key) if profitable else 0
        
        unprofitable = [r <= 0 for r in returns]
        max_consecutive_losses = max(sum(1 for _ in group) for key, group in itertools.groupby(unprofitable) if key) if unprofitable else 0
        
        metrics['max_consecutive_wins'] = max_consecutive_wins
        metrics['max_consecutive_losses'] = max_consecutive_losses
        
        return metrics
    
    def _plot_walk_forward_returns(self, period_results, symbol, output_dir):
        """Plot returns for each walk-forward period"""
        periods = [p['period'] for p in period_results]
        returns = [p['metrics']['total_return'] for p in period_results]
        
        plt.figure(figsize=(12, 6))
        
        # Create bar colors based on return values
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        plt.bar(periods, returns, color=colors)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Add average line
        avg_return = np.mean(returns)
        plt.axhline(y=avg_return, color='blue', linestyle='--', label=f'Average: {avg_return:.2f}%')
        
        plt.title(f'Walk-Forward Test Returns: {symbol}')
        plt.xlabel('Period')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'period_returns.png'))
        plt.close()
    
    def _plot_walk_forward_metric(self, period_results, metric, symbol, output_dir):
        """Plot a specific metric across walk-forward periods"""
        periods = [p['period'] for p in period_results]
        values = [p['metrics'][metric] for p in period_results]
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(periods, values, marker='o', linestyle='-', linewidth=2)
        
        # Add average line
        avg_value = np.mean(values)
        plt.axhline(y=avg_value, color='red', linestyle='--', label=f'Average: {avg_value:.2f}')
        
        # Add standard deviation band if more than 1 period
        if len(values) > 1:
            std_dev = np.std(values)
            plt.fill_between(periods, avg_value - std_dev, avg_value + std_dev, 
                           color='red', alpha=0.1, label=f'±1 StdDev: {std_dev:.2f}')
        
        title_map = {
            'total_return': 'Return (%)',
            'win_rate': 'Win Rate (%)',
            'profit_factor': 'Profit Factor'
        }
        
        metric_title = title_map.get(metric, metric.replace('_', ' ').title())
        
        plt.title(f'Walk-Forward {metric_title} Consistency: {symbol}')
        plt.xlabel('Period')
        plt.ylabel(metric_title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'{metric}_consistency.png'))
        plt.close()
    
    def _save_results(self, results, filename):
        """Save results to JSON file, handling non-serializable types"""
        # Create a deep copy to avoid modifying original data
        results_copy = deepcopy(results)
        
        # Helper function to convert non-serializable types
        def convert_for_json(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp) or isinstance(obj, datetime.datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, datetime.date):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            elif isinstance(obj, float) and np.isinf(obj):
                return "Infinity"
            else:
                return obj
            
        # Convert all data
        results_json = convert_for_json(results_copy)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


# Helper functions for command-line interface
def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Signal Bot Backtesting Tool')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Backtest command')
    
    # Single backtest command
    single_parser = subparsers.add_parser('single', help='Backtest a single instrument')
    single_parser.add_argument('--symbol', type=str, required=True, help='Instrument symbol/key')
    single_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    single_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    single_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    single_parser.add_argument('--commission', type=float, default=0.0015, help='Commission rate')
    
    # Portfolio backtest command
    portfolio_parser = subparsers.add_parser('portfolio', help='Run portfolio backtest')
    portfolio_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (default: use first 10 from config)')
    portfolio_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    portfolio_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    portfolio_parser.add_argument('--capital', type=float, default=1000000, help='Initial portfolio capital')
    portfolio_parser.add_argument('--max-positions', type=int, default=5, help='Maximum concurrent positions')
    portfolio_parser.add_argument('--rebalance', type=str, default='monthly', choices=['daily', 'weekly', 'monthly'],
                                help='Portfolio rebalancing frequency')
    portfolio_parser.add_argument('--allocation', type=str, default='equal', choices=['equal', 'risk_parity'],
                                help='Capital allocation method')
    
    # Walk-forward testing command
    wf_parser = subparsers.add_parser('walkforward', help='Run walk-forward test')
    wf_parser.add_argument('--symbol', type=str, required=True, help='Instrument symbol/key')
    wf_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    wf_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    wf_parser.add_argument('--in-sample', type=int, default=120, help='In-sample period length (days)')
    wf_parser.add_argument('--out-sample', type=int, default=30, help='Out-of-sample period length (days)')
    
    # Add no-plots option to all commands
    for p in [single_parser, portfolio_parser, wf_parser]:
        p.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    if not args.command:
        print("Please specify a command. Use --help for options.")
        return
    
    runner = BacktestRunner()
    
    # Initialize Upstox client
    if not runner.initialize_upstox():
        print("Failed to initialize Upstox client. Please check your API credentials.")
        return
    
    if args.command == 'single':
        # Run single backtest
        results = runner.run_single_backtest(
            args.symbol,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            commission=args.commission,
            plot=not args.no_plots
        )
        
        if results:
            metrics = results['metrics']
            print("\n===== Backtest Results =====")
            print(f"Symbol: {results['name']} ({results['symbol']})")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Avg Trade: {metrics['avg_trade_pct']:.2f}%")
            print("===========================")
    
    elif args.command == 'portfolio':
        # Run portfolio backtest
        symbols = None
        if args.symbols:
            symbols = args.symbols.split(',')
        
        results = runner.run_portfolio_backtest(
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            max_positions=args.max_positions,
            rebalance_freq=args.rebalance,
            allocation_method=args.allocation,
            plot=not args.no_plots
        )
        
        if results:
            metrics = results['metrics']
            print("\n===== Portfolio Backtest Results =====")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            print("===================================")
    
    elif args.command == 'walkforward':
        # Run walk-forward test
        results = runner.run_walk_forward_test(
            args.symbol,
            in_sample_days=args.in_sample,
            out_sample_days=args.out_sample,
            start_date=args.start,
            end_date=args.end,
            plot=not args.no_plots
        )
        
        if results:
            metrics = results['metrics']
            print("\n===== Walk-Forward Test Results =====")
            print(f"Symbol: {results['name']} ({results['symbol']})")
            print(f"Total Periods: {metrics['total_periods']}")
            print(f"Profitable Periods: {metrics['profitable_periods']} ({metrics['robustness']*100:.2f}%)")
            print(f"Average Return: {metrics['avg_return']:.2f}%")
            print(f"Return Std Dev: {metrics['return_std']:.2f}%")
            print(f"Min/Max Return: {metrics['min_return']:.2f}% / {metrics['max_return']:.2f}%")
            print(f"Average Win Rate: {metrics['avg_win_rate']:.2f}%")
            print(f"Max Consecutive Profitable: {metrics.get('max_consecutive_wins', 0)}")
            print(f"Max Consecutive Unprofitable: {metrics.get('max_consecutive_losses', 0)}")
            print("====================================")


if __name__ == "__main__":
    main()
