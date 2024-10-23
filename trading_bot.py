import os
import asyncio
import time
from binance import AsyncClient
from binance.enums import *
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger

# Load environment variables from .env file
load_dotenv()

# Configure main logging with JSON formatter
main_log_handler = logging.FileHandler(filename='main_log.json')
main_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
main_log_handler.setFormatter(main_formatter)
logger = logging.getLogger('main_logger')
logger.setLevel(logging.INFO)
logger.addHandler(main_log_handler)

# Configure trade logging
trade_log_handler = logging.FileHandler(filename='trade_log.json')
trade_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
trade_log_handler.setFormatter(trade_formatter)
trade_logger = logging.getLogger('trade_logger')
trade_logger.setLevel(logging.INFO)
trade_logger.addHandler(trade_log_handler)

# Constants (set in .env file or use defaults)
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')        # Trading pair
INTERVAL = os.getenv('INTERVAL', '15m')         # Candlestick interval

# Trending Market Parameters
MACD_FAST_TRENDING = int(os.getenv('MACD_FAST_TRENDING', '12'))
MACD_SLOW_TRENDING = int(os.getenv('MACD_SLOW_TRENDING', '26'))
MACD_SIGNAL_TRENDING = int(os.getenv('MACD_SIGNAL_TRENDING', '9'))
RSI_PERIOD_TRENDING = int(os.getenv('RSI_PERIOD_TRENDING', '14'))
STOP_LOSS_PERCENTAGE_TRENDING = float(os.getenv('STOP_LOSS_PERCENTAGE_TRENDING', '0.03'))
TAKE_PROFIT_PERCENTAGE_TRENDING = float(os.getenv('TAKE_PROFIT_PERCENTAGE_TRENDING', '0.1'))
TRAILING_STOP_PERCENTAGE_TRENDING = float(os.getenv('TRAILING_STOP_PERCENTAGE_TRENDING', '0.03'))

# Ranging Market Parameters
MACD_FAST_RANGING = int(os.getenv('MACD_FAST_RANGING', '12'))
MACD_SLOW_RANGING = int(os.getenv('MACD_SLOW_RANGING', '26'))
MACD_SIGNAL_RANGING = int(os.getenv('MACD_SIGNAL_RANGING', '9'))
RSI_PERIOD_RANGING = int(os.getenv('RSI_PERIOD_RANGING', '14'))
STOP_LOSS_PERCENTAGE_RANGING = float(os.getenv('STOP_LOSS_PERCENTAGE_RANGING', '0.02'))
TAKE_PROFIT_PERCENTAGE_RANGING = float(os.getenv('TAKE_PROFIT_PERCENTAGE_RANGING', '0.05'))
TRAILING_STOP_PERCENTAGE_RANGING = float(os.getenv('TRAILING_STOP_PERCENTAGE_RANGING', '0.02'))

# General Risk Management
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))

async def get_historical_data(client, symbol, interval, start_str, end_str=None):
    """
    Fetch historical klines from Binance.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            klines = await client.get_historical_klines(
                symbol,
                interval,
                start_str=start_str,
                end_str=end_str
            )
            data = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            data['close'] = data['close'].astype(float)
            data['open'] = data['open'].astype(float)
            data['high'] = data['high'].astype(float)
            data['low'] = data['low'].astype(float)
            data['volume'] = data['volume'].astype(float)
            data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')

            expected_candles = calculate_expected_candles(start_str, end_str, interval)
            if len(data) < expected_candles:
                logger.warning(f"Data fetched has missing candles. Attempt {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
                else:
                    logger.error("Max retries reached. Data is insufficient.")
                    return None
            else:
                return data
        except BinanceAPIException as e:
            logger.error(f"Error fetching historical data: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
                continue
            else:
                return None
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data: {e}")
            return None

def calculate_expected_candles(start_str, end_str, interval):
    """
    Calculate the expected number of candles between start and end times.
    """
    interval_minutes = convert_interval_to_minutes(interval)
    start_time = datetime.strptime(start_str, '%d %b %Y %H:%M:%S')
    if end_str:
        end_time = datetime.strptime(end_str, '%d %b %Y %H:%M:%S')
    else:
        end_time = datetime.utcnow()
    total_minutes = (end_time - start_time).total_seconds() / 60
    expected_candles = int(total_minutes / interval_minutes)
    return expected_candles

def convert_interval_to_minutes(interval):
    """
    Convert Binance interval string to minutes.
    """
    amount = int(interval[:-1])
    unit = interval[-1]
    if unit == 'm':
        return amount
    elif unit == 'h':
        return amount * 60
    elif unit == 'd':
        return amount * 60 * 24
    else:
        raise ValueError(f"Unsupported interval: {interval}")

def calculate_macd(data, macd_fast, macd_slow, macd_signal):
    """
    Calculate MACD and Signal Line indicators.
    """
    exp1 = data['close'].ewm(span=macd_fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=macd_slow, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
    return data

def calculate_rsi(data, rsi_period):
    """
    Calculate RSI indicator using Wilder's smoothing.
    """
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data

def calculate_adx(data, n=14):
    """
    Calculate the Average Directional Index (ADX).
    """
    df = data.copy()
    df['TR'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1).shift()
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                         np.maximum(df['low'].shift() - df['low'], 0), 0)
    df['TR'] = df['TR'].fillna(0)
    df['+DM'] = df['+DM'].fillna(0)
    df['-DM'] = df['-DM'].fillna(0)

    tr_n = df['TR'].rolling(window=n).sum()
    plus_dm_n = df['+DM'].rolling(window=n).sum()
    minus_dm_n = df['-DM'].rolling(window=n).sum()

    df['+DI'] = 100 * (plus_dm_n / tr_n)
    df['-DI'] = 100 * (minus_dm_n / tr_n)

    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=n).mean()
    return df

def generate_signals(data):
    """
    Generate buy/sell signals based on MACD crossover confirmed by RSI.
    """
    data['Signal'] = 0.0
    data.loc[1:, 'Signal'] = np.where(
        (data.loc[1:, 'MACD'] > data.loc[1:, 'Signal_Line']) &
        (data.loc[1:, 'MACD'].shift(1) <= data.loc[1:, 'Signal_Line'].shift(1)) &
        (data.loc[1:, 'RSI'] < 70),
        1.0,
        np.where(
            (data.loc[1:, 'MACD'] < data.loc[1:, 'Signal_Line']) &
            (data.loc[1:, 'MACD'].shift(1) >= data.loc[1:, 'Signal_Line'].shift(1)) &
            (data.loc[1:, 'RSI'] > 30),
            -1.0,
            0.0
        )
    )
    data['Position'] = data['Signal'].replace(0, np.nan).ffill().fillna(0)
    return data

async def execute_trade(client, side, quantity, symbol, precision, trade_type):
    """
    Execute a trade on Binance.
    """
    try:
        adjusted_quantity = round(quantity, precision)
        start_time = time.time()
        order = await client.create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=adjusted_quantity
        )
        execution_time = time.time() - start_time
        logger.info(f"{side} order executed in {execution_time:.4f} seconds: {order}")

        fills = order.get('fills', [])
        if fills:
            executed_qty = sum(float(fill['qty']) for fill in fills)
            trade_price = sum(float(fill['price']) * float(fill['qty']) for fill in fills) / executed_qty
            commission = sum(float(fill['commission']) for fill in fills)
            commission_asset = fills[0].get('commissionAsset', '')
        else:
            logger.warning("No fills returned in order response.")
            executed_qty = adjusted_quantity
            trade_price = 0.0
            commission = 0.0
            commission_asset = ''

        # Log trade details to trade log
        trade_logger.info({
            'message': f'{side} Trade Executed',
            'side': side,
            'quantity': executed_qty,
            'price': trade_price,
            'commission': commission,
            'commission_asset': commission_asset,
            'execution_time': execution_time,
            'trade_type': trade_type  # 'entry' or 'exit'
        })

        return {
            'executed_qty': executed_qty,
            'trade_price': trade_price,
            'commission': commission,
            'commission_asset': commission_asset
        }
    except BinanceAPIException as e:
        logger.error(f"Error executing {side} order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

async def get_time_until_next_candle(interval_minutes, client):
    """
    Calculate the time in seconds until the next candle starts.
    """
    server_time = await client.get_server_time()
    server_timestamp = server_time['serverTime']
    now = datetime.utcfromtimestamp(server_timestamp / 1000.0)
    delta = timedelta(minutes=interval_minutes)
    remainder = (now - datetime.min) % delta
    next_candle_time = now + (delta - remainder)
    time_until_next_candle = (next_candle_time - now).total_seconds()
    return time_until_next_candle

async def main():
    # Load API keys from .env file
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logger.error("API keys not found. Please set them in the .env file.")
        return

    # Initialize Binance client
    client = await AsyncClient.create(
        api_key,
        api_secret,
        testnet=True
    )

    # Fetch symbol info
    symbol_info = await client.get_symbol_info(SYMBOL)
    step_size = float(next(filter(lambda f: f['filterType'] == 'LOT_SIZE', symbol_info['filters']))['stepSize'])
    precision = int(round(-np.log10(step_size), 0))
    base_asset = symbol_info['baseAsset']
    quote_asset = symbol_info['quoteAsset']

    in_position = False
    position_quantity = 0.0
    position_entry_price = 0.0
    trailing_stop_loss = None

    total_trades = 0
    profitable_trades = 0

    interval_minutes = convert_interval_to_minutes(INTERVAL)

    try:
        while True:
            loop_start_time = time.time()

            # Calculate required minutes and candles
            min_required_data = max(MACD_SLOW_TRENDING, MACD_SLOW_RANGING, RSI_PERIOD_TRENDING, RSI_PERIOD_RANGING) + 50
            required_candles = min_required_data + 5
            required_minutes = required_candles * interval_minutes

            # Get current time and calculate start time for historical data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=required_minutes)
            start_str = start_time.strftime('%d %b %Y %H:%M:%S')
            end_str = end_time.strftime('%d %b %Y %H:%M:%S')

            # Fetch historical data
            data = await get_historical_data(client, SYMBOL, INTERVAL, start_str, end_str)

            if data is not None and not data.empty:
                data_length = len(data)
                logger.info(f"Fetched {data_length} data points; Expected at least {min_required_data}")

                if len(data) < min_required_data:
                    logger.warning("Not enough data to compute indicators.")
                    await asyncio.sleep(60)
                    continue

                # Calculate ADX to detect market condition
                data = calculate_adx(data)
                latest_adx = data['ADX'].iloc[-1]

                if latest_adx >= 25:
                    market_condition = 'trending'
                else:
                    market_condition = 'ranging'

                logger.info(f"Market condition detected: {market_condition} (ADX: {latest_adx:.2f})")

                # Set parameters based on market condition
                if market_condition == 'trending':
                    macd_fast = MACD_FAST_TRENDING
                    macd_slow = MACD_SLOW_TRENDING
                    macd_signal = MACD_SIGNAL_TRENDING
                    rsi_period = RSI_PERIOD_TRENDING
                    stop_loss_percentage = STOP_LOSS_PERCENTAGE_TRENDING
                    take_profit_percentage = TAKE_PROFIT_PERCENTAGE_TRENDING
                    trailing_stop_percentage = TRAILING_STOP_PERCENTAGE_TRENDING
                else:
                    macd_fast = MACD_FAST_RANGING
                    macd_slow = MACD_SLOW_RANGING
                    macd_signal = MACD_SIGNAL_RANGING
                    rsi_period = RSI_PERIOD_RANGING
                    stop_loss_percentage = STOP_LOSS_PERCENTAGE_RANGING
                    take_profit_percentage = TAKE_PROFIT_PERCENTAGE_RANGING
                    trailing_stop_percentage = TRAILING_STOP_PERCENTAGE_RANGING

                # Calculate MACD and RSI with adaptive parameters
                data = calculate_macd(data, macd_fast, macd_slow, macd_signal)
                data = calculate_rsi(data, rsi_period)
                data = generate_signals(data)

                latest_signal = data.iloc[-1]

                # Get current price
                ticker = await client.get_symbol_ticker(symbol=SYMBOL)
                current_price = float(ticker['price'])

                # Log indicators
                logger.info({
                    'message': 'Indicator Values',
                    'price': latest_signal['close'],
                    'MACD': latest_signal['MACD'],
                    'Signal_Line': latest_signal['Signal_Line'],
                    'RSI': latest_signal['RSI'],
                    'ADX': latest_adx,
                    'current_price': current_price
                })

                # Risk Management: Adjust based on volatility using ATR
                data['ATR'] = data['TR'].rolling(window=14).mean()
                latest_atr = data['ATR'].iloc[-1]
                volatility = latest_atr / current_price

                # Adjust stop-loss and take-profit based on volatility
                adjusted_stop_loss_percentage = stop_loss_percentage * (1 + volatility)
                adjusted_take_profit_percentage = take_profit_percentage * (1 + volatility)
                adjusted_trailing_stop_percentage = trailing_stop_percentage * (1 + volatility)

                if in_position:
                    price_change = (current_price - position_entry_price) / position_entry_price

                    # Update trailing stop-loss
                    if not trailing_stop_loss:
                        trailing_stop_loss = position_entry_price * (1 - adjusted_trailing_stop_percentage)
                    else:
                        new_trailing_stop_loss = current_price * (1 - adjusted_trailing_stop_percentage)
                        if new_trailing_stop_loss > trailing_stop_loss:
                            trailing_stop_loss = new_trailing_stop_loss

                    # Check if trailing stop-loss is hit
                    if current_price <= trailing_stop_loss:
                        logger.info("Trailing stop-loss triggered.")
                        quantity = position_quantity
                        sell_order = await execute_trade(client, SIDE_SELL, quantity, SYMBOL, precision, 'exit')
                        if sell_order:
                            in_position = False
                            position_quantity = 0.0
                            position_entry_price = 0.0
                            trailing_stop_loss = None
                            # Check if trade was profitable
                            if current_price > position_entry_price:
                                profitable_trades += 1
                            total_trades += 1
                        else:
                            logger.error("Failed to execute trailing stop-loss sell order.")
                        sleep_duration = await get_time_until_next_candle(interval_minutes, client)
                        logger.info(f"Sleeping for {sleep_duration} seconds until next candle.")
                        await asyncio.sleep(sleep_duration)
                        continue

                    # Check for take-profit condition
                    if price_change >= adjusted_take_profit_percentage:
                        logger.info("Take-profit target reached.")
                        quantity = position_quantity
                        sell_order = await execute_trade(client, SIDE_SELL, quantity, SYMBOL, precision, 'exit')
                        if sell_order:
                            in_position = False
                            position_quantity = 0.0
                            position_entry_price = 0.0
                            trailing_stop_loss = None
                            profitable_trades += 1
                            total_trades += 1
                        else:
                            logger.error("Failed to execute take-profit sell order.")
                        sleep_duration = await get_time_until_next_candle(interval_minutes, client)
                        logger.info(f"Sleeping for {sleep_duration} seconds until next candle.")
                        await asyncio.sleep(sleep_duration)
                        continue

                    # Check for stop-loss condition
                    if price_change <= -adjusted_stop_loss_percentage:
                        logger.info("Stop-loss triggered.")
                        quantity = position_quantity
                        sell_order = await execute_trade(client, SIDE_SELL, quantity, SYMBOL, precision, 'exit')
                        if sell_order:
                            in_position = False
                            position_quantity = 0.0
                            position_entry_price = 0.0
                            trailing_stop_loss = None
                            total_trades += 1
                        else:
                            logger.error("Failed to execute stop-loss sell order.")
                        sleep_duration = await get_time_until_next_candle(interval_minutes, client)
                        logger.info(f"Sleeping for {sleep_duration} seconds until next candle.")
                        await asyncio.sleep(sleep_duration)
                        continue

                # Check for buy/sell signals
                if latest_signal['Signal'] == 1.0 and not in_position:
                    logger.info("Buy signal detected.")
                    balance = await client.get_asset_balance(asset=quote_asset)
                    balance_free = float(balance['free'])
                    risk_amount = balance_free * RISK_PER_TRADE
                    stop_loss_price = current_price * (1 - adjusted_stop_loss_percentage)
                    risk_per_unit = current_price - stop_loss_price
                    if risk_per_unit <= 0:
                        logger.error("Risk per unit is zero or negative, cannot calculate position size.")
                        sleep_duration = await get_time_until_next_candle(interval_minutes, client)
                        logger.info(f"Sleeping for {sleep_duration} seconds until next candle.")
                        await asyncio.sleep(sleep_duration)
                        continue
                    position_size = risk_amount / risk_per_unit
                    quantity = round(position_size, precision)
                    max_quantity = balance_free / current_price
                    if quantity > max_quantity:
                        quantity = max_quantity
                    order_details = await execute_trade(client, SIDE_BUY, quantity, SYMBOL, precision, 'entry')
                    if order_details:
                        in_position = True
                        position_quantity = float(order_details['executed_qty'])
                        position_entry_price = float(order_details['trade_price'])
                        trailing_stop_loss = None
                    else:
                        logger.error("Failed to execute buy order.")
                elif latest_signal['Signal'] == -1.0 and in_position:
                    logger.info("Sell signal detected.")
                    quantity = position_quantity
                    order_details = await execute_trade(client, SIDE_SELL, quantity, SYMBOL, precision, 'exit')
                    if order_details:
                        in_position = False
                        position_quantity = 0.0
                        position_entry_price = 0.0
                        trailing_stop_loss = None
                        if current_price > position_entry_price:
                            profitable_trades += 1
                        total_trades += 1
                    else:
                        logger.error("Failed to execute sell order.")
                else:
                    logger.info("No trade signal.")

                # Performance Tracking
                if total_trades > 0:
                    success_rate = (profitable_trades / total_trades) * 100
                    logger.info({
                        'message': 'Performance Metrics',
                        'total_trades': total_trades,
                        'profitable_trades': profitable_trades,
                        'success_rate': f"{success_rate:.2f}%"
                    })
            else:
                logger.warning("No data fetched or data incomplete.")

            sleep_duration = await get_time_until_next_candle(interval_minutes, client)
            loop_execution_time = time.time() - loop_start_time
            logger.info({
                'message': 'Loop Execution Time',
                'execution_time': loop_execution_time,
                'sleep_duration': sleep_duration
            })
            logger.info(f"Sleeping for {sleep_duration} seconds until next candle.")
            await asyncio.sleep(sleep_duration)

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        await client.close_connection()
        logger.info("Bot stopped.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Critical error: {e}")
