import os
import asyncio
from binance import AsyncClient, BinanceAPIException
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Constants (can be set in .env file or use defaults)
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')        # Trading pair
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', '0.001'))  # Amount to trade
INTERVAL = os.getenv('INTERVAL', '1m')         # Candlestick interval
MACD_FAST = int(os.getenv('MACD_FAST', '12'))  # Fast EMA period
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))  # Slow EMA period
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))  # Signal line EMA period

async def get_historical_data(client, symbol, interval, start_str):
    """
    Fetch historical klines from Binance.

    Args:
        client (AsyncClient): Binance client.
        symbol (str): Trading pair symbol.
        interval (str): Candlestick interval.
        start_str (str): Start time in string format.

    Returns:
        pd.DataFrame: DataFrame containing historical data.
    """
    try:
        # Fetch klines
        klines = await client.get_historical_klines(
            symbol,
            interval,
            start_str=start_str
        )

        # Create DataFrame
        data = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])

        # Preprocess data
        data['close'] = data['close'].astype(float)
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
        return data

    except BinanceAPIException as e:
        logging.error(f"Error fetching historical data: {e}")
        return None

def calculate_macd(data):
    """
    Calculate MACD and Signal Line indicators.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' prices.

    Returns:
        pd.DataFrame: DataFrame with MACD and Signal Line columns added.
    """
    # Calculate EMAs
    exp1 = data['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = data['close'].ewm(span=MACD_SLOW, adjust=False).mean()

    # MACD Line
    data['MACD'] = exp1 - exp2

    # Signal Line
    data['Signal_Line'] = data['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()

    return data

def generate_signals(data):
    """
    Generate buy/sell signals based on MACD crossover.

    Args:
        data (pd.DataFrame): DataFrame containing MACD and Signal Line.

    Returns:
        pd.DataFrame: DataFrame with 'Signal' and 'Position' columns added.
    """
    data['Signal'] = 0
    data['Signal'][1:] = np.where(
        data['MACD'][1:] > data['Signal_Line'][1:], 1, 0
    )
    data['Position'] = data['Signal'].diff()
    return data

async def execute_trade(client, side, quantity, symbol):
    """
    Execute a trade on Binance.

    Args:
        client (AsyncClient): Binance client.
        side (str): 'BUY' or 'SELL'.
        quantity (float): Amount to trade.
        symbol (str): Trading pair symbol.
    """
    try:
        order = await client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        logging.info(f"{side} order executed: {order}")
    except BinanceAPIException as e:
        logging.error(f"Error executing {side} order: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

async def main():
    # Load API keys from .env file
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logging.error("API keys not found. Please set them in the .env file.")
        return

    # Initialize Binance client
    client = await AsyncClient.create(api_key, api_secret)

    try:
        while True:
            # Get current time and calculate start time for historical data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=200)  # Adjust as needed
            start_str = start_time.strftime('%d %b %Y %H:%M:%S')

            # Fetch historical data
            data = await get_historical_data(client, SYMBOL, INTERVAL, start_str)

            if data is not None and not data.empty:
                # Calculate MACD and Signal Line
                data = calculate_macd(data)

                # Generate trading signals
                data = generate_signals(data)

                # Get the latest signal
                latest_signal = data.iloc[-1]

                # Log indicators
                logging.info(f"Price: {latest_signal['close']}, "
                             f"MACD: {latest_signal['MACD']}, "
                             f"Signal Line: {latest_signal['Signal_Line']}")

                # Check for buy/sell signals
                if latest_signal['Position'] == 1:
                    # Buy signal
                    logging.info("Buy signal detected.")
                    await execute_trade(client, 'BUY', TRADE_QUANTITY, SYMBOL)
                elif latest_signal['Position'] == -1:
                    # Sell signal
                    logging.info("Sell signal detected.")
                    await execute_trade(client, 'SELL', TRADE_QUANTITY, SYMBOL)
                else:
                    logging.info("No trade signal.")

            else:
                logging.warning("No data fetched.")

            # Wait for the next interval
            await asyncio.sleep(60)  # Sleep for 60 seconds

    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
    finally:
        await client.close_connection()
        logging.info("Bot stopped.")

if __name__ == '__main__':
    asyncio.run(main())
