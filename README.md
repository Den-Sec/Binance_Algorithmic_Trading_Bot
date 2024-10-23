# Binance Algorithmic Trading Bot

This is a Python-based algorithmic trading bot that uses the Binance API to execute trades based on technical indicators such as **Moving Average Convergence Divergence (MACD)**, **Relative Strength Index (RSI)**, and **Average Directional Index (ADX)**.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Set Up the `.env` File](#3-set-up-the-env-file)
- [Running the Bot](#running-the-bot)
- [Code Overview](#code-overview)
- [Important Notes](#important-notes)
- [Usage Guide](#usage-guide)
  - [Step-by-Step Instructions](#step-by-step-instructions)
- [Performance Metrics](#performance-metrics)
- [Additional Tips](#additional-tips)
- [Disclaimer](#disclaimer)

---

## Prerequisites

- **Python 3.7** or higher
- A **Binance account** with API access
- Basic knowledge of **Python** and trading concepts
- Installed **Docker** if you wish to run the bot inside a container

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/binance-trading-bot.git
cd binance-trading-bot
```

### 2. Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Set Up the `.env` File

Create a `.env` file in the project root directory and add your Binance API credentials and configuration settings:

```ini
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Optional configurations (default values are shown)
SYMBOL=BTCUSDT
INTERVAL=15m
MACD_FAST_TRENDING=12
MACD_SLOW_TRENDING=26
MACD_SIGNAL_TRENDING=9
RSI_PERIOD_TRENDING=14
STOP_LOSS_PERCENTAGE_TRENDING=0.03
TAKE_PROFIT_PERCENTAGE_TRENDING=0.1
TRAILING_STOP_PERCENTAGE_TRENDING=0.03

MACD_FAST_RANGING=12
MACD_SLOW_RANGING=26
MACD_SIGNAL_RANGING=9
RSI_PERIOD_RANGING=14
STOP_LOSS_PERCENTAGE_RANGING=0.02
TAKE_PROFIT_PERCENTAGE_RANGING=0.05
TRAILING_STOP_PERCENTAGE_RANGING=0.02

RISK_PER_TRADE=0.01
```

- **BINANCE_API_KEY**: Your Binance API key.
- **BINANCE_API_SECRET**: Your Binance API secret.
- **SYMBOL**: The trading pair symbol (e.g., `BTCUSDT`).
- **INTERVAL**: Candlestick interval (e.g., `15m`, `1h`).
- **MACD/RSI Parameters**: These control the calculations for technical indicators in trending and ranging markets.
- **RISK_PER_TRADE**: Percentage of the total balance to risk per trade.

> **Important**: Keep your `.env` file secure and never share it or commit it to version control.

---

## Running the Bot

Run the trading bot using the following command:

```bash
python trading_bot.py
```

The bot will start executing and will log its activities to `main_log.json` and `trade_log.json`.

---

## Code Overview

**File: `trading_bot.py`**

- **Imports**:
  - `os`, `asyncio`, `logging`, `datetime`, `pandas`, `numpy`: Standard libraries and data handling.
  - `binance`: Binance API client for asynchronous operations.
  - `dotenv`: Loads environment variables from the `.env` file.
  - `pythonjsonlogger`: Handles JSON formatted logging.

- **Configuration**:
  - Loads environment variables and sets default values if not provided.
  - Configures two types of logging: `main_log.json` for general events and `trade_log.json` for trade-specific events.

- **Functions**:
  - `get_historical_data`: Fetches historical candlestick data from Binance.
  - `calculate_macd`, `calculate_rsi`, `calculate_adx`: Calculate MACD, RSI, and ADX indicators.
  - `generate_signals`: Generates buy/sell signals based on MACD crossover and RSI levels.
  - `execute_trade`: Executes buy or sell orders on Binance.
  - `get_time_until_next_candle`: Calculates the time until the next candle forms.

- **Main Loop (`main` function)**:
  - Initializes the Binance client using API keys.
  - Continuously fetches data, calculates indicators, generates signals, and executes trades based on the current market condition (trending or ranging).
  - Implements stop-loss, take-profit, and trailing stop mechanisms to manage risks.

---

## Important Notes

- **Testing**: It is highly recommended to test the bot on Binance's testnet before using real funds. To use the testnet, modify the client initialization in `trading_bot.py`:

  ```python
  client = await AsyncClient.create(api_key, api_secret, testnet=True)
  ```

- **Risk Management**: Adjust the `RISK_PER_TRADE` and other parameters according to your risk tolerance.

- **Security**: Ensure your API keys have appropriate permissions and are stored securely.

- **Logging**: Check `main_log.json` and `trade_log.json` for detailed logs of the bot's activities.

---

## Usage Guide

### Step-by-Step Instructions

1. **Set Up Binance API Keys**

   - Log in to your Binance account.
   - Navigate to **API Management**.
   - Create a new API key labeled "TradingBot".
   - Enable only the necessary permissions:
     - **Enable Reading**: Yes.
     - **Enable Spot & Margin Trading**: Yes.
     - **Enable Withdrawals**: No.

2. **Install Python and Dependencies**

   - Ensure you have **Python 3.7** or higher installed.
   - Install dependencies using the provided `requirements.txt` file:

     ```bash
     pip install -r requirements.txt
     ```

3. **Configure the `.env` File**

   - Create a file named `.env` in the project directory.
   - Add your Binance API keys and optional configurations as shown above.

4. **Ensure Sufficient Funds**

   - **For Buying**: Have enough of the quote asset (e.g., USDT) to buy the base asset.
   - **For Selling**: Have enough of the base asset (e.g., BTC) to sell.

5. **Run the Trading Bot**

   - Execute the bot by running:

     ```bash
     python trading_bot.py
     ```

   - Monitor the console output and the log files for activity.

6. **Stopping the Bot**

   - To stop the bot, interrupt the process (e.g., press `Ctrl+C` in the terminal).

---

## Performance Metrics

- **Total Trades**: The total number of trades executed.
- **Profitable Trades**: The number of trades that were profitable.
- **Success Rate**: Percentage of profitable trades relative to the total number of trades.

The bot continuously tracks these metrics and logs them to `main_log.json` for review.

---

## Additional Tips

- **Customizing the Strategy**: Feel free to modify the `calculate_macd`, `calculate_rsi`, and `generate_signals` functions to implement different trading strategies.

- **Backtesting**: Before deploying the bot with real funds, consider backtesting the strategy using historical data to assess its performance.

- **Paper Trading**: Use Binance's testnet to simulate trades without risking real funds. Set `testnet=True` when creating the client in `trading_bot.py`.

- **Security Best Practices**:
  - Never share your API keys.
  - Do not commit the `.env` file to any public repositories.
  - Regularly rotate your API keys.

- **Enhancements**:
  - Implement additional indicators or strategies.
  - Add risk management features like more advanced stop-loss and take-profit mechanisms.
  - Incorporate notifications (e.g., email, SMS) for trade alerts.

---

## Disclaimer

*This trading bot is for educational purposes only. Trading cryptocurrencies involves significant risk and may not be suitable for all investors. You should carefully consider your investment objectives, level of experience, and risk appetite. The author is not responsible for any financial losses incurred while using this bot. Use it at your own risk.*

---

**If you have any further questions or need assistance, feel free to ask!**

