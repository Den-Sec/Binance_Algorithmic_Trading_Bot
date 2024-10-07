# Binance Algorithmic Trading Bot

This is a Python-based algorithmic trading bot that uses the Binance API to execute trades based on the **Moving Average Convergence Divergence (MACD)** indicator.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Set Up the `.env` File](#3-set-up-the-env-file)
- [Understanding Funds and Trading Operations](#understanding-funds-and-trading-operations)
- [Running the Bot](#running-the-bot)
- [Code Overview](#code-overview)
- [Important Notes](#important-notes)
- [Usage Guide](#usage-guide)
  - [Step-by-Step Instructions](#step-by-step-instructions)
- [Additional Tips](#additional-tips)
- [Disclaimer](#disclaimer)

---

## Prerequisites

- **Python 3.7** or higher
- A **Binance account** with API access
- Basic knowledge of **Python** and trading concepts

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
TRADE_QUANTITY=0.001
INTERVAL=1m
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9
```

- **BINANCE_API_KEY**: Your Binance API key.
- **BINANCE_API_SECRET**: Your Binance API secret.

### **Optional Configurations Explained**

- **SYMBOL**: The trading pair symbol (e.g., `BTCUSDT`).

  - **Default**: `BTCUSDT`
  - **Description**: Specifies the cryptocurrency pair to trade.

- **TRADE_QUANTITY**: The amount to trade per transaction.

  - **Default**: `0.001`
  - **Description**: Amount of the base asset to buy or sell in each trade.

- **INTERVAL**: Candlestick interval (e.g., `1m`, `5m`, `1h`).

  - **Default**: `1m`
  - **Description**: Timeframe for market data analysis.

- **MACD_FAST**: Fast EMA period for MACD.

  - **Default**: `12`
  - **Description**: Number of periods for the fast moving average.

- **MACD_SLOW**: Slow EMA period for MACD.

  - **Default**: `26`
  - **Description**: Number of periods for the slow moving average.

- **MACD_SIGNAL**: Signal line EMA period for MACD.

  - **Default**: `9`
  - **Description**: Number of periods for the signal line moving average.

> **Important**: Keep your `.env` file secure and never share it or commit it to version control.

---

## Understanding Funds and Trading Operations

### **Source of Funds**

- The bot uses funds from your **Binance Spot Wallet** associated with your API keys.

### **What Does the Bot Trade?**

- Trades the cryptocurrency pair specified in `SYMBOL` (default is `BTCUSDT`).

### **How Does the Bot Use Funds?**

- **Buying**: Uses the quote asset (e.g., USDT) to buy the base asset (e.g., BTC).

- **Selling**: Sells the base asset back into the quote asset.

- **Trade Quantity**: Determined by `TRADE_QUANTITY`, which specifies how much of the base asset to trade per transaction.

---

## Running the Bot

Run the trading bot using the following command:

```bash
python trading_bot.py
```

The bot will start executing and will log its activities to `trading_bot.log`.

---

## Code Overview

**File: `trading_bot.py`**

- **Imports**:
  - `os`, `asyncio`, `logging`, `datetime`, `pandas`, `numpy`: Standard libraries and data handling.
  - `binance`: Binance API client for asynchronous operations.
  - `dotenv`: Loads environment variables from the `.env` file.

- **Configuration**:
  - Loads environment variables and sets default values if not provided.
  - Configures logging to record events to `trading_bot.log`.

- **Functions**:
  - `get_historical_data`: Fetches historical candlestick data from Binance.
  - `calculate_macd`: Calculates the MACD and Signal Line indicators.
  - `generate_signals`: Generates buy/sell signals based on MACD crossover.
  - `execute_trade`: Executes buy or sell orders on Binance.

- **Main Loop (`main` function)**:
  - Initializes the Binance client using API keys.
  - Continuously fetches data, calculates indicators, generates signals, and executes trades.
  - Handles exceptions and ensures the client connection is properly closed.

---

## Important Notes

- **Testing**: It is highly recommended to test the bot on Binance's testnet before using real funds. To use the testnet, modify the client initialization in `trading_bot.py`:

  ```python
  client = await AsyncClient.create(api_key, api_secret, testnet=True)
  ```

- **Risk Management**: Adjust the `TRADE_QUANTITY` and other parameters according to your risk tolerance.

- **Security**: Ensure your API keys have appropriate permissions and are stored securely.

- **Logging**: Check `trading_bot.log` for detailed logs of the bot's activities.

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

   - Monitor the console output and the `trading_bot.log` file for activity.

6. **Monitoring and Logs**

   - The bot logs detailed information about its operations.
   - Check `trading_bot.log` regularly to monitor performance and troubleshoot issues.

7. **Stopping the Bot**

   - To stop the bot, interrupt the process (e.g., press `Ctrl+C` in the terminal).

---

## Additional Tips

- **Customizing the Strategy**: Feel free to modify the `calculate_macd` and `generate_signals` functions in `trading_bot.py` to implement different trading strategies.

- **Backtesting**: Before deploying the bot with real funds, consider backtesting the strategy using historical data to assess its performance.

- **Paper Trading**: Use Binance's testnet to simulate trades without risking real funds. Set `testnet=True` when creating the client in `trading_bot.py`.

- **Security Best Practices**:
  - Never share your API keys.
  - Do not commit the `.env` file to any public repositories.
  - Regularly rotate your API keys.

- **Enhancements**:
  - Implement additional indicators or strategies.
  - Add risk management features like stop-loss and take-profit orders.
  - Incorporate notifications (e.g., email, SMS) for trade alerts.

---

## Disclaimer

*This trading bot is for educational purposes only. Trading cryptocurrencies involves significant risk and may not be suitable for all investors. You should carefully consider your investment objectives, level of experience, and risk appetite. The author is not responsible for any financial losses incurred while using this bot. Use it at your own risk.*

---

**If you have any further questions or need assistance, feel free to ask!**

---

## Final Notes

By understanding where the bot gets its funds and how it uses them, as well as knowing the purpose of each configuration setting, you'll be better equipped to use the trading bot effectively and safely. Always exercise caution and make sure to test thoroughly before committing real funds.
