import os
import asyncio
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def main():
    # Load Testnet API keys from .env file
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("API keys not found. Please set them in the .env file.")
        return

    # Initialize Binance client for Testnet
    client = await AsyncClient.create(
        api_key,
        api_secret,
        testnet=True,
    )

    try:
        # Fetch account information
        account_info = await client.get_account()

        print("=== Account Balances ===")
        for balance in account_info['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                print(f"{asset}: Free={free}, Locked={locked}")

    except BinanceAPIException as e:
        print(f"Error fetching account information: {e}")
    finally:
        await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())
