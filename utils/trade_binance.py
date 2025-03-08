from datetime import datetime, timedelta
import pandas as pd
import logging
import ccxt
import requests
import argparse

logger = logging.getLogger(__name__)


class BinanceFuturesTrader:
    def __init__(self, api_key, api_secret, sandbox=True):
        self.api_url = "http://10.0.137.230:6378/enqueue"
        
        #print('Initializing exchange')
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
        })
        self.exchange.set_sandbox_mode(sandbox)
        self.exchange.load_markets()
        self.sl_order = {}
        self.tp_order = {}

        # Load spot market
        self.spot_exchange = ccxt.binance({
            'options': {'defaultType': 'spot'},
        })
        self.spot_exchange.load_markets()
    

    def get_dataframe(self, symbol: str, n_candles: int) -> pd.DataFrame:
        """
        Fetch spot dataframe with last n_candles for given symbol
        """
        ohlcv = self.spot_exchange.fetch_ohlcv(symbol, '1h', limit=n_candles)
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(ohlcv, columns=cols)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['date'] = df['date'].dt.tz_localize('UTC')

        return df


    def call_api(self, symbol, n_candles) -> list:
        """
        Call api to compute predictions for symbol
        """
        # Get spot data
        dataframe = self.get_dataframe(symbol, n_candles)
        
        # Prepare JSON 
        data_json = {
            "pair": symbol.replace('/USDT', ''),
            "data": {
                "open": dataframe['open'].tolist(),
                "high": dataframe['high'].tolist(),
                "low": dataframe['low'].tolist(),
                "close": dataframe['close'].tolist(),
                "volume": dataframe['volume'].tolist(),
                "date": dataframe['date'].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x).tolist() 
                }
        }

        # solicitud POST a la API
        try:
            response = requests.post(self.api_url, json=data_json)
            response.raise_for_status()  # Comprobar si hay errores en la solicitud
            predictions = response.json().get('prediction', [])

            logger.info('REQUEST PREDICTION')
            logger.info(str(datetime.now()))
            logger.info(str(predictions))

            return predictions
        
        except requests.exceptions.RequestException as e:
            print(f"Error al llamar a la API: {e}")
            return []
    

    def set_leverage(self, symbol, leverage):
        """
        Set leverage parameter for a given symbol
        """
        try:
            symbol_binance = symbol.replace('/', '')
           
            margin_mode_response = self.exchange.set_margin_mode(
                marginMode="ISOLATED",
                symbol=symbol_binance,
                params={'leverage': leverage}  
            )
            response = self.exchange.fapiPrivatePostLeverage({
                'symbol': symbol_binance,
                'leverage': leverage
            })

            #print(f"Leverage set to {leverage}x for {symbol}")

        except ccxt.BaseError as e:
            print(f"Failed to set leverage for {symbol}: {e}")
    

    def place_orders(self, params):
        """
        Places market, stop-loss, and take-profit orders based on provided parameters.

        Args:
            params (dict): A dictionary containing:
                - 'symbol' (str): The trading pair (e.g., "BNB/USDT").
                - 'amount_in_usdt' (float): Amount to trade in USDT.
                - 'leverage' (int): Leverage to apply.
                - 'sl_pct' (float): Stop-loss percentage multiplier.
                - 'tp_pct' (float): Take-profit percentage multiplier.
        """
        try:
            symbol = params['symbol']
            amount_in_usdt = params['amount_in_usdt']
            leverage = params['leverage']
            sl_pct = params['sl_pct']
            tp_pct = params['tp_pct']
            
            self.set_leverage(symbol, leverage)

            # Fetch current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate the amount to buy
            amount_in_symbol = amount_in_usdt / current_price
            #print(f"Buying {amount_in_symbol:.8f} {symbol.split('/')[0]} (~{amount_in_usdt} USDT)")

            # Set SL and TP prices
            sl_price = current_price * sl_pct
            tp_price = current_price * tp_pct

            # Place market buy order
            market_order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side='buy',
                amount=amount_in_symbol,
                params={'leverage': leverage}
            )
            print("Market buy order placed")

            # Place stop-loss order
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side='sell',
                amount=amount_in_symbol,
                price=sl_price,
                params={'leverage': leverage,
                        'stopPrice': sl_price,  # stopPrice triggers the market order
                        'reduceOnly': True,     # Ensures this order only closes the position
                        'timeInForce': 'GTE_GTC'  # Keeps the order valid until triggered or canceled
                        }
            )
            print("Stop-loss order placed")

            # Place take-profit order
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side='sell',
                amount=amount_in_symbol,
                price=tp_price,
                params={'leverage': leverage,
                        'stopPrice': tp_price,  # takeProfit triggers the market order
                        'reduceOnly': True,     # Ensures this order only closes the position
                        'timeInForce': 'GTE_GTC'  # Keeps the order valid until triggered or canceled
                        }
            )
            print("Take-profit order placed")

        except ccxt.BaseError as e:
            print(f"An error occurred while placing orders: {e}")

    def update_orders(self, symbol):
        """
        Closes stop-loss and take-profit that were left open for a symbol.
        """
        open_orders = self.exchange.fetch_open_orders(symbol)
        
        self.sl_order[symbol] = None
        self.tp_order[symbol] = None
        
        # Assign orders if found
        for order in open_orders:
            if order['info']['origType'] == 'TAKE_PROFIT_MARKET':
                self.sl_order[symbol] = order
            elif order['info']['origType'] == 'STOP_MARKET':
                self.tp_order[symbol] = order
        
        sl_order = self.sl_order[symbol]
        tp_order = self.tp_order[symbol]
        
        # Case 1: Both SL and TP are open
        if sl_order and tp_order:
            return False  # No need to trade
        
        # Case 2: No open orders
        if not sl_order and not tp_order:
            return True  # Ready to trade
        
        # Case 3: One order is left open, cancel the stray order
        try:
            if not sl_order:
                self.exchange.cancel_order(tp_order['id'], symbol)
            elif not tp_order:
                self.exchange.cancel_order(sl_order['id'], symbol)
            return True  # After cancellation, ready to trade
        except ccxt.BaseError as e:
            print(f"Error while closing orders for {symbol}: {e}")
            return False  # Do not trade if error occurs
        
    def check_cooldown(self, symbol, n_hours):
        """
        Checks if stop-loss order was triggered in the last n_hours for the symbol, activating cooldown
        """
        try:
            closed_orders = self.exchange.fetch_closed_orders(symbol)
            cutoff_time = datetime.now() - timedelta(hours=n_hours)

            # Filter closed orders within last n_hours
            recent_orders = [
                order for order in closed_orders 
                if datetime.fromtimestamp(order['lastTradeTimestamp'] / 1000) >= cutoff_time
            ]

            # Sort by timestamp in descending order
            recent_orders.sort(key=lambda x: x['lastTradeTimestamp'], reverse=True) 
            return len(recent_orders) > 0 and recent_orders[0]['info']['origType'] == 'STOP_MARKET'
        
        except Exception as e:
            print(f"Error in check_cooldown for {symbol}: {e}")
            return False

    def check_entry_signal(self, params):
        """
        Checks for entry signal based on provided parameters. If signal is detected, orders are placed.

        Args:
            params (dict): A dictionary containing:
                - 'symbol' (str): The trading pair (e.g., "BNB/USDT").
                - 'cooldown' (int): Time in hours to wait between trades.
                - 'n_candles' (int): Number of candles to analyze.
                - 'leverage' (int): Leverage to apply.
                - 'sl_pct' (float): Stop-loss percentage multiplier.
                - 'tp_pct' (float): Take-profit percentage multiplier.
                - 'usdt_amount' (float): Amount to trade in USDT.       
        """

        symbol = params['symbol']
        cooldown = params.get('cooldown', 24)
        n_candles = params.get('n_candles', 50)

        # Check and update order status, skip trading if position is open
        #print('Checking open orders...')
        is_ready_to_trade = self.update_orders(symbol)
        if not is_ready_to_trade:
            print(f"Skipping trade for {symbol} because of open position.")
            return
        
        # Check for cooldown
        if self.check_cooldown(symbol, cooldown):
            print(f"Skipping trade for {symbol} because of recent loss.")
            return 
        
        # Compute prediction
        #print('Computing predictions...')
        predictions = self.call_api(symbol=symbol, n_candles=n_candles)

        if len(predictions) > 0 and predictions[-1]:
            print(f"Entry signal detected for {symbol}. Placing orders...")
            order_params = {
                'symbol': symbol,
                'amount_in_usdt': params.get('usdt_amount', 50),
                'leverage': params.get('leverage', 5),
                'sl_pct': params.get('sl_pct', 0.97),
                'tp_pct': params.get('tp_pct', 1.05)
            }
            self.place_orders(order_params)
        else:
            print(f"No entry signal detected for {symbol}.")

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Binance Futures Trading Bot")
    parser.add_argument("--api_key", required=True, help="Binance API key")
    parser.add_argument("--api_secret", required=True, help="Binance API secret")
    parser.add_argument("--sandbox", action="store_true", help="Enable sandbox mode")

    args = parser.parse_args()

    trade_params = {
        'symbol': 'BNB/USDT',
        'cooldown': 24,
        'n_candles': 50,
        'leverage': 5,
        'sl_pct': 0.97,
        'tp_pct': 1.05,
        'usdt_amount': 100
    }
    # Initialize and run trader
    trader = BinanceFuturesTrader(api_key=args.api_key, api_secret=args.api_secret, sandbox=args.sandbox)
    trader.check_entry_signal(trade_params)