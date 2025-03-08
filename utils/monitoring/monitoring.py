import asyncio
import json
import argparse
import subprocess
import telegram
from binance.client import Client
from datetime import datetime, timedelta


class Account:
    def __init__(self, name:str, api_key:str, api_secret:str):
        self.name = name
        self.api_key = api_key
        self.api_secret = api_secret

# Obtiene la lista de trades abiertos en freqtrade de la cuenta
def get_open_trades():
    rest_api_dir = './rest_client.py'
    config_dir = './config.json'
    command = ['python3', rest_api_dir, '--config', config_dir, 'status']
    result = subprocess.run(command, capture_output=True, text=True).stdout
    result = json.loads(result)
    trades_dict = {}
    for trade in result:
        if trade['is_open']:
            trades_dict[trade['base_currency']] = trade['amount']
    return trades_dict

# Telegram bot
bot_token = "6792270720:AAEZj57ZJBdu2sxFaqWzSyGcxsqlBK8oxYk"
# https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id
channel_id = -4062262919

async def send_message(message):
    bot = telegram.Bot(token=bot_token)
    await bot.send_message(chat_id=channel_id, text=message)

# Verifica que el balance de Freqtrade y Binance coincidan dada una cuenta
def check_matching_balance(account):
    balance_missmatch = False
    client = Client(account.api_key, account.api_secret) 
    
    freqtrade_trades = get_open_trades()
    binance_assets = client.get_account()['balances']
    binance_assets = {asset['asset']: asset['free'] for asset in binance_assets}
    deposits = client.get_deposit_history()
    withdraws = client.get_withdraw_history()

    # Mensaje a enviar en caso de que los balances no coincidan
    msg = f"Nombre de usuario: {account.name} \n"
    
    # Chequea si hubo depósitos o retiros en el último día
    deposit, msg_deposit = check_history(deposits, True)
    withdraw, msg_withdraw = check_history(withdraws, False)
    msg += msg_deposit + msg_withdraw
    balance_missmatch = deposit or withdraw

    for coin in freqtrade_trades:
        binance_balance = float(binance_assets[coin]) 
        freqtrade_balance = freqtrade_trades[coin]
        if abs(binance_balance - freqtrade_balance) > 1e-5:
            msg += f"El balance de {coin} es de {binance_balance} en binance y {freqtrade_balance} en freqtrade.\n"
            balance_missmatch = True
    
    if balance_missmatch:
        asyncio.run(send_message(msg))

# Comprueba si hubo transacciones de depósito o retiro en el último día
def check_history(transactions, deposit):
    if len(transactions) > 0:
        last_transaction = transactions[0]
        transaction_time = datetime.fromtimestamp(last_transaction['insertTime']/1000.0)
        
        # Chequea que la transacción haya sido en el último día
        if transaction_time >= datetime.now() - timedelta(days=1):
            transaction_time = transaction_time.strftime("%d/%m/%Y")
            action = "depósito" if deposit else "retiro"
            return True, f"Se realizó un {action} de {last_transaction['amount']} {last_transaction['coin']} el {transaction_time}.\n"
    
    return False, ""  


if __name__ == "__main__":
    # Parser para argumentos pasados por cmd
    description = ("Given an user name, api key and api secret, it sends an alert if the account balance in freqtrade and binance don't match.\n" 
                   "In order to execute the script, config.json and rest_client.py must be in the same directory as the script.")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter) 
    parser.add_argument("name", help="User name", type=str)
    parser.add_argument("api_key", help="API key", type=str)
    parser.add_argument("api_secret", help="API secret", type=str)

    args = parser.parse_args()
    account = Account(args.name, args.api_key, args.api_secret)
    check_matching_balance(account)

