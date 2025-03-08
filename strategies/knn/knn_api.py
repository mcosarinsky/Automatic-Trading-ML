# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from tqdm import tqdm
import sys

# additional imports required
from datetime import datetime
from freqtrade.persistence import Trade
import requests

from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)
from freqtrade.enums import CandleType

import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, merge_informative_pair, informative, stoploss_from_absolute


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


import logging
logger = logging.getLogger(__name__)


class KNN_API(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Indicators startup period
    startup_candle_count = 100

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
    }
    
    trailing_stop = False
    use_custom_stoploss = False
    stoploss = -1

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            }
        }
    
    def informative_pairs(self):
        return [("BTC/USDT", "1h", "spot")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf_pair, inf_tf, inf_ct = self.informative_pairs()[0]
        informative = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=inf_tf, candle_type=inf_ct) #CandleType.SPOT
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        temp_dataframe = dataframe.copy()

        url = "http://127.0.0.1:8080/predict"
        cols = ['open_1h', 'low_1h', 'high_1h', 'close_1h', 'volume_1h']

        for col in temp_dataframe.columns:
            if temp_dataframe[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                temp_dataframe[col] = temp_dataframe[col].astype(str)
        
        payload = {
            "data": temp_dataframe.to_dict(orient='records'), 
            "cols": cols
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
                result = response.json()  
                preds = np.array(result['predictions'])
                zeros = np.zeros(29)  # window_size+1=29
                preds = np.concatenate((zeros, preds))
                print(f'{len(preds)} predictions computed')
                dataframe['enter_long'] = preds
        else:
            print(f"Error: {response.status_code} - {response.text}")

        # Cooldown
        candles_coldown = 30
        time_diference = candles_coldown
        trades = Trade.get_trades()
        diff_open_close_rates = 0  # Para saber si el profit fue negativo o no (hacemos solo la cuenta de la diferencia entre close rate y open rate que nos ahorra hacer una cuenta mas)
        # Parametro dataframe
        activado = 0  # open_date mayo a close_date es decir hay trade abierto

        # Aca nos vamos fijando todos los trades cerrados que se hicieron para cierto par y agarramos el ultimo para saber la diferencia de tiempo, y la diferencia de profits
        # (Aca se podria optimizar -> pero no transformarlo como diccionario ni lista porque el objeto trades pierde informacion por alguna razón)
        try:
            for trade in trades:
                if (trade.pair == metadata["pair"]) and (trade.close_date):
                    current_candle = dataframe.iloc[-1]
                    current_candle_time = current_candle.date.replace(second=0, microsecond=0, tzinfo=None)
                    close_date_pure = trade.close_date  # necesito los segundos asi que pongo esta variable
                    last_trade_time = close_date_pure.replace(second=0, microsecond=0, tzinfo=None)
                    seconds_div = 3600  # Si fuera de a 1 minuto le pones 60
                    time_diference = int((current_candle_time - last_trade_time).total_seconds() / seconds_div)  # (3600 porque es en horas -> en el de minutos solo se divide por 60)
                    close_rate_con_comision = trade.close_rate * (1 - 0.01 / 100)
                    open_rate_con_comision = trade.open_rate * (1 + 0.01 / 100)
                    diff_open_close_rates = close_rate_con_comision - open_rate_con_comision  # Hace menos cuentas asi

                # Agregado, ultimo trade_abierto
                ultimo_trade_abierto = trade.open_date  # como necesito los segundos no le pongo ningun replace
                activado = 0
                if close_date_pure <= ultimo_trade_abierto:  # ultimo trade cerrado y ultimo trade abierto
                    activado = 1
        except:
            print("Error en la obtencion de trades antiguos")

        # Ii la diferencia de velas entre ahora y el ultimo trade es menor al coldown_candles y el profit fue negativo, se aplica el cooldown
        if (((time_diference < candles_coldown) and (diff_open_close_rates < 0)) or activado == 1):  
            dataframe.at[dataframe.index[-1], 'enter_long'] = 0  # Señal de no comprar en la ultima
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:     
        return dataframe
    
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return 5.0
    
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str:
        exit = None
        if current_profit >= 0.05*5:
            exit = "TP"
        elif current_profit <= -0.025*5:
            exit = "SL"

        return exit