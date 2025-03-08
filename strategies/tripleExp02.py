# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from joblib import load

# additional imports required
from datetime import datetime
from freqtrade.persistence import Trade


from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


import logging
logger = logging.getLogger(__name__)



class TripleExp02(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1d'

    # Indicators startup period
    startup_candle_count = 15

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 10000
    }

    # Establecer el stop loss absoluto
    stoploss = -1
    trailing_stop = False
    use_custom_stoploss = True
    trailing_stop_positive: None
    trailing_stop_positive_offset: 0.0
    trailing_only_offset_is_reached: False

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


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['emaFast'] = ta.EMA(dataframe, timeperiod=4) 
        dataframe['emaSlow'] = ta.EMA(dataframe, timeperiod=8) 
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        df_new = dataframe.loc[15:, :].copy()

        # Use .loc for assignment
        df_new['enter_long'] = 0
        df_new.loc[
            (df_new["emaFast"] > df_new["emaSlow"]),
            'enter_long'] = 1

        return df_new

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> Union[float, None]:
        """
        Calculates stop loss based on ATR.
        """
        ""# Obtener el DataFrame analizado para el par y el marco temporal actual
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Calcular el ATR y obtener el valor del último cierre
        tr1 = dataframe['high'].iloc[:-1] - dataframe['low'].iloc[:-1] 
        tr2 = abs(dataframe['high'].iloc[:-1] - dataframe['close'].iloc[:-1].shift())
        tr3 = abs(dataframe['low'].iloc[:-1] - dataframe['close'].iloc[:-1].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(9).mean() 
        stop_loss = dataframe["close"].iloc[-1] - (atr.iloc[-1] * 3) 

        # Calcular el porcentaje de pérdida basado en el valor del stoploss y el precio actual (porcentaje no porque esta divido 100)
        loss_percent = (current_rate - stop_loss) / current_rate
        return loss_percent

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        
        stakes = {'BTC/USDT':0.5,'ETH/USDT':0.25,'BNB/USDT':0.05,'DOGE/USDT':0.05,'SOL/USDT':0.05,'MATIC/USDT':0.05,'ADA/USDT':0.05}
        try:
            if self.config['stake_amount'] == 'unlimited' or self.config['stake_amount'] > 0:
                # Use entire available wallet during favorable conditions when in compounding mode.
                return self.wallets.get_total_stake_amount() * stakes[pair]
            else:
                # Compound profits during favorable conditions instead of using a static stake.
                return self.wallets.get_total_stake_amount() / self.config['max_open_trades']
        except:
            # Use default stake amount.
            return proposed_stake
        
