# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

import numpy as np
import ast
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
from camel_relaxed import fibo_retracement, find_entry_from

import logging
logger = logging.getLogger(__name__)



class Camel_opt(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Indicators startup period
    startup_candle_count = 100

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.

    # Establecer el stop loss absoluto
    stoploss = -1
    trailing_stop = False
    use_custom_stoploss = False
    trailing_stop_positive: None
    trailing_stop_positive_offset: 0.0
    trailing_only_offset_is_reached: False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Define sell parameters search space
    SL = CategoricalParameter(["out", "optimal", "aggressive"], default="optimal", space="sell", optimize=True)
    TP = DecimalParameter(2.0, 3.0, default=2.6, space="sell", decimals=1, optimize=True)
    rebounce_level = DecimalParameter(0.48, 0.53, default=0.5, space="buy", decimals=2, optimize=True)
    fail_level = DecimalParameter(0.78, 0.855, default=0.85, space="buy", decimals=2, optimize=True)

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
        # Add fractals indicators
        for i in range(2, len(dataframe) - 2):
            sides = [i-2,i-1,i+1,i+2]
            
            # Fractal Up (Bearish)
            dataframe.loc[i, 'fractal_up'] = 1 if all(dataframe.loc[i, 'high'] > dataframe.loc[j, 'high'] for j in sides) else 0
            
            # Fractal Down (Bullish)
            dataframe.loc[i, 'fractal_down'] = 1 if all(dataframe.loc[i, 'low'] < dataframe.loc[j, 'low'] for j in sides) else 0

        dataframe = dataframe.fillna(0)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        entries = []
    
        self.SL.value = 'optimal'
        self.TP.value = 2.75
        self.fail_level.value = 0.85
        self.rebounce_level.value = 0.5
        
        for i in range(len(dataframe)):
            if dataframe.loc[i, 'fractal_down']:
                entry_points = find_entry_from(dataframe, i, rebounce_level=self.rebounce_level.value, fail_level=self.fail_level.value)
                if len(entry_points) > 0: entries.append(entry_points)

        entries = list(set(tuple(lst) for lst in entries))
        entries.sort()

        for i in range(len(entries)):
            A, C, E = dataframe.loc[entries[i][0], 'low'], dataframe.loc[entries[i][2], 'low'], dataframe.loc[entries[i][4], 'low']
            B, D = dataframe.loc[entries[i][1], 'high'], dataframe.loc[entries[i][3], 'high']
            
            SL = -1

            if self.SL.value == 'out':
                SL = fibo_retracement(A, B, 1.272)
            elif self.SL.value == 'optimal':
                SL = fibo_retracement(A, B, 0.855)
            elif self.SL.value == 'aggressive':
                SL = fibo_retracement(C, D, 0.855)

            TP = fibo_retracement(D, E, self.TP.value)
            difference = TP/SL

            if difference > 1:
                dataframe.loc[entries[i][-1], ['enter_long', 'enter_tag']] = (1, str([int(TP), int(SL)]))
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        return dataframe
    

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        vals = ast.literal_eval(trade.enter_tag)
        TP, SL = vals

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        #if last_candle['close'] > dataframe.loc[F, 'close']*1.3: return 'TP'
        if last_candle['high'] > TP: return 'TP'
        if last_candle['low'] < SL: return 'SL'
   
