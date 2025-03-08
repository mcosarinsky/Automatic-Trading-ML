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
import os
sys.path.append('/home/matias/freqtrade/dev/crypto-trading/utils')
from strategy_funcs import *
# additional imports required
from datetime import datetime
from freqtrade.persistence import Trade


from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)
from freqtrade.enums import CandleType

import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, merge_informative_pair


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


import logging
logger = logging.getLogger(__name__)


class KNN_test(IStrategy):

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

    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.profits_df = pd.read_csv('dev/crypto_trading/modelos/datos/btc_curado_28h.csv', parse_dates=['startDate', 'endDate'])
        self.profits_df['startDate'] = self.profits_df['startDate'].dt.tz_localize('UTC')
        self.profits_df['endDate'] = self.profits_df['endDate'].dt.tz_localize('UTC')
        self.profits_df = self.profits_df.sort_values(by='endDate', ascending=False)

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


    def get_n_closest(self, startDate, n=1000, starting_count=48):
        previous_data = self.profits_df[self.profits_df['endDate'] < startDate - pd.Timedelta(hours=starting_count)]
        res = previous_data.head(n) #+ starting_count).iloc[starting_count:]
        
        drop_cols = ['startDate', 'endDate', 'profits', 'profit_curado']
        X_train, y_train = res.drop(columns=drop_cols), res['profit_curado']
        
        return np.array(X_train), np.array(y_train), np.array(res.index)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['emaFast'] = ta.EMA(dataframe, timeperiod=4) 
        dataframe['emaSlow'] = ta.EMA(dataframe, timeperiod=8) 
        
        return dataframe
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        window_size = 28
        n_points = 2500
        k = 7
        threshold = 6/7
        starting_count = 1
        cols = ['open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h']

        inf_pair, inf_tf, inf_ct = self.informative_pairs()[0]
        informative = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=inf_tf, candle_type=inf_ct) 
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        data = process_dataframe(dataframe, lags=window_size, cols=cols)
        dates = data.index

        for date in tqdm(dates):
            X, y, row_indexes = self.get_n_closest(date, n=n_points, starting_count=starting_count)
            current_data = np.array(data.loc[date])
            
            if len(y) == n_points:
                knn_clf = FaissKNNClassifier(n_neighbors=k)
                knn_clf.fit(X, y)
                pred_proba, idx = knn_clf.predict_proba(current_data)
                pred = pred_proba[0,1] >= threshold

                dataframe.loc[dataframe['date'] == date, 'enter_long'] = int(pred)
        
        dataframe.to_csv('backtest_results/test_data_updated.csv', index=False)
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