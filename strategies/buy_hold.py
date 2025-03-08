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



class Buy_Hold(IStrategy):

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

    # Establecer el stop loss absoluto
    stoploss = -1
    trailing_stop = False
    use_custom_stoploss = False
    trailing_stop_positive: None
    trailing_stop_positive_offset: 0.0
    trailing_only_offset_is_reached: False

    # These values can be overridden in the config.
    use_exit_signal = False
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
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df_new = dataframe.loc[29:, :].copy()

        # Use .loc for assignment
        df_new['enter_long'] = 1

        return df_new

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        stakes = {'BTC/USDT':1.0,'ETH/USDT':0,'BNB/USDT':0,'DOGE/USDT':0,'TRX/USDT':0,
                    'LTC/USDT':0,'XRP/USDT':0,'SOL/USDT':0,'MATIC/USDT':0,'ADA/USDT':0}
        

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
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 5.0