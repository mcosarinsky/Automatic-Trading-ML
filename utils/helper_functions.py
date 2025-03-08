import sys
sys.path.append('/home/matias/freqtrade')

import pandas as pd
import numpy as np
import subprocess
import os
import ta
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import plotly.figure_factory as ff
import redis
import gymnasium as gym
import faiss
from typing import Optional
#from stable_baselines3 import DQN
#from stable_baselines3.common.env_util import make_vec_env
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset
from flask_caching import Cache
from dash import dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from collections import OrderedDict

class FaissKNNClassifier:
    """A multiclass exact KNN classifier implemented using the FAISS library."""

    def __init__(
        self, n_neighbors: int, n_classes: Optional[int] = None, device: str = "cpu"
    ) -> None:
        """Instantiate a faiss KNN Classifier.

        Args:
            n_neighbors: number of KNN neighbors
            n_classes: (optional) number of dataset classes
                (otherwise derive from the data)
            device: a torch device, e.g. cpu, cuda, cuda:0, etc.
        """
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes

        if device == "cpu":
            self.cuda = False
            self.device = None
        else:
            self.cuda = True
            if ":" in device:
                self.device = int(device.split(":")[-1])
            else:
                self.device = 0

    def create_index(self, d: int) -> None:
        """Create the faiss index.

        Args:
            d: feature dimension
        """
        if self.cuda:
            self.res = faiss.StandardGpuResources()
            self.config = faiss.GpuIndexFlatConfig()
            self.config.device = self.device
            self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)
        else:
            self.index = faiss.IndexFlatL2(d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Store train X and y.

        Args:
            X: input features (N, d)
            y: input labels (N, ...)

        Returns:
            self
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        self.create_index(X.shape[-1])
        self.index.add(X)
        self.y = y.astype(int)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        return self

    def __del__(self) -> None:
        """Cleanup helpers."""
        if hasattr(self, "index"):
            self.index.reset()
            del self.index
        if hasattr(self, "res"):
            self.res.noTempMemory()
            del self.res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict int labels given X.

        Args:
            X: input features (N, d)

        Returns:
            preds: int predicted labels (N,)
        """
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        preds = np.argmax(counts, axis=1)
        return preds, idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict float probabilities for labels given X.

        Args:
            X: input features (N, d)

        Returns:
            preds_proba: float probas per labels (N, c)
        """
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=class_idx.astype(np.int16),
        )

        preds_proba = counts / self.n_neighbors
        return preds_proba, idx

def add_indicators(df, drop_cols=['open', 'high', 'low', 'close', 'volume']):
    df['rsi'] = ta.momentum.rsi(df.close)/100
    df['stoch'] = ta.momentum.stoch(df.high, df.low, df.close)/100
    df['adx'] = ta.trend.adx(df.high, df.low, df.close)/100
    df['macd'] = ta.trend.macd(df.close)/ta.trend.ema_indicator(df.close, window=26)  # macd = (ema12 - ema26)/ema26
    df['adi'] = ta.volume.acc_dist_index(df.high, df.low, df.close, df.volume)  # adi = adi_prev + volume*CLV  
    df['adi'] = df['adi'].pct_change()
    df.drop(columns=drop_cols, inplace=True)
    
    return df

def triple_barrier(dataframe, SL=None, TP=None, tl=None):
    data = dataframe.copy()
    
    # Precompute entry prices and date for the horizon window
    entry_prices = data['close'].values
    start_dates = data['date'].values
    
    # Precompute the take-profit and stop-loss thresholds
    tp_levels = entry_prices * (1 + TP) if TP else None
    sl_levels = entry_prices * (1 - SL) if SL else None
    
    # Precompute future dates and prices for time horizon (vertical barrier)
    labels = []
    end_dates = []
    end_profits = []  # List to store end profit values
    hit_barriers = []  # List to store whether a barrier was hit
    
    for row_idx in tqdm(range(len(data) - 1)):
        entry_price = entry_prices[row_idx]
        start_date = start_dates[row_idx]
        
        # Define time horizon index
        if tl:
            end_idx = min(row_idx + tl, len(data))
        else:
            end_idx = len(data)
        
        future_data = data.iloc[row_idx + 1:end_idx]
        future_prices = future_data['close'].values
        future_highs = future_data['high'].values
        future_lows = future_data['low'].values
        future_dates = future_data['date'].values
        
        hit_barrier = False
        end_date = future_dates[-1]
        end_profit = None  # Initialize end_profit
        
        for idx in range(len(future_data)):
            if TP and future_highs[idx] >= tp_levels[row_idx]:
                labels.append(1)
                end_date = future_dates[idx]
                end_profit = (future_highs[idx] - entry_price) / entry_price  
                hit_barrier = True
                break
                
            if SL and future_lows[idx] <= sl_levels[row_idx]:
                labels.append(0)
                end_date = future_dates[idx]
                end_profit = (future_lows[idx] - entry_price) / entry_price
                hit_barrier = True
                break
        
        # If no barrier was hit, set label to 0
        if not hit_barrier:
            """
            if tl:
                final_price = future_prices[-1]
                label = 1 if final_price > entry_price * (1 + min_profit) else 0
                labels.append(label)
                end_profit =(final_price - entry_price) / entry_price
            """
            end_profit = (future_prices[-1] - entry_price) / entry_price
            labels.append(0) 
        
        end_dates.append(end_date)
        end_profits.append(end_profit)
        hit_barriers.append(hit_barrier)  # Append whether a barrier was hit
    
    # Add the last row data
    labels.append(0)
    end_dates.append(start_dates[-1])
    end_profits.append(entry_prices[-1] - entry_prices[-1])  # End profit for the last row
    hit_barriers.append(False)  # No barrier hit for the last row
    
    # Add the results back to the dataframe
    data['label'] = labels
    data['endDate'] = end_dates
    data['profits'] = end_profits
    data['hit_barrier'] = hit_barriers  # Add hit_barrier column
    data = data.rename(columns={'date': 'startDate'})
    
    # Remove the last `tl` rows where hit_barrier is False
    if tl:
        last_tl_rows = data.iloc[-tl:]
        data = pd.concat([data.iloc[:-tl], last_tl_rows[last_tl_rows['hit_barrier'] == True]])
    
    return data.drop(columns='hit_barrier')

def curado_profits(data):
    diff = data["profits"]
    diff = (diff > 0).astype(int)
    resultado = diff.copy()
    n = len(resultado)

    activaciones = resultado.copy()
    for i in range(4,n-6):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 1 and activaciones[i+1] == 1 and activaciones[i+2] == 1:
                # Regla 8: 0 0 0 0 1 1 1 0 0 0 0
                if (activaciones[i - 1] == 0 and activaciones[i - 2] == 0  and activaciones[i - 3] == 0 and activaciones[i - 4] == 0) and (activaciones[i + 3] == 0 and activaciones[i + 4] == 0  and activaciones[i + 5] == 0 and activaciones[i + 6] == 0):
                    resultado[i] = 0; resultado[i+1] = 0; activaciones[i+2] == 0;

    # 8: 1 1 1 1 0 0 0 1 1 1 1
    activaciones = resultado.copy()
    for i in range(4,n-5):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 0 and activaciones[i+1] == 0 and activaciones[i+2] == 0:
                # Regla 7: 1 1 1 1 0 0 0 1 1 1 1
                if (activaciones[i - 1] == 1 and activaciones[i - 2] == 1  and activaciones[i - 3] == 1 and activaciones[i - 4] == 1) and (activaciones[i + 3] == 1 and activaciones[i + 4] == 1  and activaciones[i + 5] == 1 and activaciones[i + 6] == 1):
                    resultado[i] = 1; resultado[i+1] = 1; activaciones[i+2] == 1;

    # 7:  1 1 1 0 0 1 1 1
    activaciones = resultado.copy()
    for i in range(3,n-4):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 0 and activaciones[i+1] == 0:
                # Regla 5: 1 1 1 0 0 1 1 1
                if activaciones[i - 1] == 1 and activaciones[i - 2] == 1 and activaciones[i - 3] == 1  and activaciones[i + 2] ==1 and activaciones[i + 3]== 1 and activaciones[i + 4]== 1:
                    resultado[i] = 1; resultado[i+1] = 1

    # 6: 1 1 1 0 0 1 1 1
    activaciones = resultado.copy()
    for i in range(3,n-4):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 0 and activaciones[i+1] == 0:
                # Regla 4: 1 1 1 0 0 1 1 1
                if activaciones[i - 1] == 1 and activaciones[i - 2] == 1 and activaciones[i - 3] == 1 and activaciones[i + 2] == 1 and activaciones[i + 3] == 1 and activaciones[i + 4] == 1:
                    resultado[i] = 1; resultado[i+1] = 1
    # 5: 0 1 1 0
    activaciones = resultado.copy()
    for i in range(n-1):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 1 and activaciones[i+1] == 1:
                # Regla 3:  0 1 1 0
                if activaciones[i - 1] == 0 and activaciones[i + 2] == 0:
                    resultado[i] = 0; resultado[i+1] = 0

    # 4: 1 0 1
    activaciones = resultado.copy()
    for i in range(n):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 0:
                # Regla 2: 1 0 1
                if activaciones[i - 1] == 1 and activaciones[i + 1] == 1:
                    resultado[i] = 1

    # 3: 0 1 0
    activaciones = resultado.copy()
    for i in range(n):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 1:
                # Regla 1: 0 1 0
                if activaciones[i - 1] == 0 and activaciones[i + 1] == 0:
                    resultado[i] = 0

    #################################
    ### Reglas repetidas al final ###
    #################################

    # 2: 0 0 0 0 1 1 1 0 0 0 0
    activaciones = resultado.copy()
    for i in range(4,n-6):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 1 and activaciones[i+1] == 1 and activaciones[i+2] == 1:
                # Regla 8: 0 0 0 0 1 1 1 0 0 0 0
                if (activaciones[i - 1] == 0 and activaciones[i - 2] == 0  and activaciones[i - 3] == 0 and activaciones[i - 4] == 0) and (activaciones[i + 3] == 0 and activaciones[i + 4] == 0  and activaciones[i + 5] == 0 and activaciones[i + 6] == 0):
                    resultado[i] = 0; resultado[i+1] = 0; activaciones[i+2] == 0;

    # 1: 0 1 0
    activaciones = resultado.copy()
    for i in range(n):
        if (i > 0 and i < n - 1):
            if activaciones[i] == 1:
                # Regla 1: 0 1 0
                if activaciones[i - 1] == 0 and activaciones[i + 1] == 0:
                    resultado[i] = 0
    return resultado

def get_results_df(optimizer):
    dic = pd.DataFrame(optimizer.cv_results_).set_index('rank_test_score').sort_index()
    params = dic['params'].iloc[0]
    cols = ['mean_test_score', 'mean_train_score', 'params'] + [f'param_{key}' for key in params.keys()] 
    dic = dic[cols].rename(columns=lambda x: x.replace('param_', ''))

    def rename_keys(dic):
        return OrderedDict((key.split('__')[1] if '__' in key else key, value) for key, value in list(dic.items()))
    dic['params'] = dic['params'].apply(rename_keys)
    
    return dic

def get_trials(study):
    dic = study.trials_dataframe().sort_values(by='value', ascending=False)
    params = [col for col in dic.columns if col.startswith('params')]
    params_dic = [{x.replace('params_', ''):dic[x].iloc[i] for x in params} for i in range(len(dic))] #diccionario con parametros por fila

    cols = ['value', 'user_attrs_train_score'] + params
    dic = dic[cols].rename(columns=lambda x: x.replace('params_', ''))
    dic = dic.rename(columns={'value': 'val_score', 'user_attrs_train_score': 'train_score'})
    dic['params'] = params_dic
    return dic


def process_info(info_dic):
    total_rew = info_dic['total_reward']
    profits = pd.DataFrame(list(info_dic['profits'].items()), columns=['coin_name', 'profit']).set_index('coin_name')
    trade_history = info_dic['trades']
    if info_dic['open_trades']:
        open_trades = pd.DataFrame([{'coin_name': coin, 'profit': info_dic['open_trades'][coin].profit} 
                                    for coin in info_dic['open_trades']]).set_index('coin_name')
    else:
        open_trades = pd.DataFrame(columns=['coin_name', 'profit']).set_index('coin_name')
    return total_rew, profits, trade_history, open_trades



def check_entry_point(row, buy_dates_dict):
    # Get the coin and start date for the current row
    coin = row['coin']
    start_date = row['date']

    # Check if the start_date is in the buy_dates of the coin
    if start_date in buy_dates_dict.get(coin, []):
        return 1
    return 0

def check_exit_point(row, sell_dates_dict):
    # Get the coin and start date for the current row
    coin = row['coin']
    start_date = row['date']

    # Check if the start_date is in the sell_dates of the coin
    if start_date in sell_dates_dict.get(coin, []):
        return 1
    return 0


def get_trades_rl(df, trade_history):
    data_dir = '/home/matias/freqtrade/user_data/data/binance'
    dfs = []
    for coin in np.unique(trade_history.coin):
        start = df[df.coin==coin].startDate.iloc[0]
        end = df[df.coin==coin].startDate.iloc[-1]
        df_coin = pd.read_feather(data_dir + f'/{coin}_USDT-1d.feather')
        df_coin['date'] = pd.to_datetime(df_coin['date']).dt.tz_localize(None)
        extra_days = trade_history.sell_date.iloc[-1].tz_localize(None) - df.startDate.iloc[-1].tz_localize(None) + pd.Timedelta(days=120)
        df_coin = df_coin.loc[(df_coin.date >= start) & (df_coin.date <= end+extra_days), ['date', 'close']]
        df_coin['coin'] = coin
        df_coin['entry_point'] = df_coin['date'].isin(trade_history.loc[trade_history.coin==coin, 'buy_date']).astype(int)
        df_coin['exit_point'] = df_coin['date'].isin(trade_history.loc[trade_history.coin==coin, 'sell_date']).astype(int)
        df_coin['profit'] = None
        n =  len(df_coin[df_coin.exit_point==1])
        df_coin.loc[df_coin.exit_point==1, 'profit'] = trade_history.loc[trade_history.coin==coin, 'profit'][:n].values
        dfs.append(df_coin)
    return pd.concat(dfs).reset_index(drop=True)


def get_profit(coin, df_coin, strategy='hold'):
    data_dir = '/home/matias/freqtrade/user_data/data/binance'
    df = pd.read_feather(data_dir + f'/{coin}_USDT-1d.feather')
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    start = df_coin.startDate.iloc[0]
    end = df_coin.startDate.iloc[-1]
    df['emaSlow'] = ta.trend.ema_indicator(df['close'], window=8)
    df['emaFast'] = ta.trend.ema_indicator(df['close'], window=4)
    df = df.loc[(df.date >= start) & (df.date <= end), :].reset_index(drop=True)
    if strategy == 'hold':
        start_price = df.loc[df.date==start, 'close'].item()
        end_price = df.loc[df.date==end, 'close'].item()
        profit = (end_price - start_price) / start_price 
    elif strategy == 'exp':
        df['endDate'] = df_coin.endDate.reset_index(drop=True)
        strategy = TripleExp(df)
        profit = strategy.apply_strategy()
    return profit


def add_coins(df):
    new_df = df.copy().reset_index(drop=True)
    new_df['index'] = df.index
    new_df = new_df.sort_values(by=['startDate','index'])
    coins = ["BTC","BNB","LTC","DOGE","XRP","ETH","ADA","MATIC","SOL","TRX"]
    
    def assign_coins(group):
        group['coin'] = coins[:len(group)]
        return group

    new_df = new_df.groupby('startDate').apply(assign_coins)
    #new_df = new_df.drop(columns=['index']).reset_index(drop=True)
    new_df.index = new_df.index.droplevel('startDate') 
    new_df = new_df.sort_index()
    return new_df


def create_app(df_trades):
    r = redis.Redis(host='localhost', port=6379, db=0)
    app = dash.Dash(__name__)
    cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379'
    })

    coins = df_trades['coin'].unique()

    app.layout = html.Div([
        dcc.Dropdown(
            id="coin-dropdown",
            options=[{'label': i, 'value': i} for i in coins],
            value='BTC' if 'BTC' in coins else coins[0],  # Default value
            style={"width": "30%"}  
        ),
        dcc.RadioItems(
            id='days-radioitems',
            options=[
                {'label': '1M', 'value': 30},
                {'label': '3M', 'value': 90},
                {'label': '6M', 'value': 180}
            ],
            value=180,  # Default value
            labelStyle={'display': 'inline-block'}  # Display the options inline
        ),
        dcc.Graph(id='graph')
    ])

    @app.callback(
        Output('graph', 'figure'),
        [Input('coin-dropdown', 'value'),
        Input('days-radioitems', 'value')]
    )
    @cache.memoize()
    def update_graph(selected_coin, selected_days):
        
        # Filter data based on selected coin
        df_filtered = df_trades[df_trades['coin'] == selected_coin].reset_index(drop=True)

        x = df_filtered.date
        y = df_filtered.close

        # Calculate the min and max of y for the current period of days
        y_min = y[:selected_days].min()
        y_max = y[:selected_days].max()

        # Calculate a margin for better visualization
        y_margin = (y_max - y_min) * 0.05

        entry_trace = go.Scatter(
            x=df_filtered[df_filtered['entry_point'] == 1]['date'][:selected_days],
            y=df_filtered[df_filtered['entry_point'] == 1]['close'][:selected_days],
            mode='markers',
            marker=dict(
                color='green',
                symbol='triangle-up',
                size=10
            ),
            name='Buy'
        )

        exit_trace = go.Scatter(
            x=df_filtered[df_filtered['exit_point'] == 1]['date'][:selected_days],
            y=df_filtered[df_filtered['exit_point'] == 1]['close'][:selected_days],
            mode='markers',
            marker=dict(
                color='red',
                symbol='triangle-down',
                size=10
            ),
            name='Sell',
            hovertemplate=
            '<b>Date</b>: %{x}<br>' +
            '<b>Price</b>: %{y}<br>' +
            '<b>Profit</b>: %{text}<br>' +
            '<extra></extra>',
            text=df_filtered[df_filtered['exit_point'] == 1]['profit']
        )
        
        # Define buttons and slider
        buttons_dic = dict(type="buttons",
                        buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 300, "redraw": True},
                                                                                    "fromcurrent": True,
                                                                                    "transition": {"duration": 50}}]),
                                    dict(label="Reset", method="animate", args=[["frame0"], {"frame": {"duration": 0, "redraw": True},
                                                                                            "mode": "immediate",
                                                                                            "transition": {"duration": 0}}])])
        num_ticks = 50
    
        # If there are more data points than ticks that can fit, subsample the dates to be shown
        if len(x) > num_ticks:
            tick_indices = np.linspace(0, len(x)-1, num_ticks, dtype=int)
            tickvals = tick_indices
            ticktext = [x[i].strftime('%Y-%m-%d') for i in tick_indices]  # Adjust the date format as needed
        else:
            tickvals = list(range(len(x)))
            ticktext = [date.strftime('%Y-%m-%d') for date in x]
        
        last_possible_start = max(x) - pd.Timedelta(days=selected_days)
        ticktext_dates = pd.to_datetime(ticktext, format='%Y-%m-%d')

        # Create a mask for the dates that are within the allowable range
        mask = [date <= last_possible_start for date in ticktext_dates]
        
        # Apply the mask when creating the steps for the slider
        sliders_dic = dict(
            steps=[dict(
                method='animate', 
                args=[[f'frame{k}'], 
                    dict(mode='immediate', 
                        frame=dict(duration=300, redraw=True),
                        transition=dict(duration=50))],
                label=ticktext[i]) for i, k in zip(range(len(ticktext)), tickvals) if mask[i]],  # Use the mask here
            transition=dict(duration=0),
            currentvalue=dict(visible=True, prefix='Date: ')
        )                

        # Create figure with frames
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x[:selected_days], 
                    y=y[:selected_days], 
                    mode='lines', 
                    line=dict(color='royalblue'), 
                    showlegend=False,
                    hovertemplate=
                    '<b>Date</b>: %{x}<br>' +
                    '<b>Price</b>: %{y}<br>' +
                    '<extra></extra>'
                ),
                entry_trace,
                exit_trace
            ],
            layout=go.Layout(
                title=f"Stock Price Over Time for {selected_coin}",
                title_x=0.5,
                xaxis=dict(title="Date", title_standoff=5, range=[min(x), min(x)+pd.Timedelta(days=selected_days)], autorange=False, 
                        gridcolor='rgba(128, 128, 128, 0.1)', gridwidth=0.5, griddash='solid', tickformat="%d %B %Y",
                        fixedrange=False),
                yaxis=dict(title="Price", range=[y_min - y_margin, y_max + y_margin], autorange=False, 
                        gridcolor='rgba(128, 128, 128, 0.1)', gridwidth=0.5, griddash='solid', fixedrange=False),
                sliders=[sliders_dic],
                updatemenus=[buttons_dic],
                height=600,
                margin=dict(t=25, b=35)
            ),
            frames=[go.Frame(data=[go.Scatter(x=x[k:k+selected_days], y=y[k:k+selected_days], mode='lines', line=dict(color='royalblue'))],
                 layout=go.Layout(xaxis=dict(range=[x[k], x[k+selected_days-1]], autorange=False,
                                              gridcolor='rgba(128, 128, 128, 0.1)', gridwidth=0.5, griddash='solid'),
                                  yaxis=dict(range=[min(y[k:k+selected_days]) - y_margin, max(y[k:k+selected_days]) + y_margin], autorange=False,
                                              gridcolor='rgba(128, 128, 128, 0.1)', gridwidth=0.5, griddash='solid')),
                 name=f'frame{k}') for k in range(0, len(x)-selected_days, 1)]
        )
        fig.update_layout(xaxis=dict(nticks=8))
        return fig
    return app


def make_model_1():
    env = make_vec_env(lambda: Dummy_env_1(), seed=1, n_envs=1)
    model = DQN.load("/home/matias/freqtrade/dev/modelos/rl_model_1", env=env)
    return model


class Dummy_env_1(gym.Env):
    def __init__(self, ncols=5*14):
        self.ncols = ncols
        super(Dummy_env_1, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        # Observation space: [low, high, open, close, volume] for the last window_size days 
        self.observation_space = gym.spaces.Box(low=-4, high=4, shape=(ncols,), dtype=np.float64)

def process_dataframe_rl(dataframe: DataFrame, window_size=14) -> DataFrame:
    cols = ['open', 'high', 'low', 'close', 'volume']
    candles = dataframe.loc[:, cols]  
    dfClean = candles.pct_change().dropna()
    dfClean = dfClean.iloc[:-1,:]
    window_dfs = []
    scaler = StandardScaler()

    # Itera creando DataFrames de ventanas de 14 dias
    for i in range(len(dfClean) - window_size + 1):
        window_df = dfClean.iloc[i:i + window_size]
        window_dfs.append(window_df)

    # Lo une en un solo DataFrame de 70 columnas
    result_df = pd.concat(window_dfs, axis=0)
    for i in range(len(dfClean) - window_size + 1):
        for j in range(len(cols)):
            current_window = result_df.iloc[i:i + window_size, j]
            scaled = scaler.fit_transform(current_window.values.reshape(-1,1)).flatten()
            result_df.iloc[i:i + window_size, j] = scaled
    data = np.array(result_df).reshape(-1,window_size*5)   
    df = pd.DataFrame(data=data)
    df.columns = [f'{name}_{i}' for i in range(1, 15) for name in ['open', 'high', 'low', 'close', 'volume']]
    open_columns = [col for col in df.columns if col.startswith('open')]
    close_columns = [col for col in df.columns if col.startswith('close')]
    volume_columns = [col for col in df.columns if col.startswith('volume')]
    low_columns = [col for col in df.columns if col.startswith('low')]
    high_columns = [col for col in df.columns if col.startswith('high')]
    features = [open_columns, close_columns, volume_columns, low_columns, high_columns]

    scaler = StandardScaler()
    df_scaled = df.copy()

    # Standardize each feature within its own window
    for cols in features:
        for i in range(len(df)):
            data = df.loc[i, cols]
            df_scaled.loc[i, cols] = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

    data_scaled = np.array(df_scaled).reshape(-1,window_size*5) 
    return data_scaled


