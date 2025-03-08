import sys
sys.path.append('~/freqtrade')

import pandas as pd
import numpy as np
import subprocess
import os
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import plotly.figure_factory as ff
import redis
from pandas import DataFrame
from pandas.tseries.offsets import DateOffset
import ta
from flask_caching import Cache
from dash import dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.resolvers import ExchangeResolver, StrategyResolver


def get_backtest_results(start_date, dir, strategy_list, n_months=6, script_path="./run_backtest.sh"):
    start_date = pd.to_datetime(start_date) #- pd.Timedelta(days=15)
    end_date = min(start_date + DateOffset(months=n_months), pd.to_datetime("2024-07-01"))
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    args = [script_path, start_str, end_str] + strategy_list

    try:
        # Run the shell script
        subprocess.run(args, cwd=os.path.dirname(script_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return load_backtest_stats(dir)
    except Exception as e:
        print("An error occurred while executing the shell script. Arguments:", args)
        print("Error details:", e)

def get_comparison(stats):
    df = pd.DataFrame(stats['strategy_comparison']).rename(columns={'key':'strategy'}).set_index('strategy')
    float_columns = df.select_dtypes(include='float64').columns
    df[float_columns] = df[float_columns].round(3)
    return df.drop(df.filter(like='pct', axis=1).columns.union(['max_drawdown_abs']), axis=1)

def get_results_per_pair(stats, strategy):
    df = pd.DataFrame(stats['strategy'][strategy]['results_per_pair']).rename(columns={'key':'pair'}).set_index('pair')
    float_columns = df.select_dtypes(include='float64').columns
    df[float_columns] = df[float_columns].round(3)
    return df.drop(df.filter(like='pct', axis=1), axis=1)

def get_total_profit_by_month(dir, strategies, months=6, start="2022-03-16", end="2024-03-01", ndays=30):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    res, cols = [], []
    while start + DateOffset(months=months) <= end:
        res_strat = []
        try:
            current_res = get_backtest_results(start, dir, strategies, n_months=months)
            current_profits = get_comparison(current_res).profit_total * 100
            current_end = min(start + DateOffset(months=months), end)
            cols.append(start.strftime('%Y/%m/%d') + ' - ' + current_end.strftime('%Y/%m/%d'))
            res.append(current_profits)
            start += DateOffset(days=ndays)
        except Exception as e:
            print(e)
            start += DateOffset(days=ndays)
            continue  
    
    res_df = pd.concat(res, axis=1)
    res_df.columns = cols
    return res_df


def get_stats_by_month(dir, strategies, months=6, start="2022-03-16", end="2024-03-01", ndays=30):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    res, cols = [], []
    while start+DateOffset(months=months) <= end:
        res_strat = []
        current_res = get_backtest_results(start, dir, strategies, n_months=months)
        for strat in strategies:
            current_stats = get_results_per_pair(current_res, strat)
            current_stats['strategy'] = strat
            current_stats = current_stats.set_index(['strategy'], append=True).profit_total*100
            res_strat.append(current_stats)
        
        current_end = min(start+DateOffset(months=months), end)
        cols.append(start.strftime('%Y/%m/%d') + ' - ' + current_end.strftime('%Y/%m/%d'))
        res.append(pd.concat(res_strat, axis=0))
        start += DateOffset(days=ndays)
    
    res_df = pd.concat(res, axis=1)
    res_df.columns = cols
    return res_df

def get_trades(stats, strategy):
    df = pd.DataFrame(stats['strategy'][strategy]['trades']).rename(columns={'key':'pair'}).set_index('pair')
    float_columns = df.select_dtypes(include='float64').columns
    df[float_columns] = df[float_columns].round(3)
    columns_to_keep = ['pair', 'stake_amount', 'amount', 'open_date',
                       'close_date', 'open_rate', 'close_rate', 
                       'profit_ratio', 'profit_abs', 'exit_reason', 
                       'min_rate', 'max_rate']
    df['open_date'] = pd.to_datetime(df['open_date'])
    df['close_date'] = pd.to_datetime(df['close_date'])

    return df.filter(columns_to_keep)


def get_entry_ticks(config, start_date, strategies, n_months=24, data_dir="~/freqtrade/user_data/data/binance"):
    start_date = (pd.to_datetime(start_date)).tz_localize('UTC')
    end_date = start_date + DateOffset(months=n_months)
    pairs = [s.replace('/', '_') for s in config['pairs']]
    results = []
    for strat in strategies:
        config["strategy"] = strat
        strategy = StrategyResolver.load_strategy(config)
        strat_results = []
        for pair in pairs:
            # Load pair data for given timerange
            candles = pd.read_feather(f"{data_dir}/{pair}-1d.feather")
            candles = candles[(candles.date >= start_date) & (candles.date <= end_date)].reset_index()
            
            # Get entry ticks for strategy
            enter_counts = strategy.analyze_ticker(candles, {'pair': pair})['enter_long'].value_counts().sort_index(ascending=False)
            enter_counts.name = (strat, pair)
            strat_results.append(enter_counts)
        
        # Add total row for each strategy
        total_counts = pd.concat(strat_results).groupby(level=0).sum()
        total_counts.name = (strat, 'Total')
        strat_results.append(total_counts)
        results.extend(strat_results)
    
    results_df = pd.concat(results, axis=1).T
    results_df.index.names = ['strategy', 'pair']
    results_df.columns = ['entry', 'no_entry']
    return results_df


def plot_daily_profit(stats, strategy):
    strategy_stats = stats['strategy'][strategy]

    df = pd.DataFrame(columns=['dates','equity'], data=strategy_stats['daily_profit'])
    df['equity_daily'] = df['equity'].cumsum()

    fig = px.line(df, x="dates", y="equity_daily", title=strategy)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(height=600)
    fig.show()

def plot_profit_ratio(stats, strategy):
    trades = pd.DataFrame(stats['strategy'][strategy]['trades'])

    hist_data = [trades.profit_ratio]
    group_labels = ['profit_ratio']  # name of the dataset
    
    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01)
    fig.update_layout(height=500)
    fig.show()

def plot_profits(trades_dfs, strategies):
    pairs = np.unique([pair for df in trades_dfs for pair in df.index])
    colors = ['blue','red','sandybrown','greenyellow','darkviolet','springgreen','turquoise','teal','skyblue','olive','yellow','teal',
             'chocolate','chartreuse','coral','brown','crimson','goldenrod','indigo']
    color_dict = {strategies[i]:colors[i] for i in range(len(strategies))}
    r = redis.Redis(host='localhost', port=6379, db=0)
    app = dash.Dash(__name__)
    cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379'
    })
    app.layout = html.Div([
        html.Div([
            html.Label('Select Pair', style={'display': 'inline-block', 'width': '130px'}),
            dcc.Dropdown(
                id='pair-dropdown',
                options=[{'label': i, 'value': i} for i in ['None'] + list(pairs)],
                value=None,  # Default to None
                style={'width': '200px', 'display': 'inline-block'}  # Adjust the width here
            )
        ]),
        html.Div([
            html.Label('Select Strategies', style={'display': 'inline-block', 'width': '130px'}),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[{'label': i, 'value': i} for i in strategies],
                value=strategies[0],  # Default to the first strategy
                multi=True,  # Allow multiple selection
                style={'width': '800px', 'display': 'inline-block'}  # Adjust the width here
            )
        ]),
        dcc.Graph(id='profit-plot')
    ])

    @app.callback(
        Output('profit-plot', 'figure'),
        [Input('pair-dropdown', 'value'), Input('strategy-dropdown', 'value')]
    )
    def update_output(pair, selected_strategies):
        fig = go.Figure()
    
        if not isinstance(selected_strategies, list):
            selected_strategies = [selected_strategies]  # Make sure selected_strategies is a list
    
        for strategy in selected_strategies:
            idx = strategies.index(strategy)
            trades = trades_dfs[idx]
            if pair is not None:
                trades = trades[trades.index == pair]
    
            df_pair = trades.loc[:, ['close_date', 'profit_abs']].groupby('close_date').sum().reset_index()

            
            if pair == 'None':
                df_pair = trades_dfs[idx].loc[:, ['close_date', 'profit_abs']].groupby('close_date').sum().reset_index()

            
            # Add 5 days prior to first close_date
            first_close_date = min([df['close_date'].min() for df in trades_dfs])
            #first_close_date = min([trades_dfs[i]['close_date'].min() for i in range(len(trades_dfs))])
            prior_5_days = pd.date_range(start=first_close_date - pd.DateOffset(days=5), end=first_close_date, freq='D')
            df_prior = pd.DataFrame({'close_date': prior_5_days, 'profit_abs': 0})
            df_pair = pd.concat([df_prior, df_pair])
    
            # Add 5 days after the last close_date
            last_close_date = max([df['close_date'].max() for df in trades_dfs])
            after_5_days = pd.date_range(start=last_close_date, end=last_close_date + pd.DateOffset(days=5), freq='D')
            df_after = pd.DataFrame({'close_date': after_5_days, 'profit_abs': 0})
            df_pair = pd.concat([df_pair, df_after])
    
            # Reset profit_abs to 0 for the 5 days after the last close_date
            df_pair.loc[df_pair['close_date'] > last_close_date, 'profit_abs'] = 0
    
            df_pair['profit_daily'] = df_pair.profit_abs.cumsum()
            fig.add_trace(go.Scatter(x=df_pair["close_date"], y=df_pair["profit_daily"], mode='lines', name=strategy, 
                                     line=dict(color=color_dict[strategy])))
    
        if pair is None:
            fig.update_layout(title="Total profit for selected strategies", yaxis=dict(rangemode="tozero"), height=600)
        else:
            fig.update_layout(title=f"Profits for {pair} for selected strategies", yaxis=dict(rangemode="tozero"), height=600)
    
        return fig
    return app

def plot_by_month(df, strategies):

    app = dash.Dash(__name__)
    
    pairs = list(np.unique(df.pair))  
    colors = ['blue','red','sandybrown','greenyellow','darkviolet','springgreen','turquoise','teal','skyblue','olive','yellow','teal',
             'chocolate','chartreuse','coral','brown','crimson','goldenrod','indigo']
    color_dict = {strategies[i]:colors[i] for i in range(len(strategies))}
    
    app.layout = html.Div([
        dcc.Dropdown(
            id='pair-dropdown',
            options=[{'label': i, 'value': i} for i in pairs],
            value='TOTAL'  # Default value
        ),
        dcc.Dropdown(
            id='strategy-dropdown',
            options=[{'label': i, 'value': i} for i in strategies],
            value=strategies,
            multi=True
        ),
        dcc.Graph(id='graph')
    ])
    
    @app.callback(
        Output('graph', 'figure'),
        [Input('pair-dropdown', 'value'),
         Input('strategy-dropdown', 'value')]
    )
    def update_graph(selected_pair, selected_strategies):
        df_filt = df[df.pair == selected_pair]
        
        fig = go.Figure()
    
        for strategy in selected_strategies:
            fig.add_trace(go.Scatter(
                x=df_filt.columns[2:],  # Dates
                y=df_filt[df_filt.strategy == strategy].iloc[:, 2:].values.flatten(),  # Values
                mode='lines',
                name=strategy,
                marker_color=color_dict[strategy]
            ))
    
        fig.update_layout(
            title='Strategy Performance Over Time',
            xaxis_title='Date',
            yaxis_title='Profit',
            legend_title='Strategy',
            height=650,
            width=1500
        )
    
        return fig
    return app

def plot_boxplots(strategies, df_3_month, df_6_month, df_year):
    df_3_month_melt = pd.DataFrame()
    df_6_month_melt = pd.DataFrame()
    df_year_melt = pd.DataFrame()
    
    for strategy in strategies:
        temp_df_3_month = df_3_month[df_3_month.loc[:, 'strategy'] == strategy].iloc[:, 2:].melt(var_name='variable', value_name='profit')
        temp_df_3_month['strategy'] = strategy
        df_3_month_melt = pd.concat([df_3_month_melt, temp_df_3_month])
    
        temp_df_6_month = df_6_month[df_6_month.loc[:, 'strategy'] == strategy].iloc[:, 2:].melt(var_name='variable', value_name='profit')
        temp_df_6_month['strategy'] = strategy
        df_6_month_melt = pd.concat([df_6_month_melt, temp_df_6_month])
    
        temp_df_year = df_year[df_year.loc[:, 'strategy'] == strategy].iloc[:, 2:].melt(var_name='variable', value_name='profit')
        temp_df_year['strategy'] = strategy
        df_year_melt = pd.concat([df_year_melt, temp_df_year])
    
    # Create subplots with custom heights
    fig = make_subplots(rows=3, cols=1, subplot_titles=['Profit Distribution (3 Months)', 'Profit Distribution (6 Months)', 'Profit Distribution (1 Year)'])
    
    # Add box plots for each time period
    for i, df_melt in enumerate([df_3_month_melt, df_6_month_melt, df_year_melt]):
        for strategy in strategies:
            fig.add_trace(go.Box(x=df_melt[df_melt['strategy'] == strategy]['strategy'],
                                 y=df_melt[df_melt['strategy'] == strategy]['profit'],
                                 name=strategy,
                                 boxpoints='outliers',
                                 jitter=0.3,
                                ), row=i + 1, col=1)
    
        # Center y-axis around zero
        #fig.update_yaxes(range=[-1.2*max(df_melt['profit'].abs()), 1.1*max(df_melt['profit'].abs())], row=i + 1, col=1)
    
    # Customize subplot heights (adjust as needed)
    fig.update_layout(height=1000, showlegend=True, legend_title_text='Strategy')
    
    # Set subplot titles
    fig.update_xaxes(title_text='Strategy', row=3, col=1)
    fig.update_yaxes(title_text='Profit', row=2, col=1)
    
    # Combine legends into one
    fig.update_traces(showlegend=False)
    
    fig.show()

"""
def create_scatter(data,column_name,color,direction):
    if column_name in data.columns:
        df_short = data[data[column_name] == 1]
        if len(df_short) > 0:
            shorts = go.Scatter(
                x=df_short.date,
                y=df_short.close,
                mode='markers',
                name=column_name,
                marker=dict(
                    symbol=f"triangle-{direction}-dot",
                    size=9,
                    line=dict(width=1),
                    color=color,
                )
            )
            return shorts
        else:
            logger.warning(f"No {column_name}-signals found.")

    return None

def create_trades_plot(stats, config):

    strategy_list = list(stats['strategy'].keys())
    pair_list = ['BTC_USDT', 'ADA_USDT', 'BNB_USDT', 'DOGE_USDT', 'ETH_USDT', 'LTC_USDT', 'MATIC_USDT', 'SOL_USDT', 'TRX_USDT', 'XRP_USDT']
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Dropdown(
            id='pair-dropdown',
            options=[{'label': i, 'value': i} for i in pair_list],
            value=pair_list[0]
        ),
        dcc.Dropdown(
            id='strategy-dropdown',
            options=[{'label': i, 'value': i} for i in strategy_list],
            value=strategy_list[0]
        ),
        dcc.Graph(id='candlestick-graph')
    ])
    
    @app.callback(
        Output('candlestick-graph', 'figure'),
        [Input('pair-dropdown', 'value'),
         Input('strategy-dropdown', 'value')]
    )
    def update_graph(selected_pair, selected_strategy):
        # Load the data for the selected pair
        candles = pd.read_feather(f'/home/matias/freqtrade/user_data/data/binance/{selected_pair}-1d.feather')
        trades = pd.DataFrame(stats['strategy'][selected_strategy]['trades'])
        pair_name = str.split(selected_pair, '_')
        pair_name = pair_name[0] + '/' + pair_name[1]
        trades = trades[trades.pair == pair_name]
    
        config["strategy"] = selected_strategy
        config["timeframe"] = "1d"
    
        strategy = StrategyResolver.load_strategy(config)
        candles = candles[candles.date >= pd.to_datetime("2022-03-01").tz_localize("UTC")].reset_index()
        entry_ticks = strategy.analyze_ticker(candles, {'pair': selected_pair})
    
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.0001)
        fig['layout'].update(title=selected_pair)
        fig['layout']['yaxis1'].update(title='Price')
    
        # Candlestick plot
        candles_plot = go.Candlestick(
            x=candles.date,
            open=candles.open,
            high=candles.high,
            low=candles.low,
            close=candles.close,
            name='Price'
        )
        fig.add_trace(candles_plot, 1, 1)
        longs = create_scatter(candles, 'enter_long', 'green', 'up')
        exit_longs = create_scatter(candles, 'exit_long', 'red', 'down')
        shorts = create_scatter(candles, 'enter_short', 'blue', 'down')
        exit_shorts = create_scatter(candles, 'exit_short', 'violet', 'up')
    
        for scatter in [longs, exit_longs, shorts, exit_shorts]:
            if scatter:
                fig.add_trace(scatter, 1, 1)
        # Add trades
        if trades is not None and not trades.empty:
            trades['desc'] = trades.apply(
                lambda row: f"{row['profit_ratio']:.2%}, " +
                            (f"{row['enter_tag']}, " if row['enter_tag'] is not None else "") +
                            f"{row['exit_reason']}, " +
                            f"{row['trade_duration']} min",
                axis=1)
    
            trade_entries = go.Scatter(
                x=trades["open_date"],
                y=trades["open_rate"],
                mode='markers',
                name='Trade entry',
                text=trades["desc"],
                marker=dict(
                    symbol='circle-open',
                    size=11,
                    line=dict(width=2),
                    color='cyan'
                )
            )
    
            trade_exits = go.Scatter(
                x=trades.loc[trades['profit_ratio'] > 0, "close_date"],
                y=trades.loc[trades['profit_ratio'] > 0, "close_rate"],
                text=trades.loc[trades['profit_ratio'] > 0, "desc"],
                mode='markers',
                name='Exit - Profit',
                marker=dict(
                    symbol='square-open',
                    size=11,
                    line=dict(width=2),
                    color='green'
                )
            )
    
            trade_exits_loss = go.Scatter(
                x=trades.loc[trades['profit_ratio'] <= 0, "close_date"],
                y=trades.loc[trades['profit_ratio'] <= 0, "close_rate"],
                text=trades.loc[trades['profit_ratio'] <= 0, "desc"],
                mode='markers',
                name='Exit - Loss',
                marker=dict(
                    symbol='square-open',
                    size=11,
                    line=dict(width=2),
                    color='red'
                )
            )
    
            fig.add_trace(trade_entries, 1, 1)
            fig.add_trace(trade_exits, 1, 1)
            fig.add_trace(trade_exits_loss, 1, 1)
        
        fig.update_layout(height=650) 
        return fig
    app.run_server(debug=True)
"""