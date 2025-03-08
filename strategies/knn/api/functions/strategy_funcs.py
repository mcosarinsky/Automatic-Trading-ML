import numpy as np
import pandas as pd


def process_dataframe(dataframe: pd.DataFrame, lags=14, cols = ['open', 'high', 'low', 'close', 'volume']) -> pd.DataFrame:
    candles = dataframe.set_index('date').loc[:,cols] 
    dfClean = candles.pct_change().dropna() # Calcula variación porcentual 

    # Itera creando DataFrames de ventanas de 14 dias
    dfs_to_concat = list()

    for i in range(1, lags+1):
        df_shifted = dfClean.shift(lags - i + 1)
        df_shifted.columns = [f"{col}_{i}" for col in list(dfClean.columns)]
        dfs_to_concat.append(df_shifted)

    # Lo une en un solo DataFrame  
    result_df = pd.concat(dfs_to_concat, axis=1).dropna()
    return result_df


def triple_barrier(dataframe, SL=None, TP=None, tl=None, min_profit=0.005):
    """
    Run triple-barrier labeling method on dataframe
    """
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
    
    for row_idx in range(len(data) - 1):
        entry_price = entry_prices[row_idx]
        
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
                end_profit = (future_prices[idx] - entry_price) / entry_price  
                hit_barrier = True
                break
                
            if SL and future_lows[idx] <= sl_levels[row_idx]:
                labels.append(0)
                end_date = future_dates[idx]
                end_profit = (future_prices[idx] - entry_price) / entry_price
                hit_barrier = True
                break
        
        # If no barrier was hit, determine the label based on the final price
        if not hit_barrier:
            final_price = future_prices[-1]
            label = 1 if final_price > entry_price * (1 + min_profit) else 0
            labels.append(label)
            end_profit =(final_price - entry_price) / entry_price
        
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
    data = data.drop(columns='hit_barrier')

    return data.reset_index(drop=True)


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


def prepare_data(df, SL=0.025, TP=0.05, tl=48, window_size=28):
    """
    Run labeling method and process dataframe 
    """
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    data_labeled = triple_barrier(df, SL=SL, TP=TP, tl=tl).reset_index(drop=True)
    data_labeled['profit_curado'] = curado_profits(data_labeled)
    data_labeled = data_labeled.drop(columns=['open', 'high', 'low', 'close', 'volume', 'label'])
    
    data_processed = process_dataframe(df, lags=window_size, cols=['open', 'high', 'low', 'close', 'volume'])
    data_processed = data_processed.reset_index().rename(columns={'date': 'startDate'})
    
    profits_df = pd.merge(data_processed, data_labeled, on='startDate', how='inner')
    ensure_utc(profits_df, ['startDate', 'endDate'])
    profits_df = profits_df.sort_values(by='endDate', ascending=False)

    return profits_df

def ensure_utc(df, col_names):
    """
    Ensure the datetime columns are in UTC.
    """
    if isinstance(col_names, str):  # If a single column is passed as a string convert to list for uniform processing
        col_names = [col_names]     
    
    for col_name in col_names:
        df[col_name] = pd.to_datetime(df[col_name])
        if df[col_name].dt.tz is None:
            df[col_name] = df[col_name].dt.tz_localize('UTC')  # Localize to UTC if no timezone
        else:
            df[col_name] = df[col_name].dt.tz_convert('UTC')  