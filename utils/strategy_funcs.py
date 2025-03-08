import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from yahooquery import Ticker
import ta
import faiss

# Modelo FCN
class FCN_custom(nn.Module):
    def __init__(self, input_dim, output_dims: List[int]=[128, 256, 128], kernel_sizes: List[int]=[8,5,3]):
        """
        Initialize the FCN model.

        Args:
            input_dim: The input dimension of the data.
            output_dims: A list of output dimensions for each convolutional layer.
            kernel_sizes: A list of kernel sizes for each convolutional layer.
        """
        super(FCN_custom, self).__init__()
        torch.manual_seed(42) 
        self.layers = nn.ModuleList()
        for output_dim, kernel_size in zip(output_dims, kernel_sizes):
            self.layers.append(nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size),
                nn.BatchNorm1d(output_dim)
            ))
            input_dim = output_dim
        self.fc = nn.Linear(output_dims[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = F.avg_pool1d(x, x.shape[2]).squeeze(2)  # Global Average Pooling
        x = self.fc(x)
        return x.squeeze(1)

FEAR_ADN_GREED_API = 'https://api.alternative.me/fng/?limit=0'


# call the Fear and Greed API 
def create_fng_dataset(url=FEAR_ADN_GREED_API) -> pd.DataFrame:
    """
    Function to create the Fear and Greed Dataset
    """
    response = requests.get(url)
    df = pd.DataFrame(response.json()['data'])
    df['value'] = df['value'].astype(int) /100 
    df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
    df['FnG'] = df['value']
    df = df.drop(columns=['value','time_until_update','timestamp','value_classification'])
    df = df[::-1]
    return df

# ------------------------------------------------------------------------------------------
# Function with the call of the Fear and Greed API
# ------------------------------------------------------------------------------------------

def process_dataframe_fng(df:pd.DataFrame, lags:int=14):
    """
    Function to transform a DataFrame for training a Machine Learning Model 
    creating lag Features 
    parameters:
        - df -> DataFrame 
        - lags -> number of lags you want
    """
    df_original = df.copy()
    df_original['date'] = pd.to_datetime(df_original['date'])

    #create and merge the Fear and Greed dataset
    df_fng = create_fng_dataset()
    merged_df = df_original.merge(df_fng, on='date')
    merged_df = merged_df.loc[:, ['open', 'high', 'low', 'close', 'FnG']]
    merged_df = merged_df.pct_change().dropna()
    
    #create list to concat lags dataframes
    dfs_to_concat = list()

    for i in range(lags,0,-1):
        df_shifted = merged_df.shift(i)
        df_shifted.columns = [f"{col}_{lags-i+1}" for col in list(merged_df.columns)]
        dfs_to_concat.append(df_shifted)

    result_df = pd.concat(dfs_to_concat, axis=1).dropna()  
    return np.array(result_df)


# ------------------------------------------------------------------------------------------
# Function without the call of the Fear and Greed API
# ------------------------------------------------------------------------------------------


def process_dataframe(dataframe: pd.DataFrame, lags=14, apply_pct_change=True, cols=['open', 'high', 'low', 'close']) -> pd.DataFrame:
    """
    Function to transform a DataFrame for training a Machine Learning Model 
    creating lag Features
    """
    # Indexa por fecha
    dfClean = dataframe
    try: 
        dfClean = dfClean.set_index('date')
        #dfClean.index = dfClean.index.tz_localize(None)
    except:
        pass
    
    if apply_pct_change : 
        dfClean = dfClean.loc[:,cols].pct_change().dropna() # Calcula variación porcentual 
    else:
        dfClean = dfClean.loc[:,cols].dropna()

    # Itera creando DataFrames de ventanas de 14 dias
    dfs_to_concat = list()
  
    for i in range(lags,0,-1):   
        df_shifted = dfClean.shift(i)
        df_shifted.columns = [f"{col}_{lags-i+1}" for col in list(dfClean.columns)]
        dfs_to_concat.append(df_shifted)

    # Lo une en un solo DataFrame  
    result_df = pd.concat(dfs_to_concat, axis=1).dropna()  
    return result_df


def process_dataframe_ind(df, indicators=True, lags=14):
    df_copy = df.copy()
    if indicators: add_indicators(df_copy)
    
    df_copy.drop(columns=['emaSlow', 'emaFast'], inplace=True)
    numeric = df_copy.select_dtypes('number')
    dfs_to_concat = list()
  
    for i in range(1, lags + 1):     
        df_shifted = numeric.shift(lags - i + 1)
        df_shifted.columns = [f"{col}_{i}" for col in list(numeric.columns)]
        dfs_to_concat.append(df_shifted)

    res = pd.concat(dfs_to_concat, axis=1).dropna()

    return res

def add_indicators(df):
    df['rsi'] = ta.momentum.rsi(df.close)/100
    df['stoch'] = ta.momentum.stoch(df.high, df.low, df.close)/100
    df['adx'] = ta.trend.adx(df.high, df.low, df.close)/100
    df['macd'] = ta.trend.macd(df.close)/ta.trend.ema_indicator(df.close, window=26)  # macd = (ema12 - ema26)/ema26
    #df['adi'] = ta.volume.acc_dist_index(df.high, df.low, df.close, df.volume)  # adi = adi_prev + volume*CLV
    #df['adi'] = df['adi'].pct_change()
    df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    
    return df

def get_ticker(start, end, symbol):    
    # Traigo algunos dias extra por si las fechas caen en dia no habil
    end += pd.Timedelta(days=5)
    start -= pd.Timedelta(days=5)

    ticker = Ticker(symbol, asynchronous=True)
    data = ticker.history(start=start, end=end).loc[:,['close']]
    df = data.reset_index(level='symbol', drop=True)
    df.index = pd.to_datetime(df.index)

    # Llenamos los valores faltantes con el del anterior dia disponible
    df = df.resample('D').asfreq().ffill()
    return df.add_suffix(f'_{symbol}')

def get_full_dataframe(dataframe, symbols, lags=14, cols=['open', 'high', 'low', 'close']):
    """
    Function to add symbols from given list to the default dataframe
    """
    # Traigo los datos de cada simbolo
    start = dataframe.date.iloc[0]
    end = dataframe.date.iloc[-1]
    full_df = process_dataframe(dataframe, lags=lags, cols=cols)

    for symbol in symbols:
        symbol_df = get_ticker(start, end, symbol) # Traemos los datos de yahoofinance
        symbol_df_lags = process_dataframe(symbol_df, lags=lags, cols=[f"close_{symbol}"]) # Procesamos los datos en formato de ventanas
        full_df = full_df.join(symbol_df_lags, how='inner') 

    return full_df


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