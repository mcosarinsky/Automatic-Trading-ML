# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# --------------------------------
#from funciones_estrategia_Markov import process_dataframe # Sin encoder
from joblib import parallel_backend

###################################################################################################################

import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras import backend as K
from keras.optimizers import Adam

import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib

### FUNCIONES

## Para las matrices de probabilidades
def normalizar_matriz(matriz): # Normalizar por fila
  row_sums = matriz.sum(axis=1, keepdims=True)  # Sumar por filas
  normalized_matriz = matriz / row_sums  # Dividir cada elemento por la suma de la fila
  matriz_normalizada = np.nan_to_num(normalized_matriz, nan=0.0).round(2)
  return matriz_normalizada

def matrices_probas(dataframe,rango_dias_para_atras,rolling_window,step_in_rolling,step_in_map,percentages):
  step_by_rolling = rolling_window-step_in_rolling
  list_closes = list(dataframe.close)#[75:105]
  try:
    list_dates = list(dataframe.date)
  except:
    list_dates = [x for x in range(len(dataframe))]

  matrices_all = []
  dates_all = []
  indice_nuevo = 0
  dataframe_len = len(dataframe.close)
  cant_matrices_mapa_final = (dataframe_len-rolling_window)//step_in_map+1 # +1 ya que se cuenta el primero

  for k in range(cant_matrices_mapa_final): # Aca seria para tener la cantidad de imagenes de la crypto
    #if k%100 ==0: print(k)
    closes = list_closes[step_in_map*k:step_in_map*k+rolling_window] # 20 dias
    fechas_start_finish = list_dates[step_in_map*k:step_in_map*k+rolling_window]
    dates_all.append((fechas_start_finish[0],fechas_start_finish[-1]))
    # Limites y cierres prebucle-2
    matrices = [] #
    matrices_normalized = [] 
    for i in range(step_by_rolling): # 6 (20-14)
      actual_closes = closes[i:i+rolling_window] # Se toman los cierres de hoy a 30 dias
      limites = [actual_closes[0]*(1+per/100) for per in percentages] # Se toma el primer close como referencia a lo que sucedera luego en la transicion
      dim_limites = len(limites)
      # Variables pre-bucle 2
      # Closes
      anterior_close = actual_closes[0] # Se toma el primer close como referencia a lo que sucedera luego en la transicion -> No se usa igual
      actual_close = actual_closes[1] # Del primer close no se empieza, se empieza desde el segundo porque tiene un cambio con respecto a antes( en cambio el primero por ser el primero no)
      # Indexs
      anterior_index = percentages.index(0) # Es la variacion 0 porciento en los limites ya que es ahi donde empezamos, entocnes nunca vario!
      #actual_index: No se pone, hay que buscarlo en el codigo
      matriz = np.zeros((dim_limites,dim_limites)).astype(int) # Matriz transicion # Aca estaran las probabilidades # !!!!!!!!!!!!!!! aca no seria dim_limites-1

      # Bucle 2
      for d in range(1,step_in_rolling): # d de dias -> hasta el dia 14 en nuestro caso
        actual_close = actual_closes[d]
        ######print(actual_close)
        limite = anterior_index
        if anterior_index == 0:
          lim_inferior = -math.inf # Inferior
          lim_superior = limites[1]
        elif anterior_index == dim_limites-1:
          lim_inferior = limites[-2]
          lim_superior = +math.inf
        else:
          lim_inferior = limites[limite-1]
          lim_superior = limites[limite+1]

        # Si la posicion actual se encuentra entre los limites de los nodos superior e inferior anterior nos quedamos con el actual nodo!
        if lim_inferior<actual_close<lim_superior:
          indice_nuevo = anterior_index

        else: # Caso contrario es porque paso la barrera inferior o la superior
          if actual_close > limites[-1]: indice_nuevo = dim_limites-1
          elif actual_close < limites[0]: indice_nuevo = 0
          else:
            indice_nuevo = anterior_index
            if actual_close > limites[anterior_index]:
              while(actual_close > limites[indice_nuevo]):
                indice_nuevo+=1
              indice_nuevo=indice_nuevo-1
            elif actual_close < limites[anterior_index]:
              while(actual_close < limites[indice_nuevo]):
                indice_nuevo-=1
              indice_nuevo=indice_nuevo+1
        actual_index = indice_nuevo
        # Sumamos uno en la transicion del nodo al que le sigue en la matriz
        matriz[anterior_index,actual_index] +=1
        anterior_index = actual_index
      # Guardamos las matrices tanto normalizadas como no
      matrices.append(matriz)
      #matrices_normalized.append(normalizar_matriz(matriz))
    # Guardamos el conjunto de matrices dentro de ese mes
    matrices_all.append(matrices)

  #Matrices unidas
  matrices_all_united_normalizada = []
  matrices_all_united = []
  fechas = []
  for i in range(len(matrices_all)):
    dates_actual_start_finish = dates_all[i]
    try:
      start = dates_actual_start_finish[0].strftime('%Y/%m/%d (%H')+" hs)"
      finish = dates_actual_start_finish[1].strftime('%Y/%m/%d (%H')+" hs)"
    except:
      start = dates_actual_start_finish[0]
      finish = dates_actual_start_finish[1]
    fechas.append(str(start)+"\nto\n"+str(finish))
    matriz_suma_pack = sum(matrices_all[i])
    ########matriz_normalizada_united = normalizar_matriz(matriz_suma_pack)
    ########matrices_all_united_normalizada.append(matriz_normalizada_united)
    #matrices_all_united.append(matriz_suma_pack) # ESTA ES LA ORIGINAL
    # Esta es la que voy a usar para el encoder decoder, se divide por la cantidad de elementos totales en la matriz
    matrices_all_united.append(np.nan_to_num(np.round(matriz_suma_pack/sum(matriz_suma_pack.flatten()),2),0))

  return matrices_all_united,matrices_all_united_normalizada,fechas
  

scaler = StandardScaler()


###################################################################################################################

### Estrategia
class Markov(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3
    # Optimal timeframe for the strategy.
    timeframe = '1h'
    # Can this strategy go short?
    can_short: bool = False
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 1000
    }

    stoploss = -1

    # Trailing stoploss
    trailing_stop = False
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        nombre = "svm_model_2.pkl"
        self.modelo_cargado = joblib.load('dev/crypto_trading/modelos/clasificadores/' + nombre)
        self.loaded_encoder = load_model('dev/crypto_trading/modelos/clasificadores/encoder_model.keras')

    def process_dataframe(self, dataframe: pd.DataFrame, umbral) -> pd.DataFrame:
        # Deje el volumen porque mis datos los entrene con esa columna
        # umbral = 0.3365
        cols = ['open', 'high', 'low', 'close', "volume"]
        dataframe = dataframe.loc[:, cols]

        codificado_all = []
        predicciones = []
        # variables
        rango_dias_para_atras = 29  # 32 # multiplo de 3 si step in map es 3
        percentages = [-3.0, -1.5, -0.5, 0, 0.5, 1.5, 3.0]
        rolling_window = 20  # 1 mes o 30 dias # defecto 30
        step_in_rolling = 17  # #como se va moviendo en ese rolling_window # defecto 14 # 18 era el normal
        step_in_map = 3  # Se va moviendo cada 15 dias el analisis general #defecto 15

        matrices_all = []

        # Aca aplicamos la funcion de markov para luego hacer las predicciones
        for indice in range(len(dataframe) - 29):  # ya que lo hacemos solo una vez #saque el mas 1 porque en el dataframe no aparece la ultima hora
            # if indice % 100: print(indice)
            dataframe_parcial = dataframe.iloc[indice:indice + 29].copy()  # 32 a 2
            matrices_all_united, _, _ = matrices_probas(dataframe_parcial, rango_dias_para_atras, rolling_window, step_in_rolling, step_in_map, percentages)  # Solo se usa matrices_all_united
            matrices_all.append(matrices_all_united)

        matrices_all = np.array(matrices_all).reshape(-1, 7, 7)  # llevamos a dimensiones de input del encoder
        codificado = np.round(self.loaded_encoder.predict(matrices_all), 3).reshape(-1, 4 * 10)  # llevamos a dimensiones de input del modelo
        probas = self.modelo_cargado.predict_proba(codificado)[:, 1]
        predicciones = probas > umbral

        return predicciones

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        umbral = 0.3365 #umbral = 0.3365 o 0.36
        predicciones = self.process_dataframe(dataframe,umbral)
        windows_size = 29 
        dataframe.loc[windows_size:,'enter_long'] = predicciones

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str:
        """
        Custom exit logic. Defines custom exit signal based on the optimized take profit.
        """
        # Esto para la optimizacion de parametros
        #with parallel_backend('loky', n_jobs=1):
        if current_profit >= 0.04:#self.take_profit:
            return "TP"
        # 
        if current_profit <= -0.04:#self.exit_loss:
            return "SL"

        return None
    
 