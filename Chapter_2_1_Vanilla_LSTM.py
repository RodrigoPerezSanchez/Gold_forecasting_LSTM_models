# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:46:44 2023

@author: Rodrigo Pérez Sánchez
"""

#%%
from matplotlib import pyplot
import pandas as pd
import pathlib
import datetime
import numpy as np
from plotnine import *


# cargamos el csv una vez comprobadas las correlaciones en el fichero "Chapter_1_2_Correlations_and_graphs"
dataset = pd.read_csv("../Capitulo_1_indicadores/indicadores_cap_1_v3_DosMes.csv")
dataset = dataset.set_index("Time")
values = dataset.values

groups = [0, 1, 2, 3, 4, 5, 6, 7, 8]
i = 1

# representar cada columna
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

dataset.columns
dataset = dataset.reindex(columns=['Close_crude', 'Close_bitcoin', 'Close_usd', 'Close_sp','Close_ausdolar', 'Close_plata', 'Close_dowjones', 'Close_nasdaq', 'Close_oro'])
dataset.drop(['Close_ausdolar','Close_sp','Close_crude','Close_dowjones', 'Close_nasdaq'],axis=1,inplace=True) #!!!! que variables se van a usar (extraemos las que no necesitamos)

# representamos únicamente los gráficos de las varibales que vamos a usar

df_indicadores_2 = dataset.reset_index()
df_indicadores_2 = df_indicadores_2.reset_index()
#ggplot(aes(x="index", y="Close_crude"),df_indicadores_2) + geom_line() + ggtitle("Crude Oil")
ggplot(aes(x="index", y="Close_bitcoin"),df_indicadores_2) + geom_line() + ggtitle("Bitcoin")
ggplot(aes(x="index", y="Close_usd"),df_indicadores_2) + geom_line() + ggtitle("USD")
#ggplot(aes(x="index", y="Close_sp"),df_indicadores_2) + geom_line() + ggtitle("S&P500")
ggplot(aes(x="index", y="Close_oro"),df_indicadores_2) + geom_line() + ggtitle("Oro")

#ggplot(aes(x="index", y="Close_ausdolar"),df_indicadores_2) + geom_line() + ggtitle("Dolar Australiano")
ggplot(aes(x="index", y="Close_plata"),df_indicadores_2) + geom_line() + ggtitle("Plata")
#ggplot(aes(x="index", y="Close_dowjones"),df_indicadores_2) + geom_line() + ggtitle("Dow Jones 30")
#ggplot(aes(x="index", y="Close_nasdaq"),df_indicadores_2) + geom_line() + ggtitle("Nasdaq 100")

#%% #####################################################################

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
#from pandas import read_csv
#from pandas import DataFrame
#from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import SGD
 
# convertir las series a SUPERVISED LEARNING (explicacion teórica de la necesidad de usarlo)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True): #n_in=1, n_out=1
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
#%% ######################################################################

values = dataset.values # transformar a una matriz de arrays
values = values.astype('float32') # comprobar que son float

scaler = MinMaxScaler(feature_range=(0, 1)) #escalamos las variables para que esten en un rango de 0 a 1
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1) #aplicamos supervised learning (lagging the data)
reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True) # quitamos las columnas que no usamos #!!!!
print(reframed.head())

# en este momento, tenemos un dataset con los valores anteriores del pasado (un minuto antes), para estudiarlo con el
# valor real de tiempo del oro(t) en train_y, el unico sin (t-1). Muy recomendado para series temporales

#%% ######################################################################

# aplicamos el train y test del dataframe para probar el modelo (SPLIT)
values = reframed.values
n_train_time = int(reframed.shape[0]*0.80) # 75% para entrenamiento #!!!!
train = values[:n_train_time, :]
test = values[n_train_time:, :]

# split entradas X y salida Y (el precio del oro)
train_X, train_y = train[:, 0:-1], train[:, -1] # tomamos(-1) o no () el propio valor del oro en train
test_X, test_y = test[:, 0:-1], test[:, -1] # tomamos(-1) o no () el propio valor del oro en test

# reshape entrada para 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])) # aqui modificamos los timesteps 
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1])) # aqui modificamos los timesteps 
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



#%% ######################################################################

"""DISEÑO DEL MODELO => Stacked LSTM """

model = Sequential()
model.add(LSTM(50, activation='tanh',input_shape=(train_X.shape[1], train_X.shape[2]))) 
model.add(Dense(1))
optimizer = Adam(learning_rate = 0.0001) #0.01-0.0001 #!!!!
model.compile(loss ='mse', optimizer = optimizer) #cambiar parametros loss="mae"/"mse" #!!!

# una vez ajustado los parámetros, aplicamos el modelo a los datos
history = model.fit(train_X, train_y, epochs=100, batch_size=128, validation_data=(test_X, test_y), verbose=1, shuffle=False)

# plot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


#%% #####################################################################

""" Evaluación del modelo RMSE y MAPE"""


yhat = model.predict(test_X)                                 
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invertimos la escala para hacer el pronóstico      
inv_yhat = concatenate((test_X[:, :-1], yhat), axis=1) 
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

# invertimos la escala para los valores reales
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :-1], test_y), axis=1) 
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


########################################

import numpy as np
def calculate_mape(actual, predicted):
    """Calcula el error de porcentaje absoluto medio (MAPE) entre dos series temporales. """
    actual = np.array(actual)
    predicted = np.array(predicted)
    # Calcula el porcentaje de error absoluto para cada punto
    errors = np.abs((actual - predicted) / actual)
    # Ignora los casos en los que el valor real sea cero para evitar divisiones por cero
    errors = errors[actual != 0]
    # Calcula el MAPE promediando los errores
    mape = np.mean(errors) * 100
    return mape


mape = calculate_mape(inv_y, inv_yhat)
print("Test MAPE:", mape)


##########################################

# hacemos una representación del valor real del oro test y su pronóstico con ggplot

df_resultado1 = pd.DataFrame()
df_resultado1["prediction"] = inv_yhat
df_resultado1.reset_index(inplace=True)

df_resultado2 = pd.DataFrame()
df_resultado2["real"] = inv_y
df_resultado2.reset_index(inplace=True)

df = pd.DataFrame()
df["prediction"] = inv_yhat
df["real"] = inv_y
df.reset_index(inplace=True)
df = df.rename(columns={"index": 'index1'})
df.reset_index(inplace=True)
df = df.rename(columns={"index": 'index2'})



columna_valores = pd.concat([df["prediction"],df["real"]],ignore_index=True)
columna_minutos = pd.concat([df["index1"],df["index2"]],ignore_index=True)

df_final = pd.DataFrame({"valores":columna_valores,"minutos_test":columna_minutos})

df_final["Tipo"] = "Real"
df_final.iloc[:12191,2] = "Predicción"


ggplot(aes(x="minutos_test", y="valores"),df_final) + geom_line(aes(color = "Tipo"), size = 1) +\
       xlab("Minutos Test") + ylab("Precio Oro") + ggtitle("Vanilla LSTM - Gold")


###########################################################################################
##########  CONSTRUCCIÓN DEL CSV - VANILLA LSTM - CON HIPERPARÁMETROS #####################
###########################################################################################

       
import pandas as pd
import itertools
import time

values = dataset.values # transformar a una matriz de arrays
values = values.astype('float32') # comprobar que son float

scaler = MinMaxScaler(feature_range=(0, 1)) #escalamos las variables para que esten en un rango de 0 a 1
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1) #aplicamos supervised learning (lagging the data)
reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True) # quitamos las columnas que no usamos 
print(reframed.head())

###################################

list_split = [75,80]
list_opt = [Adam, RMSprop, SGD] 
list_lr = [0.01,0.0001]
list_epochs = [20, 50, 100]
list_activation = ["relu","tanh"]
list_batch = [32,72,128]
list_loss = ["mae","mse"]

# Crear una lista para almacenar los resultados parciales
results = []

# Generar todas las combinaciones posibles
combinations = list(itertools.product(list_split, list_opt, list_lr, list_epochs, list_activation, list_batch, list_loss))

# Iterar sobre las combinaciones
for combo in combinations:
    
        # Iniciar el temporizador
        start_time = time.time()
        
        values = reframed.values
        n_train_time = int(reframed.shape[0]*(combo[0]*0.01)) #!!!!
        train = values[:n_train_time, :]
        test = values[n_train_time:, :]
        train_X, train_y = train[:, 0:-1], train[:, -1]
        test_X, test_y = test[:, 0:-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
                   
        model = Sequential()
        model.add(LSTM(50, activation=combo[4], input_shape=(train_X.shape[1], train_X.shape[2]))) #!!!!
        # model.add(LSTM(50, activation=combo[4]))#!!!!
        model.add(Dense(1))
        optimizer = combo[1](learning_rate = combo[2]) #!!!!
        model.compile(loss = combo[6], optimizer = optimizer)
        model.fit(train_X, train_y, epochs=combo[3], batch_size=combo[5], validation_data=(test_X, test_y), verbose=1, shuffle=False) #!!!

        yhat = model.predict(test_X)                                
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))                               
        inv_yhat = concatenate((test_X[:, :-1], yhat), axis=1) 
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,-1]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_X[:, :-1], test_y), axis=1) 
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,-1]

        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        # print('Test RMSE: %.3f' % rmse)
        mape = calculate_mape(inv_y, inv_yhat)
        # print("Test MAPE:", mape)
        
        #time per cycle
        elapsed_time = time.time() - start_time
        
        result = {
            "Class": "Vanilla",
            "Split": combo[0],
            "Optimizador": combo[1],
            "Learning_Rate": combo[2],
            "Epochs": combo[3],
            "Activation": combo[4],
            "Batch": combo[5],
            "Loss": combo[6],
            "RMSE": rmse,
            "MAPE": mape,
            "Time": round(elapsed_time,2)
        }
        results.append(result)

# Construir el DataFrame final
df3 = pd.DataFrame(results) 


df3.to_csv("optimization_vanilla_lstm.csv")


"""PROBLEMAS FUTUROS"""
# investigar sobre GridsearchCV (puede ser más rapido)






