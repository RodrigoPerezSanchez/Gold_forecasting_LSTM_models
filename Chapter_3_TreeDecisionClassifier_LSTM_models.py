# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:07:21 2023

@author: perez
"""

import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import numpy as np
import pathlib
import datetime
from plotnine import *


# implementamos los csv creados en los archivos del "Chapter_2" para su posterior análisis en el modelo predictivo de clasificación

df1 = pd.read_csv("optimization_stacked_lstm.csv",index_col=0)
df1.dtypes

df2 = pd.read_csv("optimization_bidirectional_lstm.csv",index_col=0)
df2.dtypes

df3 = pd.read_csv("optimization_vanilla_lstm.csv",index_col=0)
df3.dtypes

df_aux = pd.concat([df1,df2,df3])
df_aux.isnull().sum()

# creamos una columna "Good" para definir los resultados buenos de predicción (basandose en unos rangos impuestos de tiempo y RMSE) 

df_aux["Optimizador"].value_counts()
df_aux["Optimizador"] = df_aux["Optimizador"].replace({"<class 'keras.optimizers.legacy.adam.Adam'>":"Adam","<class 'keras.optimizers.legacy.rmsprop.RMSprop'>":"RMSprop","<class 'keras.optimizers.legacy.gradient_descent.SGD'>":"SGD"})
df_aux["Optimizador"].value_counts()
df_aux["RMSE"] = df_aux["RMSE"].round(3)
df_aux["MAPE"] = df_aux["MAPE"].round(3)
df_aux = df_aux.reset_index(drop=True)

df_best = df_aux.loc[(df_aux["RMSE"] <= 10.000) & (df_aux["Time"] <= 60.00),:] #!!!!
list_best = df_best.index.to_list()
df_best["Class"].value_counts()


df_aux["Good"] = 0
for i in list_best:
    df_aux["Good"][i] = 1  
df_aux["Good"].value_counts()
df_aux.dtypes


#####################################

# guardamos los mejores resultados en archivos de excel por separado cada modelo

df_prueba = df_aux.copy()
df_prueba_V = df_prueba.loc[(df_prueba.Class == "Vanilla") & (df_prueba.Good == 1),:]
df_prueba_B = df_prueba.loc[(df_prueba.Class == "Bidirectional") & (df_prueba.Good == 1),:]
df_prueba_S = df_prueba.loc[(df_prueba.Class == "Stacked") & (df_prueba.Good == 1),:]

df_prueba_V.sort_values(by=["RMSE"],inplace=True)
df_prueba_B.sort_values(by=["RMSE"],inplace=True)
df_prueba_S.sort_values(by=["RMSE"],inplace=True)

df_prueba_V.to_excel("vanilla_resultados.xlsx")
df_prueba_B.to_excel("bidirectional_resultados.xlsx")
df_prueba_S.to_excel("stacked_resultados.xlsx")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from sklearn import datasets
from scipy.cluster import hierarchy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#######################################################################

""" VARIABLE CATEGÓRICA A NUMÉRICA """
# estos modelos no funcionan correctamente con variables categóticas, por ello debemos hacer una transformación a variable numéricas 
# esta tranformación puede afectar a modelos basados en distancias para el cálculo del error. Para este estduio no influye este inconveniente.

from sklearn.preprocessing import LabelEncoder

leb1 = LabelEncoder()
leb1.fit(df_aux.Class)
df_aux.Class = leb1.transform(df_aux.Class)
# Stacked = 1
# Bidirectional = 0
# Vanilla = 2
leb2 = LabelEncoder()
leb2.fit(df_aux.Optimizador)
df_aux.Optimizador = leb2.transform(df_aux.Optimizador)
# Adam = 0
# SGD = 2
# RMSprop = 1
leb3 = LabelEncoder()
leb3.fit(df_aux.Activation)
df_aux.Activation = leb3.transform(df_aux.Activation)
# relu = 0
# tanh = 1
leb4 = LabelEncoder()
leb4.fit(df_aux.Loss)
df_aux.Loss = leb4.transform(df_aux.Loss)
# mse = 1
# mae = 0
df_aux.head()
df_aux.dtypes
df_aux.columns

# df_aux.Class = leb1.inverse_transform(df_aux.Class) # para recuperar las variables originales

######################################################################

""" APLICAMOS EL MODELO DE ÁRBOL DE DECISIÓN """

X = df_aux.loc[:,['Class', 'Split', 'Optimizador', 'Learning_Rate', 'Epochs','Activation', 'Batch', 'Loss']] #!!!!
Y = df_aux.loc[:,['Good']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=42)

# Ejecucción de modelo
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(max_depth = 3,class_weight="balanced") # podemos modificar parámetros y con balanced tomamos valores 1 con mayor peso
tree_model = tree_model.fit(X_train, Y_train)

tree_model.feature_importances_

predicciones = tree_model.predict(X_test)
confusion_matrix(Y_test, predicciones)
tree_model.classes_  # las columnas son las predicciones (precision) y las filas el caso real (recall)
#tree_model.predict_proba(X_test)


tree_model.score(X_test,Y_test)
importancia = pd.DataFrame({"variable":X_train.columns,"importancia":tree_model.feature_importances_})
ggplot(importancia,aes(x="variable",y="importancia")) + geom_bar(stat="identity")
# Observamos las variables más influyentes

##################

# Visualización de importancia de valores de variables a la hora predecir resultados idóneos 
import graphviz
dot_data = tree.export_graphviz(
    tree_model,
    out_file=None,
    feature_names=X.columns.values,
    # class_names = Y.iloc[:,0].values.astype(str),
    filled=True, rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)
graph

# graph.render(Name for the PDF file) to save the tree in a document



