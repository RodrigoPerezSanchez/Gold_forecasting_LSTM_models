# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:55:58 2023

@author: Rodrigo Pérez Sánchez
"""

import pandas as pd
import pathlib
import datetime
import numpy as np
from plotnine import *


df_indicadores_2 = pd.read_csv("indicadores_cap_1_v3_DosMes.csv")
df_indicadores_2
df_indicadores_2.columns
df_indicadores_2 = df_indicadores_2.reset_index()


# representamos los indicadores a partir del csv creado en el fichero "Data_Base_implementation_cap_1"

ggplot(aes(x="index", y="Close_crude"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Crude Oil") + xlab("Tiempo - Minutos")
ggplot(aes(x="index", y="Close_bitcoin"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Bitcoin")+ xlab("Tiempo - Minutos") + ylab("Precio (USD)")
ggplot(aes(x="index", y="Close_usd"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("USD")+ xlab("Tiempo - Minutos") + ylab("Precio (USD)")
ggplot(aes(x="index", y="Close_sp"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("S&P500")+ xlab("Tiempo - Minutos")
ggplot(aes(x="index", y="Close_oro"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Oro")+ xlab("Tiempo - Minutos") + ylab("Precio (USD/onza)")

ggplot(aes(x="index", y="Close_ausdolar"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Dolar Australiano")+ xlab("Tiempo - Minutos")
ggplot(aes(x="index", y="Close_plata"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Plata")+ xlab("Tiempo - Minutos") + ylab("Precio (USD/onza)")
ggplot(aes(x="index", y="Close_dowjones"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Dow Jones 30")+ xlab("Tiempo - Minutos")
ggplot(aes(x="index", y="Close_nasdaq"),df_indicadores_2) + geom_line(size=0.2) + ggtitle("Nasdaq 100")+ xlab("Tiempo - Minutos")



##################################################### CORRELACION ENTRE INDICADORES ##################################################

#%%
"""Análisis de linealidad"""

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
fig = plt.figure()
scatter_matrix(df_indicadores_2.iloc[:,2:],figsize =(25,25),alpha=0.9,diagonal="kde",marker="o");


#%%
"""Análisis de normalidad => shapiro, jarque_bera, ks_test"""

# debe pasar por los menos uno de los test para considerar la variable normalmente distribuida

"""Shapiro"""

from scipy import stats
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
for item in df_indicadores_2.columns[2:]:
    statistic, pval = stats.shapiro(df_indicadores_2[item])
    if pval > 0.05:
        print("{} is normal distributed, as we failt to reject the null hyposthesis for having a {:.4f} p-value".
              format(item, pval))
    else:
        print(item+" is "+color.BOLD + 'NOT' + color.END+" normal distributed, as the pvalue is "+"{:.4f}".
              format(pval))
        
"""Jarque_bera"""     

for item in df_indicadores_2.columns[2:]:
    statistic, pval = stats.jarque_bera(df_indicadores_2[item])
    if pval > 0.05:
        print("{} is normal distributed, as we failt to reject the null hyposthesis for having a {:.4f} p-value".
              format(item, pval))
    else:
        print(item+" is "+color.BOLD + 'NOT' + color.END+" normal distributed, as the pvalue is "+"{:.4f}".
              format(pval)) 
        
"""Ks_test"""

for item in df_indicadores_2.columns[2:]:
    statistic, pval = stats.kstest(df_indicadores_2[item], cdf='norm')
    if pval > 0.05:
        print("{} is normal distributed, as we failt to reject the null hyposthesis for having a {:.4f} p-value".
              format(item, pval))
    else:
        print(item+" is "+color.BOLD + 'NOT' + color.END+" normal distributed, as the pvalue is "+"{:.4f}".
              format(pval))     

# No son normalizadas, habrá que normalizar las variables posteriormente en la implementación del modelo.
        
#%%        
        
"""Análisis de correlación => heatmap, pearson"""

"""heatmap con seaborn"""

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import stats

df_indicadores_2.columns[2:]

import seaborn as sns
sns.set(context="paper",font="monospace")
df_corr_matrix = df_indicadores_2[['Close_crude', 'Close_bitcoin', 'Close_usd', 'Close_sp', 'Close_oro','Close_ausdolar', 'Close_plata','Close_dowjones', 'Close_nasdaq']].corr()
# figura matplotlib
fig, axe = plt.subplots(figsize=(12,8))
# paleta de colores
cmap = sns.diverging_palette(220,10,center = "light", as_cmap=True)
# heatmap dibujo
sns.heatmap(df_corr_matrix,vmax=1,square =True, cmap=cmap,annot=True );  
        
"""pearson"""       
for var1 in ['Close_crude', 'Close_bitcoin', 'Close_usd', 'Close_sp', 'Close_oro','Close_ausdolar', 'Close_plata','Close_dowjones', 'Close_nasdaq']:
        for var2 in ['Close_crude', 'Close_bitcoin', 'Close_usd', 'Close_sp', 'Close_oro','Close_ausdolar', 'Close_plata','Close_dowjones', 'Close_nasdaq']:
            zscore, pvalue =stats.pearsonr(df_indicadores_2[var1],df_indicadores_2[var2])
            if(pvalue<0.05):
                print(var1,var2,"<-CORRELATED")
                print("-"*50)
            else:
                print(var1,var2)
            print("-"*100)        
        

"""
vamos a tomar los indicadores = dolar, bitcoin, plata, oro
"""

"""
NEXT STEPS: depende de los meses que estudiemos, las correlaciones cambian y habria que estudiar para cada modelo
diferentes franjas temporales y ver las 3,4 o 5 variables mas correlacionadas.
Para un caso practico de empresa, (crear una aplicacion de trading automatica) se estudiaria una vez descargada 
la base de datos, las variables mas correlacionadas, aplicarar el modelo, etc.
"""




