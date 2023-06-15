# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:11:13 2023

@author: Rodrigo Pérez Sánchez
"""

import pandas as pd
import pathlib
import datetime
import numpy as np
from plotnine import  *
import json
import requests


# Una vez implementadas las librerías, vamos aextraer los datos de los csv descargados de internet para su prepocesamiento y adapatación a los modelos.

#%%
################################################ CRUDE OIL ############################################

# importamos el data frame y lo ajustamos 

crude_oil_columns = ["Time","Open","High","Low","Close","Volume"]
df_crude_oil = pd.read_csv("indicadores_csv/BRENTCMDUSD1_crudo.csv", sep="\t",header=None,names=crude_oil_columns)
df_crude_oil.dtypes

df_crude_oil["Time"] = df_crude_oil["Time"].astype("datetime64[ns]")
df_crude_oil
df_crude_oil = df_crude_oil.set_index("Time")

# tomamos dia dos meses completos -> valores desde:
df_crude_oil_17 = df_crude_oil.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_crude_oil_17[pd.isna(df_crude_oil_17[n])])
    
df_crude_oil_17 = df_crude_oil_17.drop(["Open","High","Low","Volume"],axis=1)   
df_crude_oil_17.info()
df_crude_oil_17 = df_crude_oil_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_crude_oil_17) + geom_line()

#%%
################################################ BITCOIN DOLLAR ############################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_bitcoin = pd.read_csv("indicadores_csv/BTCUSD1_Bitcoin.csv", sep="\t",header=None,names=columns)
df_bitcoin.dtypes


df_bitcoin["Time"] = df_bitcoin["Time"].astype("datetime64[ns]")
df_bitcoin
df_bitcoin = df_bitcoin.set_index("Time")

# tomamos dia dos meses completos -> valores desde:
df_bitcoin_17 = df_bitcoin.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_bitcoin_17[pd.isna(df_bitcoin_17[n])])
  
df_bitcoin_17 = df_bitcoin_17.drop(["Open","High","Low","Volume"],axis=1)
df_bitcoin_17.info()
df_bitcoin_17 = df_bitcoin_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_bitcoin_17) + geom_line()


#%%
################################################ VALOR DOLAR-EURO ############################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_usd = pd.read_csv("indicadores_csv/EURUSD1_EuroDolar.csv", sep="\t",header=None,names=columns)
df_usd.dtypes

df_usd["Time"] = df_usd["Time"].astype("datetime64[ns]")
df_usd
df_usd = df_usd.set_index("Time")


# tomamos dia dos meses completos -> valores desde:
df_usd_17 = df_usd.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_usd_17

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_usd_17[pd.isna(df_usd_17[n])])
         
df_usd_17 = df_usd_17.drop(["Open","High","Low","Volume"],axis=1)
df_usd_17.info()
df_usd_17 = df_usd_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_usd_17) + geom_line()


#%%
################################################ VALOR S_and_P_500 ############################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_sp500 = pd.read_csv("indicadores_csv/USA500IDXUSD1_sp500.csv", sep="\t",header=None,names=columns)
df_sp500.dtypes

df_sp500["Time"] = df_sp500["Time"].astype("datetime64[ns]")
df_sp500
df_sp500 = df_sp500.set_index("Time")


# tomamos dia dos meses completos -> valores desde:
df_sp500_17 = df_sp500.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_sp500_17

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_sp500_17[pd.isna(df_sp500_17[n])])


df_sp500_17 = df_sp500_17.drop(["Open","High","Low","Volume"],axis=1)
df_sp500_17.info()
df_sp500_17 = df_sp500_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_sp500_17) + geom_line()


#%%
################################################ VALOR nasdaq100 ############################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_nasdaq = pd.read_csv("indicadores_csv/USATECHIDXUSD1_nasdaq100.csv", sep="\t",header=None,names=columns)
df_nasdaq.dtypes

df_nasdaq["Time"] = df_nasdaq["Time"].astype("datetime64[ns]")
df_nasdaq
df_nasdaq = df_nasdaq.set_index("Time")

# tomamos dia dos meses completos -> valores desde:
df_nasdaq_17 = df_nasdaq.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_nasdaq_17
  
# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_nasdaq_17[pd.isna(df_nasdaq_17[n])])

    
df_nasdaq_17 = df_nasdaq_17.drop(["Open","High","Low","Volume"],axis=1)
df_nasdaq_17.info()
df_nasdaq_17 = df_nasdaq_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_sp500_17) + geom_line()


#%%
################################################ VALOR DowJones30 ############################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_dowjones = pd.read_csv("indicadores_csv/USA30IDXUSD1_DowJones30.csv", sep="\t",header=None,names=columns)
df_dowjones.dtypes

df_dowjones["Time"] = df_dowjones["Time"].astype("datetime64[ns]")
df_dowjones
df_dowjones = df_dowjones.set_index("Time")


# tomamos dia dos meses completos -> valores desde:
df_dowjones_17 = df_dowjones.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_dowjones_17

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_dowjones_17[pd.isna(df_dowjones_17[n])])


df_dowjones_17 = df_dowjones_17.drop(["Open","High","Low","Volume"],axis=1)
df_dowjones_17.info()
df_dowjones_17 = df_dowjones_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_dowjones_17) + geom_line()


#%%
################################################ VALOR plata ########################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_plata = pd.read_csv("indicadores_csv/XAGUSD1_plata.csv", sep="\t",header=None,names=columns)
df_plata.dtypes

df_plata["Time"] = df_plata["Time"].astype("datetime64[ns]")
df_plata
df_plata = df_plata.set_index("Time")


# tomamos dia dos meses completos -> valores desde:
df_plata_17 = df_plata.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_plata_17

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_plata_17[pd.isna(df_plata_17[n])])


df_plata_17 = df_plata_17.drop(["Open","High","Low","Volume"],axis=1)
df_plata_17.info()
df_plata_17 = df_plata_17.reset_index()


# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_plata_17) + geom_line()


#%%
###################################### VALOR australian_dolar ###################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_ausdolar = pd.read_csv("indicadores_csv/AUDUSD1_AustraliaDolar.csv", sep="\t",header=None,names=columns)
df_ausdolar.dtypes

df_ausdolar["Time"] = df_ausdolar["Time"].astype("datetime64[ns]")
df_ausdolar
df_ausdolar = df_ausdolar.set_index("Time")


# tomamos dia dos meses completos -> valores desde:
df_ausdolar_17 = df_ausdolar.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_ausdolar_17

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_ausdolar_17[pd.isna(df_ausdolar_17[n])])

  
df_ausdolar_17 = df_ausdolar_17.drop(["Open","High","Low","Volume"],axis=1)
df_ausdolar_17.info()
df_ausdolar_17 = df_ausdolar_17.reset_index()


# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_ausdolar_17) + geom_line()


#%%
################################################ VALOR valor_oro_usd ############################################

# importamos el data frame y lo ajustamos 

columns = ["Time","Open","High","Low","Close","Volume"]
df_oro = pd.read_csv("indicadores_csv/XAUUSD1_oro.csv", sep="\t",header=None,names=columns)
df_oro.dtypes

df_oro["Time"] = df_oro["Time"].astype("datetime64[ns]")
df_oro
df_oro = df_oro.set_index("Time")


# tomamos dia dos meses completos -> valores desde:
df_oro_17 = df_oro.loc["2023-01-25 19:59:00":"2023-03-25 19:59:00",:]
df_oro_17

# comprobar si hay NaN:
for n in ["Open","High","Low","Close","Volume"]:
    print(df_oro_17[pd.isna(df_oro_17[n])])
    
df_oro_17 = df_oro_17.drop(["Open","High","Low","Volume"],axis=1)
df_oro_17.info()
df_oro_17 = df_oro_17.reset_index()

# representamos la gráfica con la libreria de ggplot
ggplot(aes(x="Time", y="Close"),df_oro_17) + geom_line()


#%%
#################################################
######### indicadores de miedo ##################
#################################################


################################################ Fear and Greed index CNN ############################################

"""
import requests
import pandas as pd
import time

url = "https://api.alternative.me/fng/"

def get_fng_value():
    response = requests.get(url)
    data = response.json()
    fng_value = data['data'][0]['value']
    return fng_value

columns = ['date', 'fng_value']

fng_df = pd.DataFrame(columns=columns)

while True:
    current_time = pd.Timestamp.now()
    fng_value = get_fng_value()
    fng_df = fng_df.append({'date': current_time, 'fng_value': fng_value}, ignore_index=True)
    time.sleep(60)
    if len(fng_df) == 1: # 900 = 15 horas recopilando datos
        break
        
fng_df.to_csv('fng_values.csv', index=False)
"""

################################################ GVZ (xpath - web scraping) ############################################

# //*[@id="__next"]/div[2]/div/div/div[2]/main/div/div[1]/div[2]/div[1]/span
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

url = 'https://www.investing.com/indices/cboe-gold-volatitity'
xpath = '//*[@id="__next"]/div[2]/div/div/div[2]/main/div/div[1]/div[2]/div[1]/span'

columns = ['date', 'value']

data_df_gvz = pd.DataFrame(columns=columns)

while True:
    current_time = pd.Timestamp.now()
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    value = soup.select_one(xpath).text
    numeric_value = float(re.sub('[^\d\.]', '', value))
    data_df_gvz = data_df_gvz.append({'date': current_time, 'value': numeric_value}, ignore_index=True)
    time.sleep(30)
    if len(data_df_gvz) == 4:
        break

print(data_df_gvz)
# data_df.to_csv('valor_cambiante.csv', index=False)
"""

################################################  TWITTER API  ############################################

# A la espera de aceptación twitter developer account

"""
import tweepy
import pandas as pd
import time

consumer_key = 'TU_CONSUMER_KEY'
consumer_secret = 'TU_CONSUMER_SECRET'
access_token = 'TU_ACCESS_TOKEN'
access_token_secret = 'TU_ACCESS_TOKEN_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

keywords = '#federalreserve'
columns = ['date', 'tweets_per_minute']

tweets_df = pd.DataFrame(columns=columns)

while True:
    current_time = pd.Timestamp.now()
    tweets = tweepy.Cursor(api.search, q=keywords, lang='en').items(100)
    count = 0
    for tweet in tweets:
        if tweet.created_at > (current_time - pd.Timedelta(seconds=60)):
            count += 1
    tweets_df = tweets_df.append({'date': current_time, 'tweets_per_minute': count}, ignore_index=True)
    time.sleep(60)

tweets_df.to_csv('tweets_per_minute.csv', index=False)
"""


#%%
#################################################



"""Tenemos que hacer el merge de columnas y rellenar los NaN con las medias de interpolación"""


df_crude_oil_ex_17 = df_crude_oil_17
df_bitcoin_ex_17 = df_bitcoin_17
df_usd_ex_17 = df_usd_17
df_sp500_ex_17 = df_sp500_17
df_oro_ex_17 = df_oro_17

df_ausdolar_ex_17 = df_ausdolar_17
df_plata_ex_17 = df_plata_17
df_dowjones_ex_17 = df_dowjones_17
df_nasdaq_ex_17 = df_nasdaq_17

#######
df_oro_ex_17 = df_oro_ex_17.rename(columns={"Close":"Close_oro"})
df_bitcoin_ex_17 = df_bitcoin_ex_17.rename(columns={"Close":"Close_bitcoin"})
df_crude_oil_ex_17 = df_crude_oil_ex_17.rename(columns={"Close":"Close_crude"})
df_sp500_ex_17 = df_sp500_ex_17.rename(columns={"Close":"Close_sp"})
df_usd_ex_17 = df_usd_ex_17.rename(columns={"Close":"Close_usd"})

df_ausdolar_ex_17 = df_ausdolar_ex_17.rename(columns={"Close":"Close_ausdolar"})
df_plata_ex_17 = df_plata_ex_17.rename(columns={"Close":"Close_plata"})
df_dowjones_ex_17 = df_dowjones_ex_17.rename(columns={"Close":"Close_dowjones"})
df_nasdaq_ex_17 = df_nasdaq_ex_17.rename(columns={"Close":"Close_nasdaq"})


df_crude_oil_ex_17 = df_crude_oil_ex_17.reset_index()
df_bitcoin_ex_17 = df_bitcoin_ex_17.reset_index()
df_usd_ex_17 = df_usd_ex_17.reset_index()
df_sp500_ex_17 = df_sp500_ex_17.reset_index()
df_oro_ex_17 = df_oro_ex_17.reset_index()
df_ausdolar_ex_17 = df_ausdolar_ex_17.reset_index()
df_plata_ex_17 = df_plata_ex_17.reset_index()
df_dowjones_ex_17 = df_dowjones_ex_17.reset_index()
df_nasdaq_ex_17 = df_nasdaq_ex_17.reset_index()


df_crude_oil_ex_17["Time"] = pd.to_datetime(df_crude_oil_ex_17["Time"])
df_bitcoin_ex_17["Time"] = pd.to_datetime(df_bitcoin_ex_17["Time"])
df_usd_ex_17["Time"] = pd.to_datetime(df_usd_ex_17["Time"])
df_sp500_ex_17["Time"] = pd.to_datetime(df_sp500_ex_17["Time"])
df_oro_ex_17["Time"] = pd.to_datetime(df_oro_ex_17["Time"])
df_ausdolar_ex_17["Time"] = pd.to_datetime(df_ausdolar_ex_17["Time"])
df_plata_ex_17["Time"] = pd.to_datetime(df_plata_ex_17["Time"])
df_dowjones_ex_17["Time"] = pd.to_datetime(df_dowjones_ex_17["Time"])
df_nasdaq_ex_17["Time"] = pd.to_datetime(df_nasdaq_ex_17["Time"])

df_crude_oil_ex_17 = df_crude_oil_ex_17.set_index("Time")
df_bitcoin_ex_17 = df_bitcoin_ex_17.set_index("Time")
df_usd_ex_17 = df_usd_ex_17.set_index("Time")
df_sp500_ex_17 = df_sp500_ex_17.set_index("Time")
df_oro_ex_17 = df_oro_ex_17.set_index("Time")
df_ausdolar_ex_17 = df_ausdolar_ex_17.set_index("Time")
df_plata_ex_17 = df_plata_ex_17.set_index("Time")
df_dowjones_ex_17 = df_dowjones_ex_17.set_index("Time")
df_nasdaq_ex_17 = df_nasdaq_ex_17.set_index("Time")

df_crude_oil_ex_17.drop(["index"],axis=1,inplace=True)
df_bitcoin_ex_17.drop(["index"],axis=1,inplace=True)
df_usd_ex_17.drop(["index"],axis=1,inplace=True)
df_sp500_ex_17.drop(["index"],axis=1,inplace=True)
df_oro_ex_17.drop(["index"],axis=1,inplace=True)
df_ausdolar_ex_17.drop(["index"],axis=1,inplace=True)
df_plata_ex_17.drop(["index"],axis=1,inplace=True)
df_dowjones_ex_17.drop(["index"],axis=1,inplace=True)
df_nasdaq_ex_17.drop(["index"],axis=1,inplace=True)


df1 = pd.concat([df_crude_oil_ex_17,df_bitcoin_ex_17], axis=1, join="outer").sort_values("Time")
df2 = pd.concat([df1,df_usd_ex_17], axis=1, join="outer").sort_values("Time")
df3 = pd.concat([df2,df_sp500_ex_17], axis=1, join="outer").sort_values("Time")
df4 = pd.concat([df3,df_oro_ex_17], axis=1, join="outer").sort_values("Time")
df5 = pd.concat([df4,df_ausdolar_ex_17], axis=1, join="outer").sort_values("Time")
df6 = pd.concat([df5,df_plata_ex_17], axis=1, join="outer").sort_values("Time")
df7 = pd.concat([df6,df_dowjones_ex_17], axis=1, join="outer").sort_values("Time")
df8 = pd.concat([df7,df_nasdaq_ex_17], axis=1, join="outer").sort_values("Time")

df8.isnull().sum()

# rellenamos los mising values con la media de cada variable respectivamente
# (la media movil o interpolacion lineal)
df8_ex_2 = df8 
df8.columns

df8_ex_2["Close_crude"] = df8_ex_2["Close_crude"].interpolate(method='linear')
df8_ex_2["Close_bitcoin"] = df8_ex_2["Close_bitcoin"].interpolate(method='linear')
df8_ex_2["Close_oro"] = df8_ex_2["Close_oro"].interpolate(method='linear')
df8_ex_2["Close_usd"] = df8_ex_2["Close_usd"].interpolate(method='linear')
df8_ex_2["Close_sp"] = df8_ex_2["Close_sp"].interpolate(method='linear')

df8_ex_2["Close_ausdolar"] = df8_ex_2["Close_ausdolar"].interpolate(method='linear')
df8_ex_2["Close_plata"] = df8_ex_2["Close_plata"].interpolate(method='linear')
df8_ex_2["Close_dowjones"] = df8_ex_2["Close_dowjones"].interpolate(method='linear')
df8_ex_2["Close_nasdaq"] = df8_ex_2["Close_nasdaq"].interpolate(method='linear')

dfaux = df8_ex_2.reset_index()
df8_ex_2["Close_crude"][0] = 86.40
df8_ex_2.isnull().sum()
df8_ex_2



#ponerse en le directorio capitulo_1_indicadores

df8_ex_2.to_csv("indicadores_cap_1_v3_DosMes.csv")
    
############################################






























