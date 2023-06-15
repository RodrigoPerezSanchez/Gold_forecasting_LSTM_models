# Gold_forecasting_LSTM_models
Diseño de un modelo de predicción del valor del oro implementando redes neuronales recurrentes LSTM utilizando índices externos correlacionados

Este proyecto ha sido realizado con propósitos academicos para el trabajo de fin de de carrera (TFG), para la Universidad Poletécnica de Madrid en la Escuela Técnica Superior de Ingenieros de Telecomunicaciones (ETSIT)

### Modelos LSTM utilizados: 
Stacked LSTM, BiLSTM y Vanilla LSTM
### Modelo predictivo de clasificación: 
DecisionTreeClassifier
### Plataforma de programación: 
Spyder - Anaconda

## RESUMEN
El oro siempre ha sido una de las materias primas más importantes y valoradas dentro del
mercado financiero. Usado por los bancos centrales como uno de los recursos estratégicos
para medir la riqueza, protegerse ante las fluctuaciones económicas inesperadas y el control
de los precios dentro de un país. Los inversores consideran que el oro es un activo esencial
para contrarrestar la inflación y protegerse frente a la caída del valor en el mercado de
otros recursos, lo que convierte al oro en la materia prima de inversión más común dentro
de los metales preciosos. Por ello es importante predecir y estimar el valor de este activo
para obtener los máximos beneficios posibles y estables a largo plazo. Sin embargo, el
oro es volátil en la naturaleza y está sometido a numerosos factores externos que influyen
en su valor, por lo que pronosticar su precio en el mercado se ha convertido en un gran
desafío.

El objetivo de este trabajo es diseñar un modelo de predicción del valor del oro. Debido
a la existencia de numerosos métodos para llevar a cabo este proyecto, tales como la
red neuronal recurrente BiGRU, este trabajo se centrará en implementar nuevos tipos
de RNN nunca usados para este tipo de aplicación. Se utilizarán las redes recurrentes
Stacked LSTM, BiLSTM y Vanilla LSTM para analizar sus ventajas e inconvenientes,
demostrando qué tipo de red es más eficaz a la hora de elaborar dicho modelo de predicción
comparando resultados. Se utilizará el lenguaje de Python y sus librerías para llevar a
cabo este proyecto.

Para mejorar la predicción del valor del oro, se analizará la influencia de factores externos
que condicionan su valor a lo largo del tiempo, se tomarán en cuenta numerosos índices
económicos: el SP 500, como influencia directa sobra la capitalización de las mayores
empresas de USA; el valor del petróleo y la plata; ya que se ha visto una enorme correlación
con respecto el valor del oro en el transcurso de los años; el valor del Bitcoin y del dólar
estadounidense, como medida de protección frente a la subida de precios en el mercado y
el índice Spot, entre otros. Se evaluarán en un periodo mínimo de 30 días estos modelos,
analizando aquellos que tengan una mayor repercusión, tanto positiva como negativa, a
la hora de recopilar la información. Se identificarán aquellos con la máxima correlación
posible y se tomarán los valores en el transcurso del tiempo de los índices publicados en
plataformas públicas.

Tras el diseño e implementación de cada uno de los modelos propuestos se estudiará, dentro
de los resultados obtenidos tras un proceso de optimización de hiper-parámetros, qué
modelo obtiene mejores resultados gracias al uso de un algoritmo predictivo de clasificación
basado en árbol de decisión.
