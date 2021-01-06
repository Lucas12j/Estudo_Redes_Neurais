import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento_ex.csv')
base = base.dropna()

#Pré-processamento
base_treinamento = base.iloc[:, 1].values
base_treinamento = np.reshape(base_treinamento, (len(base_treinamento), 1))

normalizador = MinMaxScaler(feature_range=(0,1)) 
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

periodo = 90
generator = TimeseriesGenerator(base_treinamento_normalizada, base_treinamento_normalizada, length = periodo, batch_size = 1)

previsores = np.array([i[0] for i in generator])
previsores = np.reshape(previsores, (len(previsores), periodo, 1))

preco_real = np.array([i[1] for i in generator])
preco_real = np.reshape(preco_real, (len(preco_real)))


#Criação da rede

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))
regressor.add(Dense(units = 1))
regressor.compile( optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)

#PREVISÂO
base_teste = pd.read_csv('petr4_teste_ex.csv')

preco_real_teste = base_teste.iloc[:, 1].values
preco_real_teste = np.reshape(preco_real_teste, (len(preco_real_teste), 1))
preco_real_teste_normalizado = normalizador.transform(preco_real_teste)

base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

entradas = base_completa[len(base_completa) - len(base_teste) - periodo:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

generator2 = TimeseriesGenerator(entradas, entradas, length = periodo, batch_size = 1)
X_teste = np.array([i[0] for i in generator2])
X_teste = np.reshape(X_teste,(len(X_teste), len(X_teste[0][0]), len(X_teste[0][0][0])))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)


plt.plot(preco_real_teste, color='red', label = 'Preço real')
plt.plot(previsoes, color='blue', label = 'Preço previsto')
plt.title('Previsão preço da ação')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()

###TENTANDO PREVER COM VETOR COLOCADO POR MIM
'''
array = X_teste[0] 
array_real = normalizador.inverse_transform(array) #Array enviado para a previsão
array_formatado = np.reshape(array, (1,90,1))
resultado = regressor.predict(array_formatado)
resultado_real = normalizador.inverse_transform(resultado) #Resultado cuspido pela rede
'''