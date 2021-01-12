import pandas as pd
import numpy as np
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

base_treinamento = base.iloc[:, 1:2].values
base_valor_maximo = base.iloc[:, 2:3].values


normalizador = MinMaxScaler(feature_range=(0,1)) #Já tem 0,1 como padrão, mas quis colocar assim mesmo.
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)

periodo = 90

from keras.preprocessing.sequence import TimeseriesGenerator

generator_treinamento = TimeseriesGenerator(base_treinamento_normalizada, base_treinamento_normalizada, length = periodo, batch_size = 1)
generator_valor_maximo = TimeseriesGenerator(base_valor_maximo_normalizada, base_valor_maximo_normalizada, length = periodo, batch_size = 1)

previsores = np.array([i[0] for i in generator_treinamento])
preco_real1 = np.array([i[1] for i in generator_treinamento])
preco_real2 = np.array([i[1] for i in generator_valor_maximo])

previsores, preco_real1, preco_real2 = np.reshape(previsores, (len(previsores), periodo, 1)), np.reshape(preco_real1, (len(preco_real1))), np.reshape(preco_real2, (len(preco_real2)))

preco_real = np.column_stack((preco_real1, preco_real2))

regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1 )))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 2))

regressor.compile( optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)


base_teste = pd.read_csv('petr4_teste.csv')
preco_real_open = base_teste.iloc[:, 1:2].values
preco_real_high = base_teste.iloc[:, 2:3].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

entradas = base_completa[len(base_completa) - len(base_teste) - periodo:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

generator_entradas = TimeseriesGenerator(entradas, entradas, length = periodo, batch_size = 1)

X_teste = np.array([i[0] for i in generator_entradas])
X_teste = np.reshape(X_teste, (len(X_teste), len(X_teste[0][0]), 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

plt.plot(preco_real_open, color='pink', label = 'Preço real open')
plt.plot(preco_real_high, color='black', label = 'Preço real high')
plt.plot(previsoes[:, 0:1], color='yellow', label = 'Preço previsto open')
plt.plot(previsoes[:, 1:2], color='blue', label = 'Preço previsto high')
plt.title('Previsão preço da ação')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()