import pandas as pd
import numpy as np
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()


base_treinamento = base.iloc[:, 1::].values

normalizador = MinMaxScaler(feature_range=(0,1)) 
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

normalizador_previsao = MinMaxScaler(feature_range=(0,1)) 
normalizador_previsao.fit_transform(base_treinamento[:, 0:1])

periodo = 90
generator = TimeseriesGenerator(base_treinamento_normalizada, base_treinamento_normalizada, length = periodo, batch_size = 1)

previsores = np.array([i[0] for i in generator])
previsores = np.reshape(previsores, (len(previsores), periodo, 6 )) #dimensão do vetor da última dimensão

preco_real = np.array([i[1][0][0] for i in generator])


regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6 )))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'sigmoid'))

regressor.compile( optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

#Meios de você definir critérios de parada no treinamento, como por exemplo quando a loss function não consegue diminuir mais.
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=10)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
mc = ModelCheckpoint(filepath='pesos.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32, callbacks=[es,rlr,mc])


base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
frames = [base, base_teste]
base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis = 1)


entradas = base_completa[len(base_completa) - len(base_teste) - periodo:].values
entradas = normalizador.transform(entradas)

generator2 = TimeseriesGenerator(entradas, entradas, length = periodo, batch_size = 1)


X_teste = np.array([i[0] for i in generator2])
X_teste = np.reshape(X_teste, (len(X_teste), periodo, 6 ))


previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

plt.plot(preco_real_teste, color='red', label = 'Preço real')
plt.plot(previsoes, color='blue', label = 'Preço previsto')
plt.title('Previsão preço da ação')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()




