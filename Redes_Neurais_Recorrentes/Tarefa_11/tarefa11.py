import pandas as pd
import numpy as np
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('poluicao.csv')
base = base.dropna()

base = base.drop(columns = ['No', 'year', 'month', 'day', 'hour', 'cbwd'])

base_treinamento = base.iloc[:len(base) - 50, :].values


normalizador = MinMaxScaler(feature_range=(0,1)) 
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

normalizador_previsao = MinMaxScaler(feature_range=(0,1)) 
normalizador_previsao.fit_transform(base_treinamento[:, 0:1])


periodo = 10
generator = TimeseriesGenerator(base_treinamento_normalizada, base_treinamento_normalizada, length = periodo, batch_size = 1)

previsores = np.array([i[0] for i in generator])
previsores = np.reshape(previsores, (len(previsores), periodo, len(previsores[0][0][0]) )) 


classe_poluicao = np.array([i[1][0][0] for i in generator])

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], previsores.shape[2] )))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))
regressor.add(Dense(units = 1, activation = 'sigmoid'))
regressor.compile( optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=10)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
mc = ModelCheckpoint(filepath='pesos.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, classe_poluicao, epochs = 50, batch_size = 64, callbacks=[es,rlr,mc])

base_teste = base.iloc[len(base) - 50:,:]
base_teste_normalizada = normalizador.transform(base_teste)

generator2 = TimeseriesGenerator(base_teste_normalizada, base_teste_normalizada, length = periodo, batch_size = 1)

X_teste = np.array([i[0] for i in generator2])
X_teste = np.reshape(X_teste, (40,10,7))

y_teste = np.array([i[1] for i in generator2])
y_teste = np.reshape(y_teste, (40,7))
y_teste = np.array([[i[0]] for i in y_teste]) #Converter o formato de 40, para 40, 1

previsoes = regressor.predict(X_teste)

previsoes = normalizador_previsao.inverse_transform(previsoes)
y_teste = normalizador_previsao.inverse_transform(y_teste)

plt.plot(previsoes, color='red', label = 'Poluição Previsão')
plt.plot(y_teste, color='blue', label = 'Poluição Real')
plt.title('Previsão Poluição')
plt.xlabel('Tempo')
plt.ylabel('Nível de poluição')
plt.legend()
plt.show()