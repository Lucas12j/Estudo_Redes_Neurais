import pandas as pd
import numpy as np
from keras.layers import Dropout, Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

#Separando a base
base_treinamento = base.iloc[:, 1:2].values
###base_treinamento = np.reshape(base_treinamento, (len(base_treinamento), 1))  Faz mais sentido para mim do que iloc[:, 1:2]

#Objeto para normalização
normalizador = MinMaxScaler(feature_range=(0,1)) #Já tem 0,1 como padrão, mas quis colocar assim mesmo.

base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

#periodo a ser considerado para a previsão
periodo = 90

#Cria uma matriz que cada linha tem 90 periodos, onde uma linha tem o mesmo intervalo de valores, porém com um dia a frente. EX: 01/01 - 07/01 == 02/01 - 08/01
previsores = [base_treinamento_normalizada[i-periodo : i, 0]  for i in range(periodo, len(base_treinamento_normalizada))]
#Cria um vetor com os valores da classe. OBS: O primeiro valor é 1 "periodo" a frente, pois é necessário ter um periodo antes para fazer o calculo.
#Processo semelhante a como a média movel funciona
preco_real = [base_treinamento_normalizada[i, 0]  for i in range(periodo, len(base_treinamento_normalizada))]


#Conversoes
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (len(previsores), len(previsores[0]), 1))

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


base_teste = pd.read_csv('petr4_teste.csv')
#Separa a coluna correspondente
preco_real_teste = base_teste.iloc[:, 1:2].values
#Junta as bases para criar uma completa
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

#Pega a quantidade de previsores necessária para se fazer a previsão da base de teste. Ou seja, 1 "periodo" antes da base de treinamento inicial
entradas = base_completa[len(base_completa) - len(base_teste) - periodo:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

#Mesmo processo feito para os previsores, só que na base de testes (previsores de teste)
X_teste = [ entradas[i-periodo : i, 0] for i in range(periodo, len(entradas)) ]
#conversão
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (len(X_teste), len(X_teste[0]), 1))


previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

plt.plot(preco_real_teste, color='red', label = 'Preço real')
plt.plot(previsoes, color='blue', label = 'Preço previsto')
plt.title('Previsão preço da ação')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()