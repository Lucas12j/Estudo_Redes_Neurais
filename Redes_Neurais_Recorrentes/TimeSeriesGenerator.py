import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

base_treinamento = base.iloc[:, 1].values
base_treinamento = np.reshape(base_treinamento, (len(base_treinamento), 1))

periodo = 90
generator = TimeseriesGenerator(base_treinamento, base_treinamento, length = periodo, batch_size = 1)

previsores = np.array([i[0] for i in generator])
previsores = np.reshape(previsores, (len(previsores), periodo, 1))

preco_real = np.array([i[1] for i in generator])
preco_real = np.reshape(preco_real, (len(preco_real)))







'''
teste = np.array([[1,2,3,4,5,6,7,8,9,10]])
a = np.reshape(teste, (5, 2))
'''