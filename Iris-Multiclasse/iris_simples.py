import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')

previsores = base.iloc[:, 0:4].values

classe = base.iloc[:, 4].values


    
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

classificador = Sequential()

#Adicionando camadas ocultas
classificador.add(Dense(units=4, activation = 'relu', input_dim = 4)) #units -> Númro neuronioscamada oculta.  input_dim -> Númro de entradas
classificador.add(Dense(units=4, activation = 'relu'))
classificador.add(Dense(units=3, activation = 'softmax'))


#Compilando
classificador.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5)


#Criando uma matriz de confusão para melho visualização
#Será necessário ser uma matriz 3x3, por se tratar de 3 classe. Fazer da maneira anterior, com classificação binaria, com um neuronio de saida, dará erro, por ser uma matriz 2x2 (true or false)
from sklearn.metrics import confusion_matrix
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)