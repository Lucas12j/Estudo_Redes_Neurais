import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(X_treinamento, y_treinamento),(X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treinamento[7], cmap= 'gray')

#ETAPA NECESSÁRIO PARA O TENSOR FLOW ENTENDER
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

#CONVERSÃO PARA FLOAT
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#MUDANDO A ESCALA PARA 0 A 1
previsores_treinamento /= 255
previsores_teste /= 255

#CONVERTENDO PARA VARIÁVEIS CATEGORICAS
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

#OPERADOR DE CONVOLUÇÃO
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = 'relu'))
##Normalizando o mapa de características
classificador.add(BatchNormalization())

#POOLING
classificador.add(MaxPooling2D(pool_size= (2,2)))

#ADICIONANDO UMA SEGUNDA CAMADA DE CONVOLUÇÃO
classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2,2)))

#FLATTENING
##Só é necessário usar o Flattening uma única vez, no último passo da camada de convolução
classificador.add(Flatten())

#CRIAÇÃO DA REDE NEURAL DENSA
##Diminui o número de neurônios e gerações em relação ao do professor, para diminuir o tempo de processamento.
classificador.add(Dense(units = 100, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 100, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#TREINAMENTO
classificador.fit(previsores_treinamento, classe_treinamento, batch_size= 128, epochs = 3, validation_data = (previsores_teste, classe_teste))
