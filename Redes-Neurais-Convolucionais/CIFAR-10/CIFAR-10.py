import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization 

#Dividindo as variáveis
(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

#ETAPA NECESSÁRIO PARA O TENSOR FLOW ENTENDER (Talvez não seja mais necessário com as novas versões)
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

#Conversão para float
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#normalizando
previsores_treinamento /= 255
previsores_teste /= 255

#Variaveis tipo dummy, por ter mais de uma classe (boa prática)
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()

#Primeira camada
classificador.add(Conv2D(64, (3,3), input_shape = (32, 32, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2,2)))

#Segunda camada
classificador.add(Conv2D(64, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2,2)))

#Aplicando o flattenig
classificador.add(Flatten())


classificador.add(Dense(units = 256, activation = 'relu'))
classificador.add(Dropout(0.25))

classificador.add(Dense(units = 256, activation = 'relu'))
classificador.add(Dropout(0.25))

classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size= 128, epochs = 5, validation_data = (previsores_teste, classe_teste))

#USEI O COLAB DA GOOGLE, PARA IR MAIS RÁPIDO.
