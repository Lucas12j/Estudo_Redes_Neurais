import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
    
labelencoder = LabelEncoder()
saidas = labelencoder.fit_transform(classe)

def cria_classificador():

    classificador = Sequential()    
    classificador.add(Dense(units = 12, activation= 'relu', kernel_initializer= 'normal', input_dim = 4))    
    classificador.add(Dropout(0.2))    
    classificador.add(Dense(units = 12, activation= 'relu', kernel_initializer= 'normal'))    
    classificador.add(Dropout(0.2))    
    classificador.add(Dense(units = 3, activation='softmax')) 
            
    classificador.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])    
    classificador.fit(previsores, saidas, batch_size = 15, epochs = 150)
    
    return classificador

def salvar(classificador):
    
    classificador_iris_json = classificador.to_json()
    with open('rede_iris_config.json','w') as file:
        file.write(classificador_iris_json)
    classificador.save_weights('pesos_iris.h5')
    
def carregar():
    
    with open('rede_iris_config.json','r') as classificador_iris_json:
        estrutura_rede_iris = classificador_iris_json.read()
    
    classificador2 = model_from_json(estrutura_rede_iris) 
    classificador2.load_weights('pesos_iris.h5')
     
    classificador2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return classificador2

def classificar_um_registro(classificador_um_registro):
    novo_registro = np.array([[7.5, 3.4, 6.2, 2.1]])
    return classificador_um_registro.predict(novo_registro)



print('Criando Classificador')
classificador = cria_classificador()

print('Salvando...')
salvar(classificador)

print('Carregando...')
classificador_um_registro = carregar()

print('Classificando um registro:')
resultado_classificação = classificar_um_registro(classificador_um_registro)

