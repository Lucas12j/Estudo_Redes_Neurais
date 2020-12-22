import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()  
  
classificador.add(Dense(units = 8, activation= 'relu', kernel_initializer= 'normal', input_dim = 30))
    
classificador.add(Dropout(0.2))
    
classificador.add(Dense(units = 9, activation= 'relu', kernel_initializer= 'normal'))
    
classificador.add(Dropout(0.2))
    
classificador.add(Dense(units = 1, activation='sigmoid')) 
    
    
classificador.compile(optimizer='adam', loss= 'binary_crossentropy', metrics = ['binary_accuracy'])
    
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

classificador_json = classificador.to_json()
with open('rede_config.json','w') as file:
    file.write(classificador_json)
classificador.save_weights('pesos_breast_cancer.h5')