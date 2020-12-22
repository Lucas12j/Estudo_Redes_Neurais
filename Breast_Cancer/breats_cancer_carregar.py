from keras.models import model_from_json
import numpy as np
import pandas as pd

with open('rede_config.json','r') as classificador_json:
    estrutura_rede = classificador_json.read()
    
classificador = model_from_json(estrutura_rede)

classificador.load_weights('pesos_breast_cancer.h5')

entradas = np.array(pd.read_csv('entradas_breast.csv'))

saidas = np.array(pd.read_csv('saidas_breast.csv'))

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

resultado = classificador.evaluate(entradas, saidas)





