import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criarRede(neurons, optimizer, activation, loss, kernel_initializer, drop):
    classificador = Sequential()  
  
    classificador.add(Dense(units = neurons, activation= activation, kernel_initializer= kernel_initializer, input_dim = 4))
    
    classificador.add(Dropout(drop))
    
    classificador.add(Dense(units = neurons, activation= activation, kernel_initializer= kernel_initializer))
    
    classificador.add(Dropout(drop))
    
    classificador.add(Dense(units = 3, activation='softmax')) 
    
    
    classificador.compile(optimizer=optimizer, loss= loss, metrics = ['accuracy'])
    
    return classificador


classificador = KerasClassifier(build_fn = criarRede)

parametros = {'batch_size': [15, 30], 'epochs': [100,150], 'optimizer': ['adam'], 'loss': ['sparse_categorical_crossentropy'],
              'kernel_initializer': ['normal'], 'activation' : ['relu'], 'neurons': [10,12], 'drop': [0.2,0.3]}



grid_search = GridSearchCV(estimator=classificador, param_grid=parametros , cv =5  )

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
 