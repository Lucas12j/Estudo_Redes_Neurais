import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')


#Tratamento dos dados
base = base.drop(columns = ['Other_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Developer', 'Name'])
base = base.dropna(axis = 0)

base = base.loc[base['Global_Sales'] > 1]

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
global_sales = base.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


def criar_rede(neurons, loss, optimizer):
    #USANDO FUNÇÕES DE ATICAÇÃO COMO OBJETOS:
    activ1 = Activation('sigmoid')
    activ2 = Activation('linear')
    
    camada_entrada = Input(shape=(99,)) #numero de entradas
    camada_oculta1 = Dense(units=neurons, activation = activ1)(camada_entrada)

    camada_oculta2 = Dense(units=neurons, activation = activ1)(camada_oculta1)
  
    camada_saida = Dense(units=1,activation = activ2)(camada_oculta2)
    
    regressor = Model(inputs=camada_entrada, outputs=camada_saida)
    regressor.compile(optimizer = optimizer, loss = loss)

    return regressor

'''
regressor = KerasRegressor(build_fn = criar_rede,  batch_size = 100, epochs = 1000)

parametros = {'loss': ['mse', 'squared_hinge'],'neurons':[50,25], 'optimizer':'adam'}

grid_search = GridSearchCV(estimator=regressor, param_grid=parametros , cv = 2  )
grid_search = grid_search.fit(previsores, global_sales)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
'''

regressor = criar_rede(50, 'mse', 'adam')

regressor.fit(previsores, global_sales, batch_size = 100, epochs = 5000)

previsao_global = regressor.predict(previsores)

m_previsao_global = previsao_global.mean()
m_global_sales = global_sales.mean()