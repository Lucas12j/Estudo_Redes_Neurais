import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model


base = pd.read_csv('games.csv')

nome_jogos = base.Name

base = base.drop(columns = ['Other_Sales', 'Global_Sales', 'Developer', 'Name'])

base = base.dropna(axis = 0)

base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

#onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
#previsores = onehotencoder.fit_transform(previsores).toarray()

#ERRO! O CÓDIGO ATUALIZADO SE ENCONTRA NA PASTA BAIXADA DE CONSIDERAÇÕES FINAIS.
#PORÉM AINDA SIM HÁ UM ERRO, SUSPEITO QUE SEJA A VERSÃO DO DAS BIBLIOTECAS, AGUARDANDO RESPOSTA DOS INSTRUTORES

#SE UTILIZAR DESTA MANEIRA, IRÉ FUNCIONAR
#onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8], n_values='auto')

#Darei continuidade, usando a solução acima, ainda que esteja ultrapassada, obsoleta, Deprecation

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8], n_values='auto')
previsores = onehotencoder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(61,)) #numero de entradas
camada_oculta1 = Dense(units=32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)
camada_saida1 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida3 = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer='adam', loss = 'mse')

regressor.fit(previsores, [venda_na, venda_eu, venda_jp], batch_size = 100, epochs = 5000)


previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)


