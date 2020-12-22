import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

####  PRE PROCESSAMENTO

base = pd.read_csv('autos.csv', encoding=('ISO-8859-1'))

#base = base.drop('dateCrawled', axis = 1)

#Dropando colunas irrelevantes
base = base.drop(columns = ['dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen', 'name', 'seller', 'offerType'])

#Retirando valores anormais da dase
i1 = base.loc[base.price <= 10]
base = base.loc[base.price > 10]

i2 = base.loc[base.price >= 350000]
base = base.loc[base.price < 350000]

#Tratando valores nulos

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

valores = {'vehicleType' : 'limousine', 'gearbox' : 'manuell', 'model' : 'golf', 'fuelType' : 'benzin', 'notRepairedDamage' : 'nein'}
base = base.fillna(value = valores)

#Separando classe e previsores

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_previsores = LabelEncoder()

previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_encoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = label_encoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_encoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = label_encoder_previsores.fit_transform(previsores[:, 10])

#Colocando variável dummy, para atributos categoricos

onehotencoder = OneHotEncoder(categorical_features= [0, 1, 3, 5, 8, 9, 10])
previsores = onehotencoder.fit_transform(previsores).toarray()

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def criar_rede(loss):
    regressor = Sequential()

    regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
    regressor.add(Dense(units = 158, activation = 'relu'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    
    regressor.compile(loss = loss, optimizer = 'adam', metrics=['mean_absolute_error'])
    
    return regressor


regressor = KerasRegressor(build_fn = criar_rede, batch_size = 500, epochs = 20)

parametros = {'loss': ['mean_squared_error' , 'mean_absolute_error' , 
                       'mean_absolute_percentage_error' ,'mean_squared_logarithmic_error', 'squared_hinge']}



grid_search = GridSearchCV(estimator=regressor, param_grid=parametros , cv = 2  )

grid_search = grid_search.fit(previsores, preco_real)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
 