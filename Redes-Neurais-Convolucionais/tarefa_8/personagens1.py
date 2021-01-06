import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('personagens.csv')

previsores = df.iloc[:,0:6]
classe_antes = df.iloc[:,6]

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe_antes)

X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size = 0.25)

classificador = Sequential()
classificador.add(Dense(units=60, activation = 'tanh',kernel_initializer= 'random_uniform', input_dim = 6))
classificador.add(Dropout(0.25))
classificador.add(Dense(units=60, activation = 'tanh',kernel_initializer= 'random_uniform'))
classificador.add(Dropout(0.25))
classificador.add(Dense(units=60, activation = 'tanh',kernel_initializer= 'random_uniform'))
classificador.add(Dropout(0.25))
classificador.add(Dense(units=1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(X_train, y_train, batch_size = 5, epochs = 500)

resultado = classificador.evaluate(X_test, y_test)
previsoes = classificador.predict(X_test)
