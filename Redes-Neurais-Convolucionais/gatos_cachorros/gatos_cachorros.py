from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#Para ler imagens externas
import numpy as np
from keras.preprocessing import image


classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))


classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())



classificador.add(Dense(units=128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

'''
IMPORTANTE: Mesmo que eu não queira adicionar mais imagens ao dataset com o ImageDataGenerator, é importante usa-lo, 
para não precisar criar matrizes numpy das imagens. Bastaria utilizar o "rescale" para normalizar os pixels e deixar 
o restante dos parametros em branco, que as imagens são carregadas automaticamente.
'''
gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, 
                                         shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size=32, class_mode='binary')
base_teste = gerador_treinamento.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')

classificador.fit_generator(base_treinamento, steps_per_epoch = 4000/25,  epochs = 5,  validation_data = base_teste, validation_steps=1000)







#####Para classificar imagens externas

imagem_teste = image.load_img('previsao_externa/2.jpeg', target_size=(64, 64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)

previsao = classificador.predict(imagem_teste)

#visualizando as classes
base_treinamento.class_indices
