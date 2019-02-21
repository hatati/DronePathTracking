from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import to_categorical

import pickle

NAME = "Forward-vs-left-vs-right-CNN-RGB"

#Load data
pickle_in = open("Dataset/X_RGB.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Dataset/y_RGB.pickle", "rb")
y = pickle.load(pickle_in)

#Normalize
X = X/255

#Transform labels to categorical
y = to_categorical(y)

#Create model graph
model = Sequential()

model.add(Conv2D(64, (5, 5), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors

#FC layers
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))


#Create Tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


#Compile
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

#Train
model.fit(X, y, batch_size=10, epochs=10, validation_split=0.3, callbacks=[tensorboard])

#Save model as a Keras file
KERAS_FILE = "model_files/forward-left-right-CNN-RGB.h5"
save_model(model, KERAS_FILE)

