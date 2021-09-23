from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf

(trainX, trainY), (testX, testY) = mnist.load_data()

#Reshaping the the numpy array from 3 dimensional to 4 dimensional
trainX = trainX.reshape(trainX.shape[0],28,28,1)
testX = testX.reshape(testX.shape[0],28,28,1)
input_shape = (28,28,1)
#Making the values floats so that we can get decimal points after division
trainX = trainX.astype('float32')
testX = testX.astype('float32')
#Normalizing the RGB codes bij dividing it to the max RGB value
trainX /= 255
testX /= 255
#Creating a Sequential Model and adding the layers for the network
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
#Setting up the optimizer and fitting the model with the training data
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=trainX, y=trainY, epochs=10)
#evaluating the model with the test dataset
model.evaluate(testX,testY)
model.save('model.h5')

image_index = 4444
plt.imshow(testX[image_index].reshape(28, 28))
plt.show()
pred2 = model.predict(testX[image_index].reshape(1, 28, 28, 1))
print(pred2.argmax())


