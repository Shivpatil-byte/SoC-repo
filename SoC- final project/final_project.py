#importing all important liabraries and mnist dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#loading and splitting dataset to xtrain,yrain,xtest,ytest
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

#checking the size of the array of each
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

#confirming the image in dataset matches to its given value
plt.imshow(xtrain[0])
print(ytrain[0])

#converting the values of xtrain,xtest in between 0-1 and also reshaping it to obtain  better results
xtrain=xtrain.reshape((-1,28,28,1)).astype('float32')/255
xtest=xtest.reshape((-1,28,28,1)).astype('float32')/255

#one-hot encode the labels
from tensorflow.keras.utils import to_categorical
ytrain=to_categorical(ytrain,num_classes=10)
ytest=to_categorical(ytest,num_classes=10)

#building the CNN model
from tensorflow.keras import models,layers

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')   # 10 classes for digits 0-9
])

#compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#fitting or training the model
history=model.fit(xtrain,ytrain,epochs=5,batch_size=64,validation_split=0.1)


#evaluating the model
test_loss,test_acc=model.evaluate(xtest,ytest)
print(f"\nTest accuracy: {test_acc:.4f}")


#graph of accuracy vs epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()