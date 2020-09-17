import os
import numpy as np
import pandas as pd
import cv2
import glob
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

image=np.load("image.npy",allow_pickle=True)
labels=np.load("labels.npy",allow_pickle=True)

s=np.arange(image.shape[0])
np.random.shuffle(s)
image=image[s]
labels=labels[s]

num_classes=len(np.unique(labels))
len_data=len(image)

x_train,x_test=image[(int)(0.1*len_data):],image[:(int)(0.1*len_data)]
y_train,y_test=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]


y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

l2_reg=0.001

#Baseline of a 3 layers of Conv2D for CNN model with dropout regularization on each layers
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(50,50, 3), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu',kernel_regularizer=l2(l2_reg)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


from keras.callbacks import ModelCheckpoint

filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(x_train,y_train,batch_size=128,epochs=70,verbose=1,validation_split=0.33,callbacks=[checkpoint])

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


#Training visualization
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])

#not Necessarily need to run the code below
'''
#use tensorflow GradientTape to see single neural loss.
#learnt from MIT Videos, use their library for convenience.
#run the next column to install mitdeeplearning library
#!pip install mitdeeplearning
import mitdeeplearning as mdl
import tensorflow as tf
from tqdm import tqdm

cnn_model = model

batch_size = 12
loss_history = mdl.util.LossHistory(smoothing_factor=0.95) # to record the evolution of the loss
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2) # define our optimizer

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for idx in tqdm(range(0, x_train.shape[0], batch_size)):
  # First grab a batch of training data and convert the input images to tensors
  (images, labels) = (x_train[idx:idx+batch_size], y_train[idx:idx+batch_size])
  images = tf.convert_to_tensor(images, dtype=tf.float32)

  # GradientTape to record differentiation operations
  with tf.GradientTape() as tape:
    logits = cnn_model(images)
    loss_value = tf.keras.backend.sparse_categorical_crossentropy(labels, logits) 

  loss_history.append(loss_value.numpy().mean()) # append the loss to the loss_history record
  plotter.plot(loss_history.get())
  '''