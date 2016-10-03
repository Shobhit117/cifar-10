import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def one_hot(labels):
	rows = labels.shape[0]
	one_hot_labels = np.zeros((rows,10))
	index_offset = np.arange(rows) * 10
	one_hot_labels.flat[index_offset + labels.ravel().astype(int)-1] = 1
	return one_hot_labels	

def get_cifar10_data(num_validate=5000):
	# Read the red channel data.
	red_channel = pd.read_csv('train-data/red_data.csv').iloc[:,:].values
	# Separate the labels and the pixel data.
	labels = one_hot(red_channel[:,0].ravel())
	red_channel = red_channel[:,1:]
	# Load the blue and green channel data.
	green_channel = pd.read_csv('train-data/green_data.csv').iloc[:,1:].values
	blue_channel = pd.read_csv('train-data/blue_data.csv').iloc[:,1:].values
	# Combine the channels together.
	X = np.c_[red_channel,green_channel,blue_channel]
	X = X.reshape((50000,3,32,32))
	# Train on 40000 samples and validate on 10000 samples.
	return (X[num_validate:],labels[num_validate:]),(X[:num_validate],labels[:num_validate])

def display(img):
	plt.figure()
	plt.imshow(np.transpose(img,(1,2,0)))
	plt.show()

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def make_network():
	model = Sequential()
	model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,32,32)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32,3,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(64,3,3,border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64,3,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	return model

def save_model(model):
	model_json = model.to_json()
	open('cifar10_architecture.json','w').write(model_json)
	model.save_weights('cifar10_weights.h5',overwrite=True)

def main():
	# Load the data:
	print('Loading data...')
	(X_train,y_train),(X_val,y_val) = get_cifar10_data()
	print('Done!')
	# Create the model:
	print('Creating Model...')
	model = make_network()
	print('Done!')
	# Check for existing trained weights:
	if os.path.exists('./cifar10_weights.h5'):
		print('Loading Existing Weights...')
		model.load_weights('cifar10_weights.h5')
		print('Done!')
	# Train the model:
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	model.fit(X_train,y_train,nb_epoch=2,batch_size=32,validation_data=(X_val,y_val),verbose=1)
	# Save the model:
	print('Saving results...')
	save_model(model)
	print('Done!')


if __name__ == '__main__': main()

