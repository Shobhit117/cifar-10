import numpy as np
import pandas as pd
from keras.models import model_from_json

def get_model():
	# Load the model architecture.
	model = model_from_json(open('cifar10_architecture.json').read())
	# Load the model weights.
	model.load_weights('cifar10_weights.h5')
	return model

def get_cifar10_validation(validation_size=5000):
	# Load the red channel.
	red_data = pd.read_csv('train-data/red_data.csv')
	# Separate the labels and pixels:
	labels = red_data.iloc[:validation_size,0].values
	labels = labels.astype(int) - 1
	red_pixels = red_data.iloc[:validation_size,1:]
	# Load the blue and green channels.
	blue_pixels = pd.read_csv('train-data/blue_data.csv').iloc[:validation_size,1:].values
	green_pixels = pd.read_csv('train-data/green_data.csv').iloc[:validation_size,1:].values
	# Combine the channels:
	X = np.c_[red_pixels,green_pixels,blue_pixels]
	X = X.reshape((validation_size,3,32,32))
	# Return the data:
	return X, labels

def predict_labels(model,X):
	y = np.argmax(model.predict(X,batch_size=10,verbose=0),axis=1)
	return y

def calc_accuracy(y_val,y_pred):
	matches = np.sum(y_val == y_pred)
	accuracy = matches/float(y_val.shape[0])
	return accuracy

def main():
	# Load the validation data:
	X_val, y_val = get_cifar10_validation()
	# Load the model:
	model = get_model()
	# Predict the labels.
	y_pred = predict_labels(model,X_val)
	# Calculate accuracy:
	accuracy = calc_accuracy(y_val,y_pred)
	# Print the answer:
	print('Validation Accuracy = '+str(accuracy))

if __name__ == '__main__': main()