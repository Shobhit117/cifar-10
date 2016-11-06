import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from keras.models import model_from_json

categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def read_image(address):
	img = cv2.imread(address,1)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	return img

def preprocess_input(image):
	img = cv2.resize(image,(32,32))
	img = np.transpose(img,(2,0,1))
	img = np.reshape(img,(1,3,32,32))
	img = img / 255.0
	return img

def probability_plot(predictions):
	y_pos = np.arange(len(categories))
	plt.figure()
	plt.barh(y_pos,predictions,align='center',alpha=0.5)
	plt.yticks(y_pos,categories)
	plt.xlabel('Probability')
	plt.show()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-image',type=str,required=True,
		help='Specify the address of an image.')
	args = parser.parse_args()
	# Load the image.
	address = args.image
	image = read_image(address)
	input_image = preprocess_input(image)
	# Load the model.
	model = model_from_json(open('cifar10_architecture.json').read())
	model.load_weights('cifar10_weights.h5')
	# Get the predicitons.
	predictions = model.predict(input_image,verbose=0).flatten()
	top_results = predictions.argsort()
	print('Top predictions: %s: %0.4f, %s: %0.4f, %s: %0.4f' % 
		(categories[top_results[9]], predictions[top_results[9]],
			categories[top_results[8]], predictions[top_results[8]], 
			categories[top_results[7]],predictions[top_results[7]]))
	# Plot the probability bar chart.
	probability_plot(predictions)


if __name__ == '__main__': main()
