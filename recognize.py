import numpy as np
import argparse
import cv2
from keras.models import model_from_json

def read_image(address):
	img = cv2.imread(address)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	return img

def preprocess_input(image):
	img = cv2.resize(image,(32,32))
	img = np.transpose(img,(2,0,1))
	img = img / 255.0
	img = img.reshape((1,3,32,32))
	return img

def predict(input):
	categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
	model = model_from_json(open('cifar10_architecture.json').read())
	model.load_weights('cifar10_weights.h5')
	prediction = np.argmax(model.predict(input,verbose=0),axis=1)
	return categories[prediction[0]]

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i','--image',required=True,help='path to input image')
	address = vars(ap.parse_args())
	# Read the imput image:
	image = read_image(address['image'])
	# Predict the class:
	prediction = predict(preprocess_input(image))
	# Write the class on the image:
	cv2.putText(image,prediction,(10,20),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2)
	# Display the image:
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	cv2.imshow('image',image)
	cv2.waitKey(0)

if __name__ == '__main__': main()

