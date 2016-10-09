import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import argparse

# Get the channel to extract.
ap = argparse.ArgumentParser()
ap.add_argument('-i','--channel',required=True,help="Channel Number: Red = 0, Green = 1, Blue = 2")
args = vars(ap.parse_args())
channel = int(args['channel'])

if channel != 0 and channel != 1 and channel != 2:
	print('incorrect channel!')
	quit()

# The number of images:
num_images = 50000

# Category keys:
categories = {'airplane':1,'automobile':2,'bird':3,'cat':4,'deer':5,'dog':6,'frog':7,'horse':8,'ship':9,'truck':10}

# Get the labels:
labels_data = pd.read_csv('trainLabels.csv')
labels = list(labels_data['label'].apply(str))
# labels = labels[:num_images]

# Process the images and change the labels to their respective keys:
images_data = []
for i in range(num_images):
	print('processing image '+str(i+1)+' ...')
	image_source = 'train/'+str(i+1)+'.png'
	current_image = mpimg.imread(image_source)
	current_image = current_image[:,:,channel]
	# Rescale the image:
	current_image = current_image.reshape(1,1024)
	# Add the image to the list:
	images_data.append(current_image)
	# Convert the label to key:
	labels[i] = categories[labels[i]]
	print('done')

images_data = np.array(images_data)
images_data = images_data.reshape((num_images,1024))
labels = np.array(labels)
labels = labels.reshape((num_images,1))

# Join labels and image data:
data = np.c_[labels,images_data]

# Prepare to save the csv file.
# Decide the filename.
if channel == 0:
	filename = 'red_data.csv'
elif channel == 1:
	filename = 'green_data.csv'
elif channel == 2:
	filename = 'blue_data.csv'

# Make the header for csv file:
header_c ='label'
for i in range(1024):
	label_h = ',pixel'+str(i)
	header_c = header_c + label_h

# Save the data:
np.savetxt(filename,data,delimiter=',',header=header_c,comments='',fmt='%f')


