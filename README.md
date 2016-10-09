# CIFAR-10

This is an attempt to solve Kaggle's CIFAR-10 challenge using Python and Keras. The model used is a convolutional neural network which after 25 epochs of training gives approximately 81% training accuracy and 77% validation accuracy.

## The Dataset

You can download the 3 csv files from this [google drive link](https://drive.google.com/open?id=0ByCZreDktfuea3NWSHJHd3FYR2M) and place them inside the `train-data` folder. 

Alternately, you can download the training data from [Kaggle](https://www.kaggle.com/c/cifar-10) and place the expanded folder inside the `train-data` folder. Then, use the `make_csv.py` script to generate the csv files corresponding to each channel. To use `make_csv.py`, run `python make_csv.py --channel color` where color can be specified as red, green and blue. Once, the three csv files are generated you can proceed to train and test the model. 

## Recognizing images of any shape (OpenCV required)

The script `recognize.py` can be used to recognize images of any shape using the weights learned from training the model. The usage is as follows: `python recognize.py --image address-of-image`. 
