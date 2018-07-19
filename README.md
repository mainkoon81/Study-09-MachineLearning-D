# Study-09-MachineLearning-D
DeepLearning (CNN, RNN)

----------------------------------------------------------------------------------------------------------------------------------------

# CNN
#### Applications ?
 - ComputerVision
 - NaturalLanguageProcessing (WaveNet-model)
 - Let SelfDrivingCars read Road Signs
 - limitless....

### MNIST database
As arguably the most famous database in the field of deep learning, it contains 70,000 images of hand-written digits(0-9).
```
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
Sometimes some digits are less legible than others. How to conquer these difficulty? How our algorithm examine the images and discover patterns? These pattern can then be used to decipher the digits in images that it hasn't seen before. Let's see how computer sees when we input these images.  
<img src="https://user-images.githubusercontent.com/31917400/42932985-9d125f86-8b3b-11e8-976e-dfda4faeba7e.jpg" />

For computer, an image is a matrix with one entry for each image pixel. Each image in MNIST database is **28 x 28**(pixel). White pixels are encoded as **255**, black pixels are encoded as **0**, and grey pixels appear in the matrix as an integer somewhere in between. We rescale each image to have values in the range from 0 to 1.   
<img src="https://user-images.githubusercontent.com/31917400/42933905-d9982a42-8b3d-11e8-9674-792685ddb6aa.jpg" />

Before supplying the data to our deeplearning network, we'll need to preprocess the labels because each image currently has a label that's integer-valued. So we convert this to an one-hot encoding. Each label will be transformed to a vector with mostly 0. 
<img src="https://user-images.githubusercontent.com/31917400/42939141-947659a2-8b4c-11e8-9d2e-91606b40d004.jpg" />

However, recall that our MLP only takes **vectors** as input. So if we want to use MLP with images, we have to first convert all of our matirces to vectors. For example, in the case of 28 x 28 image, we can get a vector with 784 entries. This is called **flattening**. After encoding our images as vectors, they can be fed into the input layer of our MLP.  
<img src="https://user-images.githubusercontent.com/31917400/42935288-2cb6273a-8b41-11e8-9b6c-1e203d957233.jpg" />

How to create our MLP for discovering the patterns in our data?


































