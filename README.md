# Study-09-MachineLearning-D
DeepLearning (CNN, RNN)

----------------------------------------------------------------------------------------------------------------------------------------

# CNN
#### Applications ?
 - ComputerVision
 - NaturalLanguageProcessing (WaveNet-model)
 - Let SelfDrivingCars read Road Signs
 - limitless....

### Introduction with MNIST database
As arguably the most famous database in the field of deep learning, it contains 70,000 images of hand-written digits(0-9).
```
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
Sometimes some digits are less legible than others. How to conquer these difficulty? How our algorithm examine the images and discover patterns? These pattern can then be used to decipher the digits in images that it hasn't seen before. Let's see how computer sees when we input these images.  
<img src="https://user-images.githubusercontent.com/31917400/42932985-9d125f86-8b3b-11e8-976e-dfda4faeba7e.jpg" />

## 1. [MLP]
### X
For computer, an image is a matrix with one entry for each image pixel. Each image in MNIST database is **28 x 28**(pixel). White pixels are encoded as **255**, black pixels are encoded as **0**, and grey pixels appear in the matrix as an integer somewhere in between. We rescale each image to have values in the range from 0 to 1.   
<img src="https://user-images.githubusercontent.com/31917400/42933905-d9982a42-8b3d-11e8-9674-792685ddb6aa.jpg" />

### y
Before supplying the data to our deeplearning network, we'll need to preprocess the **labels**(y_train, y_test) because each image currently has a label that's integer-valued. So we convert this to an one-hot encoding. Each label will be transformed to a vector with mostly 0. 
<img src="https://user-images.githubusercontent.com/31917400/42939141-947659a2-8b4c-11e8-9d2e-91606b40d004.jpg" />

### X again?
However, recall that our MLP only takes **vectors** as input. So if we want to use MLP with images, we have to first convert all of our matirces to vectors. For example, in the case of a 28 x 28 image, we can get a vector with 784 entries. This is called **flattening**. After encoding our images as vectors, they can be fed into the input layer of our MLP.  
<img src="https://user-images.githubusercontent.com/31917400/42935288-2cb6273a-8b41-11e8-9b6c-1e203d957233.jpg" />

How to create our MLP for discovering the patterns in our data? Let's say...
 - Since our data pt are vectors with 784 entries -> the input layer with 784 nodes
 - Let's say -> the 2 hidden layers each containing 572 nodes
 - The model needs to distinguish 10 different digits -> the output layer with 10 nodes, activation='softmax' 
   - the softmax function ensures that our network outputs an estimate for the probability that each potential digit is depicted in the image(multi-class classification).

Here, we just add the 'flatten layer' before we specify our MLP. It takes the image matrix's input and convert it to a **vector**.  
<img src="https://user-images.githubusercontent.com/31917400/42940231-9b24ec88-8b50-11e8-8d2e-4beafbee2351.jpg" />

### After first fitting...
This model is great as a first draft, but we can make it better.
 - To lift model accuracy:
   - add **'relu'** activation function to all of our hidden layers. 'relu' helps with the **vanishing gradients** problem.
 - To minimize overfitting:
   - add **dropout layer**. This layer must be supplied a parameter(from 0 to 1) which is a probability that any node in the network is dropped during training.  
<img src="https://user-images.githubusercontent.com/31917400/42946281-6a7bd846-8b62-11e8-919e-a7794c06818b.jpg" />
 
Now our data is processed and our model is specified. Currently, all of the 600,000 weights have random values so the model has random predictions. By training our model, we'll modify these weights and improve predictions. Before training the model, we need to specify a **loss function**. Since we are constructing a **multi-class classifier**, we will use **categorical_crossentropy** loss function. 

### categorical_crossentropy 
This loss function checks to see if our model has done a good job with classifying an image by comparing the model's predictions to the **true label**. As seen above, each label became a vector with 10 entries. Our model outputs a vector also with 10 entries which are predictions. 
 - Here our model predicts there's an '8' in the image with 90% probability and a '3' in the image with 10% probability. But in the label, there is a 100% probability that the imagedepicts a '3'.
 - the categorical_crossentropy_loss looks at these two vectors and returns a **lower value** if the two vectors agree regarding what's in the image. 
<img src="https://user-images.githubusercontent.com/31917400/42950341-b1ad4624-8b6b-11e8-929e-75d3e695d035.jpg" />

In this example, our model is pretty sure that the digit is an '8', but the label knows for certain it's a '3' thus, will return a higher value for the loss. If the model later returns the next output where it changes to being 90% sure thata the image depicts a '3' the loss value will be lower. To sum, if the model predictions **agree** with the labels, the loss is **lower**.

We want its predictions to agree with the labels. We'll try to find the **model parameters** that gives us predictions that minimizing the loss function. And the standard method for descending a loss function is called **Gradient Descent**. There are lots of ways to perform Gradient Descent and **each method** in Keras has a corresponding **optimizer**. The surface depicted here is an example of a loss function and all of the optimizers are racing towards the minimum of the function, and some are better than others.  
<img src="https://user-images.githubusercontent.com/31917400/42951046-3e05cb5e-8b6d-11e8-88b4-ab75663259f1.jpg" />

### Compiling_Step
 - we'll specify the **'loss', 'optimizer', and 'metrics'**. 
> For loss function...
> - 'mean_squared_error' (y_true, y_pred)
> - 'binary_crossentropy' (y_true, y_pred)
> - 'categorical_crossentropy' (y_true, y_pred), etc...

> For optimizer...
> - 'sgd' : Stochastic Gradient Descent
> - 'rmsprop' : RMSprop (for recurrent neural networks??)
> - 'adagrad' : Adagrad that adapts learning rates, accumulating all past gradients?
> - 'adadelta' : a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates?
> - 'adam' : Adam
> - 'adamax' : Adamax
> - 'nadam' : Nesterov Adam optimizer
> - 'tfoptimizer' : TFOptimizer

### Training_Step
<img src="https://user-images.githubusercontent.com/31917400/42966788-21303aea-8b96-11e8-89e3-5c76c967093b.jpg" />

To understand the modification of training code, we'll first need to understand the idea of 'model validation'. So far, we've checked model performance based on how the loss in accuracy changed with epoch. When we design our model, it's not always clear **how many layers** the network should have or **how many nodes** to place in each layer or **how many epochs and what batch_size** should we use. So we break our dataset into three set: train + validation + test. Our model looks only at the training set when deciding how to modify its weights. And every epoch, the model checks how it's doing by checking its accuracy on the validation set (but it does not use any part of the validation set for the backpropagation step). We use the training set to find all the patterns we can, and the validation set tells us if our chosen model is performing well. Since our model does not use the validation set for deciding the weights, it can tell us if we're overfitting to the training set. For example, around certain epoch, we can get some evidence of overfitting where the **training loss** starts to decrease but the **validation loss** starts to increase. Then we will want to keep the weights around the epoch and discard the weights from the later epochs. This kind of process can prove useful if we have multiple potential architectures to choose from. 
<img src="https://user-images.githubusercontent.com/31917400/42973988-611d7840-8bad-11e8-9e96-62d7daddd1a9.jpg" />

 - **How many layers? nodes? epochs? batch_size?**: we save the weights from each potential architecture for later comparison. Then we will always pick the model that gets the lowest validation loss. 
 - **but why we create the test set?**: When we go to test the model, it looks at data that it has truly never seen before. Eventhough the model doesn't use the validation set to update its weights, our model selection process can be biased in favor of the validation set. 
 - Notice the `fit()` takes `validation_split` argument. `ModelCheckpoint()` class allows us to save the model weights after each epoch. `save_best_only` parameter says that save the weights to get the best accuracy on the validation set. 
### But in the case of real-world messy image data(random size, complex shapes, patterns..), do you think MLP works? Nope!
   - MLP only use fully connected layers...it uses too many parameters..a risk of **overfitting**
   - MLP only accepts vectors as input
<img src="https://user-images.githubusercontent.com/31917400/43022609-b6e8521a-8c5f-11e8-879f-7ed4a048a99b.jpg" />

## 2. [CNN]
<img src="https://user-images.githubusercontent.com/31917400/42973712-1fe68142-8bac-11e8-8e0e-c94178e64919.jpg" />

What's the convolutional layer?
<img src="https://user-images.githubusercontent.com/31917400/43023382-973db9e8-8c62-11e8-8557-aff4c924ce0d.jpg" />

 - Break the image up into smaller pieces.
   - first select a width and height defining a convolution window
   - then simply slide this window horizontally, vertically over the matrix of the pixels
   - At each position, the window specifies a small piece within the image, and define a collection of pixels to which we connect a single hidden node.

In detail, how a regional collection of input nodes influences the value of a node in a convolutional layer?   
 - We could represent the weights connecting the nodes by writing a numerical value on top of the arrows. 
 - Then in order to get the value of a node in the convolutional layer for this image input we operate as we did with MLP with multiplying the input nodes by their corresponding weights and summing up the result. When we do that, we get zero just as with 




























