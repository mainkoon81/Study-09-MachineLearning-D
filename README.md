# Study-09-MachineLearning-D
DeepLearning (CNN, RNN)

----------------------------------------------------------------------------------------------------------------------------------------

# CNN
## Applications ?
 - ComputerVision
 - NaturalLanguageProcessing (WaveNet-model)
 - Let SelfDrivingCars read Road Signs
 - limitless....

## MNIST database
As arguably the most famous database in the field of deep learning, it contains 70,000 images of hand-written digits(0-9).
```
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
Sometimes some digits are less legible than others. How to conquer these difficulty? Our algorithm examine the images and discover patterns. 
<img src="https://user-images.githubusercontent.com/31917400/42932985-9d125f86-8b3b-11e8-976e-dfda4faeba7e.jpg" />








































