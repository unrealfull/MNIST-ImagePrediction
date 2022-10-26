# MNIST-ImagePrediction

Handwritten numbers (0-9) on images (9x9) can be correctly identified

Two applications - one for training the model and the other consumes (uses / applies) the model

## Training application:
 - use public (MNIST) resources as an input
 - trains a model to correlty classify images with KNN or SVM
 - persists the trained model 

## Consumer application:
 - load a model
 - load an image
 - (return) print out the predicted number (additional returns list from 0 to 9 with all probabilities)

 Libraries: opencv, scikit-learn, matplotlib