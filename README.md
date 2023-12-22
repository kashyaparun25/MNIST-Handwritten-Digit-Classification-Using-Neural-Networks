# MNIST-Handwritten-Digit-Classification-Using-Neural-Networks
Develop a neural network for MNIST digit recognition, including steps like data download, preprocessing, network setup, feedforward, backpropagation, training, and evaluation. The process involves image normalization, label one-hot encoding, and building a two-layer network for digit classification.

In this programming problem, you will get familiar with building a neural network using backpropagation. You will write a program that learns how to recognize the handwritten digits using stochastic gradient descent and the MNIST training data.
The MNIST database (Modified National Institute of Standards and Technology database is a large database of handwritten digits that is commonly used for training various image processing systems

![image](https://github.com/kashyaparun25/MNIST-Handwritten-Digit-Classification-Using-Neural-Networks/assets/52271759/1859c4e6-affe-48e1-9a36-4dd370ec145d)

Step 1 Data Acquisition and Visualization: In this step, you need to:

(a) Download the “MNIST” dataset and extract the files. You will get four files with extension .gz (e.g., train-images-idx3-ubyte.gz). You can use the provided function read_idx below to read in the dataset. As its official description, the dataset is split into 60000 training images and 10000 images. The four file corresponds to the training images, training labels, testing images and testing labels. You need to print out their shape to finish this step.

![image](https://github.com/kashyaparun25/MNIST-Handwritten-Digit-Classification-Using-Neural-Networks/assets/52271759/c9a83493-ecef-439d-a8a5-baa2a84f2721)

(b)To further understand what the dataset is, you need to use the ‘matplotlib’ library to print out a random data with code plt.imshow together with its label. You will see something like this:

![image](https://github.com/kashyaparun25/MNIST-Handwritten-Digit-Classification-Using-Neural-Networks/assets/52271759/227335de-f17a-4c1c-94df-8a8d20172f85)

Step 2 Data Preprocessing : In this step, you need to:

(a) Normalize the pixel values of images to be between 0 and 1.

(b) Convert the labels from categorical data into numerical values using one-hot encoding. hint: you can explore the eye function in Numpy.

Step 3 Network Initialization: We will work with a neuron network with two hidden layers, using Sigmoid function as the activation functions for hidden layers and softmax activation function for the output layer. To finish this, you need to:

(a) Identify the auxiliary input including the Sigmoid function and its derivative and Softmax function

(b) Initialize all the parameters in neural network uniformly. In this network, the input size is 784 dimensions (each input is a 28x28 image, so you have to flatten the data from 2D to 1D). For the two linear hidden layers, we have 128 and 64 neurons respectively. For the output layer, its size will be 10 since there are 10 classes (0-9) in MNIST. To finish this step, you need to initialize the weights and bias in random with a pre-set random seed using Numpy. Please set the seed value = 695.

Step 4 Feed Forward : In this step, you need to:

(a) Define a function named feed_forward. Given an input x, it should output the sigmoid of wx+b where w and b indicates the weights and bias defined in step 2.

Step 5 Back Propagation (15 pts): In this step, you need to implement the back propagation:

(a) You need to compute the loss for the output layer first. Here, we use categorical cross entropy loss function given below for multi-class classification problem. (5 pts) Note, to achieve this, you need to first encode the categorical labels as numerical values using one-hot encoding finished in step 2.
![image](https://github.com/kashyaparun25/MNIST-Handwritten-Digit-Classification-Using-Neural-Networks/assets/52271759/55fd55dc-7b84-4499-b301-eef7d4b899f2)

(b) Calculate the gradients for the weights and bias for each layer. Use the chain rule to compute gradients for previous layers.

Step 6 Model Training : In this step, you need to:

(a) Use mini-batch gradient descent to update the parameters including weights and bias. Notice that a complete training round consists of a feed forward process, back propagation and parameter update. Define the batch size = 128 and epoch = 100.

Step 7 Model Evaluation : In this step, you need to:

(a)Use your trained neural network to predict the labels of the test dataset and compute the accuracy on the test dataset.

Remark: if you correctly execute every step above, you will probably get a result around 90%.

(b) Plot some of the misclassified images with their predicted and true labels. This probably can give you some insights into why these images are misclassified.





