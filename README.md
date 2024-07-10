# Face Classification and Verification
A basic CNN to classify and verify the facial images

### Overview
A sample project to help learn to build CNN-based architectures for face classification and
face verification. The homework will instruct you on two key concepts:
- How to build effective convolutional neural networks.
- How to generate discriminative and generalizable feature representations for data.

#### Face Classification
Face classification is a closed-set multiclass classification problem where the subjects in the
test set has also been seen in the training set, although the precise pictures in the test set
will not be in the training set. For this to achieve high accuracy, it is only required that the
embeddings for (all pictures of) the subjects in our “vocabulary” be linearly separable from
each other.

#### Face Verification
Face verification refers to determining whether two face images are of the same
person without necessarily knowing who the person is. Face verification is an instance of a
larger class of problems where we attempt to determine if two data instances belong to the
same class without necessarily knowing (or having a model for) the class itself.
This is a common problem used in a variety of situations, for instance, when your laptop
uses facial recognition to identify you. You would have “enrolled” yourself with an enrollment
image, and later when you try to login, your system compares a picture it takes of your face
to the stored enrollment image to determine if both are from the same person. 

### Data Description
We will work with a small batch of data provided by Tom Mitchell from CMU and ponder over how convolution networks perform when benchmarked against a fully connected network. The full data set is provided in the zipped file below. Make sure to test your model with your own pictures (preferably in low-res). The data set can be downloaded 
from Tom Mitchell's website at CMU or the UCI ML repo:

http://archive.ics.uci.edu/ml/datasets/cmu+face+images
http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html

Note: In the faces dataset you will see variations of the same image, for instance:

an2i_left_angry_open.pgm: indicate a full-resolution image (128 columns by 120 rows); 
an2i_left_angry_open_2.pgm: indicates a half-resolution image (64 by 60) and
an2i_left_angry_open_4.pgm: indicates a quarter-resolution image (32 by 30).
When designing your network, you will have to choose work with images of a fixed resolution.

This is a relatively small dataset for "big data" standards. However a well-designed convolution neural network, such as LeNet5 could make a dent on this problem. Bear in mind the risk of overfitting, and if your network has too many parameters, try to adjust it to accommodate the smaller dataset. 

### Problem Statement
Consider two tasks:
- Task 1: Is this image a picture of Mitchell? 
- Task 2: What's the facial expression in the picture?

Goal: Build a CNN using LeNet-5 architecture and report how your network performs at tasks Task 1 and Task 2.


