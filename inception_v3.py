# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
#
# ## Step 1: Dataset Summary & Exploration
#
# The pickled data is a dictionary with 4 key/value pairs:
#
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
#
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results.

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(train['features'])

# TODO: Number of validation examples
n_validation = len(valid['features'])

# TODO: Number of testing examples.
n_test = len(test['features'])

# TODO: What's the shape of an traffic sign image?
image_shape = train['sizes']

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
#
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
#
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[3]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#get_ipython().magic('matplotlib inline')

import random
import numpy as np

random_train = random.randint(0, n_train)
assert(len(X_train[random_train]) == 32)
"""
plt.figure()
plt.title('Picture {} in train is classified as {}'.format(random_train, train['labels'][random_train]))
plt.imshow(X_train[random_train])

# Distribution of classes in training
unique_elements, counts_elements = np.unique(train['labels'], return_counts=True)
plt.figure()
plt.title('Distribution in train')
plt.xlabel('Class')
plt.ylabel('Count')
plt.hist(train['labels'], bins=unique_elements)

unique_elements, counts_elements = np.unique(valid['labels'], return_counts=True)
plt.figure()
plt.title('Distribution in valid')
plt.xlabel('Class')
plt.ylabel('Count')
plt.hist(valid['labels'], bins=unique_elements)

unique_elements, counts_elements = np.unique(test['labels'], return_counts=True)
plt.figure()
plt.title('Distribution in test')
plt.xlabel('Class')
plt.ylabel('Count')
plt.hist(test['labels'], bins=unique_elements)
"""

# ----
#
# ## Step 2: Design and Test a Model Architecture
#
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
#
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!
#
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission.
#
# There are various aspects to consider when thinking about this problem:
#
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
#
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project.
#
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance.
#
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[4]:

import tensorflow as tf
### Hyperparameters
### @TODO Explore hyperparamter space
GRAY_SCALE = False
NUM_CHANNELS = 3
rate = 0.001
# L2 regularisation
beta = 0.00
EPOCHS = 30
BATCH_SIZE = 128
USE_DROPOUT = True
DROPOUT_RATE = 0.5
USE_BATCH_NORM = False
USE_EXTENDED_TRAINING_DATA = False

### Constants
# Small epsilon value for the BN transform
epsilon = 1e-3


# In[5]:

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2

if GRAY_SCALE:
    NUM_CHANNELS = 1
    train['features_grayscale'] = []
    for i in range(n_train):
        train['features_grayscale'].append(cv2.cvtColor(train['features'][i], cv2.COLOR_RGB2GRAY))

    valid['features_grayscale'] = []
    for i in range(n_validation):
        valid['features_grayscale'].append(cv2.cvtColor(valid['features'][i], cv2.COLOR_RGB2GRAY))

    test['features_grayscale'] = []
    for i in range(n_test):
        test['features_grayscale'].append(cv2.cvtColor(test['features'][i], cv2.COLOR_RGB2GRAY))

    train['features_grayscale'] = np.asarray(train['features_grayscale']).reshape((n_train, 32, 32, 1))
    valid['features_grayscale'] = np.asarray(valid['features_grayscale']).reshape((n_validation, 32, 32, 1))
    test['features_grayscale'] = np.asarray(test['features_grayscale']).reshape((n_test, 32, 32, 1))

    X_train = train['features_grayscale']
    X_validation = valid['features_grayscale']
    X_test = test['features_grayscale']


# In[6]:

# Create extended training data here

# Add random offset [-2, 2] to traning data
def add_offset(X_train):
    X_train_aug = []
    for i in range(len(X_train)):
        rand_offset_X = random.randint(-2, 2)
        rand_offset_Y = random.randint(-2, 2)
        img = cv2.warpAffine(X_train[i], np.float32([[1,0,rand_offset_X],[0,1,rand_offset_Y]]),(32, 32))
        img = img.reshape(img.shape[0], img.shape[1], NUM_CHANNELS)
        X_train_aug.append(img)
    return np.asarray(X_train_aug)

# Add random rotation [-15, 15] to traning data
def add_rotation(X_train):
    X_train_aug = []
    for i in range(len(X_train)):
        rand_rot = random.randint(-15000, 15000) * 0.001
        img = cv2.warpAffine(X_train[i], cv2.getRotationMatrix2D((32/2,32/2),rand_rot,1),(32,32))
        img = img.reshape(img.shape[0], img.shape[1], NUM_CHANNELS)
        X_train_aug.append(img)
    return np.asarray(X_train_aug)

# Add random scaling [0.9, 1.1] to traning data
def add_scaling(X_train):
    X_train_aug = []
    for i in range(len(X_train)):
        rand_scale = random.randint(900, 1100) * 0.001
        img = cv2.resize(X_train[i],None,fx=rand_scale, fy=rand_scale)
        img = img.reshape(img.shape[0], img.shape[1], NUM_CHANNELS)
        if img.shape[0] < 32:
            pad = np.zeros((32, 32, NUM_CHANNELS), dtype=np.uint8)
            pad[:img.shape[0], :img.shape[1], :] = img
            img = pad
        #img.reshape(32, 32, NUM_CHANNELS)
        X_train_aug.append(img[:32, :32, :])
    return np.asarray(X_train_aug)

# Add gaussian noise to traning data
def add_noise(X_train):
    X_train_aug = []
    for i in range(len(X_train)):
        img = X_train[i]
        row,col,ch= img.shape
        mean = 5
        var = 5
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        X_train_aug.append(noisy)
    return np.asarray(X_train_aug)

# https://github.com/fastai/fastai/blob/master/fastai/transforms.py
# 50 is too aggressive and causes training diffculty
def add_lighting(X_train, b=25, c=0.8):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im

    X_train_aug = []
    for i in range(len(X_train)):
        img = X_train[i]
        mu = np.average(img)

        X_train_aug.append(np.clip((img-mu)*c+mu+b,0.,255.).astype(np.uint8))
    return np.asarray(X_train_aug)

if USE_EXTENDED_TRAINING_DATA:
    X_train_offset = add_offset(X_train)
    X_train_rot = add_rotation(X_train)
    X_train_scale = add_scaling(X_train)
    X_train_noise = add_noise(X_train)
    X_train_lighting = add_lighting(X_train)

    X_train = np.concatenate((X_train, X_train_rot, X_train_offset, X_train_scale, X_train_noise, X_train_lighting), axis=0)
    y_train = np.concatenate((train['labels'], train['labels'], train['labels'], train['labels'], train['labels'], train['labels']), axis=0)

    random_train = random.randint(0, n_train)
    print(random_train)

    """
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(2, 3, 1)
    plt.title('Original {}'.format(y_train[random_train]))
    if GRAY_SCALE:
        plt.imshow(X_train[random_train].reshape(32, 32), cmap='gray')
    else:
        plt.imshow(X_train[random_train])
    fig.add_subplot(2, 3, 2)
    plt.title('Random Offset')
    if GRAY_SCALE:
        plt.imshow(X_train_offset[random_train].reshape(32, 32), cmap='gray')
    else:
        plt.imshow(X_train_offset[random_train])
    fig.add_subplot(2, 3, 3)
    plt.title('Random Rotation')
    if GRAY_SCALE:
        plt.imshow(X_train_rot[random_train].reshape(32, 32), cmap='gray')
    else:
        plt.imshow(X_train_rot[random_train])
    fig.add_subplot(2, 3, 4)
    plt.title('Random Scale')
    if GRAY_SCALE:
        plt.imshow(X_train_scale[random_train].reshape(32, 32), cmap='gray')
    else:
        plt.imshow(X_train_scale[random_train])
    fig.add_subplot(2, 3, 5)
    plt.title('Gaussian Noise')
    if GRAY_SCALE:
        plt.imshow(X_train_noise[random_train].reshape(32, 32), cmap='gray')
    else:
        plt.imshow(X_train_noise[random_train])
    fig.add_subplot(2, 3, 6)
    plt.title('Lighting and Constract')
    if GRAY_SCALE:
        plt.imshow(X_train_lighting[random_train].reshape(32, 32), cmap='gray')
    else:
        plt.imshow(X_train_lighting[random_train])
    plt.show()
    """
from keras.applications.inception_v3 import *
import numpy as np
import keras
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

# Add random scaling [0.9, 1.1] to traning data
def scale_image(X_train, scale):
    X_train_aug = []
    for i in range(len(X_train)):
        img = cv2.resize(X_train[i],None,fx=scale, fy=scale)
        img = img.reshape(img.shape[0], img.shape[1], NUM_CHANNELS)
        X_train_aug.append(img)
    return np.asarray(X_train_aug)

model = InceptionV3(weights=None, input_shape=(299, 299, 3), classes=43)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


from sklearn.utils import shuffle

train_accuracies = []
validation_accuracies = []

for i in range(EPOCHS):
    num_examples = len(X_train)

    print("Training...")
    print()

    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
        print("batch {}".format(offset))
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        batch_x = scale_image(batch_x, 9.34375)
        one_hot_labels = keras.utils.np_utils.to_categorical(batch_y, nb_classes=43)

        print(batch_x.shape)
        print(one_hot_labels.shape)
        model.train_on_batch(batch_x, one_hot_labels)

    print("EPOCH {} ...".format(i+1))

	#train_accuracy = model.predict(
	#train_accuracies.append(train_accuracy)
	#validation_accuracy = evaluate(X_validation, y_validation, accuracy_operation, tensors['dropout_keep_prob'])
	#validation_accuracies.append(validation_accuracy)
	#print("Validation Accuracy = {:.3f}".format(validation_accuracy))
	#print()

