# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[distribution_training]: ./res/distribution_train.png "Distribution in Training"
[distribution_validation]: ./res/distribution_validation.png "Distribution in Validation"
[distribution_test]: ./res/distribution_test.png "Distribution in Test"
[before_after_greyscale]: ./res/before_after_grayscale.png "Grayscaling"
[augmentation]: ./res/augmentation.png "Augmentation"
[download]: ./res/download.png "Download"
[feature_map_70_km]: ./res/feature_map_70_km.png "Feautre Map 70 km/h"
[feature_map_no_entry]: ./res/feature_map_no_entry.png "Feautre Map No Entry"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/clarkli86/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distrubtion of each traffic sign in training/validatin/testing datasets.

We can see that distrubtion is roughly the same in all three datasets. Thus validation and test datasets should provide unbiased evalution performance. [1]

![alt text][distribution_training]
![alt text][distribution_validation]
![alt text][distribution_test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because
1. All German traffic signs are in differnet shapes by visual inspection. [2]
2. In practice Grayscale images produce higher test accuracy. [3]

Here is an example of a traffic sign image before and after grayscaling.

![alt text][before_after_grayscale]

As a last step, I normalized the image data because it speeds up the training process as the parameters will be in equal range. (equal variance) This makes it more likely for gradient descent to optimise on all parameters and coverge faster. [4] When training on large dataset or very deep neural networks, this could save the researcher a couple of epochs or days.

I decided to generate additional data because it prevent overfitting to the original training set and add robustness to new test images.

To add more data to the the data set, I used the following techniques because camera images could come with different size/position/brightness or random noise.

Here is an example of an original image and augmented images:

![alt text][augmentation]

The difference between the original data set and the augmented data set is the following ...

|Dataset                            | Size  |
|:---------------------------------:|:-----:|
|Random Offset (-2, 2)              | 34799 |
|Random Scale (0.9, 1.1)            | 34799 |
|Random Rotation (-15 deg, 15 deg)  | 34799 |
|Gaussian Noise (5, 5)              | 34799 |
|Random Brigntness/Contrast (-2, 2) | 34799 |

All augmented data set haves the same class distribution as the original one.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Fully connected		| outputs 120  									|
| Fully connected		| outputs 80									|
| Fully connected		| outputs 43									|
| Softmax				| etc       									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I have run an Adam optimizer with learning rate 0.001, batch size 12 for 30 epochs.
A table of all hyperparameters

| Hyperparameter     | Description |
|:------------------------:|:-----:|
| Learning Rate            | 0.001 |
| Batch Size               | 128   |
| Epoch                    | 30    |
| Dropout Keep Probability | 0.5   |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.984
* test set accuracy of 0.973

An iterative approach was chosen for this project:
1. Firstly the orinal LeNet5 model was used. Two convolutional layers and 2 hidden layers should be a good start for low resolution image (32x32x3) classification. Prolbem with this model was that validation accuracy (90%) was much lower than training accuracy.
2. Dropout was added to prevent reventing complex co-adaptations on training data. Validation accuracy increased to 95%.
3. Then I doubled the filter size in the first and second convluation layers to capture more features. Validation accuracy increased to >98% after this change.
4. Then I tried batch normlisation to further reduce overfit to traning data but it didn't seem to help.
5. Also tried the multiple-scale architecture which connects output to from multiple convolution layers to fully-connected layers[3], but it didn't seem to help either.
6. The next step would be to try inception model which employs convlution layers with different kernal size. Hopefully this model will capture more unique features of each different traffic sign.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web: (downsampled to 32x32 with sign in the center of camera)
![alt text][download]

- The first image might be difficult to classify because it is rotated.
- The sixth image might be difficult to classify because it is streched.
- The ninth image might be difficult to classify because it is close to camera.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 70 km/h	      		| 70 km/h  		         		 				|
| 30 km/h	      		| 30 km/h  		         		 				|
| Priority Road			| Priority Road      							|
| Keep Right      		| Keep Right             		 				|
| No entry	      		| No entry 		         		 				|
| Yield					| Yield											|
| Turn left ahead  		| Turn left ahead       		 				|
| Turn right ahead 		| Turn right ahead       		 				|
| Road work      		| Road work 									|
| Stop Sign      		| Stop sign   									|

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares less favorably to the accuracy on the test set of 97.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.
```python
# Restore model and try to predict
predicted_class_ids = None

tf.reset_default_graph()
(x, y), _, _, tensors = LeNet(is_training=False)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './lenet')
    predict_operation = tf.argmax(tensors['logits'], 1)
    predicted_class_ids = sess.run(predict_operation, feed_dict={ x: X_download, y: y_download, tensors['dropout_keep_prob'] : 1.0})

    # Read lable names from signnames.csv
    fig=plt.figure(figsize=(20, 20))
    for i in range(len(predicted_class_ids)):
        fig.add_subplot(4, 4, i + 1)
        class_id = predicted_class_ids[i]
        color = 'green' if class_id == download['labels'][i] else 'red'
        plt.title('Predicted as ' + str(class_id) + ' - ' + class_names[class_id], color=color)
        plt.imshow(download['features'][i])
    plt.show()
```

For the first image, the model is relatively sure that this is a 70 km/h speed limit sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Speed limit (70km/h) |
| 0.0 | Speed limit (20km/h) |
| 0.0 | Speed limit (30km/h) |
| 0.0 | Speed limit (50km/h) |
| 0.0 | Speed limit (60km/h) |

For the second image, the model is relatively sure that this is a 30 km/h speed limit sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Speed limit (30km/h) |
| 1.084540512952259e-10 | Speed limit (70km/h) |
| 3.4022345487758066e-14 | Speed limit (20km/h) |
| 1.4816447697105044e-14 | Speed limit (50km/h) |
| 1.0803471561359607e-14 | Keep right |

For the third image, the model is relatively sure that this is a priority road sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Priority road |
| 4.436801690668137e-36 | Yield |
| 4.747134190250079e-37 | Keep right |
| 3.251670935407931e-37 | Ahead only |
| 0.0 | Speed limit (20km/h) |

For the fourth image, the model is relatively sure that this is a keep right sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Keep right |
| 2.748486117518101e-13 | Priority road |
| 2.1430158558660413e-16 | No vehicles |
| 7.787736060597305e-17 | Speed limit (30km/h) |
| 1.9676588599607198e-17 | Yield |

For the fifth image, the model is relatively sure that this is a priority road sign (probability of 0.96). and does not contain a no entry sign. The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 0.9604279398918152 | Priority road |
| 0.030626516789197922 | Roundabout mandatory |
| 0.006541649345308542 | No entry |
| 0.0012478885473683476 | Keep right |
| 0.0011183376191183925 | End of all speed and passing limits |

For the sixth image, the model is relatively sure that this is a yield sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Yield |
| 0.0 | Speed limit (20km/h) |
| 0.0 | Speed limit (30km/h) |
| 0.0 | Speed limit (50km/h) |
| 0.0 | Speed limit (60km/h) |

For the seventh image, the model is relatively sure that this is a turn left ahead sign (probability of 0.99). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 0.9999880790710449 | Turn left ahead |
| 4.950617494614562e-06 | Keep right |
| 4.766291567648295e-06 | Ahead only |
| 1.596443780726986e-06 | Go straight or right |
| 2.9755901209682634e-07 | Yield |

For the eighth image, the model is relatively sure that this is a turn right ahead sign (probability of 0.99). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 0.9999984502792358 | Turn right ahead |
| 1.273617840524821e-06 | Yield |
| 1.9208975743367773e-07 | Ahead only |
| 4.7405013248180694e-08 | Speed limit (60km/h) |
| 1.0132477079594082e-08 | Stop |

For the ninth image, the model is relatively sure that this is a road work sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Road work |
| 3.5238897348336716e-26 | Keep right |
| 3.921407776942002e-31 | Bumpy road |
| 7.929453407308172e-36 | Traffic signals |
| 0.0 | Speed limit (20km/h) |

For the tenth image, the model is relatively sure that this is a stop sign (probability of 1.0). The top five soft max probabilities were

| Probability | Prediction |
|:-----------:|:----------:|
| 1.0 | Stop |
| 1.1558399426405063e-18 | Priority road |
| 3.478716467144854e-20 | Keep right |
| 1.7820014944128097e-20 | No vehicles |
| 5.318853329763013e-21 | No entry |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I've visuallised the feature map of two downloaded imges in the first convolution layer.

1. Successful prediction on Speed Limit 70 km/h
![alt text][feature_map_70_km]
The feature map shows that digits and circular boundary around it have been emphasised for activitation.

1. Unsuccessful prediction on No Entry
![alt text][feature_map_no_entry]
The feature map shows that even the horizantal bar is found but the feature of cicular boundary is lost. The broken up segments of the boundary may make the model believe it is a priority road sign.

## References
[1] https://en.wikipedia.org/wiki/Training,_test,_and_validation_sets
[2] https://en.wikipedia.org/wiki/Road_signs_in_Germany
[3] http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
[4] https://en.wikipedia.org/wiki/Feature_scaling