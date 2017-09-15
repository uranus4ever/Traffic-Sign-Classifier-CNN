# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)：German Traffic Sign Dataset (<http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset>)
* Explore, summarize and visualize the data set
* Design, train and test a Convenlutional Neural Network model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


## Data Set Summary & Exploration

I  calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

## Design and Test a Model Architecture

### 1. Preprocess the image data - Normalization.

My first step to process the image data is to grayscale. Because the key info conveyed in the traffic sign is the pattern instead of the color. RGB - channels increase computation complexity and helps little on recognition accurancy, although color classifies the categories and helps human to have a general warning at first glance.

I use numpy library to merge RGB channels to one and roll the axis to last.

```
gray = np.array([np.dot(img[..., :3], [0.299, 0.587, 0.114])])
NewImg = np.rollaxis(gray, 0, 3) 
```
After that, the training data set is shuffled to avoid overfitting.

### 2. My final model consisted of the following layers:

| Layer No. | Layer  		|     Description	        					| 
|:-:|:----------------:|:----------------------------:| 
|   | Input          		| 32x32x1 image 							| 
| 1 | Convolution 5x5 	| 1x1 stride, valid padding, outputs 28x28x6 	|
|   | RELU			        		|					Activation							|
|   | Max pooling	    	| 2x2 stride, outputs 14x14x6 				|
| 2 | Convolution 3x3  | 1x1 stride, valid padding, outputs 12x12x20 |
|   | RELU		           |      Activation   									|
|   | Max pooling			  	| 2x2 stride, outputs 6x6x20        				|
| 3 | Convolution 3x3  | 1x1 stride, valid padding, outputs 4x4x60 |
|   | RELU		           |      Activation   									|
|   | Max pooling			  	| 2x2 stride, outputs 2x2x60        									|
|	  | Flatten			     		|	outputs 240											|
| 4 |	Fully connected		|	outputs 160, dropout		|
|   | RELU             |      Activation       |
| 5 |	Fully connected		|	outputs 80											|
|   | RELU             |      Activation      |
|	6 | Fully connected		|	outputs 43											|
 
### 3. How to train my model. 

To train the model, I used an optimizer, batch size = 128, epochs = 20, learning rate = 0.001.
```
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)
```
Then feed data to the training model batch by batch.
```
for offset in range(0, num_examples, BATCH_SIZE):
    end = offset + BATCH_SIZE
    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
```

### 4. Model results

My final model results were:
* validation set accuracy of 94.8% 
* test set accuracy of 93.6%

An iterative approach was chosen:
#### What was the first architecture that was tried and why was it chosen?
I choose LeNet because they have the same application scenario - training the data set to learn how to classifer categories.

#### What were some problems with the initial architecture?
LeNet architecture adoption is underfitting to this project.

#### How was the architecture adjusted and why was it adjusted? 
I add one layer to make the architecture deeper, including convolution and max pooling, due to initial model accurancy is around 83%, indicating underfitting.

#### Which parameters were tuned? How were they adjusted and why?
I tune filter height and width, depth. Too large filter size results in inefficient whereas too small size underfitting.

#### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? 
  I consider adopting 1x1 conv, dropout and max pooling techniques to improve the model performance. 1x1 conv is an inexpensive way to make model deeper and have more parameters. However, it does not make my model efficient.
  In terms of dropout, it is a technique for regularization. It  makes things more robust and prevents over fitting to improve performance. Since there is a large size of training data size, I do not worry about abandon of redundant details. And dropout really helps in my model! 
  Max pooling has no risk an increase in overfitting. It is more accurate but more expensive to compute.
 
The following figure is the summary comparison. 
![Model_Comparison](https://github.com/uranus4ever/Traffic-Sign-Classifier-CNN/blob/master/ModelAccurancyCompare.png)
 
Finnally I apply dropout and max pooling in my model.

If a well known architecture was chosen:
#### What architecture was chosen?
  LeNet.

#### Why did you believe it would be relevant to the traffic sign application?
  Because they have the same application scenario - training the data set to learn how to classifer categories. And I grayscale the input image to transfer into the same problem.

#### How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final model is believed to work well after tuning, although it still has a lot of room to improve.
* validation set accuracy of 94.8% 
* test set accuracy of 93.6%


## Test a Model on New Images

### 1. Test the model with new images and analyze performance.
Test images are isolated from training data and validation data to ensure REAL effectiveness of test accuracy. My test set accuracy is 93.6%.

### 2. Choose German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Some images might be difficult to classify because they are too dark and low contract, resulting hard for the model to extract the feature to classify correctly.

