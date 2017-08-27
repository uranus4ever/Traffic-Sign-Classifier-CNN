# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)：German Traffic Sign Dataset (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Data Set Summary & Exploration


I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43



### Design and Test a Model Architecture

#### 1. Preprocess the image data - Normalization.
My first step to process the image data is to normalize. Next step is pad the data from 32x32 to 36x36.

#### 2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 36x36x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x8 				|
| Convolution 5x5	    | outputs 12x12x16      									|
| RELU		|         									|
| Max pooling				| outputs 6x6x16        									|
|	Flatten					|	outputs 576											|
|	Fully connected					|	outputs 192											|
|	Fully connected					|	outputs 80											|
|	Softmax					|	outputs 43											|
 


#### 3. How to train my model. 
To train the model, I used an optimizer, batch size = 128, epochs = 10, learning rate = 0.001.

#### 4. Model results

My final model results were:
* training set accuracy of ?
* validation set accuracy of 83% 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The last image might be difficult to classify because the contrast ratio of the image is low.


