# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[Nvidia_model]: ./output_img/Nvidia_model.JPG "Nvidia_model"
[loss_figure]: ./output_img/loss_track2_dropout_50.png "loss_figure"
[center_image]: ./output_img/center_2018_01_14_23_28_00_986.jpg "center_image"
[left_image]: ./output_img/left_2018_01_14_23_28_00_986.jpg "left_image"
[right_image]: ./output_img/right_2018_01_14_23_28_00_986.jpg "right_image"
[multiple_cameras]: ./output_img/carnd-using-multiple-cameras.png "multiple_cameras"
[flipped_image]: ./output_img/flipped_image.png "flipped_image"
[cropped_image]: ./output_img/cropped_image.png "cropped_image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_track2.h5 containing a trained convolution neural network 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_track2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model design

The model used is based on NVIDIA's paper - ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316v1.pdf). Here is the architecture of it.

![alt text][Nvidia_model]

In order to let the model fit the training data, improve the performance and reduce overfitting. I modified the model to the following architecture.

| Layer         		|     output shape	        					|               					description                 				   |
|:---------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------:| 
| Input         		| (None, 160, 320, 3)   						| normalize input image                     									   |
| Cropping2D     		| (None, 90, 320, 3) 							| Crop images to exclude no use info        									   |
| Convolution2D			| (None, 43, 158, 24)							| 1st convolution layer with 5x5 kernel size. Use max pooling 2X2. Activation RELU.|
| Convolution2D	      	| (None, 28, 60, 48)		 				    | 2nd convolution layer with 5x5 kernel size. Use max pooling 2X2. Activation RELU.|
| Convolution2D	    	| (None, 8, 37, 48)   							| 3rd convolution layer with 5x5 kernel size. Use max pooling 2X2. Activation RELU.|
| Convolution2D			| (None, 6, 35, 64) 							| 4th convolution layer with 5x5 kernel size. Use max pooling 2X2. Activation RELU.|
| Convolution2D	      	| (None, 4, 33, 64) 				    		| 5th convolution layer with 5x5 kernel size. Use max pooling 2X2. Activation RELU.|
| Flatten		        | (None, 8448)        							| flatten layer																	   |
| Fully Connected		| (None, 100)		        			        | fully connected layer. Output 100												   |
| RELU					| (None, 100)									| RELU activation																   |
| Dropout layer  	    | (None, 100)									| Dropout layer with prob 0.5                                                      |
| Fully Connected		| (None, 50)		        			        | fully connected layer. Output 50												   |
| RELU					| (None, 50)									| RELU activation																   |
| Dropout layer  	    | (None, 50)									| Dropout layer with prob 0.5                                                      |
| Fully Connected		| (None, 10)		        			        | fully connected layer. Output 10												   |
| RELU					| (None, 10)									| RELU activation																   |
| Dropout layer  	    | (None, 10)									| Dropout layer with prob 0.5                                                      |
| Fully Connected		| (None, 1)	 	        			            | fully connected layer. Output 1												   |



#### 2. Attempts to reduce overfitting in the model

As shown in the architecture above, the model contains dropout layers with prob 0.5 in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .
I tuned epochs and batch size to lower the loss and increase training speed.
After tuning, they are

batch_size = 128
epochs =50

Here is the loss figure,

![alt text][loss_figure]


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

#### 5. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps(two laps reverse) on track one using center lane driving. Here are example images of center/left/right lane driving:

![alt text][center_image]

![alt text][left_image]

![alt text][right_image]

As following image shows the angles from left and right camera are different from center camera. 

![alt text][multiple_cameras]

So `correction = +/- 0.2` is added to the steer angle. 

In order to get more data I also flipped the image to create new training sets. Original and flipped images are as follows:

![alt text][center_image]  ![alt text][flipped_image]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

In the training model, in order to exclude useless part of image, I used cropping layer to crop the input image with (cropping=((50,20), (0,0))), here are the original image and cropped image,

![alt text][center_image]

![alt text][cropped_image]

after cropping, image size becomes 90 * 320.

Then the model trained with 1st track data(model.h5) runs well on 1st track. 

#### 6. 2nd Track

Regarding the 2nd track, at first, I use the model that is trained with only 1st track images(model.h5) to run it. It failed.

I think it is because the 2nd track contains some info that 1st doesn't have(sharp turns/2 lanes/....). Then I collect some training data in 2nd track and put it together with 1st track data to the training model. 

The trained model(model_track2.h5) works fine on both 1st track(run1.mp4) and 2nd track(run2.mp4).

