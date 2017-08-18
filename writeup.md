# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nvidia_arch.png "Conv Net Architecture"
[image2]: ./images/steering_angles.png "Steering Angles Histogram"

[image3]: ./images/placeholder_small.png "Recovery Image"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* preprocess.ipynb jupyter notebook with some data preprocessing and visualizations.
* behavioral_cloning.ipynb jupyter notebook containing the model (as in model.py) and the functions I used to augment and preprocess the images.

### #2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

* My model architecture was based on the original [Nvidia post:](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and [Nvidia paper](https://arxiv.org/pdf/1604.07316v1.pdf)


![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

* The model contains one dropout layer in order to reduce overfitting
* The model was trained and validated on different data sets to ensure that the model was not overfitting. 
* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for several laps.

#### 3. Model parameter tuning

The model used an adam optimizer with a starting learning rate of 1(10)^-4

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, center lane driving in the other direction and recovering from the left and right sides of the road. The training data I used came form various sources: training files provided by udacity, recovery training data from [another student's github](https://github.com/cssomnath/udacity-sdc/tree/master/carnd-projects/CarND-Behavioral-Cloning/sharp_turn.zip) and my own training data of center lane driving on track 1 and driving in the inverse direction on track 1.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to watch and take notes from the lessons on the classroom, read Nvidia's End to end learning for self dricing cars and read their post. Also it helped a lot to read on Medium my other fellow classmates posts.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
To combat the overfitting, I used one dropout layer after all the convolutional layers on the model. 

Then I tried with different number of epochs and batch sizes to improve the accuracy of the model and avoid overfitting by early stopping.

The final step was to run the simulator to see how well the car was driving around track one. The first runs the car fell off the track on the very first curves then I tried just using the data from the driving log that had a steering angle different from 0. This process is on the jupyter notebook titled "preprocess.ipynb". Then I visualized the distribution of steering angles and made shure it looked balanced.

At the end of the process, the vehicle WAS able to drive autonomously around the track 1 several laps without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                 Output Shape              Param #   
### =================================================================

cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________

lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________

conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________

conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________

conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________

conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________

conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________

dropout_1 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________

flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________

dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________

dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________

dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________

dense_4 (Dense)              (None, 1)                 11        

### =================================================================

##### Total params: 981,819


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the driving data provided by Udacity and my own center lane driving data. Also I found on a Meidum post and then on a github some excellent driving recovery data and added it to the training set. The final steering angle distribution looked like this:

![alt text][image2]

To augment the data set I used a generator to augment data 'on the fly' and defined the functions: 'augment_brightness' to add random brighntess to the image, 'augment images' to randomly flip and tilt an image from either the center, left or right camera, and 'add shadow' to add a random shadow on the images. All of this to make the model more robust to different lighting setups and roads.

Then I used the function 'batch_generator' to take a random batch of images and apply the functions mentioned above and traing the model with them on the fly. 

I randomly shuffled the data set with sklearn's 'train_test_split' and used a testing size of 20%.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10.


