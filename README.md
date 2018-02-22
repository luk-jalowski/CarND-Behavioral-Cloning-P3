# **CarND-Behavioral-Cloning-P3**
My solution to Udacity Self-Driving Car behavioral cloning project 

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-final.h5 containing a trained convolution neural network 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for loading data and then training and validating the model. Code is commented to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For my model I've used network architecture described here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ with added dropout layers between fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. They are located between fully connected layers

Dataset has been split into train(80%) and validation(20%) set.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data I used contain mainly driving in the center of the road. I also included recovery data, which consisted of driving from the side of the road back to the center.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In the beginning I've started with LaNet architecture used in previous project. The model performed relatively well, but had problems on long curves. Then, I've moved to network architecture as described above, but without dropout layers at first. The model performed even better, but car got tendency to drift off at bridge and it didn't turn as expected near dirt road. To help with that I added more images to dataset, I've focused on things the network was struggling with. After this dataset grew large, so just to be sure I've added dropout layers between fully connected layers. This model was performing very well on the simulator and was capable of driving on the track without leaving the road, so I've decided that I'm going to submit it.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](examples/center_bridge.jpg)
![alt text](examples/center-drit-road.jpg)

I then recorded the vehicle recovering from sides of the road and driving to the center

![alt text](examples/left.jpg)
![alt text](examples/left2.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I flipped all the images and negated steering value.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Model was trained on 3 epochs
