import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split

#Python generator used for feeding data into neural network during training
def generator(samples, batch_size=36):
    batch_size = int(np.floor(batch_size/6.0))
    while 1:
        n_samples = len(samples)
        shuffle(samples)
        
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            
            for batch in batch_samples:
                folder_path = './data/IMG/'
                #Loading image from three cameras from dataset
                image = cv2.imread(folder_path + batch[0].split('/')[-1])
                image2 = cv2.imread(folder_path + batch[1].split('/')[-1])
                image3 = cv2.imread(folder_path + batch[2].split('/')[-1])
                images.extend([image, image2, image3])
                #Flipping camera images to increase number of examples
                images.extend([cv2.flip(image,1), cv2.flip(image2,1), cv2.flip(image3,1)])
                angle = float(batch[3])
                #Steering correction is used on images from left and right camera /
                #because image is shifted to side
                correction = 0.25
                angles.extend([angle, angle + correction, angle - correction])
                #Steering angles for flipped images need to be negated
                angles.extend([angle*-1.0, (angle + correction)*-1.0, (angle - correction)*-1.0])
                
            X_train = np.array(images)
            y_train = np.array(angles)
  
            yield sklearn.utils.shuffle(X_train, y_train)
            
       
lines = []

#Read contents of driving_log.csv file which contains dataset
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
 #First line in dataset desribes data in csv file so we need to get rid of it 
del lines[0] 

#Number of lines read from .csv file
n_elements = len(lines)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D

#Neural Network architecture
#Using architecture described in https://devblogs.nvidia.com/deep-learning-self-driving-cars/
#Added dropout layer after each fully connected layer
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape =(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


#Split dataset into train and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
#Initialize generators
train_generator = generator(train_samples, batch_size=30)
validation_generator = generator(validation_samples, batch_size=30)
#Train network using generator
model.fit_generator(train_generator,
                    samples_per_epoch = n_elements*6,
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples)*6,
                    nb_epoch=3)

#Save trained model on drive
model.save('model.h5')