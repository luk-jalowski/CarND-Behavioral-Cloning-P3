import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32):
    while 1:
        n_samples = len(samples)
        shuffle(samples)
        
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            
            for batch in batch_samples:
                folder_path = './data/IMG/'
                
                image = cv2.imread(folder_path + batch[0].split('/')[-1])
                image2 = cv2.imread(folder_path + batch[1].split('/')[-1])
                image3 = cv2.imread(folder_path + batch[2].split('/')[-1])
                images.extend([image, image2, image3])
                images.extend([cv2.flip(image,1), cv2.flip(image2,1), cv2.flip(image3,1)])
                angle = float(batch[3])
                correction = 0.25
                angles.extend([angle, angle + correction, angle - correction])
                angles.extend([angle*-1.0, (angle + correction)*-1.0, (angle - correction)*-1.0])
                
            X_train = np.array(images)
            y_train = np.array(angles)
  
            yield sklearn.utils.shuffle(X_train, y_train)
            
            
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
del lines[0]
n_elements = len(lines)
current_element = 0
print(len(lines))


#X_train = np.array(images)
#y_train = np.array(measurements)
#
#for image, measurement in zip(images, measurements):
#    #augmented_images.append(image)
#    #augmented_measurements.append(measurement)
#    #augmented_images.append(cv2.flip(image,1))
#    #augmented_measurements(measurement*-1.0)
#    X_train = np.append(X_train, cv2.flip(image,1))
#    y_train = np.append(y_train, measurement*-1.0)
    
#
#
#images = None
#measurements = None
#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)
print("Finished augmenting data")

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

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
#for i in range(np.int(np.ceil(n_elements/1024.0))):
#    n_images =0
#    if ((n_elements - 1024*i) % 1024) > 0:
#        n_images = 1024
#    else:
#        n_images = n_elements - 1024*i
#    X_train, y_train, current_element = getBatchElements(n_images, current_element)
#    current_element += n_images*6
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.fit_generator(train_generator,
                    samples_per_epoch = n_elements*6,
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples)*6,
                    nb_epoch=3)

model.save('model.h5')