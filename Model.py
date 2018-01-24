# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 22:25:54 2018

@author: benbe
"""
import numpy as np
import cv2
import csv
import tensorflow as tf
from os import getcwd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from PIL import Image


#use multiple cameras images by adding correction to left&right
def use_multiple_cameras(datapaths, correction):
    car_images_paths = []
    steering_angles = []
    for j in range(2):
        with open (datapaths[j] + '/driving_log.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                steering_center = float(row[3])
                
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                 # fill in the path to your training IMG directory
                 # add images and angles to data set
                img_center_path = img_path_pre[j] + row[0]
                car_images_paths.append(img_center_path)
                steering_angles.append(steering_center)
                img_left_path = img_path_pre[j] + row[1]
                img_right_path = img_path_pre[j] + row[2]
                car_images_paths.append(img_left_path)
                car_images_paths.append(img_right_path)
                steering_angles.append(steering_left)
                steering_angles.append(steering_right)
        
    return (car_images_paths,steering_angles)

def generator(samples, batch_size=32, validation_flag = False):
     
    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                if originalImage is not None:
                    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                else:
                    continue
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)
#                if offset == 0:
#                    print(images[0].shape)

            
            inputs = np.array(images)
            outputs = np.array(angles)
            yield shuffle(inputs, outputs)
    


#nvidia model
def NV_model():
    model = Sequential()
    #normalize images
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
    #crop images
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(1))
    return model

#data paths
my_data_path = './My_data/'
my_data_path_2 = './My_data_2'
uda_data_path = './uda_data/'
img_path_pre = ['','',uda_data_path]

datapaths = [my_data_path, my_data_path_2, uda_data_path]

#set hyperparameters
correction = 0.2
batch_size = 256
epoch = 100

car_image_paths, measurements = use_multiple_cameras(datapaths, correction)


# Splitting samples
samples = list(zip(car_image_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(car_image_paths))
print(len(measurements))
#img = cv2.imread(car_image_paths[0])
#image_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(image_new)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

#create generators
train_generator = generator(train_samples, batch_size, validation_flag = False)
validation_generator = generator(validation_samples, batch_size, validation_flag = True)

#create model
model = NV_model()

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch = epoch, verbose=1)

model.save('model_track2.h5')
#print(history_object.history.keys())
#print('Loss')
#print(history_object.history['loss'])
#print('Validation Loss')
#print(history_object.history['val_loss'])

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('./output_img/loss_figure_1.png')
#


#count = 0
#car_image = []
#with open (my_data_path + '/driving_log.csv') as f:
#    reader = csv.reader(f)
#    for row in reader:
#        if count < 1:
#            print(row)
#            print(row[0])
#            print(cv2.imread(row[0]).shape)
#            img = Image.open(row[0])
#            img.show()
#            #cv2.imshow(cv2.imread(row[0]))
#        count += 1
        
        #cv2.imshow(row[0])
        
        
                