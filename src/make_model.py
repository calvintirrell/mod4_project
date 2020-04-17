from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os, sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
##TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D,  Dropout, Dense, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools

def create_dir():
    data = os.path.join(os.pardir, os.pardir, 'data')
    
    train_dir = os.path.join(data, 'train')
    validation_dir = os.path.join(data, 'test')

    # Directory with our training normal pictures
    train_normal_dir = os.path.join(train_dir, 'NORMAL')
    print ('Total training normal images:', len(os.listdir(train_normal_dir)))

    # Directory with our training pneumonia pictures
    train_pna_dir = os.path.join(train_dir, 'PNEUMONIA')
    print ('Total training pneumonia images:', len(os.listdir(train_pna_dir)))

    # Directory with our validation normal pictures
    validation_normal_dir = os.path.join(validation_dir, 'NORMAL')
    print ('Total validation normal images:', len(os.listdir(validation_normal_dir)))

    # Directory with our validation pneumonia pictures
    validation_pna_dir = os.path.join(validation_dir, 'PNEUMONIA')
    print ('Total validation pneumonia images:', len(os.listdir(validation_pna_dir)))

    return train_dir,validation_dir

def create_images():
    input_path = '../../data/'
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax = ax.ravel()
    plt.tight_layout()

    for i, _set in enumerate(['train', 'val', 'test']):
        set_path = input_path + _set
        ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')
        ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
        ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')
        ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
    plt.show()
    
def create_generator(train_dir, validation_dir,image_size,batch_size):
  
    # Rescale all images by 1./255 and apply image augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255)

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
                    train_dir,  # Source directory for the training images
                    target_size=(image_size, image_size),
                    batch_size=batch_size,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary',
                    shuffle=False)

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
                    validation_dir, # Source directory for the validation images
                    target_size=(image_size, image_size),
                    batch_size=batch_size,
                    class_mode='binary',
                    shuffle=False)
    return train_generator, validation_generator

def make_model(train_generator, validation_generator,image_size, batch_size):
    ## set IMG_SHAPE for the model.fit parameter
    IMG_SHAPE = (image_size, image_size, 3)

    model2 =Sequential()
    model2.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=IMG_SHAPE))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(BatchNormalization())

    model2.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(BatchNormalization())

    model2.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(BatchNormalization())

    model2.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(BatchNormalization())

    model2.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(BatchNormalization())
    # model2.add(Dropout(0.2))

    model2.add(Flatten())
    model2.add(Dense(256, activation='relu'))
    # model2.add(Dropout(0.2))

    model2.add(Dense(128, activation='relu'))
    model2.add(Dropout(0.2))

    model2.add(Dense(1, activation = 'sigmoid'))

    model2.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    epochs = 10
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    # es = EarlyStopping(monitor='val_loss', patience=0)

    history = model2.fit(train_generator,
                        steps_per_epoch = steps_per_epoch,
                        epochs=epochs,
                        workers=4,
                        validation_data=validation_generator,
                        validation_steps=validation_steps
                                )

    return model2

# def save_model():
#     model2.save('best_model.h5')


def read_model(file_name):
    load_model(file_name)

def create_y_test_pred(model, validation_generator, threshold):
    validation_generator.reset()
    y_test = validation_generator.classes
    y_pred=model.predict_generator(validation_generator, verbose=1)
    y_pred = (y_pred>threshold).astype(int)

    return y_test, y_pred

