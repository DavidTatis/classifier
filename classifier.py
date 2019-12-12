from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
import time
import numpy as np
import cv2
#======= For only CPU ========
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#=============================

#================ GPU ===========
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
#================================

#====== load existing model =====
#new_model = tf.keras.models.load_model('./cnn-50ep-100batch.h5')
# Show the model architecture
#new_model.summary()
#================================


NB_CHANNELS = 3
BATCH_SIZE = 30
NB_TRAIN_IMG = 2000
NB_VALID_IMG = 1500

# preprocessing of train images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range = 0.2,                    
    shear_range=0.2, 
    horizontal_flip = True)

# preprocessing of validation images
validation_datagen = ImageDataGenerator(rescale=1./255)

#load train images from the folder
train_generator = train_datagen.flow_from_directory(
    '../data2/train',
    target_size=(600,450),
    class_mode='binary',
    batch_size = BATCH_SIZE)

#load validation images from the folder
validation_generator = validation_datagen.flow_from_directory(
    '../data2/validate',
    target_size=(600,450),
    class_mode='binary',
    batch_size = BATCH_SIZE)


cnn = Sequential() #A linear stack of layers.
cnn.add(Conv2D(filters=32, 
               kernel_size=(2,2), 
               strides=(1,1),
               padding='same',
               input_shape=(600,450,NB_CHANNELS),
               data_format='channels_last')) #default format that use keras for the conv2d. (height, width, channels). https://cutt.ly/Qe3TTl5
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Dropout(0.4))
cnn.add(Conv2D(filters=64,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Flatten())        
cnn.add(Dense(64))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.4))
cnn.add(Dense(1))
cnn.add(Activation('sigmoid'))

cnn.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy',tf.keras.metrics.Recall()])

checkpoint_path = "training_2/cp.ckpt"
# Create a callback that saves the model's weights between ephocs
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
start = time.time()
cnn.load_weights('cnn-50ep-100batch.h5') #load previous weights
#========== train ==========
cnn.fit_generator(
    train_generator,
    steps_per_epoch=NB_TRAIN_IMG//BATCH_SIZE,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=NB_VALID_IMG//BATCH_SIZE,
    callbacks=[cp_callback])
end = time.time()
print('Processing time:',(end - start)/60)
cnn.save('fullModel04.h5') # save the model with weights
