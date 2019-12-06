import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TerminateOnNaN, TensorBoard, EarlyStopping, ReduceLROnPlateau

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from cancer.variables import CANCER_DATA_DIR


BATCH_SIZE = 32
EPOCHS = 1000
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
VAL_SPLIT = .2

data_path = os.path.join(CANCER_DATA_DIR, 'SIPaKMeD', 'processed_data', 'full_slide_classification', 'classes_small')

# data generation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VAL_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')


logdir = "/home/seanwade/sean/cancerDetection/logs/indavidual_slide_classification/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    TensorBoard(log_dir=logdir, write_images=True, write_grads=True),
    EarlyStopping(monitor='val_loss', patience=15),
    TerminateOnNaN(),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


# Define model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = EPOCHS,
    callbacks=callbacks
)
