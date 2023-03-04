#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import pathlib
data_dir = pathlib.Path("./dataset")

image_count = len(list(data_dir.glob('*/*.png')))
print("There are {} images in the dataset.".format(image_count))


batch_size = 32
img_height = 513
img_width = 800
resized_height = 512
resized_width = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print("Class names:")
print(class_names)

for image_batch, labels_batch in train_ds:
    print("The shape of a batch of images:")
    print(image_batch.shape)
    print("The shape of a batch of labels:")
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


resizing_and_rescaling_layer = keras.Sequential([
    layers.Resizing(resized_height, resized_width),
    layers.Rescaling(1./255),
])

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(resized_height, resized_width, 3)),
    layers.RandomRotation(0.01),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
  ]
)

preprocess_layer = keras.Sequential([
    resizing_and_rescaling_layer,
    data_augmentation,
])

preprocessed_ds = train_ds.map(lambda x, y: (preprocess_layer(x), y))
preprocessed_vs = val_ds.map(lambda x, y: (preprocess_layer(x), y))


num_classes = len(class_names)

model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(resized_height,resized_width,3),
    pooling=None,
    classes=num_classes,
    classifier_activation='softmax'
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


epochs=200
history = model.fit(
  x=preprocessed_ds,
  validation_data=preprocessed_vs,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./Accuracy_Loss.png')

model.save("my_keras_model.h5")


