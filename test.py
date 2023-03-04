import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pathlib
import numpy as np

model = keras.models.load_model('my_keras_model.h5')

image_height = 513
image_width = 800
resized_height = 512
resized_width = 512
class_names = ['0','1','10','11','12','13','14','15','16','17','18','19',
                '2','20','21','22','23','3','4','5','6','7','8','9']

resizing_and_rescaling_layer = keras.Sequential([
    layers.Resizing(resized_height,resized_width),
    layers.Rescaling(1./255),
])

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(resized_height, resized_width, 3)),
    layers.RandomRotation(0.01),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

preprocess_layer = keras.Sequential([
    resizing_and_rescaling_layer,
    data_augmentation,
])

data_dir = pathlib.Path('./spectrogram-dataset1-test')

def get_class(img_dir):
    img_str = str(img_dir)
    i, j = 0, 0
    while img_str[i] != '_':
        i += 1
    j = i
    i += 1
    while img_str[j] != '.':
        j += 1
    return img_str[i:j]

accu_count = 0
total_count = len(list(data_dir.glob('*.png')))
with open('result.txt', 'w') as f:
    sys.stdout = f
    for img_dir in data_dir.glob('*.png'):
        img = tf.keras.utils.load_img(img_dir, target_size=(image_height,image_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = preprocess_layer(img_array)
        img_class = get_class(img_dir)

        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])

        print("I guess this image({}) belong to class {} with a {:.2f} percent confidence."
                .format(str(img_dir),class_names[np.argmax(score)], 100 * np.max(score)))

        if class_names[np.argmax(score)] == get_class(img_dir):
            accu_count += 1

    print("The accuracy of this model is {:.2f} percent.".format(100.*accu_count/total_count))

