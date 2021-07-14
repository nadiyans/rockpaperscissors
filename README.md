# rock-paper-scissors
My final submission for dicoding's machine learning project.

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

!wget --no-check-certificate \
  https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip

import zipfile,os
local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
 
base_dir = '/tmp/rockpaperscissors/rps-cv-images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

os.listdir('/tmp/rockpaperscissors/rps-cv-images')

pip install split_folders

import split_folders
split_folders.ratio('/tmp/rockpaperscissors/rps-cv-images', output='/tmp/rockpaperscissors/data', seed=1337, ratio=(.8,.2))
folder='/tmp/rockpaperscissors/data'
train_dir=os.path.join(folder, 'train')
validation_dir=os.path.join(folder, 'val')

train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                horizontal_flip=True,
                shear_range = 0.2,
                fill_mode = 'wrap')
 
test_datagen = ImageDataGenerator(
               rescale=1./255,
               rotation_range=20,
               horizontal_flip=True,
               shear_range = 0.2,
               fill_mode = 'wrap')

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 200),
        batch_size=8,
        class_mode='categorical')
 
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 200),
        batch_size=8,
        class_mode='categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')])

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=40,
      verbose=1)

import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

uploaded = files.upload()

for fn in uploaded.keys():
  path = fn
  img = image.load_img(path, target_size=(300,200))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=8)
  
print(fn)
if classes[0][0]==1:
  print('paper')
elif classes[0][1]==1:
  print('rock')
elif classes[0][2]==1:
  print('scissors')

