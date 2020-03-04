import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

ds = pd.read_csv('C:/Users/georg/Desktop/CNN_classification/driver_imgs_list.csv')


# Defining  train Image Generator
train_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)
train_generator = train_datagen.flow_from_directory(
    directory=r"imgs/train/",
    target_size=(240, 240),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
# Defining Validation generator 
valid_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"imgs/valid/",
    target_size=(240, 240),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
# Defining test generator 
test_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)
test_generator = test_datagen.flow_from_directory(
    directory=r"imgs/test/uploads",
    target_size=(240, 240),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


model = tf.keras.models.Sequential()
# Adding the first convolutional layer and the max-pooling layer

# CNN layer hyper-parameters:
# - filters: 32
# - kernel_size:3
# - padding: same
# - activation: relu

# MaxPool layer hyper-parameters:
# - pool_size: 2
# - strides: 2
# - padding: valid
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', input_shape = (240, 240, 3), data_format = 'channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the second convolutional layer and the max-pooling layer

# CNN layer hyper-parameters:
# - filters: 32
# - kernel_size:3
# - padding: same
# - activation: relu

# MaxPool layer hyper-parameters:
# - pool_size: 2
# - strides: 2
# - padding: valid
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the third convolutional layer and the max-pooling layer

# CNN layer hyper-parameters:
# - filters: 32
# - kernel_size:3
# - padding: same
# - activation: relu

# MaxPool layer hyper-parameters:
# - pool_size: 2
# - strides: 2
# - padding: valid
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


# Adding the flattening layer
model.add(tf.keras.layers.Flatten())


# Adding the Dense layers (1024, 256, 10)
# Dense layer hyper-parameters:

# units/neurons: 10 (number of classes)
# activation: softmax
model.add(Dense(units = 1024, activation = 'relu'))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 10, activation = 'sigmoid'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# model.fit_generator(generator=train_generator,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=valid_generator,
#                     validation_steps=STEP_SIZE_VALID,
#                     epochs=10
# )
# Saving Topology
# model_json = model.to_json()
# with open("drivers_distraction_final_v", "w") as json_file:
#     json_file.write(model_json)

#Saving network weights
# model.save_weights("drivers_distraction_final_v.h5")

# model.evaluate_generator(generator=valid_generator,
# steps=STEP_SIZE_VALID)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]