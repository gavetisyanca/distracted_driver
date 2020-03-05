# 1. Importing project dependencies

import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template



app = Flask(__name__)

@app.route('/')
def index():
    return  """ <form method=POST enctype=multipart/form-data action="index">
            <input type=file name=myfile>
            <input type=submit>
            </form> """





# 4. DEFINING image_classify function
@app.route("/index", methods = ["GET", "POST"])
def image_classify():
    with open("./drivers_distraction_final_v","r") as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("./drivers_distraction_final_v.h5")
   

    test_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)
    img_folder = "./imgs/test/uploads"
    test_generator = test_datagen.flow_from_directory(
        directory=img_folder,
        target_size=(240, 240),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    test_generator.reset()
    pred=model.predict_generator(test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)


    labels = {'safe driving': 0, 'texting - right':1, 'talking on the phone - right': 2,
                                'texting - left': 3, 'talking on the phone - left':4, 'operating the radio': 5,
                                'drinking': 6, 'reaching behind': 7, 'hair and makeup': 8, 'talking to passenger': 9}
    # labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    result = []
    for s in predictions:
        result.append(s)
    
    return render_template('index.html', result = result)
if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
