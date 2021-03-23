from flask import Flask, request, jsonify, url_for, render_template
import tensorflow as tf
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import StringIO
from io import BytesIO
import cv2
import pandas as pd

app = Flask(__name__)
model = load_model('EMNIST-Balanced-Model.h5',compile=True)
ascii_map = pd.read_csv("mapping.csv")


@app.route('/')
def index():
    return render_template("main.html")


@app.route('/predict',methods=["POST"])
def get_image():
    canvasdata = request.form['canvasimg']
    # print(canvasdata)
    encoded_data = request.form['canvasimg'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img.shape)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    gray_image = gray_image/255.0
    
    gray_image = np.expand_dims(gray_image, axis=-1)
    img = np.expand_dims(gray_image, axis=0)

    print('Image received: {}'.format(img.shape))
    prediction = model.predict(img)
    cl = list(prediction[0])
    print("Prediction : ",ascii_map["Character"][cl.index(max(cl))])

    ## INITIAL TF VERSION -> 2.3.0
    # print(prediction)

    return render_template("main.html", value=ascii_map["Character"][cl.index(max(cl))])





if __name__ == '__main__':
    app.run(debug=True)
            
