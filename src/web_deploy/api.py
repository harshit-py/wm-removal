# encoding: utf-8
"""
@author: Harshit Saxena
"""

import io
import json
from pathlib import Path
import PIL
import flask
import torch
import fastai
from fastai.vision import *
from fastai.basic_train import load_learner
from flask import jsonify
from torch import nn
import os
import sys

import_path = os.getcwd()
sys.path.insert(0, import_path)

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
use_gpu = False
model = None


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": True, "version": "0.1.0", "api_version": "v1"})


def prepare_image(image_path):
    imsize = (128, 128)
    img = PIL.Image.open(image_path)
    img = np.asarray(img.resize(imsize, resample=PIL.Image.BILINEAR).convert('RGB'),
                                 dtype=np.float32).transpose((2,0,1))/255 # pytorch dim
    img = torch.from_numpy(img)
    return Image(img)


@app.route("/addwatermark", methods=["POST"])
def add_watermark():
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]
            f.save(os.path.join('static/input.jpeg'))  # save file to disk

            # add watermark logic
            main = PIL.Image.open(f.filename)
            mark = PIL.Image.open('static/pngkit_copyright-symbol-png_185408.png')
            mask = mark.convert('L').point(lambda x: min(x, 25))
            mark.putalpha(mask)
            mark_width, mark_height = mark.size
            main_width, main_height = main.size
            aspect_ratio = mark_width / mark_height
            new_mark_width = main_width * 0.25
            mark.thumbnail((new_mark_width, new_mark_width / aspect_ratio), PIL.Image.ANTIALIAS)

            tmp_img = PIL.Image.new('RGB', main.size)

            for i in range(0, tmp_img.size[0], mark.size[0]):
                for j in range(0, tmp_img.size[1], mark.size[1]):
                    main.paste(mark, (i, j), mark)
                    main.thumbnail((8000, 8000), PIL.Image.ANTIALIAS)
            main.save('static/watermark_added.jpg', quality=100) # save watermarked file


            # Return the prediction to HTML Template
            return flask.render_template("add_watermark.html")

@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/prediction", methods=["POST"])
def html_predict():

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST":
        print(flask.request.files)
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]

            # Preprocess the image and prepare it for classification.
            image = prepare_image(f.filename)

            # Classify the input image and then initialize the list of predictions to return to the client.
            im = PIL.Image.fromarray((prediction[2].numpy().transpose((1,2,0))*255).astype(np.uint8))
            im.save('static/output.jpg')
            return flask.render_template("output.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST":
        print(flask.request.files)
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]
            f.save(f.filename)  # save file to disk

            # Preprocess the image and prepare it for classification.
            image = prepare_image(f.filename)

            # Classify the input image and then initialize the list of predictions to return to the client.
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_prediction = predicted.numpy()[0]
            data["prediction"] = classes[class_prediction]

            # Indicate that the request was a success.
            data["success"] = True
            os.remove(f.filename)

            # Return the data dictionary as a JSON response
            return flask.jsonify(data)


def load_model() -> None:
    # Define model
    global model
    try:
        model = load_learner(path='../../models/', file='wm_remove.pkl')
        model.model.eval()
    except:
        print('Error Loading model, check model')



if __name__ == "__main__":
    print("Loading PyTorch model and Flask starting server.")
    print("Please wait until server has fully started...")
    load_model()
    app.run(debug=True)


