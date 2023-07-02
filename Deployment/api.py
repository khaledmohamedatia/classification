from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import urllib.request
import io
from utils import *
import logging

app = Flask(__name__)

# Load the model and labels
model = model_arc()
model.load_weights("weights/model.h5")
labels = gen_labels()

logging.basicConfig(level=logging.INFO)
# # Define a helper function to preprocess the image
# def preprocess(image):
#     # Resize the image to the model's input size
#     img = image.resize((224, 224))
#     # Convert the image to a NumPy array
#     img_array = np.array(img)
#     # Preprocess the image array
#     img_array = img_array / 255.0
#     img_array = img_array - np.array([0.485, 0.456, 0.406])
#     img_array = img_array / np.array([0.229, 0.224, 0.225])
#     # Add a batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

@app.route('/predictImage', methods=['POST'])

def predictImage():

    logging.info('call predictImage api')

    # # Get the image file from the POST request
    file = request.files['image']

    # logging.info(jsonify({'img :': jsonify(file)}))

    # # Read the image file
    image = Image.open(io.BytesIO(file.read()))

    # #logging.info(jsonify({'img :': image}))


    # # Preprocess the image
    img = preprocess(image)

    # # Make a prediction with the model
    prediction = model.predict(img[np.newaxis, ...])

    # # Get the predicted label
    label = labels[np.argmax(prediction[0], axis=-1)]

    # # Return the predicted label as JSON
    return jsonify({'label': label})
    # return jsonify({'label': 'test'})

if __name__ == '__main__':
    app.run(port=8501)
