import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
from io import BytesIO
from PIL import Image
import re, time, base64

app = Flask(__name__)

CORS(app)
class_name = ['apple', 'banana', 'beetroot', 'bellpepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
              'chillipepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
              'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
              'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


def selected_reshape_image(image):
    # Define the target input size of the model
    target_width = 160
    target_height = 160

    # Resize the image to match the model's input size
    resized_image = cv2.resize(image, (target_width, target_height))

    # Ensure the image has 3 color channels (e.g., convert from grayscale to RGB)
    if resized_image.shape[-1] != 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    # Add a batch dimension (None) to the image
    input_image = np.expand_dims(resized_image, axis=0)

    # Optionally, you can convert the input image to the required data type (e.g., float32)
    input_image = input_image.astype(np.float32)

    # Normalize the input image if required by the model
    # For example, divide by 255.0 to scale values to the [0, 1] range
    return input_image

def reshape_image():
    # Define the target input size of the model
    image=cv2.imread('captured_image2.jpeg')
    target_width = 160
    target_height = 160

    # Resize the image to match the model's input size
    resized_image = cv2.resize(image, (target_width, target_height))

    # Ensure the image has 3 color channels (e.g., convert from grayscale to RGB)
    if resized_image.shape[-1] != 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    # Add a batch dimension (None) to the image
    input_image = np.expand_dims(resized_image, axis=0)

    # Optionally, you can convert the input image to the required data type (e.g., float32)
    input_image = input_image.astype(np.float32)

    # Normalize the input image if required by the model
    # For example, divide by 255.0 to scale values to the [0, 1] range
    return input_image


filename = 'my_model.h5'


def predicting_value(image):
    model = tf.keras.models.load_model('my_model.h5')

    return class_name[np.argmax(model.predict(image))]


def convertPITtoPng(img):
    base64_data = re.sub('^data:image/.+;base64,', '', img)
    byte_data = base64.b64decode(base64_data)
        # You can save the image as a PNG file
    with open('captured_image2.jpeg', 'wb') as f:
        f.write(byte_data)


@app.route('/', methods=['POST', 'GET'])
def home():
    return jsonify({
        'success': True,
        'file': 'Received'
    })




@app.route('/captureImage', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        image_base64 = data.get('image', '')
        convertPITtoPng(image_base64)
        image = reshape_image()
        result = predicting_value(image)

        # Decode the base64 image to binary data
        return jsonify({'message': 'Image received and stored as captured_image.png', 'Output': result})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/selectimage', methods=['POST','GET'])
def selected_image():
    # Receive the image from the request
    try:
        image = request.files['image']
        # Read the image using OpenCV
        img_data = image.read()
        nparr = np.fromstring(img_data, np.uint8)
        image_cv2 = cv2.imdecode(nparr, cv2.COLOR_GRAY2RGB)
        image=selected_reshape_image(image_cv2)
        result = predicting_value(image)
        # Send back a response
        return jsonify({'Output': result})
    except Exception as e:
        return jsonify({'error': str(e)})






if __name__ == "__main__":
    app.run(port=8080, debug=True)
