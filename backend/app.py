from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(_name_)

# Load your trained model
model = load_model('skin_model.h5')  # Replace with your model file

# Class labels
class_names = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea']  # Replace with your classes

def prepare_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize
    return img_array

@app.route('/')
def home():
    return "Skin Disease Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream)
        img_array = prepare_image(img)
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        return jsonify({
            'disease': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    app.run(debug=True)