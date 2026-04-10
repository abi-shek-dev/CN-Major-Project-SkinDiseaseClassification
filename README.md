# Skin Disease Classification from Images

## Project Description

Skin diseases affect millions of people worldwide, and early detection can make treatment faster and more effective. This project builds an AI-based skin lesion classification system using deep learning and dermoscopic images. The application allows a user to upload a skin image and receive a predicted lesion class with confidence scores.

The model is designed around the HAM10000 dataset and uses a Convolutional Neural Network approach with TensorFlow and Keras. The backend handles model loading, image preprocessing, and prediction APIs, while the frontend provides a simple web interface for image upload and result display.

This project is intended as an educational prototype and research demonstration. It does not replace a dermatologist or medical diagnosis.

## Features

- Multi-class skin lesion classification
- Real-time image prediction through a web interface
- Confidence-based ranked predictions
- Trained model support using the HAM10000 dataset
- Simple Flask backend API
- Responsive frontend for uploading and analyzing images
- Exported training metrics including accuracy, precision, recall, and F1-score

## Technologies Used

- Python
- Flask
- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- Scikit-learn
- HTML
- CSS
- JavaScript

## Dataset

This project uses the HAM10000 dataset:

- Kaggle: `https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000`

The dataset contains labeled dermoscopic images across these 7 classes:

- `akiec` - Actinic Keratoses / Intraepithelial Carcinoma
- `bcc` - Basal Cell Carcinoma
- `bkl` - Benign Keratosis-like Lesion
- `df` - Dermatofibroma
- `mel` - Melanoma
- `nv` - Melanocytic Nevus
- `vasc` - Vascular Lesion

## Project Structure

```text
CN-Major-Project-2/
|-- backend/
|   |-- app.py
|   |-- train.py
|   |-- label_info.py
|   |-- data/
|   |   |-- HAM10000_metadata.csv
|   |   |-- HAM10000_images_part_1/
|   |   |-- HAM10000_images_part_2/
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   |-- app.js
|-- requirements.txt
|-- README.md
```

## How the Project Works

1. The training script reads image metadata from the HAM10000 CSV file.
2. It matches metadata entries with the image files in the dataset folders.
3. A MobileNetV2-based deep learning model is trained on the lesion classes.
4. The trained model is saved as `skin_model.h5`.
5. The Flask backend loads the model and exposes prediction APIs.
6. The frontend sends uploaded images to the backend and displays prediction results.

## Installation and Setup

### 1. Open the project folder

Make sure your terminal is inside the root project folder:

```powershell
cd c:\Projects\CN_Intern\CN-Major-Project-2
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## How to Train the Model

Before running predictions, train the model using:

```powershell
python backend/train.py
```

### Optional training arguments

You can also run training with custom options:

```powershell
python backend/train.py --epochs 10 --batch-size 32
```

Available arguments:

- `--csv-path` for a custom metadata CSV path
- `--image-dir` for a custom image root directory
- `--model-path` for a custom model output path
- `--classes-path` for a custom class metadata output path
- `--metrics-path` for a custom training metrics output path
- `--epochs` for number of epochs
- `--batch-size` for training batch size

### Files generated after training

After training, these files will be created inside the `backend` folder:

- `skin_model.h5`
- `class_names.json`
- `training_metrics.json`

## How to Run the Application

Once the model has been trained, start the Flask application:

```powershell
python backend/app.py
```

The app will start on:

```text
http://127.0.0.1:5000
```

Open that URL in your browser.

## How to Use the Program

1. Start the backend server.
2. Open the web app in your browser.
3. Upload a skin lesion image in JPG or PNG format.
4. Click `Analyze Image`.
5. View the predicted class, confidence score, and top alternative predictions.

## API Endpoints

The backend provides these endpoints:

- `GET /` - serves the frontend
- `GET /api/health` - checks backend and model status
- `GET /api/classes` - returns supported lesion classes
- `POST /api/predict` - accepts an image file and returns prediction results

## Example Prediction Response

```json
{
  "prediction": {
    "code": "mel",
    "name": "Melanoma",
    "description": "A serious type of skin cancer that requires urgent professional evaluation.",
    "confidence": 0.873421
  },
  "topPredictions": [
    {
      "code": "mel",
      "name": "Melanoma",
      "description": "A serious type of skin cancer that requires urgent professional evaluation.",
      "confidence": 0.873421
    }
  ],
  "disclaimer": "This prototype is for educational use only and does not replace a medical diagnosis."
}
```

## Expected Outcomes

- A trained deep learning model for skin lesion classification
- A working prototype web application
- Real-time predictions from uploaded images
- Saved performance metrics for evaluation
- A demonstration of AI support in medical image analysis

## Important Note

This application is only for academic, educational, and prototype purposes. It must not be used as a replacement for professional medical advice, diagnosis, or treatment.

## Troubleshooting

### Model file not found

If you see an error saying the model file is missing, run:

```powershell
python backend/train.py
```

### TensorFlow installation issues

If TensorFlow fails to install, make sure:

- You are using a compatible Python version
- `pip` is updated

You can update pip using:

```powershell
python -m pip install --upgrade pip
```

### Backend starts but predictions fail

Check that:

- The model finished training successfully
- `skin_model.h5` exists inside the `backend` folder
- The uploaded image is valid JPG or PNG

## Author

Major project prototype for skin disease classification using deep learning and the HAM10000 dataset.
