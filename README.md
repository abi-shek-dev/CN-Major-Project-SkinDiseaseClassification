# Skin Disease Classification from Images

## Project Description

Skin diseases affect millions of people worldwide, and early detection can improve treatment outcomes. This project is an AI-based skin lesion classification system that analyzes dermoscopic images and predicts the most likely lesion category with confidence scores.

The application is built as a full-stack prototype:

- A PyTorch-based deep learning training pipeline for the HAM10000 dataset
- A Flask backend for loading the trained model and serving predictions
- A responsive frontend for image upload, status checking, and result display

This project is designed for academic and demonstration purposes. It is not a replacement for professional medical advice or diagnosis.

## Features

- Multi-class skin lesion classification
- Real-time prediction from uploaded images
- Top-3 ranked predictions with confidence values
- Frontend UI for image upload and result display
- PyTorch training pipeline with GPU support on compatible NVIDIA systems
- Exported evaluation metrics including accuracy, precision, recall, and F1-score
- HAM10000 class metadata and lesion descriptions

## Technologies Used

- Python
- PyTorch
- Torchvision
- Flask
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

The dataset contains 7 lesion classes:

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
|   |-- __init__.py
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

1. The training script loads the HAM10000 metadata CSV.
2. It maps each metadata row to the real image file path.
3. A MobileNetV2 classifier is trained using PyTorch and Torchvision.
4. The best model is saved as `skin_model.pth`.
5. The Flask backend loads the PyTorch checkpoint.
6. The frontend sends uploaded images to the backend and displays predictions.

## Installation and Setup

### 1. Open the project folder

```powershell
cd c:\Projects\CN_Intern\CN-Major-Project-2
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch

For GPU training on Windows with an NVIDIA GPU, install a CUDA-enabled PyTorch build using the official PyTorch install selector:

- Official install page: `https://pytorch.org/get-started/locally/`

Example command shown on the official PyTorch site for Windows `pip` installs may look like:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If you want CPU-only PyTorch instead:

```powershell
pip install torch torchvision torchaudio
```

### 4. Install the rest of the dependencies

```powershell
pip install -r requirements.txt
```

## Check Whether GPU Is Available

Run:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If GPU support is working, you should see `True` and your NVIDIA GPU name.

## How to Train the Model

Train the model with:

```powershell
python backend/train.py
```

Optional arguments:

```powershell
python backend/train.py --epochs 10 --batch-size 32
```

Available options:

- `--csv-path` for a custom HAM10000 metadata file
- `--image-dir` for a custom dataset image root
- `--model-path` for a custom output model path
- `--classes-path` for a custom class metadata output path
- `--metrics-path` for a custom metrics JSON output path
- `--epochs` for number of epochs
- `--batch-size` for training batch size

## Files Generated After Training

After training, these files are created in `backend/`:

- `skin_model.pth`
- `class_names.json`
- `training_metrics.json`

## How to Run the Application

After training finishes, start the Flask app:

```powershell
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How to Use the Program

1. Start the Flask server.
2. Open the app in your browser.
3. Upload a JPG or PNG skin lesion image.
4. Click `Analyze Image`.
5. Review the predicted lesion class and confidence-ranked alternatives.

## API Endpoints

- `GET /` - serves the frontend
- `GET /api/health` - returns backend and model status
- `GET /api/classes` - returns supported lesion classes
- `POST /api/predict` - accepts an uploaded image and returns prediction data

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

- A trained PyTorch model for lesion classification
- A full-stack prototype for dermoscopic image analysis
- Real-time predictions through a browser UI
- Saved model performance metrics for evaluation
- A project suitable for academic demonstration and further extension

## Troubleshooting

### `skin_model.pth` not found

Train the model first:

```powershell
python backend/train.py
```

### PyTorch does not detect the GPU

Check:

- Your NVIDIA driver is installed correctly
- You installed a CUDA-enabled PyTorch build, not CPU-only
- `torch.cuda.is_available()` returns `True`

If needed, reinstall PyTorch using the official selector:

- `https://pytorch.org/get-started/locally/`

### The backend starts but predictions fail

Check that:

- Training completed successfully
- `backend/skin_model.pth` exists
- `backend/class_names.json` exists
- The uploaded image is a valid JPG or PNG file

## Important Note

This application is only for educational, academic, and prototype use. It must not be used as a substitute for professional medical diagnosis or treatment.
