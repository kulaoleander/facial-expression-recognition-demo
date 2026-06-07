# Facial Expression Recognition Demo

This project is a beginner-friendly deep learning and computer vision demo for facial expression recognition.

It implements a complete first-version AI pipeline using PyTorch, including dataset inspection, data loading, CNN model training, evaluation, model saving/loading, single image prediction, a Streamlit web demo, and pytest-based automated tests.

The main goal of this project is not to achieve state-of-the-art accuracy, but to build a complete, explainable, reproducible AI application pipeline that can be shown on GitHub, discussed in interviews, and extended later.

## Project Overview

The project predicts facial expressions from face images.

The current model is a simple CNN baseline trained on a FER2013-style dataset with 7 emotion classes:

* angry
* disgust
* fear
* happy
* neutral
* sad
* surprise

Current input image format:

* grayscale image
* 48x48 pixels
* tensor shape: `[batch_size, 1, 48, 48]`

Model output format:

* logits shape: `[batch_size, 7]`

## Current Features

Implemented features:

* Project folder structure
* Python virtual environment setup
* Dependency management with `requirements.txt`
* Dataset folder documentation
* FER2013-style dataset structure inspection
* Image count checking for train/test splits
* Sample image mode and size inspection
* PyTorch `ImageFolder` dataset loading
* DataLoader batch creation
* Simple CNN baseline model
* Model forward pass validation
* Training loop with `CrossEntropyLoss` and Adam optimizer
* Multi-epoch training
* Test accuracy evaluation
* Model weight saving to `.pth`
* Model weight loading from `.pth`
* Single image prediction
* Streamlit web demo
* Automated tests with pytest

## Tech Stack

* Python
* PyTorch
* torchvision
* Pillow
* NumPy
* pandas
* matplotlib
* scikit-learn
* OpenCV
* Streamlit
* pytest

## Project Structure

```text
facial-expression-recognition-demo/
├── app/
│   ├── README.md
│   └── streamlit_app.py
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   └── processed/
├── notebooks/
│   └── README.md
├── outputs/
│   ├── README.md
│   └── models/
│       └── simple_cnn.pth
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── environment_check.py
│   ├── evaluate.py
│   ├── inspect_dataset.py
│   ├── load_model.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── README.md
│   ├── test_data_loader.py
│   ├── test_environment_check.py
│   ├── test_evaluate.py
│   ├── test_inspect_dataset.py
│   ├── test_load_model.py
│   ├── test_model.py
│   ├── test_predict.py
│   └── test_train.py
├── .gitignore
├── README.md
└── requirements.txt
```

Note: `data/raw/`, `data/processed/`, and model weight files such as `.pth` are ignored by Git and are not uploaded to GitHub.

## Dataset

This project is designed for a FER2013-style facial expression dataset organized as image folders.

Expected local dataset structure:

```text
data/raw/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

The dataset files are not committed to GitHub because image datasets are usually large.

## Setup

Create and activate a virtual environment first.

Example on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Check

Run:

```powershell
python -m src.environment_check
```

This checks whether the main Python dependencies can be imported successfully.

## Dataset Inspection

Run:

```powershell
python -m src.inspect_dataset
```

This checks:

* train/test folder structure
* emotion class folders
* image counts
* sample image mode
* sample image size

Expected sample image format:

```text
mode: L
size: (48, 48)
```

## Train the Model

Run:

```powershell
python -m src.train
```

The current training script:

* creates train/test DataLoaders
* trains a simple CNN baseline for multiple epochs
* evaluates test accuracy after each epoch
* saves model weights to:

```text
outputs/models/simple_cnn.pth
```

Example result from the baseline training run:

```text
Epoch 1/3 | loss: 1.6252 | test accuracy: 0.4055
Epoch 2/3 | loss: 1.4232 | test accuracy: 0.4749
Epoch 3/3 | loss: 1.3043 | test accuracy: 0.5052
```

Results may vary because model initialization and data shuffling are random.

## Load the Saved Model

Run:

```powershell
python -m src.load_model
```

This loads:

```text
outputs/models/simple_cnn.pth
```

and checks whether the loaded model can still produce output with shape:

```text
[32, 7]
```

## Single Image Prediction

Run:

```powershell
python -m src.predict
```

This loads the saved model and predicts the emotion of one sample image.

Example output:

```text
Predicted class: happy
Confidence: 0.8929
```

The prediction result depends on the selected image and the trained model weights.

## Streamlit Demo

Run:

```powershell
streamlit run app/streamlit_app.py
```

The web app allows users to:

* upload an image
* preview the uploaded image
* run emotion prediction
* see predicted emotion and confidence

Important: before running the demo, make sure the model weight file exists:

```text
outputs/models/simple_cnn.pth
```

If it does not exist, run:

```powershell
python -m src.train
```

first.

## Run Tests

Run all tests:

```powershell
python -m pytest
```

Current test coverage includes:

* environment check
* dataset inspection
* DataLoader output shape and dtype
* CNN model output shape
* training loop
* model saving
* model loading
* evaluation accuracy
* single image prediction

Current local test result:

```text
29 passed
```

## Current Baseline Result

The current SimpleCNN baseline can learn useful patterns from the FER2013-style dataset.

A sample 3-epoch training run reached around 50% test accuracy.

This is a reasonable first baseline, but it is not a final high-performance model.

## Limitations

Current limitations:

* The model is a simple CNN baseline.
* The model is trained on 48x48 grayscale face images.
* Uploaded real-world photos may not work well if the face is not cropped.
* There is no automatic face detection or face alignment yet.
* The model may struggle with class imbalance, especially minority classes such as disgust.
* Confidence scores are based on softmax and should not be treated as calibrated probabilities.
* The project currently focuses on building a complete pipeline rather than achieving state-of-the-art accuracy.

## Next Steps

Possible next improvements:

* Add face detection and automatic face cropping
* Add confusion matrix visualization
* Add per-class accuracy / precision / recall / F1-score
* Save training history
* Plot loss and accuracy curves
* Improve CNN architecture with BatchNorm and Dropout
* Add model checkpointing for best test accuracy
* Improve Streamlit UI
* Add README screenshots
* Prepare a short project explanation for CV and interviews

## Project Status

First complete working pipeline implemented.

The project currently supports:

```text
Dataset
↓
DataLoader
↓
CNN model
↓
Training
↓
Evaluation
↓
Model saving
↓
Model loading
↓
Single image prediction
↓
Streamlit demo
↓
pytest tests
```
