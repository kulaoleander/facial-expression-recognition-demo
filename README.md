# Facial Expression Recognition Demo

An end-to-end deep learning and computer vision project for facial expression recognition.

This project predicts one of seven facial expressions from face images and includes a complete AI application pipeline: dataset inspection, PyTorch data loading, CNN baselines, transfer learning with ResNet18, training and validation, final test evaluation, model saving/loading, single-image prediction, OpenCV face detection, Streamlit demo, and pytest-based automated tests.

The goal of this project is not to claim state-of-the-art performance, but to build a complete, explainable, reproducible AI application that can be shown on GitHub, discussed in interviews, and extended into real-time computer vision applications.

---

## Project Overview

The project classifies facial expressions into seven categories:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

Input format:

```text
grayscale image
48x48 pixels
tensor shape: [batch_size, 1, 48, 48]
```

Model output format:

```text
logits shape: [batch_size, 7]
```

The project started with a simple CNN baseline and was gradually improved with validation/test split, data augmentation, ImprovedCNN, transfer learning with ResNet18, robust prediction outputs, OpenCV-based face crop support, and final test set evaluation.

---

## Main Features

- FER2013-style dataset structure inspection
- PyTorch `ImageFolder` data loading
- Train / validation / test split
- Data augmentation for training data
- SimpleCNN baseline model
- ImprovedCNN with BatchNorm and Dropout
- ResNet18 transfer learning model
- Training loop with `CrossEntropyLoss` and Adam optimizer
- Validation accuracy tracking
- Best model checkpoint saving
- Final test set evaluation
- Training history saved as JSON
- Training curves saved as PNG
- Experiment comparison logging as CSV
- Detailed evaluation with classification report and confusion matrix
- Single-image prediction
- Top-3 prediction probabilities
- Low-confidence warning
- OpenCV Haar Cascade face detection
- Largest face crop with fallback to original image
- Streamlit web demo
- Automated tests with pytest

---

## Tech Stack

- Python
- PyTorch
- torchvision
- OpenCV
- Pillow
- NumPy
- pandas
- matplotlib
- scikit-learn
- Streamlit
- pytest

---

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
├── docs/
│   ├── baseline_v0.1.md
│   └── experiment_comparison.md
├── notebooks/
│   └── README.md
├── outputs/
│   ├── README.md
│   ├── figures/
│   ├── logs/
│   └── models/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── environment_check.py
│   ├── evaluate.py
│   ├── evaluate_detailed.py
│   ├── face_detection.py
│   ├── final_test_evaluation.py
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
│   ├── test_evaluate_detailed.py
│   ├── test_face_detection.py
│   ├── test_final_test_evaluation.py
│   ├── test_inspect_dataset.py
│   ├── test_load_model.py
│   ├── test_model.py
│   ├── test_predict.py
│   └── test_train.py
├── .gitignore
├── README.md
└── requirements.txt
```

Note: `data/raw/`, `data/processed/`, trained model weights, generated logs, and generated figures are ignored by Git and are not uploaded to GitHub.

---

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

The dataset is not committed to GitHub because image datasets are usually large.

---

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Environment Check

Run:

```powershell
python -m src.environment_check
```

This checks whether the main Python dependencies can be imported successfully.

---

## Dataset Inspection

Run:

```powershell
python -m src.inspect_dataset
```

This checks:

- train/test folder structure
- emotion class folders
- image counts
- sample image mode
- sample image size

Expected sample image format:

```text
mode: L
size: (48, 48)
```

---

## Train the Model

Run:

```powershell
python -m src.train
```

The training script supports multiple model choices in code:

```python
model_name = "simple_cnn"
model_name = "improved_cnn"
model_name = "resnet18"
```

For ResNet18 transfer learning:

```python
model_name = "resnet18"
use_pretrained = True
```

The training pipeline:

```text
Dataset
↓
Train / validation / test split
↓
Data augmentation on training set
↓
Model training
↓
Validation accuracy tracking
↓
Best model checkpoint saving
↓
Training history and curve saving
↓
Experiment result logging
```

Generated outputs include:

```text
outputs/models/
outputs/logs/training_history.json
outputs/logs/experiment_results.csv
outputs/figures/training_curves.png
```

---

## Model Comparison

A controlled experiment was run using the same validation split and training configuration.

| Model | Validation Accuracy |
|---|---:|
| SimpleCNN | 0.4142 |
| ImprovedCNN | 0.4644 |
| ResNet18 from scratch | 0.4715 |
| Pretrained ResNet18 | 0.5078 |

The pretrained ResNet18 model achieved the best validation accuracy in this project.

This shows the value of transfer learning: instead of learning visual features from scratch, the model can reuse general image features learned from large-scale image data and fine-tune them for facial expression recognition.

---

## Final Test Evaluation

After selecting the best model based on validation accuracy, the pretrained ResNet18 model was evaluated once on the held-out test set.

| Metric | Value |
|---|---:|
| Test samples | 7178 |
| Test accuracy | 0.5059 |
| Macro average F1-score | 0.3993 |
| Weighted average F1-score | 0.4825 |

Per-class accuracy:

| Class | Accuracy |
|---|---:|
| angry | 0.4635 |
| disgust | 0.0000 |
| fear | 0.0889 |
| happy | 0.7880 |
| neutral | 0.5053 |
| sad | 0.4106 |
| surprise | 0.6775 |

The final test result shows that the model performs best on classes such as happy and surprise, while minority or visually ambiguous classes such as disgust and fear remain challenging. This is a realistic limitation of facial expression recognition on low-resolution FER2013-style data.

The final test evaluation outputs are saved locally to:

```text
outputs/logs/final_test_metrics.json
outputs/figures/final_test_confusion_matrix.png
```

---

## Detailed Evaluation

Run:

```powershell
python -m src.evaluate_detailed
```

This generates:

- overall accuracy
- classification report
- per-class precision, recall, and F1-score
- confusion matrix
- per-class accuracy

This is useful because overall accuracy alone can hide weak classes, especially when the dataset is imbalanced.

---

## Final Test Evaluation Script

Run:

```powershell
python -m src.final_test_evaluation
```

This script:

- loads the default best model
- creates the test DataLoader
- evaluates the model on the held-out test set
- calculates test accuracy, macro F1-score, weighted F1-score, per-class accuracy, and confusion matrix
- saves the final test metrics to JSON
- saves the final test confusion matrix as PNG

The default best model is:

```text
outputs/models/resnet18.pth
```

---

## Load the Saved Model

Run:

```powershell
python -m src.load_model
```

By default, the project now loads the ResNet18 model from:

```text
outputs/models/resnet18.pth
```

Expected output shape:

```text
torch.Size([32, 7])
```

---

## Single Image Prediction

Run:

```powershell
python -m src.predict
```

By default, prediction now loads the ResNet18 model from `outputs/models/resnet18.pth`.

The prediction output includes:

- predicted class
- confidence
- top-3 predictions
- low-confidence flag
- face crop status
- number of detected faces
- selected face box

Example output:

```text
Single image prediction
----------------------------------------
Use face crop: True
Face found: False
Number of faces: 0
Face box: None
Predicted class: happy
Confidence: 0.7702
Low confidence: False
Top predictions:
- happy: 0.7702
- angry: 0.0646
- neutral: 0.0569
```

If no face is detected, the system falls back to the original image instead of crashing.

---

## Streamlit Demo

Run:

```powershell
streamlit run app/streamlit_app.py
```

By default, the Streamlit demo uses the ResNet18 model from `outputs/models/resnet18.pth`.

The web app allows users to:

- upload an image
- preview the uploaded image
- enable or disable face detection / face crop
- see whether a face was detected
- view the image used for prediction
- run emotion prediction
- see predicted emotion and confidence
- see top-3 prediction probabilities
- see a low-confidence warning when appropriate

---

## Face Detection and Face Crop

The project uses OpenCV Haar Cascade face detection.

The face crop pipeline is:

```text
Uploaded image
↓
OpenCV face detection
↓
Select largest detected face
↓
Add padding around face box
↓
Crop face region
↓
Resize to 48x48
↓
Run expression prediction
```

If no face is detected:

```text
No detected face
↓
Use original image as fallback
↓
Continue prediction
```

This makes the demo more robust for real-world uploaded images.

---

## Run Tests

Run all tests:

```powershell
python -m pytest
```

Current local test result:

```text
85 passed
```

Current test coverage includes:

- environment check
- dataset inspection
- DataLoader output shape and split logic
- data augmentation behavior
- model output shape
- SimpleCNN / ImprovedCNN / ResNet18 model creation
- training loop
- model saving
- model loading
- ResNet18 loading consistency
- training history saving
- training curve saving
- experiment summary saving
- evaluation accuracy
- detailed evaluation metrics
- final test evaluation metrics
- single-image prediction
- top-k prediction logic
- low-confidence logic
- OpenCV face detection utilities
- face crop fallback behavior

---

## Project Pipeline

```text
Dataset
↓
Dataset inspection
↓
DataLoader
↓
Train / validation / test split
↓
Data augmentation
↓
CNN baseline
↓
Improved CNN
↓
Transfer learning with ResNet18
↓
Training and validation
↓
Best model checkpoint saving
↓
Final test evaluation
↓
Detailed evaluation
↓
Single-image prediction
↓
Top-3 prediction and low-confidence warning
↓
OpenCV face detection and face crop
↓
Streamlit web demo
↓
Automated tests
```

---

## What I Learned

This project helped me practice:

- building a complete PyTorch image classification pipeline
- using train / validation / test split correctly
- comparing models with controlled experiments
- using transfer learning with ResNet18
- evaluating classification models beyond overall accuracy
- using a held-out test set for final model evaluation
- building a simple AI web demo with Streamlit
- adding OpenCV preprocessing before model inference
- writing pytest tests for machine learning project components
- keeping training, evaluation, prediction, and demo model loading consistent
- organizing a project for GitHub and interview discussion

---

## Limitations

Current limitations:

- FER2013-style 48x48 grayscale images are low-resolution.
- Facial expression labels can be ambiguous.
- Some classes are harder due to imbalance and visual similarity.
- The `disgust` class performs poorly in the final test evaluation, likely due to class imbalance and limited visual examples.
- Haar Cascade face detection is simple and may fail on low-resolution or non-frontal faces.
- Softmax confidence is not a fully calibrated probability.
- The project focuses on building a complete applied AI pipeline rather than achieving state-of-the-art performance.

---

## Possible Future Improvements

Possible future improvements include:

- use a stronger face detector
- add face alignment
- train for more epochs
- tune learning rate and batch size
- improve class imbalance handling
- add model selection in the Streamlit interface
- deploy the Streamlit demo online
- extend the project to webcam-based real-time prediction
- combine expression recognition with attention-state estimation

---

## Project Status

Version 1.1 completed.

The project now supports a complete facial expression recognition workflow from dataset loading to model training, validation-based model selection, final test evaluation, robust prediction, face crop preprocessing, web demo, and automated tests.