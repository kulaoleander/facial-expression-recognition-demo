# Facial Expression Recognition Demo

An end-to-end deep learning and computer vision project for classifying facial expressions from face images.

The project covers dataset inspection, PyTorch data loading, CNN baselines, transfer learning with ResNet18, configurable training, class-imbalance handling, evaluation, model saving/loading, single-image prediction, OpenCV face detection, a Streamlit demo, and pytest-based automated tests.

The goal is not to claim state-of-the-art performance. The goal is to demonstrate a complete, explainable, and reproducible applied-AI workflow.

---

## Key Results

The final model is a ResNet18 initialized with ImageNet-pretrained weights and then fine-tuned for seven-class facial-expression recognition.

| Metric | Value |
|---|---:|
| Test samples | 7,178 |
| Test accuracy | 0.6257 |
| Macro F1-score | 0.5996 |
| Weighted F1-score | 0.6224 |

Per-class accuracy:

| Class | Accuracy |
|---|---:|
| angry | 0.5532 |
| disgust | 0.7027 |
| fear | 0.3701 |
| happy | 0.8005 |
| neutral | 0.6667 |
| sad | 0.4747 |
| surprise | 0.8063 |

The strongest classes were `happy` and `surprise`. The most difficult classes were `fear` and `sad`, which is consistent with the ambiguity, visual similarity, class imbalance, and low resolution of FER2013-style images.

---

## Project Overview

The model predicts one of seven facial-expression classes:

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

Model output:

```text
logits shape: [batch_size, 7]
```

The project started with a SimpleCNN baseline and was gradually improved through validation/test separation, data augmentation, an ImprovedCNN, ResNet18 transfer learning, class weights, learning-rate scheduling, early stopping, final test evaluation, OpenCV face cropping, and a Streamlit interface.

---

## Main Features

- FER2013-style dataset inspection
- PyTorch `ImageFolder` data loading
- train / validation / test split
- training-time data augmentation
- SimpleCNN baseline
- ImprovedCNN with BatchNorm and Dropout
- ResNet18 transfer learning
- configurable command-line training
- automatic CPU/CUDA device selection
- class weights for imbalanced data
- learning-rate scheduler support
- early stopping
- best-checkpoint saving
- final held-out test evaluation
- classification report and confusion matrix
- single-image prediction
- top-3 prediction probabilities
- low-confidence warning
- OpenCV Haar Cascade face detection
- largest-face crop with fallback behavior
- Streamlit web demo
- pytest-based automated tests

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

## Dataset

The project expects a FER2013-style dataset organized as image folders:

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

The dataset is not committed to this repository because image datasets are usually large and may have separate licensing or access requirements.

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

Run the environment check:

```powershell
python -m src.environment_check
```

---

## Model Weights

This project uses **two different types of ResNet18 weights**.

### 1. ImageNet-pretrained initialization

When training with `--pretrained`, torchvision downloads general ImageNet-pretrained ResNet18 weights automatically and uses them as the starting point for transfer learning.

These are not the final facial-expression model weights.

### 2. Fine-tuned facial-expression checkpoint

After training on the facial-expression dataset, the project saves the best fine-tuned checkpoint to:

```text
outputs/models/resnet18.pth
```

This fine-tuned checkpoint contains the model parameters learned specifically for this seven-class facial-expression task. It is required by:

- `python -m src.predict`
- `python -m src.final_test_evaluation`
- `streamlit run app/streamlit_app.py`

The fine-tuned `resnet18.pth` file is **not currently committed to this GitHub repository**.

To run inference or the Streamlit demo, use one of the following options.

### Option A: Train the model

Run the training command below. The best checkpoint will be generated automatically at `outputs/models/resnet18.pth`.

```powershell
python -m src.train --model resnet18 --epochs 50 --batch-size 32 --lr 0.001 --pretrained --use-augmentation --class-weights --scheduler reduce_on_plateau --scheduler-patience 4 --scheduler-factor 0.5 --early-stopping-patience 8 --device auto
```

### Option B: Use an existing local checkpoint

Copy a compatible fine-tuned checkpoint into:

```text
outputs/models/resnet18.pth
```

Without this file, training and tests that do not require the final checkpoint can still run, but prediction, final evaluation, and the Streamlit demo cannot load the trained model.

A future release may provide the fine-tuned checkpoint as a separate downloadable artifact rather than storing the binary file directly in the Git repository.

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

Basic training command:

```powershell
python -m src.train
```

Example formal ResNet18 command:

```powershell
python -m src.train --model resnet18 --epochs 50 --batch-size 32 --lr 0.001 --pretrained --use-augmentation --class-weights --scheduler reduce_on_plateau --scheduler-patience 4 --scheduler-factor 0.5 --early-stopping-patience 8 --device cuda
```

Main options:

```text
--model simple_cnn | improved_cnn | resnet18
--epochs
--batch-size
--lr
--pretrained / --no-pretrained
--use-augmentation / --no-use-augmentation
--class-weights / --no-class-weights
--scheduler none | reduce_on_plateau | cosine
--scheduler-patience
--scheduler-factor
--early-stopping-patience
--device auto | cpu | cuda
```

The formal v1.2 training run used:

```text
Model: resnet18
Pretrained initialization: True
Epochs: 50
Batch size: 32
Learning rate: 0.001
Data augmentation: True
Class weights: True
Scheduler: reduce_on_plateau
Scheduler patience: 4
Scheduler factor: 0.5
Early stopping patience: 8
Device: CUDA / Tesla T4
```

Generated outputs include:

```text
outputs/models/resnet18.pth
outputs/logs/training_config.json
outputs/logs/training_history.json
outputs/logs/experiment_results.csv
outputs/figures/training_curves.png
```

---

## Model Comparison

| Model / Version | Validation Accuracy |
|---|---:|
| SimpleCNN baseline | 0.4142 |
| ImprovedCNN | 0.4644 |
| ResNet18 from scratch | 0.4715 |
| Pretrained ResNet18, v1.1 short training | 0.5078 |
| Pretrained ResNet18, v1.2 GPU training + class weights + scheduler | 0.6314 |

The v1.2 transfer-learning model achieved the best validation accuracy. This comparison demonstrates the value of reusing general visual features and then fine-tuning them for a task-specific dataset.

---

## Final Test Evaluation

Run:

```powershell
python -m src.final_test_evaluation
```

The script:

- loads `outputs/models/resnet18.pth`
- creates the held-out test DataLoader
- calculates accuracy, macro F1, weighted F1, and per-class accuracy
- saves final metrics as JSON
- saves a confusion matrix as PNG

Generated files:

```text
outputs/logs/final_test_metrics.json
outputs/figures/final_test_confusion_matrix.png
```

---

## Single-Image Prediction

Run:

```powershell
python -m src.predict
```

Prediction output includes:

- predicted class
- confidence
- top-3 predictions
- low-confidence flag
- face-crop status
- number of detected faces
- selected face box

If no face is detected, the pipeline falls back to the original image instead of crashing.

---

## Streamlit Demo

Run:

```powershell
streamlit run app/streamlit_app.py
```

The app allows users to:

- upload an image
- preview the uploaded image
- enable or disable face detection and cropping
- view the image used for prediction
- run expression prediction
- view the predicted expression and confidence
- view top-3 probabilities
- receive a low-confidence warning

The demo requires the fine-tuned checkpoint at:

```text
outputs/models/resnet18.pth
```

---

## Face Detection and Face Crop

```text
Uploaded image
↓
OpenCV face detection
↓
Select largest detected face
↓
Add padding around the face box
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

---

## Run Tests

```powershell
python -m pytest
```

Current local result:

```text
106 passed, 1 skipped
```

The tests cover dataset loading, preprocessing, model creation, training configuration, device selection, class weights, schedulers, early stopping, model saving/loading, evaluation metrics, prediction behavior, OpenCV utilities, and fallback behavior.

---

## Project Pipeline

```text
Dataset
↓
Dataset inspection
↓
DataLoader and data split
↓
Data augmentation
↓
CNN baselines
↓
ResNet18 transfer learning
↓
Class weights, scheduler, and early stopping
↓
Best checkpoint saving
↓
Final held-out test evaluation
↓
Single-image prediction
↓
OpenCV face detection and crop
↓
Streamlit demo
↓
Automated tests
```

---

## Limitations

- FER2013-style 48x48 grayscale images are low-resolution.
- Facial-expression labels can be ambiguous.
- Some classes are difficult because of imbalance and visual similarity.
- `fear` and `sad` remain challenging in the final test results.
- Haar Cascade may fail on low-resolution, rotated, or non-frontal faces.
- Softmax confidence is not a fully calibrated probability.
- The fine-tuned checkpoint is not currently distributed through this repository.
- The project demonstrates an applied-AI pipeline rather than state-of-the-art performance.

---

## Possible Future Improvements

- publish the fine-tuned checkpoint as a GitHub Release or model artifact
- use a stronger face detector
- add face alignment
- improve model calibration
- improve handling of difficult and minority classes
- add an online hosted demo

---

## Project Status

Version 1.2 completed.

The project supports a complete facial-expression-recognition workflow from dataset loading and model training to validation-based model selection, final test evaluation, prediction, face-crop preprocessing, a Streamlit demo, and automated tests.
