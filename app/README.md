# Streamlit App

This folder contains the Streamlit web demo for the facial expression recognition project.

The app allows users to upload a face image and run emotion prediction through a trained PyTorch model.

## Main Features

* Upload a JPG, JPEG, or PNG image
* Preview the uploaded image
* Enable or disable OpenCV face detection / face crop
* Show whether a face was detected
* Show the image used for prediction
* Predict one of seven facial expression classes
* Display prediction confidence
* Display top-3 prediction probabilities
* Show a low-confidence warning when appropriate

## How to Run

From the project root folder, run:

```powershell
streamlit run app/streamlit_app.py
```

Before running the app, make sure the required dependencies are installed:

```powershell
pip install -r requirements.txt
```

Also make sure a trained model file exists under:

```text
outputs/models/
```

If the model file does not exist, train a model first:

```powershell
python -m src.train
```

## App Pipeline

```text
Uploaded image
↓
Optional OpenCV face detection
↓
Optional largest face crop
↓
Fallback to original image if no face is detected
↓
Image preprocessing
↓
PyTorch model prediction
↓
Top-3 probabilities
↓
Low-confidence warning
↓
Streamlit result display
```

## Notes

This app is designed as a simple local demo for project presentation and interview discussion.

It is not intended to be a production deployment.
