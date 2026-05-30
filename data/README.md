# Data Folder

This folder stores dataset-related instructions and local data folders for the facial expression recognition project.

## Dataset Choice

This project is designed to use a FER2013-style facial expression dataset.

A typical FER2013-style dataset contains grayscale face images labeled with emotion categories such as:

* angry
* disgust
* fear
* happy
* sad
* surprise
* neutral

## Expected Local Folder Structure

The dataset should be stored locally using the following structure:

```text
data/
├── raw/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   └── neutral/
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── sad/
│       ├── surprise/
│       └── neutral/
└── processed/
```

## Important Note

The actual dataset files are not committed to GitHub.

Large image datasets should be downloaded manually and stored locally under:

```text
data/raw/
```

The `.gitignore` file is configured to ignore:

```text
data/raw/
data/processed/
```

This keeps the GitHub repository lightweight and clean.

## Current Stage

Dataset instructions are prepared. The actual dataset will be downloaded and inspected in the next stage.
