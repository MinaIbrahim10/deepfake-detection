# Deepfake Detection and Intent-Based Chatbot Project

## Table of Contents
- Project Overview
- Repository Structure
- Getting Started
    - Prerequisites
    - Installation
- Models and Notebooks
    - Deepfake Text Detection
    - Deepfake Image Detection
    - Deepfake Audio Detection
    - Intent-Based Chatbot Implementation
- API
- Usage
- Contributing


## Project Overview
This project focuses on developing and evaluating models for detecting deepfake content in various modalities and creating an intent-based chatbot for interacting with users regarding deepfake content. The following aspects are covered:

- **Deepfake Text Detection**: Models for detecting deepfake content in English and Arabic text.
- **Deepfake Image Detection**: Classification of deepfake images using CNN, scikit-learn, and OpenCV.
- **Deepfake Audio Detection**: Detection of deepfake audio content.
- **Intent-Based Chatbot Implementation**: A chatbot designed using LSTM for handling specific user intents related to deepfake content.

## Repository Structure

```bash
├── models/
│   ├── deepfake_text_detection_english.h5
│   ├── deepfake_text_detection_arabic.h5
│   ├── deepfake_image_detection.h5
│   └── deepfake_audio_detection.h5
│
├── notebooks/
│   ├── deepfake_image_classification_cnn_sklearn_opencv.ipynb
│   ├── deepfake_audio_detection.ipynb
│   ├── deepfake_text_detection_english.ipynb
│   ├── deepfake_text_detection_arabic.ipynb
│   └── pixel_intent_chatbot_lstm.ipynb
│
├── api/
│   ├── deepfake_fast_api.py
│   ├── deepfake_audio_detection_api.py
│   └── deepfake_image_detection_api.py
│
├── requirements.txt
└── README.md
└── LICENSE.md 

```

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required packages listed in `requirements.txt`

### Installation

1. Clone the Repository:

    ```bash
    git clone https://github.com/MinaIbrahim10/deepfake-detection.git
    ```

2. Navigate to the Repository:

    ```bash
    cd deepfake-text-detection
    ```

3. Install Dependencies:

    Install the required packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## Models and Notebooks

### Deepfake Text Detection

- **English**: Model for detecting deepfake content in English text.
- **Arabic**: Model for detecting deepfake content in Arabic text.
- **Notebooks**:
    - `deepfake_text_detection_english.ipynb`
    - `deepfake_text_detection_arabic.ipynb`

### Deepfake Image Detection

- **Description**: Classification of deepfake images using CNN, scikit-learn, and OpenCV.
- **Notebook**: `deepfake_image_classification_cnn_sklearn_opencv.ipynb`

### Deepfake Audio Detection

- **Description**: Detection of deepfake audio content.
- **Notebook**: `deepfake_audio_detection.ipynb`

### Intent-Based Chatbot Implementation

- **Description**: An intent-based chatbot named "Pixel" using LSTM to handle specific user intents related to deepfake content.
- **Notebook**: `pixel_intent_chatbot_lstm.ipynb`

## API

An API is provided for interfacing with the deepfake detection models. The APIs are implemented using FastAPI:

- **File**: `deepfake_fast_api.py` - FastAPI for deepfake text detection.
- **File**: `deepfake_audio_detection_api.py` - FastAPI for deepfake audio detection.
- **File**: `deepfake_image_detection_api.py` - FastAPI for deepfake image detection.


## Usage

To run the API, use the following command:

```bash
uvicorn deepfake_fast_api:app --reload
```
To access notebooks for detailed analysis and model evaluation:
jupyter notebook
## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contributing

Contributions are welcome! Please follow the standard GitHub workflow for contributions:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Contact

For any questions or feedback, you can reach me at minaibrahim190@gmail.com.
