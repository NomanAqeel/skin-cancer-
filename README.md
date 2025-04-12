# Skin Cancer Detection Application

A Streamlit-based web application for detecting potential skin cancer from images using a deep learning model.

## Features

- User-friendly interface for image uploads
- Real-time analysis of skin lesion images
- Visualization of detection confidence
- Results indicating benign or malignant classifications

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/skin-cancer-detection-app.git
   cd skin-cancer-detection-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your trained H5 model in the `model` directory:
   ```
   mkdir -p model
   # Copy your skin_cancer_model.h5 file to the model directory
   ```

### Running the Application

1. Start the Streamlit server:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

## Usage Guide

1. Upload a clear image of the skin lesion using the file uploader
2. Click the "Analyze Image" button
3. View the results showing the classification (benign or malignant)
4. Check the confidence score and additional information

## Model Information

The application uses a pre-trained deep learning model saved in H5 format. The model was trained on dermatological images to classify skin lesions as either benign or malignant.

## Important Notes

- This application is for educational purposes only
- It should not be used as a substitute for professional medical advice
- Always consult a healthcare provider for proper diagnosis and treatment

## Customization

To use your own custom model:
1. Replace the `skin_cancer_model.h5` file in the `model` directory with your trained model
2. If your model has different input requirements or output classes, update the preprocessing and prediction functions in the code accordingly

## License

This project is licensed under the MIT License - see the LICENSE file for details.