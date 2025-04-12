import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Detection App",
    page_icon="🔬",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .info-text {
        background-color: #EBF5FB;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1.5rem;
        text-align: center;
    }
    .positive-result {
        background-color: #FADBD8;
        border: 1px solid #E74C3C;
    }
    .negative-result {
        background-color: #D5F5E3;
        border: 1px solid #2ECC71;
    }
    .confidence-bar {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #7F8C8D;
        font-style: italic;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Skin Cancer Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Upload an image to detect potential skin cancer</h3>", unsafe_allow_html=True)

# Information section
with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This application uses a deep learning model to analyze skin lesion images and detect potential skin cancers.
    
    **How to use:**
    1. Upload a clear image of the skin lesion
    2. Click the 'Analyze Image' button
    3. View the results and prediction confidence
    
    **Important:** This app is for educational purposes only and should not replace professional medical advice.
    Always consult a healthcare provider for proper diagnosis and treatment.
    """)

# Function to load the model
@st.cache_resource
def load_ml_model():
    """Load the pre-trained model"""
    try:
        model = load_model('model/skin_cancer_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(img):
    """Preprocess the image for model prediction"""
    # Resize image to 224x224 (or whatever your model expects)
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# Function to make prediction
def predict_skin_cancer(model, img_array):
    """Make prediction using the model"""
    try:
        prediction = model.predict(img_array)
        return prediction[0][0]  # Assuming binary classification
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Class names (adjust based on your model)
class_names = ['Benign', 'Malignant']

# Main app functionality
def main():
    # Load model
    model = load_ml_model()
    
    if model is None:
        st.warning("Model could not be loaded. Please check your model file.")
        return
    
    # Image upload section
    st.markdown("<div class='info-text'>Please upload a clear, well-lit image of the skin lesion.</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Display the uploaded image
        with col1:
            st.subheader("Uploaded Image")
            image_data = uploaded_file.read()
            img = Image.open(io.BytesIO(image_data))
            st.image(img, width=300, caption="Uploaded Image")
        
        # Analysis button
        analyze_button = st.button("Analyze Image")
        
        if analyze_button:
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                img_array = preprocess_image(img)
                
                # Make prediction
                prediction_score = predict_skin_cancer(model, img_array)
                
                if prediction_score is not None:
                    # Display results
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Determine class and confidence
                        predicted_class = 1 if prediction_score > 0.5 else 0
                        confidence = prediction_score if predicted_class == 1 else 1 - prediction_score
                        confidence_percentage = f"{confidence * 100:.2f}%"
                        
                        # Display result with styling based on prediction
                        if predicted_class == 1:  # Malignant
                            st.markdown(f"""
                            <div class='result-box positive-result'>
                                <h3>Result: {class_names[predicted_class]}</h3>
                                <p>The model detected potential malignant characteristics.</p>
                                <p>Confidence: {confidence_percentage}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:  # Benign
                            st.markdown(f"""
                            <div class='result-box negative-result'>
                                <h3>Result: {class_names[predicted_class]}</h3>
                                <p>The model detected primarily benign characteristics.</p>
                                <p>Confidence: {confidence_percentage}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confidence visualization
                    st.subheader("Confidence Score")
                    st.markdown("<div class='confidence-bar'>", unsafe_allow_html=True)
                    st.progress(float(confidence))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional information based on result
                    if predicted_class == 1:
                        st.info("⚠️ This image shows potential signs of skin cancer. Please consult a healthcare professional promptly.")
                    else:
                        st.info("✅ This image appears to show benign characteristics, but regular skin checks are still recommended.")
    
    # Medical disclaimer
    st.markdown("""
    <div class='disclaimer'>
        <strong>Medical Disclaimer:</strong> This app is intended for educational and informational purposes only. 
        It is not a replacement for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()