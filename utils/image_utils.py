import numpy as np
from PIL import Image
import tensorflow as tf
import io

def load_image_from_upload(uploaded_file):
    """
    Load an image from a file upload
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        PIL Image object
    """
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))
    return img

def preprocess_for_model(img, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        img: PIL Image object
        target_size: Tuple of (height, width)
        
    Returns:
        Preprocessed image array
    """
    # Resize image
    img = img.resize(target_size)
    
    # Convert to RGB if not already
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Convert to array and expand dimensions
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize to [0,1]
    img_array = img_array / 255.0
    
    return img_array

def get_gradcam(model, img_array, last_conv_layer_name="conv5_block3_out"):
    """
    Generate Grad-CAM visualization
    
    Args:
        model: Keras model
        img_array: Preprocessed image array
        last_conv_layer_name: Name of the last convolutional layer
        
    Returns:
        Heatmap array
    """
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of the top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        top_class_channel = predictions[:, class_idx]
        
    # Gradient of the output with respect to the last conv layer
    grads = tape.gradient(top_class_channel, conv_outputs)
    
    # Vector of mean intensity of gradient over feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by corresponding gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()