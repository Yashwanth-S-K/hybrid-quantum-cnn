import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
from quantum_layer import QuantumLayer  # Assuming you saved the QuantumLayer class in quantum_layer.py

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess a single image"""
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to array and preprocess
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_class_labels(train_dir):
    """Get class labels from training directory"""
    classes = sorted(os.listdir(train_dir))
    return {i: class_name for i, class_name in enumerate(classes)}

def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'hybrid_qcnn_final_model.keras')
    image_path = os.path.join(current_dir, 'image.jpg')
    train_dir = os.path.join(current_dir, 'dataset', 'train')  # Path to training directory
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    try:
        # Load the model with custom objects
        print("Loading model...")
        custom_objects = {'QuantumLayer': QuantumLayer}
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Load and preprocess the image
        print("Processing image...")
        processed_image = load_and_preprocess_image(image_path)
        
        # Get class labels
        class_labels = get_class_labels(train_dir)
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(processed_image)
        
        # Get the predicted class
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        
        print("\nPrediction Results:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Print top 3 predictions
        print("\nTop 3 Predictions:")
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        for idx in top_3_indices:
            class_name = class_labels[idx]
            confidence = predictions[0][idx] * 100
            print(f"{class_name}: {confidence:.2f}%")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()