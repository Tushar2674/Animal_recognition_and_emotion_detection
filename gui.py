import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import numpy as np

# Load the models
animal_recognition_model = load_model('best_animal_recognition_model.keras')
emotion_detection_model = load_model('best_emotion_model.keras')

# Load the object detection model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Define class names for animal recognition and emotion detection
animal_classes = ["bear","bison","cat","cow","deer","dog","elephant","goat","gorilla","horse","kangaroo","leopard","lion",'parrot',"penguin","pigeon","rat",'shark',"sheep",'sparrow',"snake","tiger",'turtle',"wolf"]  # Adjust based on your actual classes
emotion_classes = [ 'Angry','Happy', 'Hungry','Sad']

def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)
    print(image)
    return image

def object_detection(image):
    image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    detections = detector(image_tensor)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    print(detection_boxes)
    print(detection_scores)
    
    # Filter out detections with low confidence
    confidence_threshold = 0.5  # Adjust threshold as needed
    boxes = detection_boxes[detection_scores >= confidence_threshold]
    
    # Convert normalized box coordinates to pixel coordinates
    width, height = image.size
    boxes = [
        (int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width))
        for ymin, xmin, ymax, xmax in boxes
    ]
    print(boxes)
    return boxes

def predict_image(filepath):
    image = Image.open(filepath)
    detections = object_detection(image)
    results = []
    
    for box in detections:
        cropped_image = image.crop(box)
        prepared_image = prepare_image(cropped_image, target_size=(128,128))
        
        # Predict animal class
        animal_predictions = animal_recognition_model.predict(prepared_image)
        animal_class_idx = np.argmax(animal_predictions, axis=1)[0]
        animal_class = animal_classes[animal_class_idx]

        # Predict emotion
        emotion_predictions = emotion_detection_model.predict(prepared_image)[0]
        emotion_class_idx = np.argmax(emotion_predictions)
        emotion_class = emotion_classes[emotion_class_idx]

        

        results.append(f"{animal_class} + {emotion_class}")

    result_message = "\n".join(results)
    return result_message

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        messagebox.showinfo("Prediction Result", result)

# Create the main application window
root = tk.Tk()
root.title("Animal and Emotion Recognition")
root.geometry("400x200")

# Add a button to upload image
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=20)

# Run the application
root.mainloop()
