from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels once with error handling
try:
    model = load_model("keras_Model.h5", compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open("labels.txt", "r") as file:
        class_names = file.readlines()
    print("Labels loaded successfully")
except Exception as e:
    print(f"Error loading labels: {e}")

def predict_image(image_file):
    try:
        # Load and process the image
        image = Image.open(image_file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Prepare data
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        return class_name, float(confidence_score)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None
