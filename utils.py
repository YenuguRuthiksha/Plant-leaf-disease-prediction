# Importing Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image

# Cleaning Image
def clean_image(image):
    image = np.array(image)

    # Resize the image to match the model's input size
    image = np.array(Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS))

    # Ensure only the first 3 channels are used (ignore alpha channel if present)
    if image.shape[-1] == 4:  # If 4 channels, keep only RGB
        image = image[:, :, :3]

    return image

# Predict Image Class
def get_prediction(model, image):
    # Normalize the image (rescale pixel values to [0, 1])
    image = image / 255.0

    # Ensure the input shape matches the model's requirements
    if len(image.shape) == 3:  # Add a batch dimension if needed
        image = np.expand_dims(image, axis=0)

    # Predict the class probabilities
    predictions = model.predict(image)
    predictions_arr = np.argmax(predictions, axis=1)  # Get the class index with the highest probability

    return predictions, predictions_arr

# Make Final Results
def make_results(predictions, predictions_arr):
    result = {}
    if int(predictions_arr) == 0:
        result = {"status": "is Healthy",
                  "prediction": f"{int(predictions[0][0].round(2) * 100)}%"}
    if int(predictions_arr) == 1:
        result = {"status": "has Multiple Diseases",
                  "prediction": f"{int(predictions[0][1].round(2) * 100)}%"}
    if int(predictions_arr) == 2:
        result = {"status": "has Rust",
                  "prediction": f"{int(predictions[0][2].round(2) * 100)}%"}
    if int(predictions_arr) == 3:
        result = {"status": "has Scab",
                  "prediction": f"{int(predictions[0][3].round(2) * 100)}%"}
    return result
