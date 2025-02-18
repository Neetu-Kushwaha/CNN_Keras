from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os


def predict_image(image_path):
    """Loads a saved model and predicts the class of a given image."""
    model_path = os.path.join("saved_models", 'cnn_cifar10.h5')
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return

    model = load_model(model_path)

    image = load_img(image_path, target_size=(32, 32))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    print(f"Predicted Class: {class_id}")


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_image(image_path)
