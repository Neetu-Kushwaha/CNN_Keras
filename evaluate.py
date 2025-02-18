from data_loader import load_data
from tensorflow.keras.models import load_model
import os


def evaluate_model():
    """Loads the saved model and evaluates it on the test set."""
    (_, _), (x_test, y_test) = load_data()

    model_path = os.path.join("saved_models", 'cnn_cifar10.h5')
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return

    model = load_model(model_path)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate_model()
