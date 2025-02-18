from data_loader import load_data
from model_builder import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os


def train_model():
    """Trains the CNN model on CIFAR-10 dataset and saves the best model."""
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    # Directory to save the model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'cnn_cifar10.h5')

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=50, callbacks=[checkpoint])

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
