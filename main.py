import argparse
from train import train_model
from evaluate import evaluate_model
from predict import predict_image


def main():
    parser = argparse.ArgumentParser(description="CNN Model for CIFAR-10")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", type=str, help="Predict an image (provide image path)")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("Please specify an action: --train, --evaluate, or --predict [image_path]")


if __name__ == "__main__":
    main()
