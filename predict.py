import argparse
import torch
from torchvision import models, transforms
from torch import nn
import json
from PIL import Image


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    # Improvement
    model = getattr(models, checkpoint["architecture"])(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


def predict(image_path, model, topk, gpu):
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model.forward(image)
        probs, classes = torch.exp(output).topk(topk)

    return probs[0].tolist(), classes[0].add(1).tolist()  # Adjusting class index


def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("input", type=str, help="Path to image")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument(
        "--top_k", type=int, default=3, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        help="Path to JSON file mapping categories to real names",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(str(cls), "Unknown") for cls in classes]

    print("Predicted Classes:", classes)
    print("Probabilities:", probs)


if __name__ == "__main__":
    main()
