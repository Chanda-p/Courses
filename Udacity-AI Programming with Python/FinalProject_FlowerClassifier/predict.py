import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=1)
    parser.add_argument('--category_names', type=str, help='Path to the category names mapping JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    pil_image = Image.open(image)

    # Resize the image where the shortest side is 256 pixels
    width, height = pil_image.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)
    pil_image = pil_image.resize((new_width, new_height))

    # Crop out the center 224x224 portion of the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert PIL image to Numpy array
    np_image = np.array(pil_image)

    # Normalize the image
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose the color channel to be the first dimension
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)
        top_probs, top_indices = probabilities.topk(topk)

    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes


def load_category_names(filename):
    with open(filename, 'r') as f:
        category_names = json.load(f)
    return category_names


def main():
    args = parse_arguments()

    # Set the device (GPU/CPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Process the image and make predictions
    top_probs, top_classes = predict(args.image_path, model, args.top_k, device)

    # Load the category names if provided
    if args.category_names:
        category_names = load_category_names(args.category_names)
        top_classes = [category_names[class_] for class_ in top_classes]

    # Print the results
    for prob, class_ in zip(top_probs, top_classes):
        print(f"Class: {class_}, Probability: {prob:.3f}")

if __name__ == '__main__':
    main()