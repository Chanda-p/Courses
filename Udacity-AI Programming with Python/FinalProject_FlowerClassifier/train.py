import argparse
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from collections import OrderedDict
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints', default='checkpoints')
    parser.add_argument('--arch', type=str, help='Model architecture (default: "vgg16")', default='vgg16')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (default: 0.001)', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the classifier (default: 512)', default=512)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train (default: 10)', default=10)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()


def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    return image_datasets, dataloaders


def build_model(arch, hidden_units):
    # Load a pre-trained model
    model = models.__dict__[arch](pretrained=True)

    # Freeze the pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with a new fully connected network
    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )

    model.classifier = classifier
    return model


def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    model.to(device)
    steps = 0
    print_every = 10

    for epoch in range(epochs):
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = validate_model(model, dataloaders['valid'], criterion, device)
                print(f"Epoch: {epoch+1}/{epochs} "
                      f"Training Loss: {running_loss/print_every:.3f} "
                      f"Validation Loss: {valid_loss:.3f} "
                      f"Validation Accuracy: {accuracy:.3f}")
                running_loss = 0


def validate_model(model, dataloader, criterion, device):
    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    return valid_loss / len(dataloader), accuracy / len(dataloader)


def save_checkpoint(model, image_datasets, save_dir, arch, hidden_units, epochs):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'epochs': epochs
    }
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    args = parse_arguments()

    # Set the device (GPU/CPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load and preprocess the data
    image_datasets, dataloaders = load_data(args.data_dir)

    # Build the model
    model = build_model(args.arch, args.hidden_units)

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)

    # Save the checkpoint
    save_checkpoint(model, image_datasets, args.save_dir, args.arch, args.hidden_units, args.epochs)

if __name__ == '__main__':
    main()