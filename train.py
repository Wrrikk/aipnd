import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import os

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Define your transforms for the training and validation sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, validloader

def build_model(arch, hidden_units):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Architecture '{arch}' not recognized. Please choose 'vgg13' or 'densenet121'.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    return model

def train_model(model, trainloader, validloader, learning_rate, epochs, gpu):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(trainloader):.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            model.train()

def save_checkpoint(model, save_dir, arch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    checkpoint = {'architecture': arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('data_directory', type=str, help='Directory of training data')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    trainloader, validloader = load_data(args.data_directory)
    model = build_model(args.arch, args.hidden_units)
    train_model(model, trainloader, validloader, args.learning_rate, args.epochs, args.gpu)
    save_checkpoint(model, args.save_dir, args.arch)

if __name__ == '__main__':
    main()