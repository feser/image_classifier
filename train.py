import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import keep_awake
from collections import OrderedDict
from utils import create_network, save_checkpoint
import argparse
import sys

parser = argparse.ArgumentParser(description='Train a network on a dataset of images')
parser.add_argument('data_dir', metavar='data_directory', help='Data directory of images')
parser.add_argument('--save_dir', dest='save_dir', default='checkpoint.pth', metavar='save_directory', help='Directory to save checkpoints')
parser.add_argument('--arch', dest='arch', default='vgg16' ,metavar='vgg16', choices=['vgg16','alexnet'], help='Architecture which is used to train the model')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, metavar='0.001', help='Learning rate')
parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=8192, metavar='8192', help='Hidden units')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.4, metavar='0.4', help='Dropout')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, metavar='10', help='Epochs')
parser.add_argument('--gpu', dest='gpu', help='Enable GPU', action='store_true')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = args.learning_rate
hidden_units = args.hidden_units
dropout = args.dropout
epochs = args.epochs
arch = args.arch

if args.gpu and not torch.cuda.is_available():
    sys.exit('GPU option is not available, please run without it.')
device = "cuda" if args.gpu else "cpu"

def create_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    validation_tranforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_tranforms)
    return train_data, validation_data

def create_loaders(train_data, validation_data):    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=120, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=30)
    return train_loader, validation_loader


def test_network(model, loader, device):
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train() 
    return test_loss, test_accuracy

def train_network(model, device, optimizer, train_loader, validation_loader, epochs):
    running_loss = 0
    for epoch in keep_awake(range(epochs)):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            # Set gradients to zero.
            optimizer.zero_grad()
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:          
            validation_loss, validation_accuracy = test_network(model, validation_loader, device) 
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_loader):.3f}.. "
                  f"Validation loss: {validation_loss/len(validation_loader):.3f}.. "
                  f"Validation accuracy: {validation_accuracy/len(validation_loader):.3f}")
            running_loss = 0
    return model, optimizer           
    
    
train_data, validation_data = create_data(data_dir)    
train_loader , validation_loader= create_loaders(train_data, validation_data)
print('Data is loaded from {}'.format(data_dir))

model, criterion, optimizer = create_network(arch, device, learning_rate, dropout, hidden_units)

print('Starting to train network')
model, optimizer = train_network(model, device, optimizer, train_loader, validation_loader, epochs)
print('Network training finished')

if save_dir != None:
    save_checkpoint(save_dir, model, arch, train_data, learning_rate, dropout, hidden_units)
    print('Checkpoint is saved to {}'.format(save_dir))
