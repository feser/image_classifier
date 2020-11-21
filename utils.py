import torch
from torch import nn
from torch import optim
from torchvision import models, transforms
from collections import OrderedDict
from PIL import Image
import json

def create_network(arch, device= 'cuda', learning_rate = 0.001, dropout = 0.4, hidden_layer_1 = 8192):
    model = None
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_layer = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_layer = 9216
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(input_layer, hidden_layer_1)),
        ("relu1", nn.ReLU()),
        ("do1", nn.Dropout(dropout)),
        ("fc2", nn.Linear(hidden_layer_1, 1096)), 
        ("relu2", nn.ReLU()),
        ("do2", nn.Dropout(dropout)),    
        ("fc3", nn.Linear(1096, 102)), 
        ("out", nn.LogSoftmax(dim=1))
    ]))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    return model, criterion, optimizer

def save_checkpoint(save_dir, model, arch, train_data, learning_rate, dropout, hidden_layer_1):
    checkpoint = {
        'learning_rate': learning_rate,
        'arch': arch, 
        'dropout': dropout,
        'hidden_layer_1': hidden_layer_1,
        'class_to_idx': train_data.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_dir)
    
    
def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    learning_rate = checkpoint['learning_rate']
    dropout = checkpoint['dropout']
    arch = checkpoint['arch']
    hidden_layer_1 = checkpoint['hidden_layer_1']
    model, criterion, optimizer = create_network(arch, device, learning_rate, dropout, hidden_layer_1)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model, criterion, optimizer    


def process_image(image):
    image = Image.open(image)
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_image(image)

def read_json(file_name):
    with open(file_name, 'r') as f:
        result = json.load(f) 
        return result