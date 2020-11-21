import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import keep_awake
from collections import OrderedDict
import argparse
from utils import load_checkpoint, process_image, read_json
import sys

parser = argparse.ArgumentParser(description='Train a network on a dataset of images')
parser.add_argument('image_path', metavar='path/to/image', help='Path of image')
parser.add_argument('checkpoint_path', metavar='path/to/checkpoint', help='Path of checkpoint')
parser.add_argument('--top_k', dest='top_k', type=int, default=1, metavar='3', help='Top K most likely classes')
parser.add_argument('--category_names', dest='category_names', metavar='cat_to_name.json', help='Maping of categories to real names')
parser.add_argument('--gpu', dest='gpu', help='Enable GPU', action='store_true')

args = parser.parse_args()
image_path = args.image_path
checkpoint_path = args.checkpoint_path
top_k = args.top_k
category_names = args.category_names

if args.gpu and not torch.cuda.is_available():
    sys.exit('GPU option is not available, please run without it.')
device = "cuda" if args.gpu else "cpu"

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}  
    
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.to(device)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(image)
        
    ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(topk, dim=1)
    top_class = [idx_to_class[v.item()] for v in top_class[0]]
    
    return top_p[0].tolist(), top_class


model, criterion, optimizer = load_checkpoint(checkpoint_path, device) 

top_ps, top_classes = predict(image_path, model, top_k, device)

if category_names != None:
    cat_to_name = read_json(category_names)
    top_classes = [cat_to_name[v] for v in top_classes]    

for ii, (p, c) in enumerate(zip(top_ps, top_classes), 1):
    print(f"{ii}. Class: {c} ,Probability: {p:.5f} ")

