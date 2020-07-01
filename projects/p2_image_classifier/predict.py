#!/usr/bin/env

################################################
# Author: Seth Patterson
#
# Use following arguments:
#      python predict.py --image <path to image for prediction>
#                        --savedir <checkpoints dir>
#                        --checkpoint <file name for loading train data>
#                        --categories <categories file name>
#                        --gpu <use gpu> 
#                        --topk <print out the top K classes along with associated probabilities>
# Example call:
#    python predict.py --image flowers/test/1/image_06764.jpg --checkpoint checkpoint.pth --savedircheckpoints --gpu True --categories cat_to_name.json --topk 3
#############################################################

from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
import torchvision
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
from pprint import pprint as pp
from collections import OrderedDict
import time
import random
import os
import argparse

def main():
    print("Welcome to predict.py")

    cuda = torch.cuda.is_available()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', 
                        type = str, 
                        default = 'flowers/test/1/image_06764.jpg', 
                        help = 'path to image to run prediction on')
    parser.add_argument('--gpu', 
                        action='store_true', 
                        help='Use gpu for calculation')
    parser.add_argument('--checkpoint',
                        type = str,
                        default = 'checkpoint.pth',
                        help = 'Archive with train data') 
    parser.add_argument('--categories', 
                        type = str, 
                        default = 'cat_to_name.json',
                        help = 'Categories to names file')
    parser.add_argument('--topk', 
                        type = int, 
                        default = '5',
                        help = 'top number of predictions to print out')
    
    cli_args = parser.parse_args()
    
    pp(cli_args)
    
    filepath = cli_args.checkpoint
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']
    
    image_path = cli_args.image
    image = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(image)
    
    if cli_args.gpu and cuda:
        inputs = Variable(img_tensor.float().cuda())
    else:       
        inputs = Variable(img_tensor)
    
    model.eval()

    if cli_args.gpu and cuda:
        model = model.cuda()
    else:
        model = model.cpu()
   
    # Format tensor for input into model
    image = inputs.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(cli_args.topk)

    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.cpu().detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    classes = []
    for label in top_labs.cpu().numpy()[0]:
        classes.append(idx_to_class[label])
        
    with open(cli_args.categories, 'r') as f:
        cat_to_name = json.load(f)
        
    for idx in range(cli_args.topk):
        print('[{}]: {} = {}'.format(classes[idx],cat_to_name[classes[idx]],top_probs[idx]))
        
if __name__ == "__main__":
    main()