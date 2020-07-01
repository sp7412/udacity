#!/usr/bin/env

################################################
# Author: Seth Patterson
#
# Use following arguments:
#      python train.py data_dir --arch "vgg19" or "vgg16"
#      python train.py data_dir --learning_rate 0.01 --hiddenunits 512 --epochs 20
#      python train.py data_dir --gpu <turns on/off gpu>
#
#   Example calls:
#    python train.py --dir flowers --gpu --learningrate 0.001 --epochs 3 --batchsize 64 --trainsteps 3 
############################################################

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
    
    print("Welcome to train.py")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', 
                        type = str, 
                        default = 'flowers', 
                        help = 'path to the folder of flowers images') 
    parser.add_argument('--arch', 
                        type = str, 
                        default = 'vgg19', 
                        help = 'which architecture to use (vgg19,)')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use gpu for calculation')
    parser.add_argument('--learningrate', 
                        type = float,
                        default = 0.003,
                        help = 'Learning rate')
    parser.add_argument('--hiddenunits',
                        type = int,
                        default = 1024,
                        help = 'Hidden units')
    parser.add_argument('--epochs', 
                        type = int, default = 1, help = 'epochs: number of iterations through training data')
    parser.add_argument('--batchsize', 
                        type = int,
                        default = 64,
                        help = 'Train batch size') 
    parser.add_argument('--trainsteps',
                        type = int,
                        default = 10,
                        help = 'Train steps')
    parser.add_argument('--dropout', 
                        type = float,
                        default = 0.2,
                        help = 'Train dropout')
    parser.add_argument('--categories', 
                        type = str, 
                        default = 'cat_to_name.json',
                        help = 'Categories to names file')
    parser.add_argument('--savedir', 
                        type = str,
                        default = 'checkpoints',
                        help = 'Checkpoints save dir')
    
    cli_args = parser.parse_args()
        
    data_dir = cli_args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # normalize to [-1,1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=cli_args.batchsize, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size=cli_args.batchsize,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=cli_args.batchsize, shuffle = True)
    
    #set the device to gpu or cpu
    if cli_args.gpu:
        # set to gpu if possible
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
       device = 'cpu'
    
    with open(cli_args.categories, 'r') as f:
        cat_to_name = json.load(f)

    model_name = cli_args.arch
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('Provided model is not supported. Defaulting to vgg19')
        model = models.vgg19(pretrained=True)

    model.name = model_name

    #freeze the weights
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, cli_args.hiddenunits)),
                              ('drop', nn.Dropout(p=cli_args.dropout)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(cli_args.hiddenunits, len(train_data.class_to_idx))),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=cli_args.learningrate)

    model.to(device)
    
    epochs = cli_args.epochs
    steps = cli_args.trainsteps
    running_loss = 0
    print_every = 1
    start = time.time()
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                # Track the loss and accuracy on the validation set to determine the best hyperparameters
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    
    model.eval()
            
    # Turn off gradients for testing, save memory and computations
    with torch.no_grad():

        # Move the network and data to current hardware config (GPU or CPU)
        model.to(device)

        test_loss = 0
        accuracy = 0

        # Looping through images, get a batch size of images on each loop
        for inputs, labels in testloader:

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass, then backward pass, then update weights
            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)
            test_loss += batch_loss.item()

            # Convert to softmax distribution
            ps = torch.exp(log_ps)

            # Compare highest prob predicted class with labels
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            # Calculate accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    # Mapping of classes to indices
    model.class_to_idx = train_data.class_to_idx
    
    # Create model metadata dictionary
    checkpoint = {
        'name': model.name,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict()
    }

    # Save to a file
    timestr = time.strftime("%Y%m%d_%H%M%S")
    arch = cli_args.arch
    save_folder = cli_args.savedir
    
    if not os.path.exists(save_folder):
       os.makedirs(save_folder)
    
    file_name = save_folder +'/model_' + arch + '_' + timestr + '.pth'
    torch.save(checkpoint, file_name)
    
    print(f"Model checkpoint saved as: {file_name}")
    
    #import pdb; pdb.set_trace()
if __name__ == "__main__":
    main()

