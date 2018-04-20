import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import os
import sys
import copy
from PIL import Image
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import ImageFile
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True
from helpers import generator, extract_features, preprocess, save_array, load_array
from model import Classifier
from os.path import isfile, isdir


use_gpu = torch.cuda.is_available()
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet152 = models.resnet152(pretrained=True)
models_dict = {'resnet34': {'model':resnet34, 'num_features': resnet34.fc.in_features}, 'resnet50': {'model': resnet50, 'num_features': resnet50.fc.in_features}, 'resnet152': {'model': resnet152, 'num_features': resnet152.fc.in_features}}


def prepdata(arch, dataLoaders):
    # apply model to input
    pretrained_model = models_dict[arch]
    num_features = pretrained_model.fc.in_features
    modules = list(pretrained_model.children())[:-1]
    pretrained_model=nn.Sequential(*modules)
    for param in pretrained_model.parameters():
        param.requires_grad = False
    if use_gpu:
        pretrained_model = pretrained_model.cuda()
    print('Extracting convolutional features....')
    conv_feat_train, labels_train = extract_features(dataLoaders['train'], pretrained_model)
    save_array('data/processed/catsanddogs/'+ arch +'conv_feat_train.bc', conv_feat_train)
    save_array('data/processed/catsanddogs/'+ arch +'labels_train.bc', labels_train)
    conv_feat_val, labels_val = extract_features(dataLoaders['valid'], pretrained_model)
    save_array('data/processed/catsanddogs/'+ arch +'conv_feat_val.bc', conv_feat_val)
    save_array('data/processed/catsanddogs/'+ arch +'labels_val.bc', labels_val)
    conv_feat_test, labels_test = extract_features(dataLoaders['test'], pretrained_model)
    save_array('data/processed/catsanddogs/'+ arch +'conv_feat_test.bc', conv_feat_test)
    save_array('data/processed/catsanddogs/'+ arch +'labels_test.bc', labels_test)

    conv_feat_train = load_array('data/processed/catsanddogs/'+ arch +'conv_feat_train.bc')
    labels_train = load_array('data/processed/catsanddogs/'+ arch +'labels_train.bc')
    conv_feat_val = load_array('data/processed/catsanddogs/'+ arch +'conv_feat_val.bc')
    labels_val = load_array('data/processed/catsanddogs/'+ arch +'labels_val.bc')
    return (conv_feat_train, labels_train, conv_feat_val, labels_val)


def train(model, dataset_sizes=None, train_features=None, train_labels=None, 
    val_features=None, val_labels=None, 
    criterion=None, optimizer=None, num_epochs=25, train=True, shuffle=None):
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    print('Training classifier.....')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_batches = generator(
            features=conv_feat_train, labels=labels_train, shuffle=shuffle, batch_size=batch_size)
        val_batches = generator(
            features=conv_feat_val, labels=labels_val, shuffle=False, batch_size=batch_size)
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                model.train()
                for data in train_batches:
                    inputs, labels = data
                    if use_gpu:
                        inputs, labels = Variable(torch.from_numpy(inputs).cuda()), Variable(
                            torch.from_numpy(labels).cuda())
                    else:
                        inputs, labels = Variable(torch.from_numpy(
                            inputs)), Variable(torch.from_numpy(labels))
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    optimizer = optimizer
                    optimizer.zero_grad()
                    _, preds = torch.max(outputs.data, 1)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            else:
                model.eval()
                for data in val_batches:
                    inputs, labels = data
                    if use_gpu:
                        inputs, labels = Variable(torch.from_numpy(inputs).cuda()), Variable(
                            torch.from_numpy(labels).cuda())
                    else:
                        inputs, labels = Variable(torch.from_numpy(
                            inputs)), Variable(torch.from_numpy(labels))
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
        print()

    run_time = time.time() - start
    print('Training completed in {:.0f}m and {:.0f}s'.format(
        run_time // 60, run_time % 60))
    print('Best Validation Accuracy: {:4f}'.format(best_accuracy))
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), model_weights_path)
    return model

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     2 command line arguements are created:
       image - Path to image
       checkpoint - path to checkpoint file
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, help="Path to class_names'", default='models/catsanddogs/checkpoint.pth.tar')
    parser.add_argument("--lr", type=float, help="Learning rate'", default=1e-2)
    parser.add_argument("--units", type=int, help="Number of hidden units'", default=0)
    parser.add_argument("--epochs", type=int, help="Number of epochs'", default=20)
    parser.add_argument("--arch", help="default- resnet152 or pick any of the following resnet34, resnet50, resnet152", default='resnet152')
    parser.add_argument("--class_names_path", type=str, help="Path to class_names'", default='models/catsanddogs/class_names.pkl')
    parser.add_argument("--image", type=str, help="Path to image'", default='data/raw/catsanddogs/test-to-org/1.jpg')
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint", default='models/catsanddogs/checkpoint.pth.tar')
    parser.add_argument("--topk", type=int, help="Top k classes to print with probabilities", default=2)
    args = parser.parse_args()
    return args

# Call to main function to run the program
if __name__ == "__main__":
    batch_size = 64
    num_classes = 2
    weight_decay=0
    momentum=0.9
    in_arg = get_input_args()
    model_weights_path = in_arg.model
    arch = in_arg.arch
    num_features = models_dict[arch]['num_features']
    learning_rate = in_arg.lr
    units = in_arg.units
    num_epochs=in_arg.epochs
    class_to_idx, class_names, image_datasets, dataLoaders, dataset_sizes = preprocess('data/raw/catsanddogs', batch_size)
    if not isdir('data/processed/catsanddogs/'+ arch +'conv_feat_train.bc'):
        conv_feat_train, labels_train, conv_feat_val, labels_val = prepdata(arch, dataLoaders)
    else:
        conv_feat_train = load_array('data/processed/catsanddogs/'+ arch +'conv_feat_train.bc')
        labels_train = load_array('data/processed/catsanddogs/'+ arch +'labels_train.bc')
        conv_feat_val = load_array('data/processed/catsanddogs/'+ arch +'conv_feat_val.bc')
        labels_val = load_array('data/processed/catsanddogs/'+ arch +'labels_val.bc')

    # create classifier 
    model = Classifier(num_features, num_classes)

    weights, biases = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            biases += [p]
        else:
            weights += [p]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
    {'params': weights},
    {'params': biases, weight_decay:0}
    ], lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    model = train(model,dataset_sizes=dataset_sizes, train_features=conv_feat_train, train_labels=labels_train, val_features=conv_feat_val, val_labels=labels_val, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, train=True, shuffle=True)