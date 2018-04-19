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
ImageFile.LOAD_TRUNCATED_IMAGES = True
from helpers import generator


def train(model, dataset_sizes=None, train_features=None, train_labels=None, val_features=None,
    val_labels=None, criterion=None, optimizer=None, num_epochs=25, train=True, shuffle=None):
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_batches = generator(
            features=train_features, labels=train_labels, shuffle=shuffle, batch_size=batch_size)
        val_batches = generator(
            features=val_features, labels=val_labels, shuffle=False, batch_size=batch_size)
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
    return model
