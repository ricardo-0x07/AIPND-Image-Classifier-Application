import numpy as np
import torch
import bcolz as bz
from torchvision import datasets, transforms
import copy
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os, sys
# %matplotlib inline


use_gpu = torch.cuda.is_available()

def preprocess(data_base_path, batch_size):
    preprocess = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 406], std=[0.229, 0.224, 0.225])
        ])
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_base_path, x), preprocess[x]) for x in ['train', 'valid', 'test']}
    dataLoaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    class_to_idx = image_datasets['train'].class_to_idx
    return (class_to_idx, class_names, image_datasets, dataLoaders, dataset_sizes)
    



def generator(features, labels, batch_size=32, shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(features))
        features = features[index]
        labels = labels[index]
    for i in range(0, len(features), batch_size):
        yield (features[i:i+batch_size], labels[i:i+batch_size])

def test_accuracy(model, dataset_size=27, conv_feat_test=None, labels_test=None, criterion=None,
    num_epochs=1, shuffle=None):
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-'*10)
        val_batches = generator(features=conv_feat_test, labels=labels_test, shuffle=False, batch_size=batch_size)
        running_loss = 0.0
        running_corrects = 0
        model.eval()
        for data in val_batches:
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(torch.from_numpy(inputs).cuda()), Variable(torch.from_numpy(labels).cuda())
            else:
                inputs, labels = Variable(torch.from_numpy(inputs)), Variable(torch.from_numpy(labels))
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)

        epoch_loss = running_loss/dataset_size
        epoch_accuracy = running_corrects/dataset_size
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Testing:', epoch_loss, epoch_accuracy))

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
        print()
    
    run_time = time.time() - start
    print('Testing completed in {:.0f}m and {:.0f}s'.format(run_time//60, run_time%60))
    print('Best Test Accuracy: {:4f}'.format(best_accuracy))
    model.load_state_dict(best_model_weights)
    return model

def sanity_checking_model(model, pretrained_model,num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(dataLoaders['test']):
        inputs, labels = data
        print('inputs.shape', inputs.shape)
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        extracted_inputs = pretrained_model(inputs)
        outputs = model(extracted_inputs)
        _, preds = torch.max(outputs.data, 1)
        for k in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('Predicted: {}'.format(test_class_names[preds[k]]))
            imshow(inputs.cpu().data[k])
            if images_so_far == num_images:
                return

def imshow(inp, title=None):
    """imshow for tensor"""
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

from torch.autograd import Variable
import numpy as np


def extract_features(dataset, pretrained_model):
    features = []
    labels_list = []
    for data in dataset:
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        x = pretrained_model(inputs)
        features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    features = np.concatenate([[feat] for feat in features])
    return (features, labels_list)

def save_array(filename, arr):
    c=bz.carray(arr, rootdir=filename, mode='w')
    c.flush()

def load_array(filename):
    return bz.open(filename)[:]