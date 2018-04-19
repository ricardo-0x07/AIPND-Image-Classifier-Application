import torch
import torch.nn as nn
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import pickle
from helpers import extract_features
from model import Classifier
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


use_gpu = torch.cuda.is_available()



def predict():
    in_arg = get_input_args()
    img_path = in_arg.image
    model_weights_path = in_arg.checkpoint
    topk = in_arg.topk
    class_names_path = in_arg.class_names_path
    # apply model to input
    pretrained_model = models.resnet152(pretrained=True)
    num_features = pretrained_model.fc.in_features
    modules = list(pretrained_model.children())[:-1]
    pretrained_model=nn.Sequential(*modules)
    for param in pretrained_model.parameters():
        param.requires_grad = False
    if use_gpu:
        pretrained_model = pretrained_model.cuda()
    
    # obtain labels
    class_names = pickle.load(open(class_names_path, mode='rb'))
    # load the image
    img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable
    if use_gpu:
        data = Variable(img_tensor.cuda())
    else:
        data = Variable(img_tensor)
    # data = Variable(img_tensor, volatile=True) 

    # extract features with pretrained model
    output = pretrained_model(data)

    # apply classifier 
    model = Classifier(num_features, 2)

    # load model weights
    state = torch.load(model_weights_path)
    model.load_state_dict(state)

    # puts model in evaluation mode
    # instead of (default)training mode
    model.eval()
    output = model(output)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()
    if topk > 0:
        _, predictedTopK = output.topk(topk)
        probs = torch.nn.functional.softmax(output, dim=1)
        probs = probs.data.numpy()
        predictedTopK = predictedTopK.data.numpy()
        _, preds = torch.max(output.data, 1)
        for k in range(topk):
            print('Predicted: {} with a Probability of: {:.2%}'.format(class_names[0][predictedTopK[0][k]], probs[0][predictedTopK[0][k]]))
            print()

    return class_names[0][pred_idx]

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
    
    parser.add_argument("--class_names_path", type=str, help="Path to class_names'", default='models/catsanddogs/class_names.pkl')
    parser.add_argument("--image", type=str, help="Path to image'", default='data/raw/catsanddogs/test-to-org/1.jpg')
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint", default='models/catsanddogs/checkpoint.pth.tar')
    parser.add_argument("--topk", type=int, help="Top k classes to print with probabilities", default='2')
    args = parser.parse_args()
    return args

# Call to main function to run the program
if __name__ == "__main__":
    predicted_class = predict()
    print('Predicted class', predicted_class)