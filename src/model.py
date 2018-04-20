import torch.nn as nn


class Classifier(nn.Module):
    """Feedforward Classifier
    """
    def __init__(self, num_features, num_classes):
        super(Classifier, self).__init__()
        self.classifier = self.make_layers(n_layers=2, num_features=num_features, num_classes=num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def make_layers(self, n_layers=0, num_features=2048, num_classes=2):
        layers = []
        if n_layers > 0:
            for _ in range(n_layers):
                layers += [nn.Linear(num_features, num_features)]
                layers += [nn.ELU(inplace=True)]
                layers += [nn.Dropout()]
        layers += [nn.Linear(num_features, num_classes)]        
        return nn.Sequential(*layers)
