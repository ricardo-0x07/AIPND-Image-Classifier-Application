import torch.nn as nn


class Classifier(nn.Module):
    """Feedforward Classifier
    """
    def __init__(self, num_features, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
