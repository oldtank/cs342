import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        softmax = torch.nn.functional.softmax(input, dim=1)
        return -softmax[torch.arange(softmax.size()[0]), target].log().mean()


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3*64*64, out_features=6)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return self.linear(x)


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [torch.nn.Linear(in_features=3 * 64 * 64, out_features=100), torch.nn.ReLU(),
                  torch.nn.Linear(in_features=100, out_features=6)]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size()[0], -1))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
