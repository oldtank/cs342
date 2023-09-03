import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer_channel_counts = [32, 64, 128]
        L= []
        input_channel = 3

        for output_channel in layer_channel_counts:
            L.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                     kernel_size=3, padding=1, stride=2))
            L.append(torch.nn.ReLU())
            input_channel = output_channel
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(in_features=input_channel, out_features=6)

    def forward(self, x):
        z = self.network(x)
        z = z.mean(dim=[2,3])
        return self.classifier(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
