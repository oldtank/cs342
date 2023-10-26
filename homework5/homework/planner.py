import torch
import torch.nn.functional as F
from torchvision import transforms as T


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=2),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

            self.skip = torch.nn.Conv2d(in_channels=n_input, out_channels=n_output, kernel_size=1, stride=2)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=2, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128]):
        super().__init__()

        super().__init__()
        self.normalize = T.Normalize(mean=[0.2788, 0.2657, 0.2628], std=[0.2058, 0.1943, 0.2246])

        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]

        c=3
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l))
            c = l

        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l))
            c = l
            c += skip_layer_size[i]

        self.classifier = torch.nn.Conv2d(c, 1, 1)

    def forward(self, img):
        z = self.normalize(img)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            z = torch.cat([z, up_activation[i]], dim=1)
        heatmap = self.classifier(z).squeeze(1)
        return spatial_argmax(heatmap)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
