import torch
import torch.nn.functional as F
from torchvision import transforms


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x)+identity

    def __init__(self, layers=[32, 64, 128], n_input_channels=3):
        super().__init__()
        self.normalize= transforms.Normalize(mean=[0.3234, 0.3310, 0.3444], std=[0.2524, 0.2219, 0.2470])
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.BatchNorm2d(32),
             torch.nn.ReLU()
             ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(in_features=c, out_features=6)

    def forward(self, x):
        # normalize inputs
        x = self.normalize(x)

        # Compute the features
        z = self.network(x)
        # Global average pooling
        z = z.mean(dim=[2, 3])
        # Classify
        return self.classifier(z)


class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x)+identity    
            
    def __init__(self, n_input_channels=3):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        # Initial convolutional layer
        self.conv1 = torch.nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

        # blocks
        self.block1 = self.Block(64, 128, stride=2)
        # self.block2 = self.Block(128, 256, stride=2)

        # up-convolutionn
        # self.upconv1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # final convo layer
        self.output_layer = torch.nn.Conv2d(128, 5, kernel_size=1)

        self.output_layer_no_stride = torch.nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        # if input size is 1*1, do not need to do anything

         # through initial layer
        # print('shape before initial layer: ' + repr(x.shape))
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        # print('shape after initial layer: ' + repr(x1.shape))

        if x.size(2) == 1 or x.size(3) == 1:
            return self.output_layer_no_stride(x1)

        # through blocks
        x2 = self.block1(x1)
        # print('shape after block1: ' + repr(x2.shape))
        # x3 = self.block2(x2)
        # print('shape after block2: ' + repr(x3.shape))

        # # up-connvo
        # x4 = self.upconv1(x3)
        # print('shape after upconvo1: ' + repr(x4.shape))
        # # skip connection
        # x4 = torch.cat([x4, x2], dim=1)
        # print('shape after skip1: ' + repr(x4.shape))
        
        x3 = self.upconv2(x2)
        # print('shape after upconvo2: ' + repr(x3.shape))
        x3 = torch.cat([x3, x1], dim=1)  # Concatenate skip connection
        # print('shape after skip2: ' + repr(x3.shape))

        final_output = self.output_layer(x3)
        # print('final output: ' + repr(final_output.shape))
        return final_output

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
