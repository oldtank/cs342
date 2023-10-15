import torch
import torch.nn.functional as F
from . import dense_transforms
from torchvision import transforms as T
from torchvision.transforms import functional as TVF


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    heatmap_formatted = heatmap[None, None]
    max_pool = F.max_pool2d(input=heatmap_formatted, kernel_size=max_pool_ks, stride=1, padding=max_pool_ks//2, return_indices=False)

    maxima_mask = (max_pool == heatmap_formatted)
    maxima_mask = maxima_mask & (heatmap_formatted > min_score)

    # find indices
    indices = torch.nonzero(maxima_mask, as_tuple=False)

    maxima_values = heatmap_formatted[maxima_mask]
    topk_values, topk_indices = torch.topk(maxima_values, min(indices.shape[0], max_det))

    # removes the added two empty columns; and reverse the remaining two columns
    top_indices = indices[topk_indices][:,2:][:, [1, 0]]

    indices_list = top_indices.tolist()
    result = []
    for index in indices_list:
        result.append((heatmap[index[1]][index[0]], index[0], index[1]))
    return result

class Detector(torch.nn.Module):
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

    def __init__(self, layers=[16, 32, 64, 128], n_output_channel=5):
        """
           Your code here.
           Setup your detection network
        """
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

        self.classifier = torch.nn.Conv2d(c, n_output_channel, 1)

    def forward(self, x):
        z = self.normalize(x)
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
        return self.classifier(z)

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        image_reshape = image.unsqueeze(0)
        output = self.forward((image_reshape))

        peaks=[]
        for i in range(3):
            peaks_one_class = []
            curr_heatmap = output[0][i]
            for detection, cx, cy in extract_peak(heatmap=curr_heatmap, max_det=30):
                peaks_one_class.append((detection.item(), cx, cy, output[0][3][cy][cx], output[0][4][cy][cx]))
            peaks.append(peaks_one_class)

        return peaks


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
