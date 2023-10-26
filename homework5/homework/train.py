from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    from os import path
    model = Planner().to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    loss = torch.nn.MSELoss().to(device)

    train_data = load_data('drive_data', num_workers=4,
                                     batch_size=32,
                                     transform=dense_transforms.Compose([
                                         dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                                         dense_transforms.RandomHorizontalFlip(),
                                         dense_transforms.ToTensor()
                                     ]))
    global_step = 0
    for epoch in range(args.epoch):
        print('Starting epoch % 3d' % epoch)
        model.train()

        for batch_image, batch_label in train_data:
            if device is not None:
                batch_image, batch_label = batch_image.to(device),batch_label.to(device)
            output = model(batch_image)

            # loss and accuracy
            loss_val = loss(output, batch_label)

            if train_logger is not None:
                train_logger.add_scalar('size loss', loss_val, global_step)

            if train_logger is not None and global_step %100 ==0:
                log(train_logger, batch_image, batch_label, output, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step+=1

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='logs/')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')
    parser.add_argument('-c', '--continue_training', default=False)

    args = parser.parse_args()
    train(args)
