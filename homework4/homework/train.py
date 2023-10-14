import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    weights=torch.tensor([26787764/631])
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(334589755/7948,dtype=torch.float)).to(device)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4,
                                     batch_size=128,
                                     transform=dense_transforms.Compose([
                                         dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                                         dense_transforms.RandomHorizontalFlip(),
                                         dense_transforms.ToTensor(),
                                         dense_transforms.ToHeatmap()
                                     ]))
    valid_data = load_detection_data('dense_data/valid', num_workers=4,
                                     batch_size=128,
                                     transform=dense_transforms.Compose([
                                         dense_transforms.ToTensor(),
                                         dense_transforms.ToHeatmap()]))

    global_step = 0
    for epoch in range(args.epoch):
        print('Starting epoch % 3d' % epoch)
        model.train()
        accuracies = []
        for img, peak, size in train_data:
            img, peak, size = img.to(device), peak.to(device), size.to(device)
            output = model(img)
            loss_val = loss(output.view(-1), peak.view(-1))

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                train_logger.add_scalar('accuracy', accuracy(output, peak).detach().cpu().item())

            if train_logger is not None and global_step %100 ==0:
                log(train_logger, img, peak, output, global_step)

            accuracies.append(accuracy(output, peak).detach().cpu().item())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step+=1

        model.eval()
        val_accuracies = []
        val_img=None
        val_peak=None
        val_output=None
        for img, peak, size in valid_data:
            val_img = img
            val_peak=peak
            img, peak, size = img.to(device), peak.to(device), size.to(device)
            output = model(img)
            val_accuracies.append(accuracy(output, peak).detach().cpu().item())
            val_output = output.detach().cpu()
        valid_logger.add_scalar('accuracy', np.mean(val_accuracies), global_step=global_step)
        if valid_logger is not None and global_step % 100 == 0:
            log(valid_logger, val_img, val_peak, val_output, global_step)

        print('epoch = % 3d   train accuracy = %0.3f   valid accuracy = %0.3f' % (
            epoch, np.mean(accuracies), np.mean(val_accuracies)))

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='logs/')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')
    args = parser.parse_args()
    train(args)
