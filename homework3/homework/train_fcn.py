import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    from os import path
    model = FCN()
    if device is not None:
        model = model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'fcn-train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'fcn-valid'), flush_secs=1)

    train_data_loader = load_dense_data('dense_data/train', flip=True, color_aug=True, crop=True)
    valid_data_loader = load_dense_data('dense_data/valid')

    n_epochs = args.epoch
    batch_size = args.batch

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=25)

    global_step = 0
    for epoc in range(n_epochs):
        model.train()
        matrix = ConfusionMatrix()
        for batch_data, batch_label in train_data_loader:
            if device is not None:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            output = model(batch_data)

            loss_val = loss(torch.softmax(output, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 5),
                            batch_label.view(-1))
            matrix.add(output.argmax(dim=1), batch_label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step=global_step)

            # Take a gradient step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        if train_logger is not None:
            train_logger.add_scalar('accuracy',matrix.global_accuracy, global_step=global_step)
            train_logger.add_scalar('iou', matrix.iou, global_step=global_step)

        model.eval()
        val_matrix = ConfusionMatrix()
        for batch_data, batch_label in valid_data_loader:
            if device is not None:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            o = model(batch_data)
            val_matrix.add(o.argmax(dim=1), batch_label)
        valid_logger.add_scalar('accuracy', val_matrix.global_accuracy, global_step=global_step)
        valid_logger.add_scalar('iou', val_matrix.iou, global_step=global_step)
        print('epoch = % 3d   train accuracy = %0.3f   train_iou = %0.3f   valid accuracy = %0.3f   valid iou = %0.3f' % (
            epoc, matrix.global_accuracy, matrix.iou, val_matrix.global_accuracy, val_matrix.iou))

        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        scheduler.step(val_matrix.iou)
    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-b', '--batch', default=128, type=int)

    args = parser.parse_args()
    train(args)
