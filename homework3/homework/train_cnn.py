from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    from os import path
    model = CNNClassifier()
    if device is not None:
        model = model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    n_epochs = args.epoch
    batch_size = args.batch

    # loss
    loss = torch.nn.CrossEntropyLoss()

    # data loader
    train_data_loader = load_data('data/train', batch_size=batch_size, random_crop=(60,60), flip=True)
    valid_data_loader = load_data('data/valid', batch_size=batch_size, resize=(60,60))

    # optimizer
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)

    global_step = 0
    for epoc in range(n_epochs):
        model.train()
        accuracies = []
        for batch_data, batch_label in train_data_loader:
            if device is not None:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)

            output = model(batch_data)

            # loss and accuracy
            loss_val = loss(output, batch_label)
            accuracies.append(accuracy(output, batch_label).detach().cpu().item())

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step=global_step)

            # Take a gradient step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        # log
        if train_logger is not None:
            train_logger.add_scalar('accuracy', np.mean(accuracies), global_step=global_step)

        model.eval()
        val_accuracies = []
        for batch_data, batch_label in valid_data_loader:
            if device is not None:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            o = model(batch_data)
            val_accuracies.append(accuracy(o, batch_label).detach().cpu().item())
        valid_logger.add_scalar('accuracy', np.mean(val_accuracies), global_step=global_step)
        print('epoch = % 3d   train accuracy = %0.3f   valid accuracy = %0.3f' % (
            epoc, np.mean(accuracies), np.mean(val_accuracies)))

        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        scheduler.step(np.mean(val_accuracies))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-b', '--batch', default=128, type=int)
    args = parser.parse_args()
    train(args)
