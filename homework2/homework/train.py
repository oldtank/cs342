from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import numpy as np

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    n_epochs = args.epoch
    batch_size = args.batch

    # loss
    loss = torch.nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters())

    # load data
    train_data_loader = load_data('data/train', batch_size=batch_size)
    valid_data_loader = load_data('data/valid', batch_size=batch_size)

    global_step = 0
    for epoc in range(n_epochs):
        accuracies = []
        for batch_data, batch_label in train_data_loader:
            output = model(batch_data)
            # compute loss
            loss_val = loss(output, batch_label)
            train_logger.add_scalar('loss', loss_val, global_step=global_step)

            accuracies.append(accuracy(output, batch_label).item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        train_logger.add_scalar('accuracy', np.mean(accuracies), global_step=global_step)

        val_accuracies = []
        for batch_data, batch_label in valid_data_loader:
            o = model(batch_data)
            val_accuracies.append(accuracy(o, batch_label).item())
        valid_logger.add_scalar('accuracy', np.mean(val_accuracies), global_step=global_step)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-b', '--batch', default=128, type=int)
    args = parser.parse_args()
    train(args)
