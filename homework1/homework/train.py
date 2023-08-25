from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data, SuperTuxDataset
import torch
import torch.utils.tensorboard as tb


def train(args):
    train_logger = tb.SummaryWriter('homework/logs/' + args.model + '/train', flush_secs=1)

    n_epochs = args.epoch
    batch_size = args.batch

    # get model
    model = model_factory[args.model]()

    # loss
    loss = ClassificationLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # load data
    train_data_loader = load_data('data/train', batch_size=batch_size)

    global_step = 0
    for epoc in range(n_epochs):
        for batch_data, batch_label in train_data_loader:
            output = model(batch_data)
            # compute loss
            loss_val = loss(output, batch_label)

            train_logger.add_scalar('train/loss', loss_val, global_step=global_step)
            train_logger.add_scalar('train/accuracy', accuracy(output, batch_label), global_step=global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-b', '--batch', default=128, type=int)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
