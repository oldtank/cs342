from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    train_logger = tb.SummaryWriter('homework/logs/linear', flush_secs=1)

    n_epochs = 100
    batch_size = 128

    # get model
    model = model_factory[args.model]()

    # loss
    loss = ClassificationLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # load data
    train_data_loader = load_data('data/train')
    valid_data_loader = load_data('data/valid')

    for iteration in range(100):
        batch_data,batch_label =train_data_loader[iteration]
        output = model(batch_data)
        # compute loss
        loss_val = loss(output, batch_label.float())

        train_logger.add_scalar('loss', loss_val, global_step=iteration)
        train_logger.add_scalar('accuracy', accuracy(output, batch_label), global_step=iteration)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


    # save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
