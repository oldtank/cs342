import torch
import numpy as np
import torch.utils.tensorboard as tb
from .utils import load_data
from .model import StateAgentModel, save_model
from os import path
import math


class CustomLoss(torch.nn.Module):
    def __init__(self, device):
        super(CustomLoss, self).__init__()
        self.acceleration_loss = torch.nn.MSELoss().to(device)
        self.steer_loss = torch.nn.MSELoss().to(device)
        self.brake_loss = torch.nn.BCELoss().to(device)

    def forward(self, acceleration, steer, brake, batch_label):
        # brake_mask = brake > 0.5
        # acceleration[brake_mask] = 0

        # brake_mask_label = batch_label[:,2] == 1

        steer_loss_val = self.steer_loss(steer, batch_label[:, 1].unsqueeze(1))
        brake_loss_val = self.brake_loss(brake, batch_label[:, 2].unsqueeze(1))
        acceleration_loss_val = self.acceleration_loss(acceleration, batch_label[:, 0].unsqueeze(1))

        return steer_loss_val + brake_loss_val + 0.5*acceleration_loss_val

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = StateAgentModel(input_size=11).to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'state_agent.th')))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    loss = CustomLoss(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    train_data = load_data(num_workers=4,batch_size=32)
    global_step = 1

    model.train()
    for epoch in range(args.epoch):
        print('starting epoch %d' % epoch)
        for batch_data, batch_label in train_data:
            # print('train data: %s' % repr(train_data))
            # print('train label: %s' % repr(train_label))
            if device is not None:
                batch_data, batch_label = batch_data.to(device),batch_label.to(device)

            # print(batch_data.shape)
            # print(batch_label.shape)
            acceleration, steer, brake = model(batch_data)

            loss_val = loss(acceleration, steer, brake, batch_label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            if global_step % 200 == 0:
                print('loss val % f' % loss_val)

            if math.isnan(loss_val.item()):
                print('nan. label: %s' % (repr(batch_label)))
                break
            # print('loss val % f' % loss_val)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step +=1

    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='logs/')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', default=20, type=int)
    parser.add_argument('-c', '--continue_training', default=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)

    args = parser.parse_args()
    train(args)
