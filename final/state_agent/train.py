import torch
import numpy as np
import torch.utils.tensorboard as tb
from .utils import load_data
from .model import StateAgentModel, save_model

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = StateAgentModel(input_size=11).to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'state_agent.th')))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    loss = torch.nn.MSELoss().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    train_data = load_data(args.file, num_workers=4,batch_size=32)
    global_step = 1

    model.train()
    for epoch in range(args.epoch):
        print('starting epoch %d' % epoch)
        for batch_data, batch_label in train_data:
            # print('train data: %s' % repr(train_data))
            # print('train label: %s' % repr(train_label))
            if device is not None:
                batch_data, batch_label = batch_data.to(device),batch_label.to(device)
                batch_data = batch_data.unsqueeze(1)

            output = model(batch_data)
            loss_val = loss(output, batch_label)

            print('loss val % f' % loss_val)
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
    parser.add_argument('-f', '--file')

    args = parser.parse_args()
    train(args)
