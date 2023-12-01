import torch
import numpy as np
from .model import StateAgentModel, save_model
from os import path
import wandb
import pickle

class CustomLoss(torch.nn.Module):
    def __init__(self, device):
        super(CustomLoss, self).__init__()
        self.acceleration_loss = torch.nn.MSELoss().to(device)
        self.steer_loss = torch.nn.MSELoss().to(device)
        self.brake_loss = torch.nn.BCELoss().to(device)

    def forward(self, acceleration, steer, brake, batch_label):

        acceleration_loss_val = self.acceleration_loss(acceleration, batch_label[:, 0].unsqueeze(1))
        steer_loss_val = self.steer_loss(steer, batch_label[:, 1].unsqueeze(1))
        brake_loss_val = self.brake_loss(brake, batch_label[:, 2].unsqueeze(1))

        return steer_loss_val + brake_loss_val + acceleration_loss_val
    
def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = StateAgentModel(n_input_features=11).to(device)
    if args.continue_training:
        # model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'state_agent.th')))
        model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt'))
        train_data = pickle.load(open('train_data.pkl', 'rb'))
    # else:
    #     train_data = pickle.load(open('super_winners_only.pkl', 'rb'))
    #     print("loaded pickle file datastet", len(train_data))

    train_data = pickle.load(open('train_data.pkl', 'rb'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    loss = CustomLoss(device)

    # train_data = load_data('recordings', num_workers=2, batch_size=args.batch_size)

    wandb.init(
    # set the wandb project where this run will be logged
        project="final_project",
        config = {
            "epochs": args.epoch,
            "optimizer": optimizer.__class__.__name__,
            "device": device
        }
    )

    global_step = 0
    model.train()
    for epoch in range(args.epoch):
        loss_vals = []
        print("EPOCH", epoch)
        for batch_data, batch_label in train_data:
            # print('train data: %s' % repr(train_data))
            # print('train label: %s' % repr(train_label))
            if device is not None:
                batch_data, batch_label = batch_data.to(device),batch_label.to(device)

            # print(batch_data.shape)
            # print(batch_label.shape)
            acceleration, steer, brake = model(batch_data)

            loss_val = loss(acceleration, steer, brake, batch_label)

            if global_step % 1000 == 0:
                # print('loss val % f' % loss_val)
                wandb.log({"train/loss": loss_val}, step=global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step +=1
            loss_vals.append(loss_val.item())
        print("average loss for epoch", np.mean(loss_vals))

    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='logs/')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', default=20, type=int)
    parser.add_argument('-c', '--continue_training', default=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=32, type=int)

    args = parser.parse_args()
    train(args)
