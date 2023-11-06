import torch

class StateAgentModel(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()

        # self.conv1 =torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.relu1 = torch.nn.ReLU()
        # self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        #
        # self.conv2 =torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.relu2 = torch.nn.ReLU()
        # self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        #
        # self.fc1 = torch.nn.Linear(128, 256)
        # self.relu3 = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(256, 64)
        # self.relu4 = torch.nn.ReLU()
        # self.fc3 = torch.nn.Linear(64, 3)

        self.fc1 = torch.nn.Linear(11, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, 128)

        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 32)

        # Output layers
        self.output1 = torch.nn.Linear(32, 1)  # First output (floating-point)
        self.output2 = torch.nn.Linear(32, 1)  # Second output (floating-point)
        self.output3 = torch.nn.Linear(32, 1)  # Third output (floating-point)

        self.network = torch.nn.Sequential(
            self.fc1,self.relu1,self.fc2,self.relu2,
            self.fc3,
            self.fc4,
            self.fc5,
        )

    def forward(self, x):
        # # print(x.shape)
        # x = self.pool1(self.relu1(self.conv1(x)))
        # # print(x.shape)
        # x = self.pool2(self.relu2(self.conv2(x)))
        # x = x.view(x.size(0), -1)  # Flatten the output from convolutional layers
        # # print(x.shape)
        # x = self.relu3(self.fc1(x))
        # x = self.relu4(self.fc2(x))
        # x = self.fc3(x)

        x= self.network(x)

        acceleration = torch.sigmoid(self.output1(x))
        steer = torch.tanh(self.output2(x))
        brake = torch.sigmoid(self.output3(x))

        # brake_mask = brake > 0.5
        # acceleration[brake_mask] = 0
        return acceleration, steer, brake

def save_model(model):
    from torch import save
    from os import path

    traced_model = torch.jit.script(model)
    torch.jit.save(traced_model, path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt'))
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'state_agent.th'))
