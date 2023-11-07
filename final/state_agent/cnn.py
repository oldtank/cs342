import torch

class ConvoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)

        self.conv2 =torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2)

        self.fc1 = torch.nn.Linear(128, 64)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 3)

        self.network = torch.nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2
        )

    def forward(self, x):
        x= self.network(x)
        x = x.view(x.size(0), -1)

        output = self.fc3(self.relu4(self.fc2(self.relu3(self.fc1(x)))))
        acceleration = torch.sigmoid(output[:,0]).unsqueeze(1)
        steer = torch.tanh(output[:,1]).unsqueeze(1)
        brake = torch.sigmoid(output[:,2]).unsqueeze(1)

        return acceleration, steer, brake

def save_model(model):
    from torch import save
    from os import path

    traced_model = torch.jit.script(model)
    torch.jit.save(traced_model, path.join(path.dirname(path.abspath(__file__)), 'cnn_agent.pt'))
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn_agent.th'))