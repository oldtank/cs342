import torch 

class StateAgentModel(torch.nn.Module):
    def __init__(self, layer_sizes = [32, 64, 128, 256], n_input_features=11, n_output_features=3):
        super().__init__()

        layers = []
        layers.append(torch.nn.Linear(n_input_features, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(torch.nn.LayerNorm(layer_sizes[i+1]))
            layers.append(torch.nn.ReLU())

        self.network = torch.nn.Sequential(*layers)

        self.classifier = torch.nn.Linear(layer_sizes[-1], n_output_features)


    def forward(self, x):
        n = self.network(x)
        z = self.classifier(n)
        acceleration = torch.sigmoid(z[:,0])
        steer = torch.tanh(z[:,1])
        brake = torch.sigmoid(z[:,2])

     
        assert torch.all((acceleration >= 0) & (acceleration <= 1)), "Values for acceleration are out of range [0, 1]"
        assert torch.all((steer >= -1) & (steer <= 1)), "Values for steer are out of range [-1, 1]"
        assert torch.all((brake >= 0) & (brake <= 1)), "Values for brake are out of range [0, 1]"
        
        return acceleration.unsqueeze(1), steer.unsqueeze(1), brake.unsqueeze(1)

def save_model(model):
    from torch import save
    from os import path

    traced_model = torch.jit.script(model)
    torch.jit.save(traced_model, path.join(path.dirname(path.abspath(__file__)), 'state_agent.pt'))
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'state_agent.th'))
