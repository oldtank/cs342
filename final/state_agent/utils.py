from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np
from os import path

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

def get_featuers_for_player(player, soccer_state, team_id, actions):
    kart_front = torch.tensor(player['kart']['front'])[[0, 2]]
    kart_center = torch.tensor(player['kart']['location'])[[0, 2]]
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle) / np.pi)

    # soccer line
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    puck_to_goal_line = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)

    # actions
    acceleration = actions['acceleration'].item()
    steer = actions['steer'].item()
    brake = actions['brake'].item()

    data = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle,
                  goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference,
                  puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)
    label = torch.tensor([acceleration, steer, brake], dtype=torch.float32)

    return data, label
class PlayerDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        from os import path
        with open(path.join('state_agent', dataset_path), "rb") as f:
            while True:
                try:
                    states = pickle.load(f)
                    # player1
                    player1_data, player1_label = get_featuers_for_player(
                        states['team1_state'][0], states['soccer_state'], 0,
                        states['actions'][0]
                    )
                    player2_data, player2_label = get_featuers_for_player(
                        states['team1_state'][1], states['soccer_state'], 1,
                        states['actions'][2]
                    )

                    self.data.append((player1_data, player1_label))
                    self.data.append((player2_data, player2_label))

                    # print(player1_data)
                    # print(player1_label)
                except EOFError:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def count(self):
        return len(self.data)

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = PlayerDataset(path.join(path.dirname(path.dirname(path.abspath(__file__))),dataset_path))
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    # image_jurgen_agent
    # yann_agent
    # jurgen_agent* 91
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    args = parser.parse_args()
    train_data = load_data(args.file)
