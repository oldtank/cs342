import os

from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np
from os import path

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

def hasNan(x):
    nan_mask = torch.isnan(x)
    nan_present = torch.any(nan_mask)
    return nan_present.item()

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
    def __init__(self):
        self.data = []
        from os import path
        home_dir = path.dirname(path.dirname(path.abspath(__file__)))
        for filename in os.listdir(home_dir):
            if filename.endswith(".pkl"):
                with open(filename, "rb") as f:
                    team_name = 'team1_state' if filename.startswith('0_') else 'team2_state'
                    team_id = 0 if filename.startswith('0_') else 1
                    player1_idx = 0 if filename.startswith('0_') else 1
                    player2_idx = 2 if filename.startswith('0_') else 3
                    while True:
                        try:
                            states = pickle.load(f)
                            # player1
                            player1_data, player1_label = get_featuers_for_player(
                                states[team_name][0], states['soccer_state'], team_id,
                                states['actions'][player1_idx]
                            )
                            #player2
                            player2_data, player2_label = get_featuers_for_player(
                                states[team_name][1], states['soccer_state'], team_id,
                                states['actions'][player2_idx]
                            )

                            if not hasNan(player1_data) and not hasNan(player1_label):
                                self.data.append((player1_data, player1_label))

                            if not hasNan(player2_data) and not hasNan(player2_label):
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

def load_data(num_workers=0, batch_size=128):
    dataset = PlayerDataset()
    print(dataset.count())

    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    # image_jurgen_agent
    # yann_agent
    # jurgen_agent* 91
    import argparse
    parser = argparse.ArgumentParser()
    train_data = load_data()
