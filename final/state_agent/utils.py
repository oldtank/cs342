import os

from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np
from os import path


def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

def hasNan(x):
    nan_mask = torch.isnan(x)
    nan_present = torch.any(nan_mask)
    return nan_present.item()

def extract_features(pstate, soccer_state, opponent_state, team_id, actions):
    eps = 1e-10  # Small constant

    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / (torch.norm(kart_front-kart_center) + eps)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0] + eps)

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / (torch.norm(puck_center-kart_center) + eps)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0] + eps) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)
 
    # features of opponents 
    opponent_center0 = torch.tensor(opponent_state[0]['kart']['location'], dtype=torch.float32)[[0, 2]]
    opponent_center1 = torch.tensor(opponent_state[1]['kart']['location'], dtype=torch.float32)[[0, 2]]

    # kart_to_opponent0 = (opponent_center0 - kart_center) / torch.norm(opponent_center0-kart_center)
    # kart_to_opponent1 = (opponent_center1 - kart_center) / torch.norm(opponent_center1-kart_center)

    # kart_to_opponent0_angle = torch.atan2(kart_to_opponent0[1], kart_to_opponent0[0]) 
    # kart_to_opponent1_angle = torch.atan2(kart_to_opponent1[1], kart_to_opponent1[0]) 

    # kart_to_opponent0_angle_difference = limit_period((kart_angle - kart_to_opponent0_angle)/np.pi)
    # kart_to_opponent1_angle_difference = limit_period((kart_angle - kart_to_opponent1_angle)/np.pi)

    
    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)
    
    if actions is None:
        features = features.unsqueeze(0)
        return features, None

    if len(actions.keys()) == 0:
        print('NO ACTIONS HERE!!!')
        return None, None
    
    acceleration = actions.get('acceleration').item()
    # if hasNan(actions.get('acceleration')) or hasNan(actions.get('steer')) or hasNan(actions.get('brake')):
    #     return None, None

    if not 0 <= acceleration <= 1:
        acceleration = .5
    assert 0 <= acceleration <= 1, "Values for acceleration are out of range [0, 1]"    
    assert -1 <= actions.get('steer').item() <= 1, "Values for steer are out of range [-1, 1]"
    assert 0 <= actions.get('brake').item() <= 1, "Values for brake are out of range [0, 1]"

    label = torch.tensor([actions.get('acceleration').item(), 
                          actions.get('steer').item(), 
                          actions.get('brake').item()], dtype=torch.float32)
    return features, label

class PlayerDataset(Dataset):
    def __init__(self, data_path):
        self.data = []

        for filename in os.listdir(data_path): 
            print(filename)
            if filename.endswith(".pkl") and filename.startswith("jurgen"):
                score_file = filename.replace('.pkl', '__score.txt')
                with open(os.path.join(data_path, score_file), "r") as s:
                    # read tuple from scorefile

                    scores = tuple(map(int, s.read().split(',')))
                recording = load_recording(os.path.join(data_path, filename))
                print(scores)
                for record in recording:
                    if scores[0] > scores[1] and scores[1] == 0:
                        # break
                        # first team is the winner, use its players for training data
                        pstates = record['team1_state']
                        opponent_state = record['team2_state']
                        team_id = 0
                    elif scores[1] > scores[0] and scores[0] == 0:
                        # second team is the winner, use its players for training data
                        pstates = record['team2_state']
                        opponent_state = record['team1_state']
                        team_id = 1
                    else:
                        # teams tied. don't add to training data
                        continue

                    actions = record['actions']
                    for i, pstate in enumerate(pstates):
                        # for each player, get the features + action label (if there is an action)
                        player_data, player_label = extract_features(pstate, record['soccer_state'], opponent_state, team_id, actions[team_id + i*2])
                        if player_data is None or player_label is None or hasNan(player_data) or hasNan(player_label):
                            break
                        self.data.append((player_data, player_label))

            # if len(self.data) > 0: 
            #     break # todo: uncomment this line to only load one recording

            #     break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def count(self):
        return len(self.data)

def load_data(data_path, num_workers=0, batch_size=128):
    dataset = PlayerDataset(data_path)
    print("num rows in dataset:", len(dataset))
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     from os import path
#     home_dir = path.dirname(path.dirname(path.abspath(__file__)))
#     # train_data = load_data(os.path.join(home_dir, 'recordings'))
#     train_data = load_data('recordings_2', batch_size=32, num_workers=2)
#     print("num batches in dataset", len(train_data))
#     pickle.dump(train_data, open('super_winners_only.pkl', 'wb'))
#     # pickle.load(open('train_data_3.pkl', 'rb'))
#     # print(len(train_data))
