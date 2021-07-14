import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from utils import TrajectoryDataset
from model import TrajectoryModel

parser = argparse.ArgumentParser()
parser.add_argument('--test_set', default='eth')
parser.add_argument('--obs_len', default=8)
parser.add_argument('--seq_len', default=20)
parser.add_argument('--num_epochs', default=300)

args = parser.parse_args()

train_set = TrajectoryDataset(args, iftrain=True)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
val_set = TrajectoryDataset(args, iftrain=False)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

model = TrajectoryModel(args, dropout=0.1)

def train():
    print('Training begin')
    test_error, test_final_error = 0, 0
    for epoch in range(args.num_epochs):
        
