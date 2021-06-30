import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--test_set', default='eth')
parser.add_argument('--obs_len', default=8)
parser.add_argument('--seq_len', default=20)

args = parser.parse_args()

