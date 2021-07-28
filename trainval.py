import argparse
import os
import torch
import yaml
import time
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from model import TrajectoryModel

parser = argparse.ArgumentParser()
parser.add_argument('--test_set', default='eth')
parser.add_argument('--obs_len', default=8)
parser.add_argument('--seq_len', default=20)
parser.add_argument('--num_epochs', default=300)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--sample_num', default=20, type=int)
parser.add_argument('--start_test', default=10, type=int)

args = parser.parse_args()

train_set = TrajectoryDataset(args, iftrain=True)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
val_set = TrajectoryDataset(args, iftrain=False)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

model = TrajectoryModel(args, dropout=0.1)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
criterion = nn.MSELoss(reduction='none')

def train(epoch):
    model.train()
    loss_epoch = 0
    for cnt, batch_data in enumerate(train_loader):
        start = time.time()
        inputs = tuple([torch.tensor(i).cuda() for i in batch_data])
        inputs = tuple([data.squeeze() for data in inputs])
        loss = torch.zeros(1).cuda()
        traj, traj_rel, social_adj = inputs
        inputs = traj[:-1], traj_rel[:-1], social_adj[:-1]
        optimizer.zero_grad()
        outputs = model.forward(inputs, iftest=False)
        loss = torch.sum(criterion(outpus, traj[1:]), dim=2)
        loss_epoch += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        end = time.time()

        if cnt % 100 == 99:
            print('train-{}/{} (epoch {}), train_loss = {:.5f}'.format(cnt, len(train_loader), epoch, loss.item()))

        return loss_epoch

@torch.no_grad()
def val(epoch):
    model.eval()
    ade_all = []
    fde_all = []
    for cnt, batch_data in tqdm(enumerate(val_loader)):
        inputs = tuple([torch.tensor(i).cuda() for i in batch_data])
        inputs = tuple([data.squeeze() for data in inputs])
        traj, traj_rel, social_adj = inputs
        inputs = traj[:-1], traj_rel[:-1], social_adj[:-1]
        ade_batch = []
        fde_batch = []
        for i in range(args.sample_num):
            outputs = model.forward(inputs, iftest=True)
            ade, fde = ade_and_fde(outputs, traj[1:])
            ade_batch.append(ade)
            fde_batch.append(fde)
        ade_all.append(min(ade_batch))
        fde_all.append(min(fde_batch))

    ade = sum(ade_all) / len(ade_all)
    fde = sum(fde_all) / len(fde_all)

    return ade, fde

print('Training Begin')
best_ade = 100
best_fde = 100
for epoch in range(args.num_epochs):
    loss = train(epoch)
    if epoch >= args.start_test:
        ade, fde = val(epoch)
        if fde < best_fde:
            best_ade = ade
            best_fde = fde
            best_epoch = epoch
        print('----epoch {}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'.format(epoch, loss, ade, fde, best_ade, best_fde, best_epoch))
    else:
        print('----epoch {}, train_loss={:.5f}'.format(epoch, loss))
