import os
import numpy as np
from torch.utils.data import Dataset


DATASET_NAME_TO_NUM = {'eth': 0, 'hotel': 1, 'zara1': 2, 'zara2': 3, 'univ': 4}
data_dirs = ['./data/eth/univ', './data/eth/hotel',
             './data/ucy/zara/zara01', './data/ucy/zara/zara02',
             './data/ucy/univ/students001', './data/ucy/univ/students003',
             './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']


class TrajectoryDataset(Dataset):
    def __init__(self, args, iftrain=True):
        super(TrajectoryDataset, self).__init__()
        self.args = args
        all_data = [i for i in range(len(data_dirs))]
        test_num = DATASET_NAME_TO_NUM[args.test_set]
        if test_num == 4:
            test_data = [4, 5]
        else:
            test_data = [test_num]
        for x in test_data:
            train_data = all_data.remove(x)
        if iftrain:
            data_path = [data_dirs[i] for i in train_data]
        else:
            data_path = [data_dirs[i] for i in test_data]
        ped_list = []
        ped_list_rel = []
        nums_in_ped_list = []
        for path in data_path:
            file_path = os.path.join(path, 'true_pos_.csv')
            data = np.genfromtxt(file_path, delimiter=',')
            frames = np.unique(data[0, :]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[:, frame == data[0, :]])
            for idx in range(len(frames) - args.seq_len + 1):
                curr_frame = np.concatenate(frame_data[idx: idx + args.seq_len], axis=1)
                peds_nums = np.unique(curr_frame[1, :]).tolist()
                peds_traj = np.zeros((args.seq_len, len(peds_nums), 2))
                peds_traj_rel = np.zeros((args.seq_len), len(peds_nums), 2)
                peds = 0
                for ped_id in peds_nums:
                    curr_ped_traj = curr_frame[:, ped_id == curr_frame[1, :]]
                    pad_front = frames.index(curr_ped_traj[0, 0]) - idx
                    pad_end = frames.index(curr_ped_traj[0, -1]) - idx + 1
                    if pad_end - pad_front != args.seq_len:
                        continue
                    curr_ped_traj = curr_ped_traj[2:, :]
                    curr_ped_traj_rel = np.zeros(curr_ped_traj.shape)
                    curr_ped_traj_rel[:, 1:] = curr_ped_traj[:, 1:] - curr_ped_traj[:, :-1]
                    peds_traj[pad_front:pad_end, peds, :] = np.transpose(curr_ped_traj)
                    peds_traj_rel[pad_front:pad_end, peds, :] = np.transpose(curr_ped_traj_rel)
                    peds += 1

                if peds > args.min_ped:
                    nums_in_ped_list.append(peds)
                    ped_list.append(peds_traj[:, :peds])
                    ped_list_rel.append(peds_traj_rel[:, :peds])

        self.traj_num = len(nums_in_ped_list)
        self.ped_list = np.concatenate(ped_list, axis=1)
        self.ped_list_rel = np.concatenate(ped_list_rel, axis=1)

        start_end = [0] + np.cumsum(nums_in_ped_list).tolist()
        self.traj_start_end = [(start, end) for start, end in zip(start_end, start_end[1:])]


    def __len__(self):
        return self.traj_num

    def __getitem__(self, index):
        start, end = self.traj_start_end[index]
        traj = self.ped_list[:, start:end]
        traj_rel = self.ped_list_rel[:, start:end]

        return traj, traj_rel