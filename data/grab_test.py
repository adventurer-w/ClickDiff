import os.path as op
import sys

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from ipdb import set_trace as st

class GrabDataset(Dataset):
    def __getitem__(self, index):
        idx = self.idxs[index]
        data = self.getitem(idx)
        return data

    def getitem(self, idx, load_rgb=True):

        data_input_all =self.data[idx,:]
        data_input_dict = {}
        data_input_dict['seq_len'] = data_input_all.shape[0]
        data_input_dict['motion'] = data_input_all.unsqueeze(0)
        data_input_dict['name'] = self.frame_names[idx]
        data_input_dict['targets'] ={}

        for key in self.targets:
            if key != 'v' and  key != 'mask' and key != 'parts_ids':
                if len(self.targets[key]) == self.targets['v_sub'].shape[0]:
                    data_input_dict['targets'][key] = self.targets[key][idx]
                else:
                    data_input_dict['targets'][key] = self.targets[key]

        return data_input_dict


    def _load_data(self):
        data_p = op.join(
            f"/data-home/arctic-master/outputs_grab/grab_final/grab_test.npy"
        )
        
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()
        # st()
        self.data = data["data_dict"]
        self.idxs = data["imgnames"]
        self.frame_names = data["frame_names"]
        self.targets = data["targets"]


    def __init__(self, args='', split='', seq=None):
        self._load_data()
        logger.info(
            f"Dataset Loaded, num samples {len(self.idxs)}"
        )

    def __len__(self):
        return len(self.idxs)

 

if __name__ == "__main__":
    ds = GrabDataset(args='',split='')
    ds.__getitem__(0)
    ds.__getitem__(1)