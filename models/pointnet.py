from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from ipdb import set_trace as st
"""
Source: https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py
"""

class PointNetfeat(nn.Module):
    def __init__(self, input_dim, shallow_dim, mid_dim, out_dim, global_feat=False):
        super(PointNetfeat, self).__init__()
        self.shallow_layer = nn.Sequential(
            nn.Conv1d(input_dim, shallow_dim, 1), nn.BatchNorm1d(shallow_dim)
        )

        self.base_layer = nn.Sequential(
            nn.Conv1d(shallow_dim, mid_dim, 1),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Conv1d(mid_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
        )

        self.global_feat = global_feat
        self.out_dim = out_dim

    def forward(self, x):
 
        x= x.squeeze(1)
        n_pts = x.size()[2]
        x = self.shallow_layer(x)
        pointfeat = x

        x = self.base_layer(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)

        trans_feat = None
        trans = None
        x = x.unsqueeze(1)
        if self.global_feat:
            return x#, trans, trans_feat


class PointNetfeat2(nn.Module):
    def __init__(self, input_dim, shallow_dim, mid_dim, out_dim, global_feat=False):
        super(PointNetfeat2, self).__init__()
        # self.shallow_layer = nn.Sequential(
        #     nn.Conv1d(input_dim, shallow_dim, 1), nn.BatchNorm1d(shallow_dim)
        # )

        self.base_layer = nn.Sequential(
            nn.Conv1d(input_dim, shallow_dim, 1),
            nn.BatchNorm1d(shallow_dim),
            nn.ReLU(),
            nn.Conv1d(shallow_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
        )

        self.global_feat = global_feat
        self.out_dim = out_dim

    def forward(self, x):
 
        x= x.squeeze(1)
        n_pts = x.size()[2]
        # x = self.shallow_layer(x)
        # pointfeat = x

        x = self.base_layer(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)

        trans_feat = None
        trans = None
        x = x.unsqueeze(1)
        if self.global_feat:
            return x#, trans, trans_feat
        # else:
        #     x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat
