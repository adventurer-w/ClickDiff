# 全数据集 heatmap到mano

import sys
import argparse
import os
from pathlib import Path
import yaml

import wandb
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import pytorch3d.transforms as transforms 
from ema_pytorch import EMA


from models.pointnet import PointNetfeat2 #as PointNetfeat #as pointnet
import torch.nn.functional as F
from ipdb import set_trace as st
import torch.nn as nn
from data.grab_test import GrabDataset
from models.cond_diffusion_model import CondGaussianDiffusion

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        pointnet_model,
        *,
        ema_decay = 0.995, 
        train_batch_size = 20480, # bs=1
        train_lr = 1e-5,
        train_num_steps = 500000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        ema_update_every = 10, 
        save_and_sample_every = 1000,
        results_folder = './results',
        use_wandb=True,
        run_demo=False,
    ):
        super().__init__()
        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.pointnet = pointnet_model
        #self.pointnet = PointNetfeat(3,64,64,1024,global_feat=True)
        self.ema2 = EMA(pointnet_model, beta=ema_decay, update_every=ema_update_every)
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        
        self.optimizer = Adam([{'params': pointnet_model.parameters(), 'lr': 1e-4},
	    {'params': diffusion_model.parameters(), 'lr': train_lr}])
        #self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 
        # self.mse_loss = nn.MSELoss(reduction="none")
        self.ds = GrabDataset()

        self.window = opt.window 
        self.mse_loss = nn.MSELoss(reduction="none")

    def prep_dataloader(self, args,seq=None):
        # Define dataset
        if seq is not None:
            split = args.run_on
        train_dataset = GrabDataset()
        val_dataset = GrabDataset()

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'pointnet': self.pointnet.state_dict(),
            'ema': self.ema.state_dict(),
            'ema2': self.ema2.state_dict(),
            'scaler': self.scaler.state_dict() # 梯度缩放器的状态字典
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        print("save sucess")

    def load(self, milestone):
        data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.pointnet.load_state_dict(data['pointnet'], strict=False)
        self.ema2.load_state_dict(data['ema2'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def load_weight_path(self, weight_path):
        data = torch.load(weight_path)
        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.pointnet.load_state_dict(data['pointnet'], strict=False)
        self.ema2.load_state_dict(data['ema2'], strict=False)
        self.scaler.load_state_dict(data['scaler'])
        self.scaler.load_state_dict(data['scaler'])

    def prep_head_condition_mask(self, data, joint_idx=15):
        mask = torch.ones_like(data).to(data.device)
        mask[:,:,61:] = torch.zeros(data.shape[0], data.shape[1], 3072).to(data.device)
        # st()
        return mask 

    def full_body_gen_cond_head_pose_sliding_window(self, input_data):
        self.ema.ema_model.eval()
        self.ema2.ema_model.eval()

        with torch.no_grad():
            obj = input_data[:,:,61:6205].view([input_data.shape[0],input_data.shape[1],2048,3]).cuda()
            feature_in = obj.transpose(3, 2).cuda()

            val_feature= self.ema2.ema_model(feature_in)  
            val_data_in = torch.cat([input_data[:,:,:61],val_feature.cuda(),input_data[:,:,6205:8253]],dim=-1).cuda()
            
            mano = input_data[:,:,:61]
            denoise_data = torch.zeros(mano.shape[0], mano.shape[1], mano.shape[2]).to(input_data.device) 
            
            data = torch.zeros(val_data_in.shape[0], val_data_in.shape[1], val_data_in.shape[2]).to(input_data.device) 
            val_data_in = val_data_in[:,:,61:]
            cond_mask = self.prep_head_condition_mask(data) 
            
            pred_x = self.ema.ema_model.sample_sliding_window_w_canonical(self.ds, x_denoise=denoise_data,\
            obj=val_data_in, x_start=data, cond_mask=cond_mask) 
    
        return pred_x 
    
    
def get_moudle(opt, run_demo=False):
    opt.window = opt.diffusion_window 

    opt.diffusion_save_dir = os.path.join(opt.diffusion_project, opt.diffusion_exp_name)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Directories
    save_dir = Path(opt.diffusion_save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    repr_dim = 61
    repr_dim_cond=3133
   
    transformer_diffusion = CondGaussianDiffusion(d_feats=repr_dim, d_condfeat=repr_dim_cond, d_model=opt.diffusion_d_model, \
                n_dec_layers=opt.diffusion_n_dec_layers, n_head=opt.diffusion_n_head, \
                d_k=opt.diffusion_d_k, d_v=opt.diffusion_d_v, \
                max_timesteps=opt.diffusion_window+1, out_dim=repr_dim, timesteps=1000, objective="pred_x0", \
                batch_size=opt.diffusion_batch_size)
    
    transformer_diffusion.to(device)
    pointnet_model = PointNetfeat2(3,64,64,1024,global_feat=True)
    #pointnet_model = pointnet(global_feat=True, feature_transform=True, channel=3)
    pointnet_model.to(device)
    
    
    trainer = Trainer(
        opt,
        transformer_diffusion,
        pointnet_model,
        train_batch_size=opt.diffusion_batch_size, # 32
        train_lr=opt.diffusion_learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False,
        run_demo=run_demo,
    )

    return trainer 