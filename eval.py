import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os
from pathlib import Path
import numpy as np
import os.path as op
import pickle 

import torch
from mano.mano_models import build_mano_aa

from torch.utils import data
from ipdb import set_trace as st
import trimesh
from models.semantic_conditional_moudle import get_moudle as  moudle_1
from models.contact_conditional_moudle import get_moudle as moudle_2
from data.grab_test import GrabDataset
 

def test(opt, device):
    diffusion_moudle_1 = moudle_1(opt)
    diffusion_moudle_2 = moudle_2(opt)
    idxs = 0

    weight_root_folder_1 = "/data-home/egoego_release-main/diffusion_ARCTIC_runs_new/v6_stage1_grab_2048x5/all_small_bs32/weights"
    diffusion_weight_path_1 = os.path.join(weight_root_folder_1, "model-1.pt")

    weight_root_folder_2 = "/data-home/egoego_release-main/diffusion_ARCTIC_runs_new/v6_stage2_grab_2048x5/all_small_bs64/weights"
    diffusion_weight_path_2 = os.path.join(weight_root_folder_2, "model-100.pt")

    diffusion_moudle_1.load_weight_path(diffusion_weight_path_1)
    diffusion_moudle_2.load_weight_path(diffusion_weight_path_2)

    val_dataset = GrabDataset()
    val_dl = data.DataLoader(val_dataset, batch_size=3000, shuffle=False, pin_memory=True, num_workers=0)

    new_item = {}
    with torch.no_grad():
       for i, item in enumerate(val_dl):
            test_input_data_dict= item  #item["motion"].shape torch.Size([32, 2531])
            one_motion = test_input_data_dict['motion']

            res_list_1 = diffusion_moudle_1.full_body_gen_cond_head_pose_sliding_window(one_motion.to(device)) 
            one_motion[:,:,6205:8253] = res_list_1
            all_res_list = diffusion_moudle_2.full_body_gen_cond_head_pose_sliding_window(one_motion.to(device)) 

            preds_new = my_process_data(preds=all_res_list,namelist=test_input_data_dict['name'])
            with open("assets/closed_mano_faces.pkl", 'rb') as f:
                hand_face = pickle.load(f)

            hand_verts = preds_new["manov3d.r"]
            exp_name = 'demo'
            save_dir = f'exp/{exp_name}'

            aa_name = test_input_data_dict['name']
            for i in range(len(hand_verts)):
                hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_face)
                parts = aa_name[i].split('/')  
                relevant_parts = [parts[2]] + parts[3].split('_') + [parts[-1].split('.')[0]]
                formatted_string = '_'.join(relevant_parts)
                formatted_string = exp_name+'_'+ formatted_string
                # st()
                hand_mesh.export(os.path.join(save_dir, f'{formatted_string}.obj'.format(i)))
                
def my_process_data(preds,namelist):
    models = {'mano_r':build_mano_aa(is_rhand=True,flat_hand=True)}

    targets=dict()
    for i in range(len(namelist)):

        rot_r = preds[i][0][:3]
        pose_r = preds[i][0][3:48]
        trans_r = preds[i][0][48:51]
        betas_r = preds[i][0][51:61]
        
        pose_r = np.concatenate((rot_r.to('cpu'), pose_r.cpu()), axis=0)
        
        if i == 0:
            targets["mano.pose.r"] = torch.from_numpy(pose_r).float().unsqueeze(0).to('cpu')
            targets["mano.beta.r"] = np.expand_dims(betas_r.cpu().numpy(), axis=0)
            targets["mano.trans.r"] = np.expand_dims(trans_r.cpu().numpy(), axis=0)                
        else:
            targets["mano.pose.r"] = torch.cat([targets["mano.pose.r"],torch.from_numpy(pose_r).float().unsqueeze(0).to('cpu')],dim=0)
            targets["mano.beta.r"] = np.concatenate([targets["mano.beta.r"],np.expand_dims(betas_r.cpu().numpy(), axis=0)],axis=0)
            targets["mano.trans.r"] = np.concatenate([targets["mano.trans.r"],np.expand_dims(trans_r.cpu().numpy(), axis=0)],axis=0)
    # st()
            
    gt_pose_r = targets["mano.pose.r"]
    gt_betas_r = targets["mano.beta.r"]
    gt_trans_r = targets["mano.trans.r"]

    temp_gt_out_r = models["mano_r"](
        betas=torch.from_numpy(gt_betas_r),
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=torch.from_numpy(gt_trans_r),
    )
    targets["manoj21.r"] = temp_gt_out_r.joints
    targets["manov3d.r"] = temp_gt_out_r.vertices

    return targets

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=0, help='the number of workers for data loading')
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--weight', default='latest')
    parser.add_argument("--gen_vis", action="store_true")
    # For AvatarPoser config 
    parser.add_argument('--kinpoly_cfg', type=str, default="", help='Path to option JSON file.')
    # Diffusion model settings
    parser.add_argument('--diffusion_window', type=int, default=1, help='horizon')
    parser.add_argument('--diffusion_batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--diffusion_learning_rate', type=float, default=1e-5, help='generator_learning_rate')

    parser.add_argument('--diffusion_n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--diffusion_n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--diffusion_d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--diffusion_d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--diffusion_d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    parser.add_argument('--diffusion_project', default='runs/test', help='project/name')
    parser.add_argument('--diffusion_exp_name', default='', help='save to project/name')

    # For data representation
    parser.add_argument("--canonicalize_init_head", action="store_true")
    parser.add_argument("--use_min_max", action="store_true")

    parser.add_argument('--data_root_folder', default='', help='')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.diffusion_save_dir = str(Path(opt.diffusion_project) / opt.diffusion_exp_name)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    test(opt, device)