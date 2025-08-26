import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_g1_23_humanoid_batch import Humanoid_G1_23_Batch
from torch.autograd import Variable
from tqdm import tqdm
import argparse

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/home/agiuser/vmp/human2humanoid/data/AMASS/AMASS_Complete")
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    g1_23_rotation_axis = torch.tensor([[
        [0, 1, 0], # l_hip_pitch
        [1, 0, 0], # l_hip_roll
        [0, 0, 1], # l_hip_yaw
        
        [0, 1, 0], # l_kneel_pitch
        [0, 1, 0], # l_ankle_pitch
        [1, 0, 0], # l_ankle_roll
            
        [0, 1, 0], # r_hip_pitch
        [1, 0, 0], # r_hip_roll
        [0, 0, 1], # r_hip_yaw
        
        [0, 1, 0], # r_kneel_pitch
        [0, 1, 0], # r_ankle_pitch
        [1, 0, 0], # r_ankle_roll
        
        [0, 0, 1], # torso

        [0, 1, 0], # l_shoulder_pitch    
        [1, 0, 0], # l_shoulder_roll
        [0, 0,1], # l_shoulder_yaw
        
        [0, 1, 0], # l_elbow_pitch
        
        [1, 0, 0], # l_wrist_roll
        
        [0, 1, 0], # r_shoulder_pitch    
        [1, 0, 0], # r_shoulder_roll
        [0, 0, 1], # r_shoulder_yaw
        
        [0, 1, 0], # r_elbow_pitch
        
        [1, 0, 0], # r_wrist_roll
    ]]).to(device)

    g1_23_joint_names = [ 'pelvis', 
                    'left_hip_pitch_link', 'left_hip_roll_link','left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link','left_ankle_roll_link',
                    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link','right_ankle_roll_link',
                    'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_rubber_hand',
                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link','right_wrist_roll_rubber_hand']


    g1_23_joint_names_augment = g1_23_joint_names + ["head_link"]
    g1_23_joint_pick = ['pelvis', "left_knee_link", "left_ankle_pitch_link",  'right_knee_link', 'right_ankle_pitch_link', "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_rubber_hand", "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_rubber_hand"]
    smpl_joint_pick = ["Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
    g1_23_joint_pick_idx = [ g1_23_joint_names_augment.index(j) for j in g1_23_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)


    shape_new, scale = joblib.load("data/g1_23/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)


    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    
    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    g1_23_fk = Humanoid_G1_23_Batch(extend_hand = True, extend_head=True, device = device)
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)


        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        pose_aa_g1_23 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 24, axis = 2), N, axis = 1)
        pose_aa_g1_23[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_g1_23 = torch.from_numpy(pose_aa_g1_23).float().to(device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

        dof_pos = torch.zeros((1, N, 23, 1)).to(device)

        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)

        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_g1_23_new = torch.cat([gt_root_rot[None, :, None], g1_23_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            fk_return = g1_23_fk.fk_batch(pose_aa_g1_23_new, root_trans_offset[None, ])
            
            diff = fk_return['global_translation_extend'][:, :, g1_23_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim = -1).mean() 
            loss = loss_g
            
            
            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            
            dof_pos_new.data.clamp_(g1_23_fk.joints_range[:, 0, None], g1_23_fk.joints_range[:, 1, None])
            
        dof_pos_new.data.clamp_(g1_23_fk.joints_range[:, 0, None], g1_23_fk.joints_range[:, 1, None])
        pose_aa_g1_23_new = torch.cat([gt_root_rot[None, :, None], g1_23_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
        fk_return = g1_23_fk.fk_batch(pose_aa_g1_23_new, root_trans_offset[None, ])

        root_trans_offset_dump = root_trans_offset.clone()

        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
        
        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_g1_23_new.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "fps": 30
                }
        
        # print(f"dumping {data_key} for testing, remove the line if you want to process all data")
        # import ipdb; ipdb.set_trace()
        joblib.dump(data_dump, "data/g1_23/test.pkl")
    
        
    # import ipdb; ipdb.set_trace()
    joblib.dump(data_dump, "data/g1_23/amass_all.pkl")