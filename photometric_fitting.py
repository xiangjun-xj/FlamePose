import numpy as np
import torch
import os, sys
import math
import imageio
import datetime
import json
import random
import time
import cv2
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from glob import glob

from poses.pose_utils import proj_geo_AA
from poses.pose_utils import get_rott_geo_AA

sys.path.append('./models/')
from FLAME import FLAME, FLAMETex
from render import Render
import util

class PhotometricFitting(object):
    def __init__(self, config, device):#config equals to args
        self.image_size = config.image_size
        self.config = config
        self.device = device
        #
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)



    def optimize(self, images, landmarks, image_masks, savefolder, Rts, uv_coords):
        bz = 1#one flame model
        N_imgs = images.shape[0]
        render_ids = np.arange(0, N_imgs, 1)
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        lights = nn.Parameter(torch.zeros(N_imgs, 9, 3).float().to(self.device))

        #axis and angle representation
        rotVec = torch.zeros(N_imgs, 3)
        rotVec[:,0] = 0.1
        rotVec = nn.Parameter(rotVec.float().to(self.device))
        trans = nn.Parameter(torch.zeros(N_imgs, 3).float().to(self.device))
        scale = torch.zeros(1, 1)
        scale[0, 0] = 0.09 
        scale = nn.Parameter(scale.float().to(self.device))

        
        e_opt0 = torch.optim.Adam([scale], lr=self.config.e_lr*.01)
        e_opt1_AA = torch.optim.Adam([rotVec, trans], lr=self.config.e_lr)
        e_opt2 = torch.optim.Adam([shape, exp, pose], lr=self.config.e_lr)
        e_opt3 = torch.optim.Adam([tex, lights], lr=self.config.e_lr)
        
        

        gt_landmark = landmarks

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(3000):
            cam_param = self.cam_params

            losses = {}
            _, landmarks3d = self.flame.forward_dynamic(
                shape_params=shape, expression_params=exp, pose_params=pose)    
            landmarks3d = landmarks3d.repeat(N_imgs,1,1)
            landmarks3d = scale.unsqueeze(0)*landmarks3d
            landmarks2d = proj_geo_AA(landmarks3d, cam_param, rotVec, trans, Rts)
            landmarks2d = landmarks2d / self.image_size*2-1
    
            losses['landmark'] = util.l2_distance(
                landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2])* self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt0.zero_grad()
            e_opt1_AA.zero_grad()
            all_loss.backward()
            e_opt0.step()
            e_opt1_AA.step()

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
            if k % 10 == 0:
                print(loss_info)

        
        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        renderer = Render(fx=self.cam_params[0,0], fy=self.cam_params[0,1], img_w=self.image_size, img_h=self.image_size, 
                          cx=self.cam_params[0,2], cy=self.cam_params[0,3], batch_size=render_ids.shape[0], 
                          device=self.cam_params.device, focal_length = self.cam_params[:, :2], 
                          pri_point = self.cam_params[:, 2:])#.to(self.device)
        for k in range(3000, 5001):
            tex_param = tex.expand(N_imgs, -1)
            cam_param = self.cam_params

            losses = {}
            vertices, landmarks3d = self.flame.forward_dynamic(
                shape_params=shape, expression_params=exp, pose_params=pose)
            landmarks3d = landmarks3d.repeat(N_imgs,1,1)
            vertices = vertices.repeat(N_imgs,1,1)
            landmarks3d = scale.unsqueeze(0)*landmarks3d
            vertices = scale.unsqueeze(0)*vertices
            landmarks2d = proj_geo_AA(landmarks3d, cam_param, rotVec, trans, Rts)
            landmarks2d = landmarks2d / self.image_size*2-1
            rott_vertices = get_rott_geo_AA(vertices, rotVec, trans)
            albedo = self.flametex(tex_param, uv_coords)

            losses['landmark'] = util.l2_distance(
                landmarks2d[:, :, :2], gt_landmark[:, :, :2])* self.config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg    # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * self.config.w_pose_reg       # *1e-4
            
            ## render
            rott_for_render = rott_vertices[render_ids].clone()
            rott_for_render[:, :, [0, 1]] *= -1
            render_imgs = renderer(rott_for_render, albedo[render_ids], lights[render_ids])
            render_imgs = render_imgs / 255.
            render_mask = render_imgs[:, 3:4, :, :]
            render_imgs = render_imgs[:, :3, :, :]
            losses['photometric_texture'] = ((
                image_masks[render_ids]*images[render_ids] - (render_imgs[:, :3, :, :])).abs()).mean() * self.config.w_pho
            #losses['mask'] = (render_mask - image_masks[render_ids]).abs().mean()*10
            

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt0.zero_grad()
            e_opt1_AA.zero_grad()
            e_opt2.zero_grad()
            e_opt3.zero_grad()
            all_loss.backward()
            e_opt1_AA.step()
            e_opt0.step()
            e_opt2.step()
            e_opt3.step()


    

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

            if k % 10 == 0:
                print(loss_info)

            
            # visualize
            if k % 1000 == 0:
                grids = {}
                visind = range(N_imgs)
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['render'] = torchvision.utils.make_grid(render_imgs[visind].detach().float().cpu())
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [0, 1, 2]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)
            
        
        single_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'verts': rott_for_render.detach().cpu().numpy(),
            'albedos':albedo.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lit': lights.detach().cpu().numpy(),
            'rotVec': rotVec.detach().cpu().numpy(),
            'trans': trans.detach().cpu().numpy(),
            'scale': scale.detach().cpu().numpy()
        }
        return single_params
    
    

    def run(self, images, focal, scenepath, landmarkspath, maskpath):
        # The implementation is able to optimize with images(batch_size>1),
        N_imgs = images.shape[0]
        input_size = images.shape[2]#square, =height=width
        focal = focal * self.image_size / input_size
        image_masks = []
        Rts = []#extra Rotation and transformation
        cam_paras = []
        for i in range(N_imgs):
            Rt = np.eye(4)
            Rts.append(torch.as_tensor(Rt).unsqueeze(0).float().cuda())
            cam_para = np.zeros(4, dtype=np.float32)
            cam_para[0] = focal
            cam_para[1] = focal
            cam_para[2] = self.image_size / 2.0
            cam_para[3] = self.image_size / 2.0
            cam_paras.append(torch.as_tensor(cam_para).unsqueeze(0).float().cuda())
        Rts = torch.cat(Rts, dim=0)
        self.cam_params = torch.cat(cam_paras, dim=0)
        uv_coords = np.loadtxt('./flame-data/uv_coords.txt')
        uv_coords = torch.from_numpy(uv_coords).float().cuda()

        scene_name = os.path.basename(scenepath)
        savefile = os.path.sep.join([self.config.savefolder, scene_name + '.npy'])

        # photometric optimization is sensitive to the hair or glass occlusions,
        # therefore we use a face segmentation network to mask the skin region out.
        #mask landmark image
        for single_mask_path in sorted(os.listdir(maskpath)):
            image_mask_path = os.path.join(maskpath, single_mask_path)
            image_mask = np.load(image_mask_path, allow_pickle=True)
            image_mask_bn = np.zeros_like(image_mask)
            image_mask_bn[np.where(image_mask != 0)] = 1.
            image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]))
        
        landmarks = np.load(landmarkspath).astype(np.float32)
        landmarks = landmarks/input_size * 2 - 1
        landmarks = torch.from_numpy(landmarks)
        landmarks = landmarks.to(self.device)
        images = F.interpolate(images, [self.image_size, self.image_size])
        image_masks = torch.cat(image_masks, dim=0).to(self.device)
        image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])
        

        savefolder = os.path.sep.join([self.config.savefolder, scene_name])

        util.check_mkdir(savefolder)
        # optimize
        single_params = self.optimize(images, landmarks, image_masks, savefolder, Rts, uv_coords)
        np.save(savefile, single_params)
        return single_params