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

from poses.pose_utils import proj_geo
from poses.pose_utils import get_rott_geo_global
from poses.pose_utils import get_rott_geo
#from poses.pose_utils import multi_proj

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
        """ #rotation and Translation vector init
        RotTranVec=torch.zeros(N_imgs,6)
        RotTranVec[:, 0] = 0.1
        RotTranVec = nn.Parameter(RotTranVec.float().to(self.device))
        #focal init
        focal = torch.Tensor([self.image_size])
        #print("check focal:", focal)
        focal = nn.Parameter(focal.float().to(self.device)) """
        #euler representation
        euler = nn.Parameter(torch.zeros(N_imgs, 3).float().to(self.device))
        trans = nn.Parameter(torch.zeros(N_imgs, 3).float().to(self.device))
        scale = torch.zeros(1, 1)
        scale[0, 0] = 0.09 
        scale = nn.Parameter(scale.float().to(self.device))
        
        #e_opt = torch.optim.Adam([shape, exp, pose, cam, tex, lights], lr=self.config.e_lr, betas=(0.9, 0.999))
        e_opt0 = torch.optim.Adam([scale], lr=self.config.e_lr*.01)
        e_opt1 = torch.optim.Adam([euler, trans], lr=self.config.e_lr)
        e_opt2 = torch.optim.Adam([shape, exp, pose], lr=self.config.e_lr)
        e_opt3 = torch.optim.Adam([tex, lights], lr=self.config.e_lr)
        
        

        gt_landmark = landmarks

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        #for k in range(1000):
        for k in range(3000):
            cam_param = self.cam_params
            """ #camera matrix interMat/exterMat init
            interMat = torch.Tensor([[focal, 0, self.image_size/2],
                                     [0, focal, self.image_size/2],
                                     [0, 0, 1]])#[3,3]
            exterMat = CamExsInit(RotTranVec)#N*3*4 """

            losses = {}
            _, landmarks3d = self.flame.forward_dynamic(
                shape_params=shape, expression_params=exp, pose_params=pose)    
            landmarks3d = landmarks3d.repeat(N_imgs,1,1)
            landmarks3d = scale.unsqueeze(0)*landmarks3d
            landmarks2d = proj_geo(landmarks3d, cam_param, euler, trans, Rts)
            landmarks2d = landmarks2d / self.image_size*2-1

            
            """ #print("cakics:",landmarks3d[0,0])
            landmarks3d_temp = landmarks3d.view(68, 3)
            #print("cakics:",landmarks3d_temp[0])
            landmarks3d_temp = torch.transpose(landmarks3d_temp, 0, 1)#[3,68]
            bottom = torch.ones(1, 68)
            landmarks3d_temp = torch.cat((landmarks3d_temp, bottom), 0)#[4,68]
            landmarks2d_es = torch.zeros(N_imgs, 68 ,2)
            for i in range(N_imgs):
                landmarks3d_pro = torch.mm(interMat, torch.mm(exterMat[i], landmarks3d_temp))#[3,68]
                landmarks3d_pro_z = landmarks3d_pro[2]
                landmarks3d_pro = landmarks3d_pro[:2, :] 
                landmarks3d_pro = torch.div(landmarks3d_pro, landmarks3d_pro_z)#[2,68]
                landmarks3d_pro = torch.transpose(landmarks3d_pro, 0, 1)#[68,2]
                landmarks3d_pro[:, 0] = landmarks3d_pro[:, 0] / float(self.image_size) * 2 - 1
                landmarks3d_pro[:, 1] = landmarks3d_pro[:, 1] / float(self.image_size) * 2 - 1
                landmarks2d_es[i] = landmarks3d_pro 
            landmarks2d_es = fiting_util.batch_orth_proj(landmarks2d_es, cam)
            landmarks2d_es[..., 1:] = - landmarks2d_es[..., 1:] """
            """ #landmarks2d_es = torch.zeros(N_imgs, 3, 68)
            landmarks2d_es = multi_proj(landmarks3d_temp, interMat, exterMat)
            #landmarks2d_es = torch.transpose(landmarks2d_es, 1, 2)
            landmarks2d_es[:,:,:2] = landmarks2d_es[:,:,:2] / float(self.image_size) * 2 - 1 """
    
            losses['landmark'] = util.l2_distance(
                landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2])* self.config.w_lmks
            """ losses['landmark'] = fiting_util.l2_distance(landmarks2d_es[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            if k==1:
                input=images[9].float().cpu().numpy().transpose((1,2,0))*255.
                cv2.imwrite('./test_results/TestFace_origin/landgt.png',input.astype(np.float32))
            if k==999:
                input=images[9].float().cpu().numpy().transpose((1,2,0))*255.
                landtest=landmarks2d_es[9].float().cpu()
                landtest=(landtest+1.)/2*float(self.image_size)
                print(input.shape)
                print("lhsvf",landtest[30])
                cv2.circle(input, (112,112), 20, (0,255,0), 3, cv2.LINE_AA)
                #for j in range(68):
                 #   cv2.circle(input, (int(landtest[j,0]),int(landtest[j,1])), 8, (0,255,0), 3 , cv2.LINE_AA)
                cv2.imwrite('./test_results/TestFace_origin/land.png',input.astype(np.float32)) """
            
        
            
                


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt0.zero_grad()
            e_opt1.zero_grad()
            all_loss.backward()
            e_opt0.step()
            e_opt1.step()

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
        #for k in range(1000,2000):
        for k in range(3000, 8000):
            tex_param = tex.expand(N_imgs, -1)
            cam_param = self.cam_params
            """ #camera matrix interMat/exterMat init
            interMat = torch.Tensor([[focal, 0, self.image_size/2],
                                     [0, focal, self.image_size/2],
                                     [0, 0, 1]])
            exterMat = CamExsInit(RotTranVec) """

            losses = {}
            vertices, landmarks3d = self.flame.forward_dynamic(
                shape_params=shape, expression_params=exp, pose_params=pose)
            landmarks3d = landmarks3d.repeat(N_imgs,1,1)
            vertices = vertices.repeat(N_imgs,1,1)

            
            """ landmarks3d_temp = landmarks3d.view(68, 3)
            landmarks3d_temp = torch.transpose(landmarks3d_temp, 0, 1)#[3,68]
            bottom = torch.ones(1, 68)
            landmarks3d_temp = torch.cat((landmarks3d_temp, bottom), 0)#[4,68]
            landmarks2d_es = torch.zeros(N_imgs, 68 ,2)
            for i in range(N_imgs):
                landmarks3d_pro = torch.mm(interMat, torch.mm(exterMat[i], landmarks3d_temp))#[3,68]
                landmarks3d_pro_z = landmarks3d_pro[2]
                landmarks3d_pro = landmarks3d_pro[:2, :] 
                landmarks3d_pro = torch.div(landmarks3d_pro, landmarks3d_pro_z)#[2,68]
                landmarks3d_pro = torch.transpose(landmarks3d_pro, 0, 1)#[68,2]
                landmarks3d_pro[:, 0] = landmarks3d_pro[:, 0] / float(self.image_size) * 2 - 1
                landmarks3d_pro[:, 1] = landmarks3d_pro[:, 1] / float(self.image_size) * 2 - 1
                landmarks2d_es[i] = landmarks3d_pro
            landmarks2d_es = fiting_util.batch_orth_proj(landmarks2d_es, cam)
            landmarks2d_es[..., 1:] = - landmarks2d_es[..., 1:] """
            landmarks3d = scale.unsqueeze(0)*landmarks3d
            vertices = scale.unsqueeze(0)*vertices

            landmarks2d = proj_geo(landmarks3d, cam_param, euler, trans, Rts)
            landmarks2d = landmarks2d / self.image_size*2-1
            rott_vertices = get_rott_geo_global(vertices, euler, trans, Rts)
            albedo = self.flametex(tex_param, uv_coords)

            losses['landmark'] = util.l2_distance(
                landmarks2d[:, :, :2], gt_landmark[:, :, :2])* self.config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * self.config.w_pose_reg
            
            ## render
            rott_for_render = rott_vertices[render_ids].clone()
            rott_for_render[:, :, [0, 1]] *= -1
            render_imgs = renderer(rott_for_render, albedo[render_ids], lights[render_ids])
            render_imgs = render_imgs / 255.
            render_mask = render_imgs[:, 3:4, :, :]
            render_imgs = render_imgs[:, :3, :, :]
            """ albedos = self.flametex(tex) / 255.
            vertices_multi = torch.zeros((N_imgs,vertices.shape[1],vertices.shape[2]))
            trans_vertices_multi = torch.zeros((N_imgs,vertices.shape[1],vertices.shape[2]))
            for i in range(N_imgs):
                vertices_temp = vertices[0]#[:,3]
                vertices_temp = torch.transpose(vertices_temp, 0, 1)
                bottom = torch.ones(1, vertices_temp.shape[1])
                vertices_temp = torch.cat((vertices_temp, bottom), 0)#[4,:]
                vertices_temp = torch.mm(exterMat[i], vertices_temp)#[3:,]
                vertices_multi[i] = torch.transpose(vertices_temp,0,1)
                vertices_temp = torch.unsqueeze(torch.transpose(vertices_temp,0,1), 0)#[1,:,3]
                trans_vertices_multi[i] = fiting_util.batch_orth_proj(vertices_temp, cam)[0]
                trans_vertices_multi[i, :, 1:] = - trans_vertices_multi[i, :, 1:]
            ops = self.render(vertices_multi, trans_vertices_multi, albedos, lights)
            predicted_images = ops['images']
            print("achsvbck:",lights[1,5])
            #print("achsvbck:", predicted_images.shape)
            predicted_image = predicted_images[9].detach().float().cpu().numpy().transpose((1,2,0))*255.
            if k % 500 ==0:
                cv2.imwrite('./test_results/TestFace_origin/{}.jpg'.format(k),predicted_image.astype(np.float32)) """
            losses['photometric_texture'] = ((
                image_masks[render_ids]*images[render_ids] - (render_imgs[:, :3, :, :])).abs()).mean() * self.config.w_pho
            #losses['mask'] = (render_mask - image_masks[render_ids]).abs().mean()*10
            

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt0.zero_grad()
            e_opt1.zero_grad()
            e_opt2.zero_grad()
            e_opt3.zero_grad()
            all_loss.backward()
            e_opt1.step()
            e_opt0.step()
            e_opt2.step()
            e_opt3.step()


    

            loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

            if k % 10 == 0:
                print(loss_info)
            
            #for quick test
            """ if k % 1000 == 0 or k == 8000-1:
                predicted_image = render_imgs[1].detach().float().cpu().numpy().transpose((1,2,0))
                predicted_image = predicted_image[:,:,:3]
                cv2.imwrite('./test_results/2-346_2/{}.jpg'.format(k),predicted_image.astype(np.float32)) """

            
            # visualize
            if k % 1000 == 0 or k == 8000-1:
                grids = {}
                visind = range(N_imgs)
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['render'] = torchvision.utils.make_grid(render_imgs[visind].detach().float().cpu())
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                #grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
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
            #'RotTranVec': RotTranVec.detach().cpu().numpy(),
            #'focal': focal.detach().cpu().numpy()
            'euler': euler.detach().cpu().numpy(),
            'trans': trans.detach().cpu().numpy(),
            'scale': scale.detach().cpu().numpy()
        }
        return single_params
    
    

    def run(self, images, focal, scenepath, landmarkspath, maskpath):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we try showing the example with multiple images fitting
        N_imgs = images.shape[0]
        input_size = images.shape[2]#square, =height=width
        focal = focal * self.image_size / input_size
        image_masks = []
        Rts = []
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
        #mask
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