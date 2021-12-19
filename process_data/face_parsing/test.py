#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp

from PIL import Image
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import configargparse


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg', save_mask_path='vis_results/parsing_map_on_im.npy',
                     img_size=(512, 512)):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])  # + 255[B,G,R]
    vis_parsing_anno_mask = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    

    num_of_class = np.max(vis_parsing_anno)
    for pi in range(1, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])
        vis_parsing_anno_mask[index[0], index[1], :] = np.array([1,1,1])

    for pi in range(14, 16):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 255, 0])
    for pi in range(16, 17):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 255])
    for pi in range(17, num_of_class+1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])
        vis_parsing_anno_mask[index[0], index[1], :] = np.array([1,1,1])

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    index = np.where(vis_parsing_anno == num_of_class-1)
    vis_im = cv2.resize(vis_parsing_anno_color, img_size,
                        interpolation=cv2.INTER_NEAREST)
    vis_mask = cv2.resize(vis_parsing_anno_mask, img_size,
                        interpolation=cv2.INTER_NEAREST)                   
    """ mask = cv2.resize(vis_parsing_anno_mask, img_size,
                        interpolation=cv2.INTER_NEAREST) """
    if save_im:
        cv2.imwrite(save_path, vis_im)
    vis_mask = vis_mask.transpose(2,0,1)
    mask = vis_mask[0].reshape(1,img_size[1],img_size[0])
    print(mask.shape)
    np.save(save_mask_path, mask)

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    Path(respth).mkdir(parents=True, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    processed_num = 0
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                img = Image.open(osp.join(dspth, image_path))
                ori_size = img.size
                image = img.resize((512, 512), Image.BILINEAR)
                image = image.convert("RGB")
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                out = net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                image_path = str(image_path[:-4])
                mask_path = str(image_path) + '.npy'
                image_path = str(image_path) + '.png'
            

                vis_parsing_maps(image, parsing, stride=1, save_im=False,
                                 save_path=osp.join(respth, image_path), save_mask_path=os.path.join(respth, mask_path), img_size=ori_size)
                processed_num = processed_num + 1
                if processed_num % 100 == 0:
                    print('processed parsing', processed_num)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--id', type=str,
                        default='2-346', help='scene name')
    parser.add_argument('--modelpath', type=str,
                        default='./process_data/face_parsing/79999_iter.pth')
    args = parser.parse_args()
    image_base_dir = './data'
    mask_base_dir = './data_mask'
    imgpath = os.path.join(image_base_dir, args.id)
    imgpath = os.path.join(imgpath, 'images_pad')
    respth = os.path.join(mask_base_dir, args.id)
    if not os.path.exists(respth):
        os.makedirs(respth)
    evaluate(respth=respth, dspth=imgpath, cp=args.modelpath)
