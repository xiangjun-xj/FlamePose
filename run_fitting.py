import numpy as np
import torch
import os, sys
import json
import random
from glob import glob
import util

from poses.pose_utils import load_colmap_pose
from poses.pose_utils import load_image
from poses.colmap_wrapper import run_colmap
from photometric_fitting import PhotometricFitting

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
    

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    """ parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs') """
    parser.add_argument("--scenedir", type=str, default='./data/2-346', 
                        help='input scene directory')
    parser.add_argument("--landmarkdir", type=str, default='./data_landmark/', 
                        help='input landmark directory')
    parser.add_argument("--maskdir", type=str, default='./data_mask/', 
                        help='input mask directory')
    parser.add_argument('--match_type', type=str, 
					default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    parser.add_argument('--factor', type=int, default=1,
                        help='downsample factor for images')

    #start config of PhotometricFitting
    parser.add_argument('--image_size', type=int, default=224,
                        help='the final size of square image, width==height')
    parser.add_argument("--flame_model_path", type=str, default='./flame-data/generic_model.pkl', # acquire it from FLAME project page
                        help='flame model path')
    parser.add_argument("--flame_lmk_embedding_path", type=str, default='./flame-data/landmark_embedding.npy', 
                        help='flame lmk_embedding path')
    parser.add_argument("--tex_space_path", type=str, default='./flame-data/FLAME_texture.npz', # acquire it from FLAME project page
                        help='tex_space_path')
    parser.add_argument('--shape_params', type=int, default=100,
                        help='shape params in flame')
    parser.add_argument('--expression_params', type=int, default=100,
                        help='expression params in flame')
    parser.add_argument('--pose_params', type=int, default=6,
                        help='pose params in flame')
    parser.add_argument('--tex_params', type=int, default=50,
                        help='tex params in flame')
    """ parser.add_argument("--use_face_contour", action='store_true', 
                        help='use_face_contour') """
    parser.add_argument("--e_lr", type=float, default=0.005, 
                        help='learning rate')
    parser.add_argument("--e_wd", type=float, default=0.0001, 
                        help='weight_decay')
    parser.add_argument("--savefolder", type=str, default='./test_results/', 
                        help='savefolder')
    # weights of losses and reg terms
    parser.add_argument('--w_pho', type=int, default=8,
                        help='weight')
    parser.add_argument('--w_lmks', type=int, default=1,
                        help='weight')
    parser.add_argument("--w_shape_reg", type=float, default=1e-4, 
                        help='weight')
    parser.add_argument("--w_expr_reg", type=float, default=1e-4, 
                        help='weight')
    parser.add_argument("--w_pose_reg", type=int, default=1e-4, 
                        help='weight')
    #end config of PhotometricFitting
                       

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    #colmap :get focal
    if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
        print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
        sys.exit()
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(args.scenedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(args.scenedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        run_colmap(args.scenedir, args.match_type)
    else:
        print('Don\'t need to run COLMAP')   
    print( 'Post-colmap')
    _, hwf = load_colmap_pose(args.scenedir)
    hwf = np.reshape(hwf, [3])
    hwf = torch.from_numpy(hwf).to(device)
    focal = hwf[2]
    print("check focal: ",focal)
    
    #images
    images = load_image(args.scenedir, factor = args.factor)
    images[:,:,:,[0, 2]] = images[:,:,:,[2, 0]]#BGR 2 RGB
    print(images.shape)#[n,H,W,3]
    print(type(images))
    gt_images = images
    gt_images = gt_images.transpose(0,3,1,2).astype(np.float32)
    gt_images = torch.from_numpy(gt_images).to(device)
    util.check_mkdir(args.savefolder)

    #fitting
    fitting = PhotometricFitting(args, device)
    scene_name = os.path.basename(args.scenedir)
    landmarkspath = args.landmarkdir+scene_name+"/landmarks.npy"
    maskpath = os.path.join(args.maskdir, scene_name)
    params = fitting.run(gt_images, focal, args.scenedir, landmarkspath, maskpath)

        
    return True



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
