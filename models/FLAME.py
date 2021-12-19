import torch.nn as nn
import pickle
import torch
import numpy as np
import torch.nn.functional as F

from lbs import lbs, batch_rodrigues, vertices2landmarks


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    def __init__(self, config):
        super(FLAME, self).__init__()
        self.dtype = torch.float32

        print('Creating the FLAME Decoder ...')

        with open(config.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.register_buffer(
            'faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer(
            'v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        # scale and trans to the original point
        self.v_template *= 1000.0
        self.v_template[:, 0] -= torch.mean(self.v_template[:, 0])
        self.v_template[:, 1] -= torch.mean(self.v_template[:, 1])
        self.v_template[:, 2] -= torch.mean(self.v_template[:, 2])
        
        shapedirs = to_tensor(to_np(flame_model.shapedirs * 1000.0), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:,:,:config.shape_params], shapedirs[:,:,300:300+config.expression_params]], 2)
        self.register_buffer('shapedirs', shapedirs)

        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))

        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # For landmark with dynamic index
        lmk_embeddings = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            'lmk_faces_idx', torch.tensor(lmk_embeddings['static_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer(
            'lmk_bary_coords', torch.tensor(lmk_embeddings['static_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer(
            'dynamic_lmk_faces_idx', torch.tensor(lmk_embeddings['dynamic_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer(
            'dynamic_lmk_bary_coords', torch.tensor(lmk_embeddings['dynamic_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer(
            'full_lmk_faces_idx', torch.tensor(lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer(
            'full_lmk_bary_coords', torch.tensor(lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype))

        neck_kin_chain = []; NECK_IDX=1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    # landmark with fixed index
    def forward(self, shape_params=None, expression_params=None, pose_params=None, land_index=None):
        batch_size = shape_params.shape[0]
        eye_pose_params = self.eye_pose.expand(batch_size, -1)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(betas, full_pose, template_vertices, 
                          self.shapedirs, self.posedirs, 
                          self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)

        landmarks3d = torch.index_select(vertices, 1, land_index)

        return vertices, landmarks3d

    # landmark with dynamic index
    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx, 
                                          dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32):
        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(
            3, device=pose.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(torch.clamp(
            rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords
    
    def forward_dynamic(self, shape_params=None, expression_params=None, pose_params=None):
        batch_size = shape_params.shape[0]
        eye_pose_params = self.eye_pose.expand(batch_size, -1)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)

        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        return vertices, landmarks2d


class FLAMETex(nn.Module):
    def __init__(self, config):
        super(FLAMETex, self).__init__()

        tex_params = config.tex_params
        tex_space = np.load(config.tex_space_path)
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        texture_mean = torch.from_numpy(texture_mean).float()[None,...]
        texture_basis = torch.from_numpy(texture_basis[:,:tex_params]).float()[None,...]

        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode, uv_coords):
        texture = self.texture_mean + (self.texture_basis*texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0,3,1,2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1 ,0], :, :]

        x = (uv_coords[:, 0] * 256).long()
        y = ((1 - uv_coords[:, 1]) * 256).long()

        return texture[:, :, y, x].contiguous()