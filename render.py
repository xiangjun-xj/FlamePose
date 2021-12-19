import torch.nn as nn
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    blending,
    MeshRenderer,
    MeshRasterizer,
    Materials,
    TexturesVertex
)
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend
from pytorch3d.structures import Meshes


class SoftSimpleShader(nn.Module):
    def __init__(self, device, cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()

        self.lights = lights if lights is not None else PointLights(
            device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

        self.cameras = self.cameras.to(device)
        self.lights = self.lights.to(device)
        self.materials = self.materials.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)

        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
        znear = 0.1     # 0.1
        zfar = 50.0       # 50
        images = softmax_rgb_blend(
            texels, fragments, blend_params, znear=znear, zfar=zfar
        )

        return images


class Render(nn.Module):
    def __init__(self, fx=1015, fy=1015, img_w=500, img_h=500, cx=250, cy=250, batch_size=1, device=torch.device('cuda:0'), focal_length = None, pri_point = None):
        super(Render, self).__init__()
        self.batch_size = batch_size
        self.fx = fx
        self.fy = fy
        self.img_h = img_h
        self.img_w = img_w
        self.cx = cx
        self.cy = cy
        self.device = device
        self.focal_length = focal_length
        self.pri_point = pri_point
        self.renderer = self.get_render(batch_size)

        tris = np.loadtxt('./flame-data/tris.txt')
        self.tris = torch.from_numpy(tris).to(self.device).long()

        vert_tris = np.loadtxt('./flame-data/vert_tris.txt')
        self.vert_tris = torch.from_numpy(vert_tris).to(self.device).long()

    def get_render(self, batch_size=1):
        R, T = look_at_view_transform(-1, 0, 0)#.to(self.device)
        R = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        T = T.repeat(batch_size, 1)
        T[:] = 0

        # cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=20, fov=2 * torch.atan(
        #     self.img_w // 2 / self.focal) * 180. / np.pi)
        img_size = min(self.img_w, self.img_h)
        focal_length = torch.Tensor((self.fx*2.0/(img_size-1), self.fy*2.0/(img_size-1))).float().to(
            self.device).reshape(1, 2).repeat(batch_size, 1)
        principal_point = torch.Tensor(((-self.cx*2+self.img_w-1)/(img_size-1), (-self.cy*2+self.img_h-1)/(img_size-1))).float().to(
            self.device).reshape(1, 2).repeat(batch_size, 1)
        if self.focal_length is not None:
            focal_length = self.focal_length.clone()
            focal_length *= 2.0/(img_size-1)
        if self.pri_point is not None:
            principal_point = self.pri_point.clone()
            principal_point[:,0] = (-principal_point[:,0]*2+self.img_w-1)/(img_size-1)
            principal_point[:,1] = (-principal_point[:,1]*2+self.img_h-1)/(img_size-1)
        cameras = PerspectiveCameras(device=self.device, R=R, T=T, focal_length=focal_length,
                                     principal_point=principal_point)

        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, -3]],
            ambient_color=[[1, 1, 1]],
            specular_color=[[0., 0., 0.]],
            diffuse_color=[[0., 0., 0.]])

        sigma = 1e-4

        raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma / 2.0,
            # blur_radius=0,
            faces_per_pixel=30,
            perspective_correct=False
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings,
                cameras=cameras
            ),
            shader=SoftSimpleShader(
                self.device,
                lights=lights,
                blend_params=blend_params,
                cameras=cameras
            ),
        )

        return renderer.to(self.device)

    def compute_normal(self, geometry):
        vert_1 = torch.index_select(geometry, 1, self.tris[:, 0])
        vert_2 = torch.index_select(geometry, 1, self.tris[:, 1])
        vert_3 = torch.index_select(geometry, 1, self.tris[:, 2])

        nnorm = torch.cross(vert_2-vert_1, vert_3-vert_1, 2)
        tri_normal = nn.functional.normalize(nnorm, dim=2)
        v_norm = tri_normal[:, self.vert_tris, :].sum(2)
        vert_normal = v_norm / v_norm.norm(dim=2).unsqueeze(2)

        return vert_normal

    @staticmethod
    def Illumination_layer(face_texture, norm, gamma):
        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        # face_color = face_texture

        return face_color

    def forward(self, rott_geometry, texture, diffuse_sh):
        diffuse_sh = diffuse_sh.reshape(self.batch_size, -1).contiguous()
        texture = texture.permute(0, 2, 1).contiguous()

        face_normal = self.compute_normal(rott_geometry)
        face_color = self.Illumination_layer(texture, face_normal, diffuse_sh)
        face_color = TexturesVertex(face_color)

        mesh = Meshes(
            rott_geometry, self.tris.float().repeat(rott_geometry.shape[0], 1, 1), face_color)
        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0, 255)

        return rendered_img.permute(0, 3, 1, 2).contiguous()
