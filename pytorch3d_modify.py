from typing import NamedTuple, Sequence, Union
import torch
import torch.nn.functional as F
import pytorch3d as p3d
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    AmbientLights,
    RasterizationSettings,
    MeshRasterizer,
    BlendParams
)
import mesh_utils as MU
from PIL import Image
import numpy as np
import cv2

def get_points(infos, dt=0, dr=0, bs=1, wrap=True, random=True, pp_ratio=0.1):
    centers = torch.tensor(infos.centers).view(-1, 1, 2).expand(-1, bs, -1)
    Rs = torch.tensor(infos.Rs).view(-1, 1, 1).expand(-1, bs, -1)
    thetas_fixed = [torch.linspace(0, torch.pi * 2, ntf + 1)[:-1] for ntf in infos.ntfs]
    radius_fixed = [torch.tensor(rf) for rf in infos.radius_fixed]
    thetas_wrap = [torch.linspace(0, torch.pi * 2, ntw + 1)[:-1] for ntw in infos.ntws]
    radius_wrap = [torch.tensor(rw) for rw in infos.radius_wrap]
    param_fixed = [torch.cartesian_prod(tf, rf)[None, :].expand(bs, -1, -1) for tf, rf in zip(thetas_fixed, radius_fixed)]
    param_wrap = [torch.cartesian_prod(tf, rf)[None, :].repeat(bs, 1, 1) for tf, rf in zip(thetas_wrap, radius_wrap)]
    if wrap:
        if random:
            dt = centers.new(size=[bs, infos.nparts]).uniform_(-1, 1) * dt
            dr = centers.new(size=[bs, infos.nparts]).uniform_(-1, 1) * dr
            for i in range(infos.nparts):
                param_wrap[i][..., 0] += dt[:, i:i+1] + torch.zeros_like(param_wrap[i][..., 0]).uniform_(-1, 1) * dt[:, i:i+1] * pp_ratio
                param_wrap[i][..., 1] *= 1 + dr[:, i:i+1] + torch.zeros_like(param_wrap[i][..., 1]).uniform_(-1, 1) * dr[:, i:i+1] * pp_ratio
        else:
            for i in range(infos.nparts):
                param_wrap[i][..., 0] += torch.zeros_like(param_wrap[i][..., 0]).fill_(infos.signs[i]) * dt
                param_wrap[i][..., 1] *= 1 + torch.zeros_like(param_wrap[i][..., 1]).fill_(infos.signs[i]) * dr
    params = [torch.cat([pf, pw], 1) for pf, pw in zip(param_fixed, param_wrap)]
    thetas = [torch.stack([p[..., 0].cos(), p[..., 0].sin()], -1) for p in params]
    radius = [p[..., 1:2] for p in params]
    target_control_points = [centers.permute(1, 0, 2)] + [R * r * t + c for c, R, r, t in zip(centers[:, :, None], Rs[:, :, None], radius, thetas)]
    target_control_points = torch.cat(target_control_points, 1)
    if infos.selected is not None:
        target_control_points = target_control_points[:, infos.selected]
    return target_control_points

def fragments_reprojection(fragments, start, end, mesh, locations, infos):
    grids_bc_kernels = infos['grids_bc_kernels']
    grids_index = infos['grids_index']
    pix_to_face = fragments.pix_to_face
    bary_coords = fragments.bary_coords
    pix_selected = (pix_to_face < end).logical_and(pix_to_face >= start)
    pf = pix_to_face[pix_selected] - start
    bc = bary_coords[pix_selected]
    faces_uvs = mesh.textures.faces_uvs_list()[0]
    faces_locs = locations[faces_uvs.view(-1)].view(-1, 3, 2)
    coords = (faces_locs[pf] * bc.unsqueeze(-1)).sum(1)
    indexes = ((coords + infos['max_range']) / infos['bin_size']).round().clamp(0, infos['bin_num']).long()
    indexes = indexes[:, 0] * (infos['bin_num'] + 1) + indexes[:, 1]
    bc_true = F.pad(coords, [0, 1], value=1.0).view(-1, 1, 1, 3).matmul(grids_bc_kernels[indexes]).squeeze(-2)
    indicator = (bc_true[..., 0] >= 0).logical_and((bc_true[..., 1] >= 0).logical_and(bc_true[..., 2] >= 0))
    inds = indicator.max(1)[1]
    pf_new = grids_index[indexes].gather(-1, inds.unsqueeze(-1)).squeeze(-1)
    bc_new = bc_true.gather(1, inds.view(-1, 1, 1).expand(-1, -1, 3)).squeeze(1)
    fragments.pix_to_face[pix_selected] = pf_new + start * (pf_new >= 0)
    fragments.bary_coords[pix_selected] = bc_new
    return fragments

def my_hard_rgb_blend(colors: torch.Tensor, fragments, blend_params) -> torch.Tensor:
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    inds = (fragments.pix_to_face >= 0).max(-1)[1][..., None]
    pix_to_face = fragments.pix_to_face.gather(3, inds)
    colors = colors.gather(3, inds[..., None].expand(-1, -1, -1, -1, colors.shape[-1]))
    is_background = pix_to_face[..., 0] < 0
    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = colors.new_tensor(background_color_)
    num_background_pixels = is_background.sum()
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )
    alpha = (~is_background).type_as(pixel_colors)[..., None]
    return torch.cat([pixel_colors, alpha], dim=-1)

class MyHardPhongShader(HardPhongShader):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            raise ValueError("Cameras must be specified either at initialization or in the forward pass of HardPhongShader")
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(meshes=meshes, fragments=fragments, texels=texels, lights=lights, cameras=cameras, materials=materials)
        images = my_hard_rgb_blend(colors, fragments, blend_params)
        return images

def view_mesh_wrapped(mesh_list, locations_list=None, infos_list=None, offset_verts=None, cameras=(0, 0, 0), lights=None, up=(0, 1, 0), image_size=512, device=None, fov=60, background=None, **kwargs):
    mesh_join = MU.join_meshes(mesh_list)
    num_faces = len(mesh_join.faces_packed())
    if device is None:
        device = mesh_join.device
    if isinstance(cameras, list) or isinstance(cameras, tuple):
        R, T = look_at_view_transform(*cameras, up=(up,))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    if len(mesh_join) == 1 and len(cameras) > 1:
        mesh_join = mesh_join.extend(len(cameras))
    elif len(mesh_join) != len(cameras):
        print('mesh num %d and camera %d num mis-match' % (len(mesh_join), len(cameras)))
        raise ValueError
    if offset_verts is not None:
        mesh_join.offset_verts_(offset_verts - mesh_join.verts_packed())
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=kwargs.get('blur_radius', 0.0),
        faces_per_pixel=kwargs.get('faces_per_pixel', 1),
        bin_size=kwargs.get('bin_size', None),
        max_faces_per_bin=kwargs.get('max_faces_per_bin', None),
    )
    if lights is None:
        lights = AmbientLights(device=device)
    blend_params = BlendParams(1e-4, 1e-4, background) if background is not None else None
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = MyHardPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    fragments = rasterizer(mesh_join)
    if locations_list is not None:
        start = 0
        for mesh, locations, infos in zip(mesh_list, locations_list, infos_list):
            end = start + mesh.faces_list()[0].shape[0]
            if locations is not None:
                for i in range(len(mesh_join)):
                    fragments = fragments_reprojection(fragments, start + i*num_faces, end + i*num_faces, mesh, locations[i], infos)
            start = end
    images = shader(fragments, mesh_join)
    return images
