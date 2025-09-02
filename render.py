import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import math
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    cameras,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    TexturesUV,
    SoftPhongShader
)
import mesh_utils as MU
import pytorch3d_modify as p3dmd
from tps import *
import itertools
from easydict import EasyDict
import glob
import random
from torchvision.transforms import ToPILImage
import uuid
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from NRSM import PrecomputedTPSDeformer

def rgb_to_grayscale(source):
    grayscale = 0.2989 * source[0, :, :] + 0.5870 * source[1, :, :] + 0.1140 * source[2, :, :]
    grayscale = grayscale.unsqueeze(0)
    return grayscale

def compute_relighting_params_tensor(source, target):
    if source.dim() == 3:
        source = source.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    target_resized = F.interpolate(target, size=source.shape[2:], mode='bilinear', align_corners=False)
    source_np = source.clone()
    source_np = source_np.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    target_resized_np = target_resized.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    mean_src = torch.mean(source)
    std_src = torch.std(source)
    mean_tar = torch.mean(target)
    std_tar = torch.std(target)
    ssim_score = ssim(source_np, target_resized_np, data_range=1, win_size=3, channel_axis=2)
    similarity_factor = 1 - ssim_score
    if std_tar < std_src:
        a, b, c = 0.5, 0, 1
        alpha_coeff = a * (similarity_factor ** 2) + b * similarity_factor + c
    else:
        a, b, c = -0.5, 0, 1
        alpha_coeff = a * (similarity_factor ** 2) + b * similarity_factor + c
    d, e, f = -0.5, 0, 1
    beta_coeff = d * (similarity_factor ** 2) + e * similarity_factor + f
    alpha = (std_tar / std_src) * alpha_coeff
    beta = (mean_tar - (std_tar / std_src) * mean_src) * beta_coeff
    alpha = torch.clamp(alpha, min=0.6, max=3)
    beta = torch.clamp(beta, min=0.1, max=1)
    return alpha, beta

def apply_relighting_tensor(new_tensor, alpha, beta):
    new_tensor = new_tensor * alpha + beta / 255.0
    return new_tensor

def resize_with_aspect_ratio_tensor(image, target_width, target_height):
    C, H, W = image.shape
    scale = min(target_width / W, target_height / H)
    new_W = int(W * scale)
    new_H = int(H * scale)
    resized_image = F.interpolate(image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
    result = torch.zeros((C, target_height, target_width), device=image.device)
    x_offset = (target_width - new_W) // 2
    y_offset = (target_height - new_H) // 2
    result[:, y_offset:y_offset + new_H, x_offset:x_offset + new_W] = resized_image
    return result

def generate_ordered_crops_diagonal_tensor(image_tensor, num_crops=3, H=416, W=416):
    a, b, aspect_ratio = 190.0, 0.8, 1 / 3
    D_min, D_max = 15, 40
    P_min, P_max = 60, 200
    crops = []
    last_D, last_cx = 0, 0
    direction = random.choice([-1, 1])
    for i in range(num_crops):
        if i == 0:
            D = random.randint(D_min, D_max)
        else:
            new_D = last_D + direction * random.randint(5, 15)
            if not (D_min <= new_D <= D_max):
                direction *= -1
                new_D = last_D + direction * random.randint(5, 15)
            D = np.clip(new_D, D_min, D_max)
        P = int(np.clip(round(a - b * D), P_min, P_max))
        W_p = max(1, round(P * aspect_ratio))
        y2 = H - D
        y1 = y2 - P
        if y1 < 0:
            y1, y2 = 0, P
        cx = random.randint(W_p // 2, W - W_p // 2) if i == 0 else np.clip(last_cx + random.randint(-20, 20), W_p // 2, W - W_p // 2)
        x1 = cx - W_p // 2
        crops.append((x1, y1, x1 + W_p, y2))
        last_D, last_cx = D, cx
    return crops

def process_image_tensor(rendered_person_tensor, background_tensor, crop_coords, use_relighting=True, plot=True):
    import torch.nn.functional as F
    device = rendered_person_tensor.device
    crop_coords = torch.tensor(crop_coords, dtype=torch.float, device=device)
    x1, y1, x2, y2 = crop_coords.int()
    background_crop = background_tensor[:, y1:y2, x1:x2].to(device)
    if rendered_person_tensor.shape[0] == 4:
        rendered_rgb = rendered_person_tensor[:3, :, :]
        rendered_alpha = rendered_person_tensor[3, :, :].unsqueeze(0)
    else:
        rendered_rgb = rendered_person_tensor
        rendered_alpha = torch.ones(1, rendered_rgb.shape[1], rendered_rgb.shape[2], device=device)
    if use_relighting:
        rendered_rgb_clamped = rendered_rgb.clone()
        background_rgb_clamped = background_crop[:3, :, :]
        rendered_gray = rgb_to_grayscale(rendered_rgb_clamped)
        background_gray = rgb_to_grayscale(background_rgb_clamped)
        rendered_rgb = rendered_rgb_clamped.clamp(0.000001, 0.999999)
        alpha_relight, beta_relight = compute_relighting_params_tensor(rendered_gray, background_gray)
        relighted_rgb = apply_relighting_tensor(rendered_rgb, alpha_relight, beta_relight)
        rendered_rgb = rendered_rgb_clamped.clamp(0.000001, 0.999999)
    else:
        relighted_rgb = rendered_rgb
        relighted_rgb = relighted_rgb.clamp(0, 1)
    resized_relighted_rgb = resize_with_aspect_ratio_tensor(relighted_rgb, x2 - x1, y2 - y1)
    resized_alpha = resize_with_aspect_ratio_tensor(rendered_alpha, x2 - x1, y2 - y1)
    mask = (resized_alpha > 0.5).float()
    non_zero = mask[0].nonzero(as_tuple=False)
    if non_zero.numel() > 0:
        x_min = non_zero[:, 1].min()
        x_max = non_zero[:, 1].max()
        new_x1 = x1 + x_min
        new_y1 = y1
        new_x2 = x1 + x_max
        new_y2 = y2
        background_crop_new = background_tensor[:, new_y1:new_y2, new_x1:new_x2].to(device)
        resized_relighted_rgb_new = resize_with_aspect_ratio_tensor(relighted_rgb, new_x2 - new_x1, new_y2 - new_y1)
        resized_alpha_new = resize_with_aspect_ratio_tensor(rendered_alpha, new_x2 - new_x1, new_y2 - new_y1)
        mask_new = (resized_alpha_new > 0.5).float()
        if plot:
            mask_pil = to_pil_image(mask_new[0].cpu())
            save_dir = 'mask_images'
            os.makedirs(save_dir, exist_ok=True)
            mask_path = os.path.join(save_dir, f"mask_{uuid.uuid4()}.png")
            mask_pil.save(mask_path)
        composite_crop_new = mask_new * resized_relighted_rgb_new + (1 - mask_new) * background_crop_new[:3, :, :]
        composite_image = background_tensor.clone()
        composite_image[:3, new_y1:new_y2, new_x1:new_x2] = composite_crop_new
        gt_new = [new_x1, new_y1, new_x2, new_y2]
    else:
        composite_crop = mask * resized_relighted_rgb + (1 - mask) * background_crop[:3, :, :]
        composite_image = background_tensor.clone()
        composite_image[:3, y1:y2, x1:x2] = composite_crop
        gt_new = [x1.item(), y1.item(), x2.item(), y2.item()]
    if plot:
        pil_image = to_pil_image(composite_image.cpu())
        save_dir = 'output_images'
        os.makedirs(save_dir, exist_ok=True)
        comp_path = os.path.join(save_dir, f'composite_image.png')
        pil_image.save(comp_path)
    return composite_image, gt_new

class ImageRenderer:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DATA_DIR = "./data"
        self.setup_renderer()
        self.h, self.w = 340, 864
        self.h_t, self.w_t = 484, 700
        self.args = args
        self.load_meshes()
        self.n_man = self.mesh_man.verts_packed().shape[0]
        self.n_tshirt = self.mesh_tshirt.verts_packed().shape[0]
        self.n_trouser = self.mesh_trouser.verts_packed().shape[0]
        self.idx_man = torch.arange(self.n_man, device=self.device)
        self.idx_tshirt = torch.arange(self.n_tshirt, device=self.device) + self.n_man
        self.idx_trouser = torch.arange(self.n_trouser, device=self.device) + self.n_man + self.n_tshirt
        if not args.disable_tps2d:
            self.initialize_tps2d()
        self.deformer = None
        self.source_control_points = None
        self.original_verts = None
        self.args.nrsm_range = getattr(self.args, 'nrsm_range', 0.02)
        self.args.disable_deformation = getattr(self.args, 'disable_deformation', False)
        self.args = args

    def update_camera(self):
        azim = np.random.uniform(-180, 180)
        R, T = look_at_view_transform(dist=2.5, elev=15, azim=azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)
        raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        )

    def sample_lights(self, r=None):
        if r is None:
            r = np.random.rand()
        theta = np.random.rand() * 2 * math.pi
        if r < 0.33:
            self.lights = AmbientLights(device=self.device)
        elif r < 0.67:
            ambient_intensity = np.random.uniform(0.6, 0.7)
            diffuse_intensity = np.random.uniform(0.4, 0.5)
            specular_intensity = np.random.uniform(0.3, 0.4)
            self.lights = DirectionalLights(
                device=self.device,
                direction=[[np.sin(np.random.uniform(-np.pi, np.pi)), np.sin(np.random.uniform(-np.pi/2, np.pi/2)), np.cos(np.random.uniform(-np.pi, np.pi))]],
                ambient_color=((ambient_intensity, ambient_intensity, ambient_intensity),),
                diffuse_color=((diffuse_intensity, diffuse_intensity, diffuse_intensity),),
                specular_color=((specular_intensity, specular_intensity, specular_intensity),)
            )
        else:
            self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
        return

    def setup_renderer(self):
        azim = 0
        R, T = look_at_view_transform(dist=2.5, elev=15, azim=azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)
        raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
        self.lights = AmbientLights(device=self.device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        )

    def load_meshes(self):
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=self.device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=self.device)
        self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=self.device)
        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]
        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
        self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]

    def sample_nrsm_displacements(self, scale=None):
        if self.deformer is None or self.source_control_points is None:
            raise RuntimeError("Deformer not initialized. Call initialize_deformer() first.")
        if scale is None:
            scale = self.args.nrsm_range
        num_pts = self.source_control_points.shape[0]
        disp = torch.empty((num_pts, 3), device=self.device).uniform_(-scale, scale)
        return disp

    def initialize_deformer(self, num_control_points=200):
        print("🚀 Initializing Non-Rigid Surface-aware Mesh deformer...")
        combined_full = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])
        self.original_verts = combined_full.verts_packed().clone().to(self.device)
        combined_clothes = MU.join_meshes([self.mesh_tshirt, self.mesh_trouser])
        verts_np = combined_clothes.verts_packed().cpu().numpy()
        faces_np = combined_clothes.faces_packed().cpu().numpy()
        self.deformer = PrecomputedTPSDeformer(verts_np, faces_np)
        self.deformer.select_and_prepare(num_points=num_control_points)
        self.source_control_points = torch.from_numpy(self.deformer.control_points_coords).float().to(self.device)
        print("✨ Deformer initialized successfully.")

    def apply_deformation(self, displacements):
        if self.deformer is None:
            raise RuntimeError("Deformer not initialized. Call initialize_deformer() first.")
        deformed_clothes_np = self.deformer.deform(displacements.detach().cpu().numpy())
        deformed_clothes = torch.from_numpy(deformed_clothes_np).float().to(self.device)
        n_t = self.n_tshirt
        deformed_tshirt = deformed_clothes[:n_t, :]
        deformed_trouser = deformed_clothes[n_t:, :]
        full = self.original_verts.clone()
        full[self.idx_tshirt] = deformed_tshirt
        full[self.idx_trouser] = deformed_trouser
        return full

    def load_background_images(self, img_dir):
        img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('.jpg', '.png'))]
        images = []
        for img_path in img_paths:
            image = Image.open(img_path).convert('RGBA')
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            images.append((img_path, image_tensor))
        return images

    def _process_single_patch(self, adv_patch_tensor, target_height, target_width, scale_factor):
        C, H_patch, W_patch = adv_patch_tensor.shape
        new_H_patch = max(1, int(H_patch * scale_factor))
        new_W_patch = max(1, int(W_patch * scale_factor))
        adv_patch_resized = F.interpolate(adv_patch_tensor.unsqueeze(0), size=(new_H_patch, new_W_patch), mode='bilinear', align_corners=False).squeeze(0)
        tiles_y = (target_height + new_H_patch - 1) // new_H_patch
        tiles_x = (target_width + new_W_patch - 1) // new_W_patch
        tiled_patch = adv_patch_resized.repeat(1, tiles_y, tiles_x)
        tiled_patch = tiled_patch[:, :target_height, :target_width]
        return tiled_patch

    def tile_tensor_patch(self, adv_patch_tensor, target_height, target_width, scale_factor=0.5):
        if adv_patch_tensor.dim() == 4:
            if adv_patch_tensor.size(0) == 1:
                adv_patch_tensor = adv_patch_tensor.squeeze(0)
            else:
                tiled_patches = []
                for patch in adv_patch_tensor:
                    tiled_patches.append(self._process_single_patch(patch, target_height, target_width, scale_factor))
                return torch.stack(tiled_patches)
        elif adv_patch_tensor.dim() == 3:
            pass
        else:
            raise ValueError(f"Expected adv_patch_tensor to have 3 or 4 dimensions [C, H_patch, W_patch] or [N, C, H_patch, W_patch], but got {adv_patch_tensor.dim()} dimensions.")
        tiled_patch = self._process_single_patch(adv_patch_tensor, target_height, target_width, scale_factor)
        return tiled_patch

    def _process_patch_four_positions(self, adv_patch_tensor, target_height, target_width, scale_factor=0.1):
        positions = [(117, 198),(367, 198),(581, 207),(774, 207)]
        C, H_patch, W_patch = adv_patch_tensor.shape
        new_H_patch = max(1, int(H_patch * scale_factor))
        new_W_patch = max(1, int(W_patch * scale_factor))
        adv_patch_resized = F.interpolate(adv_patch_tensor.unsqueeze(0), size=(new_H_patch, new_W_patch), mode='bilinear', align_corners=False).squeeze(0)
        background = torch.ones((C, target_height, target_width), device=adv_patch_tensor.device, dtype=adv_patch_tensor.dtype)
        for (center_x, center_y) in positions:
            start_y = center_y - new_H_patch // 2
            start_x = center_x - new_W_patch // 2
            background[:, start_y:start_y + new_H_patch, start_x:start_x + new_W_patch] = adv_patch_resized
        return background

    def _process_patch_six_positions(self, adv_patch_tensor, target_height, target_width, scale_factor=0.5):
        positions = [(54,277),(219,277),(414,287),(627,287),(133,64),(515,64)]
        C, H_patch, W_patch = adv_patch_tensor.shape
        new_H_patch = max(1, int(H_patch * scale_factor))
        new_W_patch = max(1, int(W_patch * scale_factor))
        patch_resized = F.interpolate(adv_patch_tensor.unsqueeze(0), size=(new_H_patch, new_W_patch), mode='bilinear', align_corners=False).squeeze(0)
        background = torch.ones((C, target_height, target_width), dtype=adv_patch_tensor.dtype, device=adv_patch_tensor.device)
        for (center_x, center_y) in positions:
            start_x = center_x - (new_W_patch // 2)
            end_x = start_x + new_W_patch
            start_y = center_y - (new_H_patch // 2)
            end_y = start_y + new_H_patch
            patch_start_x = 0
            patch_end_x = new_W_patch
            patch_start_y = 0
            patch_end_y = new_H_patch
            if start_x < 0:
                patch_start_x = -start_x
                start_x = 0
            if end_x > target_width:
                patch_end_x = new_W_patch - (end_x - target_width)
                end_x = target_width
            if start_y < 0:
                patch_start_y = -start_y
                start_y = 0
            if end_y > target_height:
                patch_end_y = new_H_patch - (end_y - target_height)
                end_y = target_height
            if patch_end_x > patch_start_x and patch_end_y > patch_start_y:
                background[:, start_y:end_y, start_x:end_x] = patch_resized[:, patch_start_y:patch_end_y, patch_start_x:patch_end_x]
        return background

    def place_tensor_patch_once(self, adv_patch_tensor, target_height, target_width, scale_factor=0.5):
        if adv_patch_tensor.dim() == 4:
            if adv_patch_tensor.size(0) == 1:
                adv_patch_tensor = adv_patch_tensor.squeeze(0)
            else:
                adv_patch_tensor = adv_patch_tensor[0]
        final_texture = self._process_patch_four_positions(adv_patch_tensor, target_height, target_width, scale_factor=scale_factor)
        return final_texture

    def place_tensor_patch_trouser(self, adv_patch_tensor, target_height, target_width, scale_factor=0.5):
        if adv_patch_tensor.dim() == 4:
            if adv_patch_tensor.size(0) == 1:
                adv_patch_tensor = adv_patch_tensor.squeeze(0)
            else:
                adv_patch_tensor = adv_patch_tensor[0]
        final_texture = self._process_patch_six_positions(adv_patch_tensor, target_height, target_width, scale_factor)
        return final_texture

    def set_adv_patch_texture(self, adv_patch_tensor):
        centered_patch_tshirt = self.place_tensor_patch_once(adv_patch_tensor, self.h, self.w, scale_factor=0.8)
        centered_patch_trouser = self.place_tensor_patch_trouser(adv_patch_tensor, self.h_t, self.w_t, scale_factor=0.8)
        prob_map = centered_patch_tshirt.unsqueeze(0).to(self.device)
        prob_trouser = centered_patch_trouser.unsqueeze(0).to(self.device)
        tex = prob_map.permute(0, 2, 3, 1)
        tex_trouser = prob_trouser.permute(0, 2, 3, 1)
        self.mesh_tshirt.textures = TexturesUV(maps=tex, faces_uvs=self.faces, verts_uvs=self.verts_uv)
        self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)
        return tex, tex_trouser

    def synthesis_image_person(self, use_nrsm=False):
        self.update_camera()
        locations_tshirt = locations_trouser = None
        if use_nrsm:
            if self.deformer is None:
                self.initialize_deformer()
            displacements = self.sample_nrsm_displacements(scale=self.args.nrsm_range)
            deformed_verts = self.apply_deformation(displacements)
            source_coordinate = deformed_verts
        images_predicted = p3dmd.view_mesh_wrapped(
            [self.mesh_man, self.mesh_tshirt, self.mesh_trouser],
            [None, locations_tshirt, locations_trouser],
            [None, self.infos_tshirt, self.infos_trouser], source_coordinate,
            cameras=self.cameras, lights=self.lights, image_size=512, fov=45, max_faces_per_bin=30000, faces_per_pixel=3
        )
        rendered_person = images_predicted[0]
        alpha_channel = rendered_person[..., 3]
        non_transparent = (alpha_channel > 0).float()
        min_y = torch.argmax(torch.any(non_transparent, dim=1).float())
        max_y = torch.argmax(torch.any(non_transparent.flip(dims=[0]), dim=1).float())
        max_y = rendered_person.shape[0] - max_y
        min_x = torch.argmax(torch.any(non_transparent, dim=0).float())
        max_x = torch.argmax(torch.any(non_transparent.flip(dims=[1]), dim=0).float())
        max_x = rendered_person.shape[1] - max_x
        cropped_rendered_person = rendered_person[min_y:max_y+1, min_x:max_x+1, :]
        cropped_rendered_person = cropped_rendered_person.permute(2, 0, 1)
        return cropped_rendered_person

    def generate_composite_image_tensor(self, bg_image_tensor):
        rendered_person = self.synthesis_image_person(use_nrsm=True)
        crop_positions = generate_ordered_crops_diagonal_tensor(bg_image_tensor, num_crops=3)
        if not crop_positions:
            print(f"Unable to generate a qualifying cropping frame in the background image")
            return None, None
        composite_images = []
        gts = []
        self.sample_lights(r=0.5)
        for crop_idx, crop_coords in enumerate(crop_positions):
            composite_image, gt_new = process_image_tensor(
                rendered_person_tensor=rendered_person,
                background_tensor=bg_image_tensor,
                crop_coords=crop_coords,
                use_relighting=False,
                plot=False,
            )
            composite_images.append(composite_image)
            crop_tensor = torch.tensor(gt_new, dtype=torch.long, device=self.device)
            gts.append(crop_tensor)
        return composite_images, gts
