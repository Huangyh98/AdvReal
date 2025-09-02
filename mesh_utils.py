import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import pytorch3d as p3d
from pytorch3d.structures import Meshes


def get_map_kernel(locations, faces_uvs_all, use_grids=True, bin_num=50, padded_default=True, batch_size=2000):
    """
    get the barycentric kernels for a map
    """
    faces_locs = locations[faces_uvs_all.view(-1)].view(-1, 3, 2)
    bc_kernel = torch.inverse(F.pad(faces_locs, [0, 1], value=1.0).to('cpu')).to('cuda')
    if not use_grids:
        return bc_kernel

    # use grids to save memory
    max_range = locations.abs().max()
    bin_size = 2 * max_range / bin_num
    bin_range = bin_size / np.sqrt(2)
    # compute the grids coordinates
    grids = torch.meshgrid(torch.linspace(-max_range, max_range, bin_num + 1),
                           torch.linspace(-max_range, max_range, bin_num + 1))
    grids = torch.stack(grids, -1).view(-1, 2).to(locations)

    # split faces in batches
    faces_uvs_list = faces_uvs_all.split(batch_size, 0)
    collect = []
    counts = []
    for fi, faces_uvs in enumerate(faces_uvs_list):
        faces_locs = locations[faces_uvs.view(-1)].view(-1, 3, 2)
        # bc_kernel = torch.inverse(F.pad(faces_locs, [0, 1], value=1.0))

        # compute the distance to triangles
        v1 = faces_locs.view(1, -1, 3, 2) - grids.view(-1, 1, 1, 2)
        v2 = grids.view(-1, 1, 1, 2) - faces_locs.roll(1, 1).view(1, -1, 3, 2)
        v3 = faces_locs.roll(1, 1).view(1, -1, 3, 2) - faces_locs.view(1, -1, 3, 2)
        lambda1 = - (v2 * v3).sum(-1, keepdim=True)
        lambda2 = (v1 * v3).sum(-1, keepdim=True)
        perp = (lambda1 * v1 + lambda2 * v2) / (v3 * v3).sum(-1, keepdim=True)
        perp_norm = perp.norm(2, -1)
        points_min = v1.norm(2, -1).minimum(v2.norm(2, -1))
        indicator = (lambda1 >= 0).logical_and(lambda2 <= 0).squeeze(-1)
        dis_to_tri = torch.where(indicator, perp_norm, points_min).min(-1)[0]

        area1 = (v1 * v2.flip(-1)).sum(-1).abs().sum(-1)
        area2 = ((faces_locs[:, 0] - faces_locs[:, 1]) * (faces_locs[:, 0] - faces_locs[:, 2]).flip(-1)).sum(
            -1).abs().unsqueeze(0)
        dis_to_tri = torch.where(area1 > area2, dis_to_tri, dis_to_tri.new([0.0]))  # Ng * B
        # dis_to_tri = perp.norm(2, -1).min(-1)[0]

        in_tri = dis_to_tri <= bin_range
        collect_i = in_tri.nonzero()
        collect_i[:, 1] += fi * batch_size
        collect.append(collect_i)

        counts_i = torch.count_nonzero(in_tri, dim=1)
        counts.append(counts_i)
    collect = torch.cat(collect, 0)
    collect = collect[collect[:, 0].argsort()]
    counts = torch.stack(counts, -1).sum(-1)
    max_num = counts.max().item()
    print('max range is %.3f, max number of the bins is %d' % (max_range, max_num))

    ids = collect[:, 0]
    values = collect[:, 1]
    num_ids = F.pad(counts.unsqueeze(1).expand(-1, len(ids)).triu().sum(0)[:-1], [1, 0], value=0)
    ids_per_bin = torch.arange(len(collect), device=locations.device) - num_ids[ids]

    grids_index_tt = ids.new_zeros(size=(len(grids), max_num)) - 1
    grids_index_tt.index_put_([ids, ids_per_bin], values)
    # return collect, counts

    kernel_fake = locations.new([[max_range + 1, max_range + 1], [max_range + 2, max_range + 1], [max_range + 1, max_range + 2]])
    kernel_fake = torch.inverse(F.pad(kernel_fake, [0, 1], value=1.0).to('cpu')).to('cuda')

    if padded_default:
        grids_index_tt = F.pad(grids_index_tt, [1, 0], value=-1)

    grids_bc_kernels_tt = torch.where(grids_index_tt[..., None, None].expand(-1, -1, 3, 3) >= 0, bc_kernel[grids_index_tt], kernel_fake[None, None, :])

    infos = {
        'grids_bc_kernels': grids_bc_kernels_tt,
        # 'grids_indicator': grids_indicator_tt,
        'grids_index': grids_index_tt,
        'bin_num': bin_num,
        'max_range': max_range,
        'bin_size': bin_size,
        'bn_tt': len(grids_index_tt),
    }
    return infos


def apply_block(tensors, black_block, block_size_map, map_size):
    angle, tx, ty = tensors
    bnum = angle.shape[0]
    scale1 = block_size_map[-1] / map_size[-1]
    scale2 = block_size_map[-2] / map_size[-2]

    cos = angle.cos()
    sin = angle.sin()
    theta = torch.cuda.FloatTensor(bnum, 2, 3).fill_(0)
    theta[:, 0, 0] = cos / scale1
    theta[:, 0, 1] = sin / scale2
    theta[:, 0, 2] = tx * cos / scale1 + ty * sin / scale2
    theta[:, 1, 0] = -sin / scale1
    theta[:, 1, 1] = cos / scale2
    theta[:, 1, 2] = -tx * sin / scale1 + ty * cos / scale2

    grid = F.affine_grid(theta, [bnum, map_size[-3], map_size[-2], map_size[-1]])
    mask = F.grid_sample(black_block, grid)
    mask = mask.sum(0, keepdim=True).clamp(0, 1)
    return mask


def update_mesh(mesh):
    new_verts_list = list(
            mesh._verts_packed.split(mesh.num_verts_per_mesh().tolist(), 0)
        )
    mesh._verts_list = new_verts_list

    # update verts padded
    if mesh._verts_padded is not None:
        for i, verts in enumerate(new_verts_list):
            if len(verts) > 0:
                mesh._verts_padded[i, : verts.shape[0], :] = verts

    # update face areas and normals and vertex normals
    # only if the original attributes are computed
    if any(
        v is not None
        for v in [mesh._faces_areas_packed, mesh._faces_normals_packed]
    ):
        mesh._compute_face_areas_normals(refresh=True)
    if mesh._verts_normals_packed is not None:
        mesh._compute_vertex_normals(refresh=True)


def transforms(mesh, vector=None, angle=None, center=(0, 0, 0), scale=None):
    mesh.verts_packed()

    if angle is not None:
        # angle = mesh._verts_packed.new(size=[1])
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        b = angle.new(center)
        mesh._verts_packed = b + (mesh._verts_packed - b).matmul(mesh._verts_packed.new([cos, sin, zero, -sin, cos, zero, zero, zero, one]).view(3, 3))

    if vector is not None:
        mesh._verts_packed += vector

    if scale is not None:
        mesh._verts_packed *= scale

    update_mesh(mesh)
    return mesh


def mesh_union(mesh_list, detach=True, join=False):
    verts_list = []
    faces_list = []
    for mesh in mesh_list:
        for verts in mesh.verts_list():
            verts_list.append(verts.detach().clone())
        for faces in mesh.faces_list():
            faces_list.append(faces.detach().clone())
    mesh = Meshes(verts=verts_list, faces=faces_list)
    if join:
        mesh = Meshes(verts=[mesh.verts_packed()], faces=[mesh.faces_packed()])
    return mesh


def join_meshes(meshes, join_maps=None):
    verts = []
    faces = []
    verts_uvs = []
    faces_uvs = []
    maps = []
    for mesh in meshes:
        verts.append(mesh.verts_packed())
        faces.append(mesh.faces_packed())
        maps.append(mesh.textures.maps_list()[0])
        verts_uvs.append(mesh.textures.verts_uvs_list()[0])
        faces_uvs.append(mesh.textures.faces_uvs_list()[0])

    w = 0
    h = 0
    pos = []
    for m in maps:
        if m.shape[0] > w:
            w = m.shape[0]
        h = h + m.shape[1]

    hi = 0
    v_num = 0
    vuv_num = 0
    for i in range(len(meshes)):
        verts_uvs[i] = torch.stack(
            [(verts_uvs[i][:, 0] * maps[i].shape[1] + hi) / h, verts_uvs[i][:, 1] * maps[i].shape[0] / w], -1)
        hi = hi + maps[i].shape[1]

        faces[i] = faces[i] + v_num
        v_num += len(verts[i])

        faces_uvs[i] = faces_uvs[i] + vuv_num
        vuv_num += len(verts_uvs[i])

    if join_maps is None:
        maps = [F.pad(m, (0, 0, 0, 0, w - m.shape[0], 0)) for m in maps]
        join_maps = [torch.cat(maps, 1)]

    verts = [torch.cat(verts)]
    faces = [torch.cat(faces)]
    verts_uvs = [torch.cat(verts_uvs)]
    faces_uvs = [torch.cat(faces_uvs)]


    textures = p3d.renderer.mesh.textures.TexturesUV(join_maps, faces_uvs, verts_uvs)
    return Meshes(verts=verts, faces=faces, textures=textures)


