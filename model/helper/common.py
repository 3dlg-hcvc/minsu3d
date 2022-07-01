from lib.common_ops.functions import common_ops
import MinkowskiEngine as ME
import torch


def clusters_voxelization(clusters_idx, clusters_offset, feats, coords, scale, spatial_shape, mode, device):
    batch_idx = clusters_idx[:, 0].cuda().long()
    c_idxs = clusters_idx[:, 1].long().cuda()
    feats = feats[c_idxs]
    clusters_coords = coords[c_idxs]
    clusters_offset = clusters_offset.cuda()
    clusters_coords_mean = common_ops.sec_mean(clusters_coords, clusters_offset)  # (nCluster, 3), float
    clusters_coords_mean_all = torch.index_select(clusters_coords_mean, 0, batch_idx)  # (sumNPoint, 3), float
    clusters_coords -= clusters_coords_mean_all

    clusters_coords_min = common_ops.sec_min(clusters_coords, clusters_offset)
    clusters_coords_max = common_ops.sec_max(clusters_coords, clusters_offset)

    # 0.01 to ensure voxel_coords < spatial_shape
    clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / spatial_shape).max(1)[0] - 0.01
    clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

    min_xyz = clusters_coords_min * clusters_scale[:, None]
    max_xyz = clusters_coords_max * clusters_scale[:, None]

    clusters_scale = torch.index_select(clusters_scale, 0, batch_idx)

    clusters_coords = clusters_coords * clusters_scale[:, None]

    range = max_xyz - min_xyz
    offset = -min_xyz + torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3, device=device)
    offset += torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3, device=device)
    offset = torch.index_select(offset, 0, batch_idx)
    clusters_coords += offset
    assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < spatial_shape)).sum()

    clusters_coords = clusters_coords.long().cpu()

    clusters_voxel_coords, clusters_p2v_map, clusters_v2p_map = common_ops.voxelization_idx(clusters_coords, clusters_idx[:, 0].to(torch.int16), int(
        clusters_idx[-1, 0]) + 1, mode)
    clusters_voxel_feats = common_ops.voxelization(feats, clusters_v2p_map.cuda(), mode)
    clusters_voxel_feats = ME.SparseTensor(features=clusters_voxel_feats,
                                           coordinates=clusters_voxel_coords.int().cuda())
    return clusters_voxel_feats, clusters_p2v_map


def get_batch_offsets(batch_idxs, batch_size, device):
    """
    :param batch_idxs: (N), int
    :param batch_size: int
    :return: batch_offsets: (batch_size + 1)
    """
    batch_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        batch_offsets[i + 1] = batch_offsets[i] + torch.count_nonzero(batch_idxs == i)
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets