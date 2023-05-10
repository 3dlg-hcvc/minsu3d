import torch
from torch.autograd import Function
import COMMON_OPS


class HierarchicalAggregation(Function):
    @staticmethod
    def forward(ctx, semantic_label, coord_shift, ball_query_idxs, start_len, batch_idxs, using_set_aggr,
                point_num_avg, radius_avg, ignored_label):
        N = start_len.size(0)

        assert semantic_label.is_contiguous()
        assert coord_shift.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        fragment_offsets = torch.empty_like(semantic_label, dtype=torch.int32)
        fragment_centers = coord_shift.new()  # float

        cluster_offsets_kept = torch.empty_like(semantic_label, dtype=torch.int32)
        cluster_centers_kept = coord_shift.new()  # float

        primary_offsets = torch.empty_like(semantic_label, dtype=torch.int32)
        primary_centers = coord_shift.new()  # float

        primary_offsets_post = torch.empty_like(semantic_label, dtype=torch.int32)

        using_set_aggr_ = int(using_set_aggr)

        point_num_avg = torch.tensor(point_num_avg, dtype=torch.float32, device="cpu")
        radius_avg = torch.tensor(radius_avg, dtype=torch.float32, device="cpu")

        cluster_idxs_kept, cluster_points_idxs_kept, primary_idxs, primary_points_idxs, fragment_idxs, fragment_points_idxs, primary_idxs_post, primary_points_idxs_post = \
            COMMON_OPS.hierarchical_aggregation(
                semantic_label, coord_shift, batch_idxs, ball_query_idxs, start_len, fragment_offsets, fragment_centers,
                cluster_offsets_kept, cluster_centers_kept, primary_offsets, primary_centers,
                primary_offsets_post, point_num_avg, radius_avg, N, using_set_aggr_, ignored_label
            )
        if using_set_aggr_ == 0:  # not set aggr
            pass
        else:
            # cut off tails
            primary_idxs_post = primary_idxs_post[:primary_offsets_post[-1]]
            primary_points_idxs_post = primary_points_idxs_post[:primary_offsets_post[-1]]
            primary_idxs = primary_idxs_post
            primary_points_idxs = primary_points_idxs_post
            primary_offsets = primary_offsets_post

        cluster_offsets = cluster_offsets_kept

        if primary_idxs.shape[0] != 0:
            # add primary
            primary_idxs += (cluster_offsets.size(0) - 1)
            primary_offsets += cluster_offsets[-1]
            cluster_idxs_kept = torch.cat((cluster_idxs_kept, primary_idxs), dim=0)
            cluster_points_idxs_kept = torch.cat((cluster_points_idxs_kept, primary_points_idxs), dim=0)

            cluster_offsets = torch.cat((cluster_offsets, primary_offsets[1:]))

        return cluster_idxs_kept, cluster_points_idxs_kept, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None

hierarchical_aggregation = HierarchicalAggregation.apply
