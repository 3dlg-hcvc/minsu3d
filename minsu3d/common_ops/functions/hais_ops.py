import torch
from torch.autograd import Function
import COMMON_OPS


class HierarchicalAggregation(Function):
    @staticmethod
    def forward(ctx, semantic_label, coord_shift, ball_query_idxs, start_len, batch_idxs, using_set_aggr, point_num_avg,
                radius_avg, ignored_label):
        '''
        :param ctx:
        :param semantic_label: (N_fg), int
        :param coord_shift: (N_fg, 3), float
        :param ball_query_idxs: (nActive), int
        :param start_len: (N_fg, 2), int
        :param batch_idxs: (N_fg), int16

        :return: cluster_idxs:  int (sumNPoint, 2), [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        '''
        N = start_len.size(0)

        assert semantic_label.is_contiguous()
        assert coord_shift.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        fragment_idxs = semantic_label.new()
        fragment_offsets = semantic_label.new()
        fragment_centers = coord_shift.new()  # float

        cluster_idxs_kept = semantic_label.new()
        cluster_offsets_kept = semantic_label.new()
        cluster_centers_kept = coord_shift.new()  # float

        primary_idxs = semantic_label.new()
        primary_offsets = semantic_label.new()
        primary_centers = coord_shift.new()  # float

        primary_idxs_post = semantic_label.new()
        primary_offsets_post = semantic_label.new()

        using_set_aggr_ = int(using_set_aggr)

        point_num_avg = torch.tensor(point_num_avg, dtype=torch.float32, device="cpu")
        radius_avg = torch.tensor(radius_avg, dtype=torch.float32, device="cpu")

        COMMON_OPS.hierarchical_aggregation(semantic_label, coord_shift, batch_idxs, ball_query_idxs, start_len,
                                            fragment_idxs, fragment_offsets, fragment_centers,
                                            cluster_idxs_kept, cluster_offsets_kept, cluster_centers_kept,
                                            primary_idxs, primary_offsets, primary_centers,
                                            primary_idxs_post, primary_offsets_post,
                                            point_num_avg, radius_avg,
                                            N, using_set_aggr_, ignored_label)
        if using_set_aggr_ == 0:  # not set aggr
            pass
        else:
            # cut off tails
            primary_idxs_post = primary_idxs_post[:primary_offsets_post[-1]]
            primary_idxs = primary_idxs_post
            primary_offsets = primary_offsets_post

        cluster_idxs = cluster_idxs_kept
        cluster_offsets = cluster_offsets_kept

        if primary_idxs.shape[0] != 0:
            # add primary
            primary_idxs[:, 0] += (cluster_offsets.size(0) - 1)
            primary_offsets += cluster_offsets[-1]
            cluster_idxs = torch.cat((cluster_idxs, primary_idxs), dim=0).cpu()
            cluster_offsets = torch.cat((cluster_offsets, primary_offsets[1:])).cpu()

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None

hierarchical_aggregation = HierarchicalAggregation.apply
