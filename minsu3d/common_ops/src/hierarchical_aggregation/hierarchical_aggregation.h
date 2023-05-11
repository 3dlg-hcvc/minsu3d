/*
Hierarchichal Aggregation Algorithm
*/

#ifndef HIERARCHICAL_AGGREGATION_H
#define HIERARCHICAL_AGGREGATION_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>


#include "../datatype/datatype.h"


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> hierarchical_aggregation(at::Tensor semantic_label_tensor, at::Tensor coord_shift_tensor, at::Tensor batch_idxs_tensor,
    at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor, 
    at::Tensor fragment_offsets_tensor, at::Tensor fragment_centers_tensor,
    at::Tensor cluster_offsets_kept_tensor, at::Tensor cluster_centers_kept_tensor,
    at::Tensor primary_offsets_tensor, at::Tensor primary_centers_tensor,
    at::Tensor primary_offsets_post_tensor,
    at::Tensor point_num_avg, at::Tensor radius_avg,
    const int N, const int using_set_aggr_, const int ignored_label);


void hierarchical_aggregation_cuda(
    int fragment_total_point_num, int fragment_num, long *fragment_points_idxs, int *fragment_offsets, float *fragment_centers,
    int primary_total_point_num, int primary_num, int *primary_idxs, long *primary_points_idxs, int *primary_offsets, float *primary_centers,
    int *primary_idxs_post, long *primary_points_idxs_post, int *primary_offsets_post, const float *class_radius_mean, const int class_num
);
#endif //HIERARCHICAL_AGGREGATION_H

