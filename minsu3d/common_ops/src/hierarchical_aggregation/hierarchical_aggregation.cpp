#include "hierarchical_aggregation.h"
#include "time.h"

/* ================================== hierarchical_aggregation ================================== */

// instance point num for each class, statistical data from the training set

ConnectedComponent find_cc(int idx, int16_t *semantic_label, float *coord_shift, uint8_t *batch_idxs,
        int *ball_query_idxs, int *start_len, int *visited, const int ignored_label){
    ConnectedComponent cc;
    cc.cls_label = ignored_label;
    cc.addPoint(idx);
    cc.accum_x += coord_shift[idx * 3 + 0];
    cc.accum_y += coord_shift[idx * 3 + 1];
    cc.accum_z += coord_shift[idx * 3 + 2];
    cc.cls_label = semantic_label[idx]; // currently cc's label is the label of the start point, convert to float
    cc.batch_idx = batch_idxs[idx]; // record batch info 
    visited[idx] = 1;
    std::queue<int> Q;
    assert(Q.empty());
    Q.push(idx);
    while(!Q.empty()){
        int cur = Q.front(); Q.pop();
        int start = start_len[cur * 2];
        int len = start_len[cur * 2 + 1];
        int16_t label_cur = semantic_label[cur];
        for(int i = start; i < start + len; i++){
            int idx_i = ball_query_idxs[i];
            if (semantic_label[idx_i] != label_cur) continue;
            if (visited[idx_i] == 1) continue;
            cc.addPoint(idx_i);
            cc.accum_x += coord_shift[idx_i * 3 + 0];
            cc.accum_y += coord_shift[idx_i * 3 + 1];
            cc.accum_z += coord_shift[idx_i * 3 + 2];
            visited[idx_i] = 1;
            Q.push(idx_i);
        }
    }
    return cc;
}

// split clusters into fragment and primary based on point num
void split_clusters(int16_t *semantic_label, float *coord_shift, uint8_t *batch_idxs,
    int *ball_query_idxs, int *start_len, const int nPoint,
    ConnectedComponents &CCs_fragment, ConnectedComponents &CCs_kept, ConnectedComponents &CCs_primary, 
    int *sumNPoint_fragment, int *sumNPoint_kept, int *sumNPoint_primary, const float *class_numpoint_mean_dict, const int ignored_label){
    int visited[nPoint] = {0};
    int _class_idx;
    float _class_numpoint_mean, low_thre, high_thre;

    for(int i = 0; i < nPoint; i++){
        if (visited[i] == 0){
            ConnectedComponent CC = find_cc(i, semantic_label, coord_shift, batch_idxs,
                 ball_query_idxs, start_len, visited, ignored_label);
            _class_idx = CC.cls_label;
            _class_numpoint_mean = class_numpoint_mean_dict[_class_idx];

            low_thre = 0.05 * _class_numpoint_mean;
            high_thre = 0.3 * _class_numpoint_mean;

            if ((int)CC.pt_idxs.size() < high_thre){
                CCs_fragment.push_back(CC);
                *sumNPoint_fragment += (int)CC.pt_idxs.size();

                // keep fragments which are large enough to be independent instances
                if ((int)CC.pt_idxs.size() >= low_thre && (int)CC.pt_idxs.size() < high_thre){
                    CCs_kept.push_back(CC);
                    *sumNPoint_kept += (int)CC.pt_idxs.size();
                }
            }
            else {
                CCs_primary.push_back(CC);
                *sumNPoint_primary += (int)CC.pt_idxs.size();
            }
        }
    }
    return;
}

// convert from ConnectedComponents to (idxs, offsets) representation
void fill_cluster_idxs_(ConnectedComponents &CCs, int *cluster_obj_idxs, long *cluster_point_idxs, int *cluster_offsets, float *cluster_centers){
    for(int i = 0; i < (int)CCs.size(); i++){
        cluster_offsets[i + 1] = cluster_offsets[i] + (int)CCs[i].pt_idxs.size();

        cluster_centers[i * 5 + 0] = CCs[i].accum_x / (float)CCs[i].pt_idxs.size();
        cluster_centers[i * 5 + 1] = CCs[i].accum_y / (float)CCs[i].pt_idxs.size();
        cluster_centers[i * 5 + 2] = CCs[i].accum_z / (float)CCs[i].pt_idxs.size();
        cluster_centers[i * 5 + 3] = (float)CCs[i].cls_label;
        cluster_centers[i * 5 + 4] = (float)CCs[i].batch_idx;

        for(int j = 0; j < (int)CCs[i].pt_idxs.size(); j++){
            long idx = (long)CCs[i].pt_idxs[j];
            int tmp_index = cluster_offsets[i] + j;
            cluster_obj_idxs[tmp_index] = i;
            cluster_point_idxs[tmp_index] = idx;
        }
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> hierarchical_aggregation(at::Tensor semantic_label_tensor, at::Tensor coord_shift_tensor, at::Tensor batch_idxs_tensor,
        at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor,
        at::Tensor fragment_offsets_tensor, at::Tensor fragment_centers_tensor,
        at::Tensor cluster_offsets_kept_tensor, at::Tensor cluster_centers_kept_tensor,
        at::Tensor primary_offsets_tensor, at::Tensor primary_centers_tensor,
        at::Tensor primary_offsets_post_tensor,
        at::Tensor point_num_avg_tensor, at::Tensor radius_avg_tensor,
        const int N, const int using_set_aggr_, const int ignored_label){
    int16_t *semantic_label = semantic_label_tensor.data_ptr<int16_t>();
    float *coord_shift = coord_shift_tensor.data_ptr<float>();
    uint8_t *batch_idxs = batch_idxs_tensor.data_ptr<uint8_t>();
    int *ball_query_idxs = ball_query_idxs_tensor.data_ptr<int>();
    int *start_len = start_len_tensor.data_ptr<int>();
    float *point_num_avg = point_num_avg_tensor.data_ptr<float>();
    float *radius_avg = radius_avg_tensor.data_ptr<float>();

    ConnectedComponents CCs_fragment;
    ConnectedComponents CCs_kept;
    ConnectedComponents CCs_primary;

    int sumNPoint_fragment = 0, sumNPoint_kept = 0, sumNPoint_primary = 0;
    split_clusters(semantic_label, coord_shift, batch_idxs, ball_query_idxs, start_len, N,
        CCs_fragment, CCs_kept, CCs_primary,
        & sumNPoint_fragment, & sumNPoint_kept, & sumNPoint_primary, point_num_avg, ignored_label);

    at::Tensor cluster_obj_idxs_tensor = torch::zeros({sumNPoint_kept}, torch::kInt32);
    at::Tensor cluster_point_idxs_tensor = torch::zeros({sumNPoint_kept}, torch::kInt64);


    cluster_offsets_kept_tensor.resize_({(int)CCs_kept.size() + 1});
    cluster_centers_kept_tensor.resize_({(int)CCs_kept.size(), 5});

    cluster_offsets_kept_tensor.zero_();
    cluster_centers_kept_tensor.zero_();

    int *cluster_obj_idxs = cluster_obj_idxs_tensor.data_ptr<int>();
    long *cluster_point_idxs = cluster_point_idxs_tensor.data_ptr<long>();


    int *cluster_offsets_kept = cluster_offsets_kept_tensor.data_ptr<int>();
    float *cluster_centers_kept = cluster_centers_kept_tensor.data_ptr<float>();
    fill_cluster_idxs_(CCs_kept, cluster_obj_idxs, cluster_point_idxs, cluster_offsets_kept, cluster_centers_kept);

    at::Tensor primary_idxs_tensor = torch::zeros({sumNPoint_primary}, torch::kInt32);
    at::Tensor primary_points_idxs_tensor = torch::zeros({sumNPoint_primary}, torch::kInt64);


    primary_offsets_tensor.resize_({(int)CCs_primary.size() + 1});
    primary_centers_tensor.resize_({(int)CCs_primary.size(), 5});
    primary_offsets_tensor.zero_();
    primary_centers_tensor.zero_();

    int *primary_idxs = primary_idxs_tensor.data_ptr<int>();
    long *primary_points_idxs = primary_points_idxs_tensor.data_ptr<long>();


    int *primary_offsets = primary_offsets_tensor.data_ptr<int>();
    float *primary_centers = primary_centers_tensor.data_ptr<float>();
    fill_cluster_idxs_(CCs_primary, primary_idxs, primary_points_idxs, primary_offsets, primary_centers);

    at::Tensor fragment_idxs_tensor = torch::zeros({sumNPoint_fragment}, torch::kInt32);
    at::Tensor fragment_points_idxs_tensor = torch::zeros({sumNPoint_fragment}, torch::kInt64);


    at::Tensor primary_idxs_post_tensor = torch::zeros({sumNPoint_fragment + sumNPoint_primary}, torch::kInt32);
    at::Tensor primary_points_idxs_post_tensor = torch::zeros({sumNPoint_fragment + sumNPoint_primary}, torch::kInt64);

    if (using_set_aggr_ == 0) { // only point aggr
        return std::make_tuple(cluster_obj_idxs_tensor, cluster_point_idxs_tensor, primary_idxs_tensor,
        primary_points_idxs_tensor, fragment_idxs_tensor, fragment_points_idxs_tensor, primary_idxs_post_tensor,
        primary_points_idxs_post_tensor);

    }


    fragment_offsets_tensor.resize_({(int)CCs_fragment.size() + 1});
    fragment_centers_tensor.resize_({(int)CCs_fragment.size(), 5}); //[:, -2] for cls_label, [:, -1] for batch_idx
    fragment_offsets_tensor.zero_();
    fragment_centers_tensor.zero_();

    int *fragment_idxs = fragment_idxs_tensor.data_ptr<int>();
    long *fragment_points_idxs = fragment_points_idxs_tensor.data_ptr<long>();

    int *fragment_offsets = fragment_offsets_tensor.data_ptr<int>();
    float *fragment_centers = fragment_centers_tensor.data_ptr<float>();


    fill_cluster_idxs_(CCs_fragment, fragment_idxs, fragment_points_idxs, fragment_offsets, fragment_centers);


    // prerare tensor for storing post-primary
    //primary_idxs_post_tensor.resize_({sumNPoint_fragment + sumNPoint_primary, 2});  //never overflow, but need to cut off tails



    primary_offsets_post_tensor.resize_({(int)CCs_primary.size() + 1});
    // primary_idxs_post_tensor.zero_();

    primary_offsets_post_tensor.zero_();
    int *primary_idxs_post = primary_idxs_post_tensor.data_ptr<int>();
    long *primary_points_idxs_post = primary_points_idxs_post_tensor.data_ptr<long>();

    int *primary_offsets_post = primary_offsets_post_tensor.data_ptr<int>();

    // set aggr
    hierarchical_aggregation_cuda(sumNPoint_fragment, (int)CCs_fragment.size(), fragment_points_idxs, fragment_offsets, fragment_centers,
        sumNPoint_primary, (int)CCs_primary.size(), primary_idxs, primary_points_idxs, primary_offsets, primary_centers,
        primary_idxs_post, primary_points_idxs_post, primary_offsets_post, radius_avg, radius_avg_tensor.sizes()[0]);

    return std::make_tuple(cluster_obj_idxs_tensor, cluster_point_idxs_tensor, primary_idxs_tensor,
    primary_points_idxs_tensor, fragment_idxs_tensor, fragment_points_idxs_tensor, primary_idxs_post_tensor,
    primary_points_idxs_post_tensor);

}