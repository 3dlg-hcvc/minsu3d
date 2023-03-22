#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "common_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    // SoftGroup
    m.def("sg_bfs_cluster", &sg_bfs_cluster, "sg_bfs_cluster");

    m.def("global_avg_pool_fp", &global_avg_pool_fp, "global_avg_pool_fp");
    m.def("global_avg_pool_bp", &global_avg_pool_bp, "global_avg_pool_bp");

    // Common
    m.def("ballquery_batch_p", &ballquery_batch_p, "ballquery_batch_p");
    m.def("sec_mean", &sec_mean, "sec_mean");
    m.def("sec_min", &sec_min, "sec_min");
    m.def("sec_max", &sec_max, "sec_max");
    m.def("roipool_fp", &roipool_fp, "roipool_fp");
    m.def("roipool_bp", &roipool_bp, "roipool_bp");
    m.def("get_iou", &get_iou, "get_iou");
    m.def("get_mask_iou_on_cluster", &get_mask_iou_on_cluster, "get_mask_iou_on_cluster");
    m.def("get_mask_iou_on_pred", &get_mask_iou_on_pred, "get_mask_iou_on_pred");
    m.def("get_mask_label", &get_mask_label, "get_mask_label");

    // PointGroup
    m.def("pg_bfs_cluster", &pg_bfs_cluster, "pg_bfs_cluster");

    // HAIS
    m.def("hierarchical_aggregation", &hierarchical_aggregation, "hierarchical_aggregation");
}