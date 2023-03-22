#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "datatype/datatype.cpp"
#include "cal_iou_and_masklabel/cal_iou_and_masklabel.cpp"
#include "bfs_cluster/bfs_cluster.cpp"
#include "roipool/roipool.cpp"
#include "get_iou/get_iou.cpp"
#include "sec_mean/sec_mean.cpp"
#include "hierarchical_aggregation/hierarchical_aggregation.cpp"
