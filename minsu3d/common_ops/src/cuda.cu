#include <ATen/ATen.h>
#include "datatype/datatype.h"
#include "cal_iou_and_masklabel/cal_iou_and_masklabel.cu"
#include "bfs_cluster/bfs_cluster.cu"
#include "roipool/roipool.cu"
#include "hierarchical_aggregation/hierarchical_aggregation.cu"
#include "get_iou/get_iou.cu"
#include "sec_mean/sec_mean.cu"
