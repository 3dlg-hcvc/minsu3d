/*
Calculate the IoU between predictions and GTs and generate mask labels
*/

#include "cal_iou_and_masklabel.h"
#include <math.h>
#include <stdio.h>

#define MAX_BLOCKS_PER_GRID 32768
#define MAX_THREADS_PER_BLOCK 512

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void
get_mask_iou_on_cluster_cuda_(int nInstance, int nProposal, long *proposals_idx,
                              int *proposals_offset, int16_t *instance_labels,
                              int *instance_pointnum, float *proposals_iou) {

  for (int proposal_id = blockIdx.x; proposal_id < nProposal;
       proposal_id += gridDim.x) {
    int start = proposals_offset[proposal_id];
    int end = proposals_offset[proposal_id + 1];
    int proposal_total = end - start;
    for (int instance_id = threadIdx.x; instance_id < nInstance; instance_id += blockDim.x) {
      int instance_total = instance_pointnum[instance_id];
      int intersection = 0;
      for (int i = start; i < end; i++) {
        long idx = proposals_idx[i];
        if (instance_labels[idx] == instance_id) {
          intersection += 1;
        }
      }
      proposals_iou[proposal_id * nInstance + instance_id] =
          (float)intersection /
          ((float)(proposal_total + instance_total - intersection) + 1e-5);
    }
  }
}

__global__ void
get_mask_iou_on_pred_cuda_(int nInstance, int nProposal, long *proposals_idx,
                           int *proposals_offset, int16_t *instance_labels,
                           int *instance_pointnum, float *proposals_iou,
                           float *mask_scores_sigmoid) {

  for (int proposal_id = blockIdx.x; proposal_id < nProposal; proposal_id += gridDim.x) {
    int start = proposals_offset[proposal_id];
    int end = proposals_offset[proposal_id + 1];
    int proposal_total = 0;

    for (int i = start; i < end; i++)
      if (mask_scores_sigmoid[i] > 0.5)
        proposal_total += 1;

    for (int instance_id = threadIdx.x; instance_id < nInstance;
         instance_id += blockDim.x) {
      int instance_total = instance_pointnum[instance_id];
      int intersection = 0;
      for (int i = start; i < end; i++) {
        long idx = proposals_idx[i];
        if (mask_scores_sigmoid[i] > 0.5) {
          if (instance_labels[idx] == instance_id)
            intersection += 1;
        }
      }
      proposals_iou[proposal_id * nInstance + instance_id] =
          (float)intersection /
          ((float)(proposal_total + instance_total - intersection) + 1e-5);
    }
  }
}

__global__ void get_mask_label_cuda_(int nInstance, int nProposal, int ignored_label,
                                     float iou_thr, long *proposals_idx,
                                     int *proposals_offset,
                                     int16_t *instance_labels, int16_t *instance_cls,
                                     float *proposals_iou, bool *mask_label, bool *mask_label_mask) {
  for (int proposal_id = blockIdx.x; proposal_id < nProposal; proposal_id += gridDim.x) {
    int start = proposals_offset[proposal_id];
    int end = proposals_offset[proposal_id + 1];
    // int proposal_total = end - start;

    // find the instance with max iou
    float max_iou = 0.;
    int max_ind = 0;
    for (int instance_id = 0; instance_id < nInstance; instance_id++) {
      if (proposals_iou[proposal_id * nInstance + instance_id] > max_iou) {
        if (instance_cls[instance_id] != ignored_label) {
          max_iou = proposals_iou[proposal_id * nInstance + instance_id];
          max_ind = instance_id;
        }
      }
    }
    // mask_label initilized with -1 (-1 means ignored)
    if (max_iou >= iou_thr) {
      for (int i = start; i < end; i++) {
        long idx = proposals_idx[i];
        if (instance_labels[idx] == max_ind) {
          mask_label[i] = true;
        }
        mask_label_mask[i] = true;
      }
    }
  }
}

void get_mask_iou_on_cluster_cuda(int nInstance, int nProposal,
                                  long *proposals_idx, int *proposals_offset,
                                  int16_t *instance_labels, int *instance_pointnum,
                                  float *proposals_iou) {
  get_mask_iou_on_cluster_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
                                  std::min(nInstance,
                                           (int)MAX_THREADS_PER_BLOCK)>>>(
      nInstance, nProposal, proposals_idx, proposals_offset, instance_labels,
      instance_pointnum, proposals_iou);
  cudaDeviceSynchronize();
}

void get_mask_iou_on_pred_cuda(int nInstance, int nProposal, long *proposals_idx,
                               int *proposals_offset, int16_t *instance_labels,
                               int *instance_pointnum, float *proposals_iou,
                               float *mask_scores_sigmoid) {
  get_mask_iou_on_pred_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
                               std::min(nInstance,
                                        (int)MAX_THREADS_PER_BLOCK)>>>(
      nInstance, nProposal, proposals_idx, proposals_offset, instance_labels,
      instance_pointnum, proposals_iou, mask_scores_sigmoid);
  cudaDeviceSynchronize();
}

void get_mask_label_cuda(int nInstance, int nProposal, int ignored_label, float iou_thr,
                         long *proposals_idx, int *proposals_offset,
                         int16_t *instance_labels, int16_t *instance_cls,
                         float *proposals_iou, bool *mask_label, bool *mask_label_mask) {
  get_mask_label_cuda_<<<std::min(nProposal, (int)MAX_BLOCKS_PER_GRID),
                         (int)1>>>(nInstance, nProposal, ignored_label, iou_thr, proposals_idx,
                                   proposals_offset, instance_labels,
                                   instance_cls, proposals_iou, mask_label, mask_label_mask);
  cudaDeviceSynchronize();
}
