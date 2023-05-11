#include "hierarchical_aggregation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_PRIMARY_NUM 1024
#define MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM  1024
#define INFINITY_DIS_SQUARE 10000
#define MAX_PER_PRIMARY_ABSORB_POINT_NUM 8192
#define MAX_THREADS_PER_BLOCK 1024
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void fragment_find_primary_(int primary_num, int *cuda_primary_offsets, float *cuda_primary_centers,
    int fragment_num, int *cuda_fragment_offsets, float *cuda_fragment_centers,
    int *cuda_primary_absorb_fragment_idx, int *cuda_primary_absorb_fragment_cnt, const float *class_radius_mean){
    int fragment_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fragment_idx >= fragment_num) return;

    // find the nearest primary for each fragment
    float nearest_dis_square = INFINITY_DIS_SQUARE;
    int nearest_idx = -1; // primary_idx
    for( int i = 0; i < primary_num; i++){
        if (abs(cuda_primary_centers[i * 5 + 3] - cuda_fragment_centers[fragment_idx * 5 + 3]) > 0.1){ //judge same cls_label or not
            continue;
        }
        if (abs(cuda_primary_centers[i * 5 + 4] - cuda_fragment_centers[fragment_idx * 5 + 4]) > 0.1){ //judge same batch_idx or not
            continue;
        }
        float temp_dis_square = pow((cuda_primary_centers[i * 5 + 0] - cuda_fragment_centers[fragment_idx * 5 + 0]), 2)
            + pow((cuda_primary_centers[i * 5 + 1] - cuda_fragment_centers[fragment_idx * 5 + 1]), 2)
            + pow((cuda_primary_centers[i * 5 + 2] - cuda_fragment_centers[fragment_idx * 5 + 2]), 2);
        if (temp_dis_square < nearest_dis_square){
            nearest_dis_square = temp_dis_square;
            nearest_idx = i;
        }
    }
    if (nearest_idx == -1) return; // fragment not belong to any primary

    // r_size
    int primary_point_num = cuda_primary_offsets[nearest_idx + 1] - cuda_primary_offsets[nearest_idx];
    float r_size = 0.01 * sqrt(float(primary_point_num));

    // r_cls
    // instance radius for each class, statistical data from the training set
    int _class_idx = (int)cuda_fragment_centers[fragment_idx * 5 + 3];
    float r_cls = class_radius_mean[_class_idx] * 1.;

    // r_set
    float r_set =  max(r_size, r_cls);

    // judge
    if ( nearest_dis_square < r_set * r_set ){
        int _offect = atomicAdd(cuda_primary_absorb_fragment_cnt + nearest_idx, 1);
        if (_offect < MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM)
            cuda_primary_absorb_fragment_idx[nearest_idx * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM + _offect] = fragment_idx;
    }
}

// input: ...
// output: cuda_concat_idxs
// output: cuda_concat_point_num,
__global__ void concat_fragments_(
    long *cuda_fragment_points_idxs, int *cuda_fragment_offsets,
    int *cuda_primary_absorb_fragment_idx, int *cuda_primary_absorb_fragment_cnt,
    int *cuda_concat_idxs, long *cuda_concat_points_idxs, int *cuda_concat_point_num,
    int primary_num){

    int primary_idx = blockIdx.x;
    if (primary_idx >= primary_num) return;

    int _accu_offset = 0; // unit is point
    for (int i=0; i<cuda_primary_absorb_fragment_cnt[primary_idx] && i<MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM; i++){
        int idx = cuda_primary_absorb_fragment_idx[primary_idx * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM + i];
        for (int j=cuda_fragment_offsets[idx]; j<cuda_fragment_offsets[idx + 1]; j++){
            if (_accu_offset < MAX_PER_PRIMARY_ABSORB_POINT_NUM) {
                int tmp_idx = primary_idx * MAX_PER_PRIMARY_ABSORB_POINT_NUM + _accu_offset;
                cuda_concat_idxs[tmp_idx] = primary_idx;
                cuda_concat_points_idxs[tmp_idx] = (long)cuda_fragment_points_idxs[j];
                _accu_offset++;
            }
        }
    }
    cuda_concat_point_num[primary_idx] = _accu_offset;
}

void hierarchical_aggregation_cuda(
    int fragment_total_point_num, int fragment_num, long *fragment_points_idxs, int *fragment_offsets, float *fragment_centers,
    int primary_total_point_num, int primary_num, int *primary_idxs, long *primary_points_idxs, int *primary_offsets, float *primary_centers,
    int *primary_idxs_post, long *primary_points_idxs_post, int *primary_offsets_post, const float *class_radius_mean, const int class_num){

    if (primary_num == 0){
        return;
    }
    // on devices, allocate and copy memory
    int *cuda_fragment_offsets;
    float *cuda_fragment_centers;
    long *cuda_fragment_points_idxs;
    cudaMalloc((void**)&cuda_fragment_points_idxs, fragment_total_point_num * sizeof(long));

    cudaMalloc((void**)&cuda_fragment_offsets, (fragment_num + 1) * sizeof(int));
    cudaMalloc((void**)&cuda_fragment_centers, fragment_num * 5 * sizeof(float));

    cudaMemcpy(cuda_fragment_points_idxs, fragment_points_idxs, fragment_total_point_num * sizeof(long), cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_fragment_offsets, fragment_offsets, (fragment_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_fragment_centers, fragment_centers, fragment_num * 5 * sizeof(float), cudaMemcpyHostToDevice);

    int *cuda_primary_offsets;
    float *cuda_primary_centers;

    cudaMalloc((void**)&cuda_primary_offsets, (primary_num + 1) * sizeof(int));
    cudaMalloc((void**)&cuda_primary_centers, primary_num * 5 * sizeof(float));
    cudaMemcpy(cuda_primary_offsets, primary_offsets, (primary_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_primary_centers, primary_centers, primary_num * 5 * sizeof(float), cudaMemcpyHostToDevice);

    float *cuda_class_radius_mean;
    cudaMalloc((void**)&cuda_class_radius_mean, class_num * sizeof(float));
    cudaMemcpy(cuda_class_radius_mean, class_radius_mean, class_num * sizeof(float), cudaMemcpyHostToDevice);


    // // for each fragment, find its primary
    int *cuda_primary_absorb_fragment_idx; // array for saving the fragment idxs
    int *cuda_primary_absorb_fragment_cnt; // array for saving the fragment nums
    cudaMalloc((void**)&cuda_primary_absorb_fragment_idx, primary_num * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM * sizeof(int));
    cudaMalloc((void**)&cuda_primary_absorb_fragment_cnt, primary_num * sizeof(int));
    cudaMemset(cuda_primary_absorb_fragment_idx, 0, primary_num * MAX_PER_PRIMARY_ABSORB_FRAGMENT_NUM * sizeof(int));
    cudaMemset(cuda_primary_absorb_fragment_cnt, 0, primary_num * sizeof(int));

    if (fragment_num != 0) {
        fragment_find_primary_<<<int(DIVUP(fragment_num, MAX_THREADS_PER_BLOCK)), (int)MAX_THREADS_PER_BLOCK>>>(
            primary_num, cuda_primary_offsets, cuda_primary_centers,
            fragment_num, cuda_fragment_offsets, cuda_fragment_centers,
            cuda_primary_absorb_fragment_idx, cuda_primary_absorb_fragment_cnt, cuda_class_radius_mean);
    }
    // concatenate fragments belonging to the same primary
    int *cuda_concat_idxs;
    long *cuda_concat_points_idxs;

    int *cuda_concat_point_num;
    cudaMalloc((void**)&cuda_concat_idxs, primary_num * MAX_PER_PRIMARY_ABSORB_POINT_NUM * sizeof(int));
    cudaMalloc((void**)&cuda_concat_points_idxs, primary_num * MAX_PER_PRIMARY_ABSORB_POINT_NUM * sizeof(long));

    cudaMalloc((void**)&cuda_concat_point_num, primary_num * sizeof(int));
    // assert(primary_num <= MAX_PRIMARY_NUM);
    concat_fragments_<<<primary_num, (int)1>>>(
        cuda_fragment_points_idxs, cuda_fragment_offsets,
        cuda_primary_absorb_fragment_idx, cuda_primary_absorb_fragment_cnt,
        cuda_concat_idxs, cuda_concat_points_idxs, cuda_concat_point_num,
        primary_num);
    cudaDeviceSynchronize();

    // merge primary instances and fragments
    int *concat_point_num = new int [primary_num + 1]; // allocate on host
    cudaMemcpy(concat_point_num, cuda_concat_point_num, primary_num * sizeof(int), cudaMemcpyDeviceToHost);
    int _accu_offset = 0;
    for (int i = 0; i < primary_num; i++){
        // add primary instances

        memcpy(primary_idxs_post + _accu_offset, primary_idxs + primary_offsets[i], (primary_offsets[i + 1] - primary_offsets[i]) * sizeof(int));
        memcpy(primary_points_idxs_post + _accu_offset, primary_points_idxs + primary_offsets[i], (primary_offsets[i + 1] - primary_offsets[i]) * sizeof(long));

        _accu_offset += (primary_offsets[i + 1] - primary_offsets[i]);

        // add absorbed fragments
        cudaMemcpy(primary_idxs_post + _accu_offset, cuda_concat_idxs + i * MAX_PER_PRIMARY_ABSORB_POINT_NUM, concat_point_num[i] * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(primary_points_idxs_post + _accu_offset, cuda_concat_points_idxs + i * MAX_PER_PRIMARY_ABSORB_POINT_NUM, concat_point_num[i] * sizeof(long), cudaMemcpyDeviceToHost);

        _accu_offset += concat_point_num[i];

        // writing offsets
        primary_offsets_post[i + 1] = _accu_offset;
    }
    cudaDeviceSynchronize();

    cudaFree(cuda_fragment_points_idxs);
    cudaFree(cuda_fragment_offsets);
    cudaFree(cuda_fragment_centers);

    cudaFree(cuda_primary_offsets);
    cudaFree(cuda_primary_centers);
    cudaFree(cuda_class_radius_mean);

    cudaFree(cuda_primary_absorb_fragment_idx);
    cudaFree(cuda_primary_absorb_fragment_cnt);

    cudaFree(cuda_concat_idxs);
    cudaFree(cuda_concat_points_idxs);
    cudaFree(cuda_concat_point_num);

}