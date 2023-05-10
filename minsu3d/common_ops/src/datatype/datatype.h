#ifndef DATATYPE_H
#define DATATYPE_H
#include <cstdint>
#include <array>
#include <vector>
#include <queue>

using Int = int32_t;

template <Int dimension> using Point = std::array<Int, dimension>;

class ConnectedComponent{
public:
    std::vector<Int> pt_idxs;

    float accum_x = 0;
    float accum_y = 0;
    float accum_z = 0;
    int16_t batch_idx = -1;
    int16_t cls_label;

    ConnectedComponent();
    void addPoint(Int pt_idx);
};

using ConnectedComponents = std::vector<ConnectedComponent>;

#endif //DATATYPE_H