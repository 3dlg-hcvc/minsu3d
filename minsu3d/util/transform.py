import numpy as np
import scipy.ndimage
import scipy.interpolate


##############################
#           Rotation         #
##############################

def jitter(intensity=0.1):
    """
    params:
        the intensity of jittering
    return:
        3x3 jitter matrix
    """
    return np.eye(3,) + np.random.randn(3, 3) * intensity


def flip(axis=0, random=False):
    """
    flip the specified axis
    params:
        axis 0:x, 1:y, 2:z
    return:
        3x3 flip matrix
    """
    m = np.eye(3)
    m[axis][axis] *= -1 if not random else np.random.randint(0, 2) * 2 - 1
    return m


def rotz(t):
    """
    Rotation about the z-axis. counter-clockwise
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def elastic(x, gran, mag):
    """
    Refers to https://github.com/Jia-Research-Lab/PointGroup/blob/master/data/scannetv2_inst.py
    """
    blur0 = np.ones((3, 1, 1), dtype=np.float32) / 3
    blur1 = np.ones((1, 3, 1), dtype=np.float32) / 3
    blur2 = np.ones((1, 1, 3), dtype=np.float32) / 3

    bb = (np.abs(x).max(0) // gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype(np.float32) for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    return x + np.hstack([i(x)[:, None] for i in interp]) * mag


def crop(pc, max_num_point, scale):
    '''
    Crop the points such that there are at most max_num_points points
    '''
    pc_offset = pc.copy()
    valid_idxs = pc_offset.min(1) >= 0
    max_pc_range = np.full(shape=3, fill_value=scale, dtype=np.uint16)
    pc_range = pc.max(0) - pc.min(0)
    while np.count_nonzero(valid_idxs) > max_num_point:
        offset = np.clip(max_pc_range - pc_range + 0.001, None, 0) * np.random.rand(3)
        pc_offset = pc + offset
        valid_idxs = np.logical_and(pc_offset.min(1) >= 0, np.all(pc_offset < max_pc_range, axis=1))
        max_pc_range[:2] -= 32
    return pc_offset, valid_idxs