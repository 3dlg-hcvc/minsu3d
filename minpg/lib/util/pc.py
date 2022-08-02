import numpy as np


# Point cloud IO
from plyfile import PlyData, PlyElement

# Mesh IO

########################
# point cloud sampling #
########################

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


##################
# Point cloud IO #
##################


def write_ply_rgb(points, colors, filename, text=True, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    ele = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ele], text=text).write(filename)


def write_ply_rgb_face(points, colors, faces, filename, text=True):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    faces = [((faces[i,0], faces[i,1], faces[i,2]),) for i in range(faces.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    face = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(face, 'face', comments=['faces'])
    PlyData([ele1, ele2], text=text).write(filename)


def write_ply_rgb_annotated(points, colors, labels, instanceIds, filename, text=True):
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    vertex_label = np.array(labels, dtype=[('label', 'i4')])
    vertex_instance = np.array(instanceIds, dtype=[('instance', 'i4')])
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(vertex_label, 'label', comments=['labels'])
    ele3 = PlyElement.describe(vertex_instance, 'instanceId', comments=['instanceIds'])
    PlyData([ele1, ele2, ele3], text=text).write(filename)


def write_ply_colorful(points, labels, filename, num_classes=None, colormap=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as ply file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))

    vertex = []
    if colormap is None:
        colormap = [ply.cm.jet(i/float(num_classes)) for i in range(num_classes)]

    for i in range(N):
        if labels[i] >= 0:
            c = colormap[labels[i]]
        else:
            c = [0, 0, 0]
        if c[0] < 1:
            c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    vertex_label = np.array(labels, dtype=[('label', 'i4')])

    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(vertex_label, 'label', comments=['labels'])
    PlyData([ele1, ele2], text=True).write(filename)
    