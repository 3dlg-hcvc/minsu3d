import numpy as np


def get_nms_instances(cross_ious, scores, threshold):
    """ non max suppression for 3D instance proposals based on cross ious and scores

    Args:
        ious (np.array): cross ious, (n, n)
        scores (np.array): scores for each proposal, (n,)
        threshold (float): iou threshold

    Returns:
        np.array: idx of picked instance proposals
    """
    ixs = np.argsort(-scores)  # descending order
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        ious = cross_ious[i, ixs[1:]]
        remove_ixs = np.where(ious > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        
    return np.array(pick, dtype=np.int32)
