# Modified from https://github.com/facebookresearch/votenet/blob/main/utils/eval_det.py
import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: xyzxyz
    Returns:
        iou
    """

    max_a = box_a[3:]
    max_b = box_b[3:]
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3]
    min_b = box_b[0:3]
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = (box_a[3:6] - box_a[:3]).prod()
    vol_b = (box_b[3:6] - box_b[:3]).prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for a
    single class.
    Input:
        pred: map of {img_id: [(sphere, score)]} where sphere is numpy array
        gt: map of {img_id: [sphere]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'sphere': sphere list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        sphere = np.array(gt[img_id], dtype=np.float32)
        det = np.zeros(shape=len(sphere), dtype=bool)
        npos += len(sphere)
        class_recs[img_id] = {'sphere': sphere, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'sphere': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for sphere, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(sphere)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd, dtype=bool)
    fp = np.zeros(nd, dtype=bool)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R['sphere'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = True
                R['det'][jmax] = 1
            else:
                fp[d] = True
        else:
            fp[d] = True

    # compute precision recall
    fp = np.cumsum(fp, dtype=np.uint32)
    tp = np.cumsum(tp, dtype=np.uint32)
    rec = tp.astype(np.float32) / npos
    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for
    multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, sphere, score)]}
        gt_all: map of {img_id: [(classname, sphere)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, sphere, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((sphere, score))
    for img_id in gt_all.keys():
        for classname, sphere in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(sphere)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname],
                                                                      gt[classname], ovthresh,
                                                                      use_07_metric, get_iou_func)

    return rec, prec, ap


def eval_sphere(pred_all, gt_all, ovthresh, use_07_metric=False, get_iou_func=get_iou):
    """Generic functions to compute precision/recall for object detection for
    multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, sphere, score)]}
        gt_all: map of {img_id: [(classname, sphere)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, sphere, score in pred_all[img_id]:

            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((sphere, score))
    for img_id in gt_all.keys():
        for classname, sphere in gt_all[img_id]:

            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(sphere)

    rec = {}
    prec = {}
    ap = {}
    tmp_list = [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
                        for classname in gt.keys() if classname in pred]
    ret_values = []
    for item in tmp_list:
        ret_values.append(eval_det_cls_wrapper(item))

    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0

    return rec, prec, ap


def get_gt_bbox(instance_cls, instance_bboxes, ignored_label):
    gt_bbox = []
    assert instance_cls.shape[0] == instance_bboxes.shape[0]
    for i in range(instance_cls.shape[0]):
        if instance_cls[i] == ignored_label:
            continue
        gt_bbox.append((instance_cls[i], instance_bboxes[i]))
    return gt_bbox


def evaluate_bbox_acc(all_preds, all_gts, class_names, print_result):
    iou_thresholds = [0.25, 0.5]  # adjust threshold here
    pred_all = {}
    gt_all = {}
    for i in range(len(all_preds)):
        img_id = all_preds[i][0]["scan_id"]
        pred_all[img_id] = [(pred["label_id"] - 1, pred["pred_bbox"], pred["conf"]) for pred in all_preds[i]]
        gt_all[img_id] = all_gts[i]
    bbox_aps = {}
    for iou_threshold in iou_thresholds:
        eval_res = eval_sphere(pred_all, gt_all, ovthresh=iou_threshold)
        aps = list(eval_res[-1].values())
        m_ap = np.mean(aps)
        eval_res[-1]["avg"] = m_ap
        bbox_aps[f"all_bbox_ap_{iou_threshold}"] = eval_res[-1]
    if print_result:
        print_results(bbox_aps, class_names)
    return bbox_aps


def print_results(bbox_aps, class_names):
    sep = ''
    col1 = ':'
    lineLen = 40

    print()
    print('#' * lineLen)
    line = ''
    line += '{:<15}'.format('what') + sep + col1
    line += '{:>12}'.format('BBox_AP_50%') + sep
    line += '{:>12}'.format('BBOX_AP_25%') + sep
    print(line)
    print('#' * lineLen)

    for (li, label_name) in enumerate(class_names):
        ap_50o = bbox_aps['all_bbox_ap_0.5'][li]
        ap_25o = bbox_aps['all_bbox_ap_0.25'][li]
        line = '{:<15}'.format(label_name) + sep + col1
        line += sep + '{:>12.3f}'.format(ap_50o) + sep
        line += sep + '{:>12.3f}'.format(ap_25o) + sep
        print(line)

    all_ap_50o = bbox_aps['all_bbox_ap_0.5']["avg"]
    all_ap_25o = bbox_aps['all_bbox_ap_0.25']["avg"]

    print('-' * lineLen)
    line = '{:<15}'.format('average') + sep + col1
    line += '{:>12.3f}'.format(all_ap_50o) + sep
    line += '{:>812.3f}'.format(all_ap_25o) + sep

    print(line)
    print('#' * lineLen)
    print()
