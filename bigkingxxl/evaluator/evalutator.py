import cupy as cp
from cupyx.scipy.ndimage.measurements import label

def label_instances(input: cp.ndarray) -> cp.ndarray:
    """"Labels connected components with an running index per layer."""
    input_short = input
    if len(input.shape) > 3:
        input_short = input.reshape(-1, input.shape[2], input.shape[3])
    result = cp.zeros_like(input_short)
    for layer in range(input_short.shape[0]):
        result[layer, :, :], _ = label(input_short[layer, :, :].reshape((input_short.shape[1], input_short.shape[2])).astype(cp.int8))
    return result.reshape(input.shape)

"""
The following functions are taken from
https://www.kaggle.com/theoviel/competition-metric-map-iou/notebook
LICENSE: Apache 2
"""

def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(cp.unique(labels))
    pred_objects = len(cp.unique(y_pred))

    # Compute intersection between all objects
    intersection = cp.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = cp.histogram(labels, bins=true_objects)[0]
    area_pred = cp.histogram(y_pred, bins=pred_objects)[0]
    area_true = cp.expand_dims(area_true, -1)
    area_pred = cp.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    
    return iou[1:, 1:]  # exclude background


def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (cp array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = cp.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = cp.sum(matches, axis=1) == 0  # Missed objects
    false_positives = cp.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        cp.sum(true_positives),
        cp.sum(false_positives),
        cp.sum(false_negatives),
    )
    return tp, fp, fn

def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """


    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in cp.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(cp.mean(prec)))

    return cp.mean(cp.array(prec))

def split_components(array: cp.ndarray):
    unique_el = cp.unique(array)
    result = []
    for el in unique_el:
        result.append(cp.where(array == el, array, 0))
    return result