import torch
EPS = 1e-10
class_num = 19

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(predictions, target):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        predictions: a tensor of shape [B, num_classes, H, W].
        target: a tensor of shape [B, H, W] or [B, 1, H, W].
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    hist = torch.zeros((class_num, class_num))  # num of classes
    for t, p in zip(target, torch.argmax(predictions, dim=1)):
        hist += _fast_hist(t.flatten().cpu(), p.flatten().cpu(), class_num)

    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(predictions, target):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        predictions: a tensor of shape [B, num_classes, H, W].
        target: a tensor of shape [B, H, W] or [B, 1, H, W].
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    hist = torch.zeros((class_num, class_num))  # num of classes
    for t, p in zip(target, torch.argmax(predictions, dim=1)):
        hist += _fast_hist(t.flatten().cpu(), p.flatten().cpu(), class_num)

    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(predictions, target):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        predictions: a tensor of shape [B, num_classes, H, W].
        target: a tensor of shape [B, H, W] or [B, 1, H, W].
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    hist = torch.zeros((class_num, class_num))  # num of classes
    for t, p in zip(target, torch.argmax(predictions, dim=1)):
        hist += _fast_hist(t.flatten().cpu(), p.flatten().cpu(), class_num)

    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(predictions, target):
    """Computes the SĂ¸rensenâ€“Dice coefficient, a.k.a the F1 score.
    Args:
        predictions: a tensor of shape [B, num_classes, H, W].
        target: a tensor of shape [B, H, W] or [B, 1, H, W].
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    hist = torch.zeros((class_num, class_num))  # num of classes
    for t, p in zip(target, torch.argmax(predictions, dim=1)):
        hist += _fast_hist(t.flatten().cpu(), p.flatten().cpu(), class_num)

    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])