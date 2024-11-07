import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


from collections import OrderedDict
import numpy as np

def compute_per_frame_AP(probs, labels, classes):
    if not isinstance(probs, torch.Tensor):
        probs = torch.Tensor(probs).cuda()
    
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels).long().cuda()

    n_classes = len(classes)

    gt = torch.zeros_like(probs).int()
    gt[torch.LongTensor(range(gt.size(0))), labels] = 1
    gt = gt.cpu().numpy()
    gt = gt.T

    probs = probs.cpu().numpy()
    probs = probs.T

    assert gt.shape == probs.shape
    assert gt.shape[0] == n_classes

    result = OrderedDict()
    result['per_class_AP'] = OrderedDict()
    result['per_class_cAP'] = OrderedDict()

    all_cls_ap, all_cls_acp = list(), list()

    for i in range(1, n_classes): # from 1 to ignore the background class
        this_cls_prob = probs[i, :]
        this_cls_gt = gt[i, :]

        if np.sum(this_cls_gt == 1) == 0: # ignore empty class
            continue

        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)

        this_cls_ap = psum / np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        result['per_class_AP'][i] = this_cls_ap
        result['per_class_cAP'][i] = this_cls_acp
        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    result['mAP'] = sum(all_cls_ap) / len(all_cls_ap)
    result['mcAP'] = sum(all_cls_acp) / len(all_cls_acp)
    print('mAP: {}'.format(result['mAP']))
    print('per_class_AP: {}'.format(result['per_class_AP']))
    print('mcAP: {}'.format(result['mcAP']))
    print('per_class_cAP: {}'.format(result['per_class_cAP']))
    return result


def evaluate(model, test_loader, device):
    sim_logits = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, (rgb, label, _) in enumerate(tqdm(test_loader)):
            rgb = rgb.to(device)
            label = label.to(device)

            similarity, _ = model(rgb)
            
            sim_logits.append(similarity.cpu())
            labels.append(label.cpu())

    sim, gt = sim_logits[0], labels[0]
    for i in range(1, len(sim_logits)): 
        sim = torch.cat((sim, sim_logits[i]), 0)
        gt = torch.cat((gt, labels[i]), 0)


    res = compute_per_frame_AP(sim, gt, test_loader.dataset.classes)
    return res