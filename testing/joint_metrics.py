import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data import DataManger

def joint_metrics(logger, config, datamanager: DataManger):
    gallery_feature, gallery_label, gallery_cam, gallery_frames = torch.load(os.path.join(
        config['testing']['ouput_dir'], 'gallery_embeddings.pt'), map_location='cpu')

    query_feature, query_label, query_cam, query_frame = torch.load(os.path.join(
        config['testing']['ouput_dir'], 'query_embeddings.pt'), map_location='cpu')

    distribution = torch.load(os.path.join(config['testing']['ouput_dir'], 'distribution.pt'), map_location='cpu')

    # normalize feature vector
    norm = query_feature.norm(p=2, dim=1, keepdim=True)
    query_feature = query_feature.div(norm.expand_as(query_feature))

    norm = gallery_feature.norm(p=2, dim=1, keepdim=True)
    gallery_feature = gallery_feature.div(norm.expand_as(gallery_feature))
    
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in tqdm(range(len(query_label))):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], query_frame[i],
                                   gallery_feature, gallery_label, gallery_cam, gallery_frames,
                                   distribution, config)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC/len(query_label)
    logger.info('with Spatial-temporal: top1: {} top5: {} top10: {} mAP: {}'.format(CMC[0], CMC[4], CMC[9], ap/len(query_label)))

def evaluate(query_f, query_l, query_c, query_fr, gallery_f, gallery_l, gallery_c, gallery_fr, distribution, config):
    """ Compute mAP and rank1 rank5 for each query
    Args:
        query_f:
        query_l:
        query_c:
        query_fr:
        gallery_f:
        gallery_l:
        gallery_c:
        gallery_fr:
    Return:
        Ap, CMC
    """
    interval = config['testing']['interval']
    lambda1 = config['testing']['lambda1']
    lambda2 = config['testing']['lambda2']
    gamma1 = config['testing']['gamma1']
    gamma2 = config['testing']['gamma2']

    score = np.dot(gallery_f, query_f)

    score_st = np.zeros(len(gallery_c))
    for i in range(len(gallery_c)):
        if query_fr > gallery_fr[i]:
            diff = query_fr-gallery_fr[i]
            hist_ = int(diff/interval)
            pr = distribution[query_c-1][gallery_c[i]-1][hist_]
        else:
            diff = gallery_fr[i] - query_fr
            hist_ = int(diff/interval)
            pr = distribution[gallery_c[i]-1][query_c-1][hist_]
        score_st[i] = pr

    score = (1/(1 + lambda1 * np.exp(-gamma1 * score))) * \
        (1/(1 + lambda2 * np.exp(-gamma2 * score_st)))

    index = np.argsort(-score)  # from large to small

    query_index = np.argwhere(gallery_l == query_l)
    camera_index = np.argwhere(gallery_c == query_c)
    
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.intersect1d(query_index, camera_index)

    return compute_mAP(index, good_index, junk_index)

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + (old_precision + precision)/2
        # ap += precision
    ap = ap*1.0/ngood
    return ap, cmc