from __future__ import print_function, absolute_import

import logging
import os

import torch
from torch.backends import cudnn

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature

logger = logging.getLogger(__name__)


# extract features for fed transform format
def extract_features(model, data_loader, device, print_freq=1, metric=None):
    cudnn.benchmark = False
    model.eval()

    features = []
    logger.info("extracting features...")
    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        _fcs, pool5s = extract_cnn_feature(model, inputs)
        features.extend(pool5s)
    return features


def pairwise_distance(query_features, gallery_features, metric=None):
    x = torch.cat([f.unsqueeze(0) for f in query_features], 0)
    y = torch.cat([f.unsqueeze(0) for f in gallery_features], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query_ids, gallery_ids, query_cams, gallery_cams, cmc_topk=(1, 5, 10, 20)):
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

    # Compute all kinds of CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('Mean AP: {:4.2%}'.format(mAP))
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'
              .format(k,
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['market1501'][0], cmc_scores['market1501'][4], cmc_scores['market1501'][9], mAP


class Evaluator(object):
    def __init__(self, model, test_data, query_id, gallery_id, device, is_print=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.test_data = test_data
        self.device = device

        gallery_path = [(self.test_data.data[gallery_id]['x'][i],
                         self.test_data.data[gallery_id]['y'][i])
                        for i in range(len(self.test_data.data[gallery_id]['y']))]
        query_path = [(self.test_data.data[query_id]['x'][i],
                       self.test_data.data[query_id]['y'][i])
                      for i in range(len(self.test_data.data[query_id]['y']))]
        gallery_cam, gallery_label = get_id(gallery_path)
        self.gallery_cam = gallery_cam
        self.gallery_label = gallery_label
        query_cam, query_label = get_id(query_path)
        self.query_cam = query_cam
        self.query_label = query_label

    def evaluate(self, query_loader, gallery_loader, metric=None):
        query_features = extract_features(self.model, query_loader, self.device)
        gallery_features = extract_features(self.model, gallery_loader, self.device)
        distmat = pairwise_distance(query_features, gallery_features, metric=metric)
        return evaluate_all(distmat, self.query_label, self.gallery_label, self.query_cam, self.gallery_cam)


def get_id(img_path):
    camera_id = []
    labels = []
    for p, v in img_path:
        filename = os.path.basename(p)
        if filename[:3] != 'cam':
            label = filename[0:4]
            camera = filename.split('c')[1]
            camera = camera.split('s')[0]
        else:
            label = filename.split('_')[2]
            camera = filename.split('_')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels
