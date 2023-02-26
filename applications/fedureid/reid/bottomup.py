import copy
import logging
import sys

import numpy as np
import torch

from .evaluators import Evaluator, extract_features
from .exclusive_loss import ExLoss
from .trainers import Trainer
from .utils.transform.transforms import TRANSFORM_VAL_LIST

logger = logging.getLogger(__name__)


class BottomUp:
    def __init__(self,
                 cid,
                 model,
                 batch_size,
                 eval_batch_size,
                 num_classes,
                 train_data,
                 test_data,
                 device,
                 embedding_feature_size=2048,
                 initial_epochs=20,
                 local_epochs=2,
                 step_size=16,
                 seed=0):
        self.cid = cid
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device

        self.seed = seed

        self.gallery_cam = None
        self.gallery_label = None
        self.query_cam = None
        self.query_label = None
        self.test_gallery_loader = None
        self.test_query_loader = None

        self.train_data = train_data
        self.test_data = test_data

        self.initial_epochs = initial_epochs
        self.local_epochs = local_epochs
        self.step_size = step_size

        self.embedding_feature_size = embedding_feature_size

        self.fixed_layer = False

        self.old_features = None
        self.feature_distance = 0

        self.criterion = ExLoss(self.embedding_feature_size, self.num_classes, t=10).to(device)

    def set_model(self, model, current_step):
        if current_step == 0:
            self.model = model.to(self.device)
        else:
            self.model.load_state_dict(model.state_dict())
            self.model = self.model.to(self.device)

    def train(self, step, dynamic_epoch=False):
        self.model = self.model.train()

        # adjust training epochs and learning rate
        epochs = self.initial_epochs if step == 0 else self.local_epochs

        init_lr = 0.1 if step == 0 else 0.01
        step_size = self.step_size if step == 0 else sys.maxsize

        logger.info("create train transform loader with batch size {}".format(self.batch_size))
        loader = self.train_data.loader(self.batch_size, self.cid, seed=self.seed, num_workers=6)

        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, self.model.CNN.base.parameters()))

        # we fixed the first three blocks to save GPU memory
        base_params_need_for_grad = filter(lambda p: p.requires_grad, self.model.CNN.base.parameters())

        # params of the new layers
        new_params = [p for p in self.model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        logger.info("number of epochs, {}: {}".format(self.cid, epochs))

        """ main training process """
        trainer = Trainer(self.model, self.criterion, self.device, fixed_layer=self.fixed_layer)
        for epoch in range(epochs):
            adjust_lr(epoch, step_size)
            stop_local_training = trainer.train(epoch, loader, optimizer, print_freq=max(5, len(loader) // 30 * 10))
            # Dynamically decide number of local epochs, based on conditions inside trainer.
            if step > 0 and dynamic_epoch and stop_local_training:
                logger.info("Dynamic epoch: in step {}, stop training {} after epoch {}".format(step, self.cid, epoch))
                break
        return self.model

    def evaluate(self, cid, model=None):
        # getting cid from argument is because of merged training
        if model is None:
            model = self.model
        model = model.eval()
        model = model.to(self.device)

        gallery_id = '{}_{}'.format(cid, 'gallery')
        query_id = '{}_{}'.format(cid, 'query')

        logger.info("create test transform loader with batch size {}".format(self.eval_batch_size))
        gallery_loader = self.test_data.loader(batch_size=self.eval_batch_size,
                                               client_id=gallery_id,
                                               shuffle=False,
                                               num_workers=6)
        query_loader = self.test_data.loader(batch_size=self.eval_batch_size,
                                             client_id=query_id,
                                             shuffle=False,
                                             num_workers=6)

        evaluator = Evaluator(model, self.test_data, query_id, gallery_id, self.device)
        rank1, rank5, rank10, mAP = evaluator.evaluate(query_loader, gallery_loader)
        return rank1, rank5, rank10, mAP

    # New get_new_train_data
    def relabel_train_data(self, device, unlabeled_ys, labeled_ys, nums_to_merge, size_penalty):
        # extract feature/classifier
        self.model = self.model.to(device)
        loader = self.train_data.loader(self.batch_size,
                                        self.cid,
                                        shuffle=False,
                                        num_workers=6,
                                        transform=TRANSFORM_VAL_LIST)
        features = extract_features(self.model, loader, device)

        # calculate cosine distance of features
        if self.old_features:
            similarities = []
            for old_feature, new_feature in zip(self.old_features, features):
                m = torch.cosine_similarity(old_feature, new_feature, dim=0)
                similarities.append(m)
            self.feature_distance = 1 - sum(similarities) / len(similarities)
            logger.info("Cosine distance between features, {}: {}".format(self.cid, self.feature_distance))
        self.old_features = copy.deepcopy(features)

        features = np.array([logit.numpy() for logit in features])

        # images of the same cluster
        label_to_images = {}
        for idx, l in enumerate(unlabeled_ys):
            label_to_images[l] = label_to_images.get(l, []) + [idx]

        dists = self.calculate_distance(features)

        idx1, idx2 = self.select_merge_data(features, unlabeled_ys, label_to_images, size_penalty, dists)

        unlabeled_ys = self.relabel_new_train_data(idx1, idx2, labeled_ys, unlabeled_ys, nums_to_merge)

        num_classes = len(np.unique(np.array(unlabeled_ys)))

        # change the criterion classifier
        self.criterion = ExLoss(self.embedding_feature_size, num_classes, t=10).to(device)

        return unlabeled_ys

    def relabel_new_train_data(self, idx1, idx2, labeled_ys, label, num_to_merge):
        correct = 0
        num_before_merge = len(np.unique(np.array(label)))
        # merge clusters with minimum dissimilarity
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
            if labeled_ys[idx1[i]] == labeled_ys[idx2[i]]:
                correct += 1
            num_merged = num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged == num_to_merge:
                break
        # set new label to the new training transform
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]

        self.train_data.data[self.cid]['y'] = label

        num_after_merge = len(np.unique(np.array(label)))
        logger.info("num of label before merge: {}, after merge: {}, sub: {}".format(
            num_before_merge, num_after_merge, num_before_merge - num_after_merge))
        return label

    def calculate_distance(self, u_feas):
        # calculate distance between features
        x = torch.from_numpy(u_feas)
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists

    def select_merge_data(self, u_feas, label, label_to_images, ratio_n, dists):
        dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas))))

        cnt = torch.FloatTensor([len(label_to_images[label[idx]]) for idx in range(len(u_feas))])
        dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))

        for idx in range(len(u_feas)):
            for j in range(idx + 1, len(u_feas)):
                if label[idx] == label[j]:
                    dists[idx, j] = 100000

        dists = dists.numpy()
        ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        idx1 = ind[0]
        idx2 = ind[1]
        return idx1, idx2
