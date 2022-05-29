import copy

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

from easyfl.models.model import BaseModel
from easyfl.models.resnet import ResNet18, ResNet50

SimSiam = "simsiam"
SimSiamNoSG = "simsiam_no_sg"
SimCLR = "simclr"
MoCo = "moco"
MoCoV2 = "moco_v2"
BYOL = "byol"
BYOLNoSG = "byol_no_sg"
BYOLNoEMA = "byol_no_ema"
BYOLNoEMA_NoSG = "byol_no_ema_no_sg"
BYOLNoPredictor = "byol_no_p"
Symmetric = "symmetric"
SymmetricNoSG = "symmetric_no_sg"

OneLayer = "1_layer"
TwoLayer = "2_layer"

RESNET18 = "resnet18"
RESNET50 = "resnet50"


def get_encoder(arch=RESNET18):
    return models.__dict__[arch]


def get_model(model, encoder_network, predictor_network=TwoLayer):
    mlp = False
    T = 0.07
    stop_gradient = True
    has_predictor = True
    if model == SymmetricNoSG:
        stop_gradient = False
        model = Symmetric
    elif model == SimSiamNoSG:
        stop_gradient = False
        model = SimSiam
    elif model == BYOLNoSG:
        stop_gradient = False
        model = BYOL
    elif model == BYOLNoPredictor:
        has_predictor = False
        model = BYOL
    elif model == MoCoV2:
        model = MoCo
        mlp = True
        T = 0.2

    if model == Symmetric:
        if encoder_network == RESNET50:
            return SymmetricModel(net=ResNet50(), stop_gradient=stop_gradient)
        else:
            return SymmetricModel(stop_gradient=stop_gradient)
    elif model == SimSiam:
        net = ResNet18()
        if encoder_network == RESNET50:
            net = ResNet50()
        return SimSiamModel(net=net, stop_gradient=stop_gradient)
    elif model == MoCo:
        net = ResNet18
        if encoder_network == RESNET50:
            net = ResNet50
        return MoCoModel(net=net, mlp=mlp, T=T)
    elif model == BYOL:
        net = ResNet18()
        if encoder_network == RESNET50:
            net = ResNet50()
        return BYOLModel(net=net, stop_gradient=stop_gradient, has_predictor=has_predictor,
                         predictor_network=predictor_network)
    elif model == SimCLR:
        net = ResNet18()
        if encoder_network == RESNET50:
            net = ResNet50()
        return SimCLRModel(net=net)
    else:
        raise NotImplementedError


def get_encoder_network(model, encoder_network, num_classes=10, projection_size=2048, projection_hidden_size=4096):
    if model in [MoCo, MoCoV2]:
        num_classes = 128

    if encoder_network == RESNET18:
        resnet = ResNet18(num_classes=num_classes)
    elif encoder_network == RESNET50:
        resnet = ResNet50(num_classes=num_classes)
    else:
        raise NotImplementedError

    if model in [Symmetric, SimSiam, BYOL, SymmetricNoSG, SimSiamNoSG, BYOLNoSG, SimCLR]:
        resnet.fc = MLP(resnet.feature_dim, projection_size, projection_hidden_size)
    if model == MoCoV2:
        resnet.fc = MLP(resnet.feature_dim, num_classes, resnet.feature_dim)

    return resnet


# ------------- SymmetricModel Model -----------------


class SymmetricModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            stop_gradient=True
    ):
        super().__init__()

        self.online_encoder = net
        self.online_encoder.fc = MLP(net.feature_dim, projection_size, projection_hidden_size)  # projector

        self.stop_gradient = stop_gradient

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))

    def forward(self, image_one, image_two):
        f = self.online_encoder
        z1, z2 = f(image_one), f(image_two)
        if self.stop_gradient:
            loss = D(z1, z2)
        else:
            loss = D_NO_SG(z1, z2)
        return loss


# ------------- SimSiam Model -----------------


class SimSiamModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            stop_gradient=True,
    ):
        super().__init__()

        self.online_encoder = net
        self.online_encoder.fc = MLP(net.feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        self.stop_gradient = stop_gradient

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))

    def forward(self, image_one, image_two):
        f, h = self.online_encoder, self.online_predictor
        z1, z2 = f(image_one), f(image_two)
        p1, p2 = h(z1), h(z2)
        if self.stop_gradient:
            loss = D(p1, z2) / 2 + D(p2, z1) / 2
        else:
            loss = D_NO_SG(p1, z2) / 2 + D_NO_SG(p2, z1) / 2

        return loss


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


def D_NO_SG(p, z, version='simplified'):  # negative cosine similarity without stop gradient
    if version == 'original':
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception


# ------------- BYOL Model -----------------


class BYOLModel(BaseModel):
    def __init__(
            self,
            net=ResNet18(),
            image_size=32,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # debug purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two):
        online_pred_one = self.online_encoder(image_one)
        online_pred_two = self.online_encoder(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        if self.stop_gradient:
            with torch.no_grad():
                if self.target_encoder is None:
                    self.target_encoder = self._get_target_encoder()
                target_proj_one = self.target_encoder(image_one)
                target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()

        else:
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            target_proj_one = self.target_encoder(image_one)
            target_proj_two = self.target_encoder(image_two)

        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two

        return loss.mean()


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# ------------- MoCo Model -----------------


class MoCoModel(BaseModel):
    def __init__(self, net=ResNet18, dim=128, K=4096, m=0.99, T=0.1, bn_splits=8, symmetric=True, mlp=False):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = net(num_classes=dim)
        self.encoder_k = net(num_classes=dim)

        if mlp:
            feature_dim = self.encoder_q.feature_dim
            self.encoder_q.fc = MLP(feature_dim, dim, feature_dim)
            self.encoder_k.fc = MLP(feature_dim, dim, feature_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def reset_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x, device):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k, device):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k, device)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss().to(device)(logits, labels)

        return loss, q, k

    def forward(self, im1, im2, device):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2, device)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1, device)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2, device)

        self._dequeue_and_enqueue(k)

        return loss


# ------------- SimCLR Model -----------------


class SimCLRModel(BaseModel):
    def __init__(self, net=ResNet18(), image_size=32, projection_size=2048, projection_hidden_size=4096):
        super().__init__()

        self.online_encoder = net
        self.online_encoder.fc = MLP(net.feature_dim, projection_size, projection_hidden_size)  # projector

    def forward(self, image):
        return self.online_encoder(image)
