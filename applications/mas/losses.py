import collections

import torch

sl = 0
nl = 0
nl2 = 0
nl3 = 0
dl = 0
el = 0
rl = 0
kl = 0
tl = 0
al = 0
cl = 0
popular_offsets = collections.defaultdict(int)
batch_number = 0

TASKS = {
    's': 'segment_semantic',
    'd': 'depth_zbuffer',
    'n': 'normal',
    'N': 'normal2',
    'k': 'keypoints2d',
    'e': 'edge_occlusion',
    'r': 'reshading',
    't': 'edge_texture',
    'a': 'rgb',
    'c': 'principal_curvature'
}

LOSSES = {
    "ss_l": 's',
    "edge2d_l": 't',
    "depth_l": 'd',
    "norm_l": 'n',
    "key_l": 'k',
    "edge_l": 'e',
    "shade_l": 'r',
    "rgb_l": 'a',
    "pc_l": 'c',
}


def parse_tasks(task_str):
    tasks = []
    for char in task_str:
        tasks.append(TASKS[char])
    return tasks


def parse_loss_names(loss_names):
    tasks = []
    for l in loss_names:
        tasks.append(LOSSES[l])
    return tasks


def segment_semantic_loss(output, target, mask):
    global sl
    sl = torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1), ignore_index=0,
                                           reduction='mean')
    return sl


def normal_loss(output, target, mask):
    global nl
    nl = rotate_loss(output, target, mask, normal_loss_base)
    return nl


def normal_loss_simple(output, target, mask):
    global nl
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask.float()
    nl = out.mean()
    return nl


def rotate_loss(output, target, mask, loss_name):
    global popular_offsets
    target = target[:, :, 1:-1, 1:-1].float()
    mask = mask[:, :, 1:-1, 1:-1].float()
    output = output.float()
    val1 = loss = loss_name(output[:, :, 1:-1, 1:-1], target, mask)

    val2 = loss_name(output[:, :, 0:-2, 1:-1], target, mask)
    loss = torch.min(loss, val2)
    val3 = loss_name(output[:, :, 1:-1, 0:-2], target, mask)
    loss = torch.min(loss, val3)
    val4 = loss_name(output[:, :, 2:, 1:-1], target, mask)
    loss = torch.min(loss, val4)
    val5 = loss_name(output[:, :, 1:-1, 2:], target, mask)
    loss = torch.min(loss, val5)
    val6 = loss_name(output[:, :, 0:-2, 0:-2], target, mask)
    loss = torch.min(loss, val6)
    val7 = loss_name(output[:, :, 2:, 2:], target, mask)
    loss = torch.min(loss, val7)
    val8 = loss_name(output[:, :, 0:-2, 2:], target, mask)
    loss = torch.min(loss, val8)
    val9 = loss_name(output[:, :, 2:, 0:-2], target, mask)
    loss = torch.min(loss, val9)

    # lst = [val1,val2,val3,val4,val5,val6,val7,val8,val9]

    # print(loss.size())
    loss = loss.mean()
    # print(loss)
    return loss


def normal_loss_base(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    out = out.mean(dim=(1, 2, 3))
    return out


def normal2_loss(output, target, mask):
    global nl3
    diff = output.float() - target.float()
    out = torch.abs(diff)
    out = out * mask.float()
    nl3 = out.mean()
    return nl3


def depth_loss_simple(output, target, mask):
    global dl
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask.float()
    dl = out.mean()
    return dl


def depth_loss(output, target, mask):
    global dl
    dl = rotate_loss(output, target, mask, depth_loss_base)
    return dl


def depth_loss_base(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask.float()
    out = out.mean(dim=(1, 2, 3))
    return out


def edge_loss_simple(output, target, mask):
    global el

    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    el = out.mean()
    return el


def reshade_loss(output, target, mask):
    global rl
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    rl = out.mean()
    return rl


def keypoints2d_loss(output, target, mask):
    global kl
    kl = torch.nn.functional.l1_loss(output, target)
    return kl


def edge2d_loss(output, target, mask):
    global tl
    tl = torch.nn.functional.l1_loss(output, target)
    return tl


def auto_loss(output, target, mask):
    global al
    al = torch.nn.functional.l1_loss(output, target)
    return al


def pc_loss(output, target, mask):
    global cl
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    cl = out.mean()
    return cl


def edge_loss(output, target, mask):
    global el
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    el = out.mean()
    return el


def get_taskonomy_loss(losses):
    def taskonomy_loss(output, target):
        if 'mask' in target:
            mask = target['mask']
        else:
            mask = None

        sum_loss = None
        num = 0
        for n, t in target.items():
            if n in losses:
                o = output[n].float()
                this_loss = losses[n](o, t, mask)
                num += 1
                if sum_loss:
                    sum_loss = sum_loss + this_loss
                else:
                    sum_loss = this_loss

        return sum_loss  # /num # should not take average when using xception_taskonomy_new

    return taskonomy_loss


def get_losses(task_str, is_rotate_loss, task_weights=None):
    losses = {}
    criteria = {}

    if 's' in task_str:
        losses['segment_semantic'] = segment_semantic_loss
        criteria['ss_l'] = lambda x, y: sl

    if 'd' in task_str:
        if not is_rotate_loss:
            losses['depth_zbuffer'] = depth_loss_simple
        else:
            losses['depth_zbuffer'] = depth_loss
        criteria['depth_l'] = lambda x, y: dl

    if 'n' in task_str:
        if not is_rotate_loss:
            losses['normal'] = normal_loss_simple
        else:
            losses['normal'] = normal_loss
        criteria['norm_l'] = lambda x, y: nl
        # criteria['norm_l2']=lambda x,y : nl2

    if 'N' in task_str:
        losses['normal2'] = normal2_loss
        criteria['norm2'] = lambda x, y: nl3

    if 'k' in task_str:
        losses['keypoints2d'] = keypoints2d_loss
        criteria['key_l'] = lambda x, y: kl

    if 'e' in task_str:
        if not is_rotate_loss:
            losses['edge_occlusion'] = edge_loss_simple
        else:
            losses['edge_occlusion'] = edge_loss
        # losses['edge_occlusion']=edge_loss
        criteria['edge_l'] = lambda x, y: el

    if 'r' in task_str:
        losses['reshading'] = reshade_loss
        criteria['shade_l'] = lambda x, y: rl

    if 't' in task_str:
        losses['edge_texture'] = edge2d_loss
        criteria['edge2d_l'] = lambda x, y: tl

    if 'a' in task_str:
        losses['rgb'] = auto_loss
        criteria['rgb_l'] = lambda x, y: al

    if 'c' in task_str:
        losses['principal_curvature'] = pc_loss
        criteria['pc_l'] = lambda x, y: cl

    if task_weights:
        weights = [float(x) for x in task_weights.split(',')]
        losses2 = {}
        criteria2 = {}

        for l, w, c in zip(losses.items(), weights, criteria.items()):
            losses[l[0]] = lambda x, y, z, l=l[1], w=w: l(x, y, z) * w
            criteria[c[0]] = lambda x, y, c=c[1], w=w: c(x, y) * w

    taskonomy_loss = get_taskonomy_loss(losses)

    criteria2 = {'Loss': taskonomy_loss}
    for key, value in criteria.items():
        criteria2[key] = value
    criteria = criteria2

    return criteria
