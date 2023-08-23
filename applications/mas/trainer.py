import copy
import logging
import time
from collections import defaultdict

import scipy.stats
import torch

from utils import AverageMeter

from easyfl.distributed.distributed import CPU

logger = logging.getLogger(__name__)

LR_POLY = "poly"
LR_CUSTOM = "custom"


class Trainer:
    def __init__(self, cid, conf, train_loader, model, optimizer, criteria, device=CPU, checkpoint=None):
        self.cid = cid
        self.conf = conf
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.criteria = criteria
        self.loss_keys = list(self.criteria.keys())[1:]
        self.device = device
        # self.args = args

        self.progress_table = []
        # self.best_loss = 9e9
        self.stats = []
        self.start_epoch = 0
        self.loss_history = []
        self.encoder_trainable = None
        # self.code_archive = self.get_code_archive()
        if checkpoint:
            if 'progress_table' in checkpoint:
                self.progress_table = checkpoint['progress_table']
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            # if 'best_loss' in checkpoint:
            #     self.best_loss = checkpoint['best_loss']
            if 'stats' in checkpoint:
                self.stats = checkpoint['stats']
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']

        self.lr0 = self.conf.optimizer.lr
        self.lr = self.lr0

        self.ticks = 0
        self.last_tick = 0
        # self.loss_tracking_window = self.conf.loss_tracking_window_initial

        # estimated loss tracking window for each client, based on their dataset size, compared with original implementation.
        if self.conf.optimizer.lr_type == LR_CUSTOM:
            self.loss_tracking_window = len(train_loader) * self.conf.batch_size / 8
            self.maximum_loss_tracking_window = len(train_loader) * self.conf.batch_size / 2
            logger.info(
                f"Client {self.cid}: loss_tracking_window: {self.loss_tracking_window}, maximum_loss_tracking_window: {self.maximum_loss_tracking_window}")

    def train(self):
        self.encoder_trainable = [
            p for p in self.model.encoder.parameters() if p.requires_grad
        ]

        transference = {combined_task: [] for combined_task in self.loss_keys}
        for self.epoch in range(self.start_epoch, self.conf.local_epoch):
            current_learning_rate = get_average_learning_rate(self.optimizer)
            # Stop training when learning rate is smaller than minimum learning rate
            if current_learning_rate < self.conf.minimum_learning_rate:
                logger.info(f"Client {self.cid} stop local training because lr too small, lr: {current_learning_rate}.")
                break
            # Train for one epoch
            train_string, train_stats, epoch_transference = self.train_epoch()
            self.progress_table.append(train_string)
            self.stats.append(train_stats)

            for combined_task in self.loss_keys:
                transference[combined_task].append(epoch_transference[combined_task])

            # # evaluate on validation set
            # progress_string = train_string
            # loss, progress_string, val_stats = self.validate(progress_string)
            #
            # self.progress_table.append(progress_string)
            # self.stats.append((train_stats, val_stats))
        # Clean up to save memory
        del self.encoder_trainable
        self.encoder_trainable = None
        return transference

    def train_epoch(self):
        average_meters = defaultdict(AverageMeter)
        display_values = []
        for name, func in self.criteria.items():
            display_values.append(name)

        # Switch to train mode
        self.model.train()

        epoch_start_time = time.time()
        epoch_start_time2 = time.time()

        batch_num = 0
        num_data_points = len(self.train_loader) // self.conf.virtual_batch_multiplier
        if num_data_points > 10000:
            num_data_points = num_data_points // 5

        starting_learning_rate = get_average_learning_rate(self.optimizer)

        # Initialize task affinity dictionary
        epoch_transference = {}
        for combined_task in self.loss_keys:
            epoch_transference[combined_task] = {}
            for recipient_task in self.loss_keys:
                epoch_transference[combined_task][recipient_task] = 0.

        for i, (input, target) in enumerate(self.train_loader):
            input = input.to(self.device)
            for n, t in target.items():
                target[n] = t.to(self.device)

            # self.percent = batch_num / num_data_points
            if i == 0:
                epoch_start_time2 = time.time()

            loss_dict = None
            loss = 0
            
            self.optimizer.zero_grad()

            _train_batch_lookahead = self.conf.lookahead == 'y' and i % self.conf.lookahead_step == 0

            # Accumulate gradients over multiple runs of input
            for _ in range(self.conf.virtual_batch_multiplier):
                data_start = time.time()
                average_meters['data_time'].update(time.time() - data_start)
                # lookahead step 10
                if _train_batch_lookahead:
                    loss_dict2, loss2, batch_transference = self.train_batch_lookahead(input, target)
                else:
                    loss_dict2, loss2, batch_transference = self.train_batch(input, target)
                loss += loss2
                if loss_dict is None:
                    loss_dict = loss_dict2
                else:
                    for key, value in loss_dict2.items():
                        loss_dict[key] += value

            if _train_batch_lookahead:
                for combined_task in self.loss_keys:
                    for recipient_task in self.loss_keys:
                        epoch_transference[combined_task][recipient_task] += (
                                batch_transference[combined_task][recipient_task] / (len(self.train_loader) / self.conf.lookahead_step))

            # divide by the number of accumulations
            loss /= self.conf.virtual_batch_multiplier
            for key, value in loss_dict.items():
                loss_dict[key] = value / self.conf.virtual_batch_multiplier

            # do the weight updates and set gradients back to zero
            self.optimizer.step()

            self.loss_history.append(float(loss))
            ttest_p, z_diff = self.learning_rate_schedule()

            for name, value in loss_dict.items():
                try:
                    average_meters[name].update(value.data)
                except:
                    average_meters[name].update(value)

            elapsed_time_for_epoch = (time.time() - epoch_start_time2)
            eta = (elapsed_time_for_epoch / (batch_num + .2)) * (num_data_points - batch_num)
            if eta >= 24 * 3600:
                eta = 24 * 3600 - 1

            batch_num += 1

            current_learning_rate = get_average_learning_rate(self.optimizer)
            to_print = {
                'ep': f'{self.epoch}:',
                f'#/{num_data_points}': f'{batch_num}',
                'lr': '{0:0.3g}-{1:0.3g}'.format(starting_learning_rate, current_learning_rate),
                'eta': '{0}'.format(time.strftime("%H:%M:%S", time.gmtime(int(eta)))),
                'd%': '{0:0.2g}'.format(100 * average_meters['data_time'].sum / elapsed_time_for_epoch)
            }
            for name in display_values:
                meter = average_meters[name]
                to_print[name] = '{meter.avg:.4f}'.format(meter=meter)
            if batch_num < num_data_points - 1:
                to_print['ETA'] = '{0}'.format(
                    time.strftime("%H:%M:%S", time.gmtime(int(eta + elapsed_time_for_epoch))))
                to_print['ttest'] = '{0:0.3g},{1:0.3g}'.format(z_diff, ttest_p)


        epoch_time = time.time() - epoch_start_time
        stats = {
            'batches': num_data_points,
            'learning_rate': current_learning_rate,
            'epoch_time': epoch_time,
        }
        for name in display_values:
            meter = average_meters[name]
            stats[name] = meter.avg

        to_print['eta'] = '{0}'.format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))

        logger.info(f"Client {self.cid} training statistics: {stats}")
        return [to_print], stats, epoch_transference

    def train_batch(self, x, target):
        loss_dict = {}
        x = x.float()
        output = self.model(x)
        first_loss = None
        for c_name, criterion_fn in self.criteria.items():
            if first_loss is None:
                first_loss = c_name
            loss_dict[c_name] = criterion_fn(output, target)

        loss = loss_dict[first_loss].clone()
        loss = loss / self.conf.virtual_batch_multiplier

        if self.conf.fp16:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss_dict, loss, {}

    def train_batch_lookahead(self, x, target):
        loss_dict = {}
        x = x.float()
        output = self.model(x)
        first_loss = None
        for c_name, criterion_fun in self.criteria.items():
            if first_loss is None:
                first_loss = c_name
            loss_dict[c_name] = criterion_fun(output, target)

        loss = loss_dict[first_loss].clone()

        transference = {}
        for combined_task in self.loss_keys:
            transference[combined_task] = {}
        if self.conf.fp16:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            for combined_task in self.loss_keys:
                preds = self.lookahead(x, loss_dict[combined_task])
                first_loss = None
                for c_name, criterion_fun in self.criteria.items():
                    if first_loss is None:
                        first_loss = c_name
                    transference[combined_task][c_name] = (
                            (1.0 - (criterion_fun(preds, target) / loss_dict[c_name])) /
                            self.optimizer.state_dict()['param_groups'][0]['lr']
                    ).detach().cpu().numpy()
            self.optimizer.zero_grad()
            loss.backward()

        # Want to invert the dictionary so it's source_task => gradients on source task.
        rev_transference = {source: {} for source in transference}
        for grad_task in transference:
            for source in transference[grad_task]:
                if 'Loss' in source:
                    continue
                rev_transference[source][grad_task] = transference[grad_task][
                    source]
        return loss_dict, loss, copy.deepcopy(rev_transference)

    def lookahead(self, x, loss):
        self.optimizer.zero_grad()
        shared_params = self.encoder_trainable
        init_weights = [param.data for param in shared_params]
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True)

        # Compute updated params for the forward pass: SGD w/ 0.9 momentum + 1e-4 weight decay.
        opt_state = self.optimizer.state_dict()['param_groups'][0]
        weight_decay = opt_state['weight_decay']

        for param, g, param_id in zip(shared_params, grads, opt_state['params']):
            grad = g.clone()
            grad += param * weight_decay
            if 'momentum_buffer' not in opt_state:
                mom_buf = grad
            else:
                mom_buf = opt_state['momentum_buffer']
                mom_buf = mom_buf * opt_state['momentum'] + grad
            param.data = param.data - opt_state['lr'] * mom_buf

            grad = grad.cpu()
            del grad

        with torch.no_grad():
            output = self.model(x)

        for param, init_weight in zip(shared_params, init_weights):
            param.data = init_weight
        return output

    def learning_rate_schedule(self):
        # don't process learning rate if the schedule type is poly, which adjusted before training.
        if self.conf.optimizer.lr_type == LR_POLY:
            return 0, 0

        # don't reduce learning rate until the second epoch has ended.
        if self.epoch < 2:
            return 0, 0

        ttest_p = 0
        z_diff = 0

        wind = self.loss_tracking_window // (self.conf.batch_size * self.conf.virtual_batch_multiplier)
        if len(self.loss_history) - self.last_tick > wind:
            a = self.loss_history[-wind:-wind * 5 // 8]
            b = self.loss_history[-wind * 3 // 8:]
            # remove outliers
            a = sorted(a)
            b = sorted(b)
            a = a[int(len(a) * .05):int(len(a) * .95)]
            b = b[int(len(b) * .05):int(len(b) * .95)]
            length_ = min(len(a), len(b))
            a = a[:length_]
            b = b[:length_]
            z_diff, ttest_p = scipy.stats.ttest_rel(a, b, nan_policy='omit')

            if z_diff < 0 or ttest_p > .99:
                self.ticks += 1
                self.last_tick = len(self.loss_history)
                self.adjust_learning_rate()
                self.loss_tracking_window = min(self.maximum_loss_tracking_window, self.loss_tracking_window * 2)
        return ttest_p, z_diff

    def adjust_learning_rate(self):
        self.lr = self.lr0 * (0.50 ** self.ticks)
        self.set_learning_rate(self.lr)

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device


def get_average_learning_rate(optimizer):
    try:
        return optimizer.learning_rate
    except:
        s = 0
        for param_group in optimizer.param_groups:
            s += param_group['lr']
        return s / len(optimizer.param_groups)
