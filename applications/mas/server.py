import copy
import logging
import os
import shutil
import time
from collections import defaultdict

import torch

from dataset import DataPrefetcher
from losses import get_losses
from utils import AverageMeter
from easyfl.distributed.distributed import CPU
from easyfl.server.base import BaseServer, MODEL, DATA_SIZE
from easyfl.tracking import metric

logger = logging.getLogger(__name__)


class MASServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(MASServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        self.train_loader = None
        self.test_loader = None

        self._progress_table = []
        self._stats = []
        self._loss_history = []
        self._current_loss = 9e9
        self._best_loss = 9e9
        self._best_model = None
        self._client_models = []

    def aggregation(self):
        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())

        # Cache client models for saving
        self._client_models = [copy.deepcopy(m).cpu() for m in models]

        # Aggregation
        model = self.aggregate(models, weights)

        self.set_model(model, load_dict=True)

    def test_in_server(self, device=CPU):
        # Validation
        val_loader = self.val_data.loader(
            batch_size=max(self.conf.server.batch_size // 2, 1),
            shuffle=False,
            seed=self.conf.seed)

        test_results, stats, progress = self.test_fn(val_loader, self._model, device)
        self._current_loss = float(stats['Loss'])
        self._stats.append(stats)
        self._loss_history.append(self._current_loss)
        self._progress_table.append(progress)
        logger.info(f"Validation statistics: {stats}")

        # Test
        if self._current_round == self.conf.server.rounds - 1:
            test_loader = self.test_data.loader(
                batch_size=max(self.conf.server.batch_size // 2, 1),
                shuffle=False,
                seed=self.conf.seed)
            _, stats, progress_table = self.test_fn(test_loader, self._model, device)
            logger.info(f"Testing statistics of last round: {stats}")

            if self._current_loss <= self._best_loss:
                logger.info(f"Last round {self._current_round} is the best round")
            else:
                _, stats, progress_table = self.test_fn(test_loader, self._best_model, device)
                logger.info(f"Testing statistics of best model: {stats}")

        return test_results

    def test_fn(self, loader, model, device=CPU):
        model.eval()
        model.to(device)

        criteria = get_losses(self.conf.client.task_str, self.conf.client.rotate_loss, self.conf.client.task_weights)

        average_meters = defaultdict(AverageMeter)
        epoch_start_time = time.time()
        batch_num = 0
        num_data_points = len(loader)

        prefetcher = DataPrefetcher(loader, device)
        # torch.cuda.empty_cache()

        with torch.no_grad():
            for i in range(len(loader)):
                input, target = prefetcher.next()

                if batch_num == 0:
                    epoch_start_time2 = time.time()

                output = model(input)

                loss_dict = {}
                for c_name, criterion_fn in criteria.items():
                    loss_dict[c_name] = criterion_fn(output, target)

                batch_num = i + 1

                for name, value in loss_dict.items():
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                eta = ((time.time() - epoch_start_time2) / (batch_num + .2)) * (len(loader) - batch_num)
                to_print = {
                    f'#/{num_data_points}': '{0}'.format(batch_num),
                    'eta': '{0}'.format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                }
                for name in criteria.keys():
                    meter = average_meters[name]
                    to_print[name] = '{meter.avg:.4f}'.format(meter=meter)


        epoch_time = time.time() - epoch_start_time

        stats = {'batches': len(loader), 'epoch_time': epoch_time}

        for name in criteria.keys():
            meter = average_meters[name]
            stats[name] = meter.avg

        to_print['eta'] = '{0}'.format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        torch.cuda.empty_cache()

        test_results = {
            metric.TEST_ACCURACY: 0,
            metric.TEST_LOSS: float(stats['Loss']),
        }

        return test_results, stats, [to_print]

    def save_model(self):
        if self._do_every(self.conf.server.save_model_every, self._current_round, self.conf.server.rounds) and \
                self.is_primary_server():
            save_path = self.conf.server.save_model_path
            if save_path == "":
                save_path = os.path.join(os.getcwd(), "saved_models", "mas", self.conf.task_id)
            os.makedirs(save_path, exist_ok=True)
            if self.conf.server.save_model_every == 1:
                save_filename = f"{self.conf.task_id}_checkpoint.pth.tar"
            else:
                save_filename = f"{self.conf.task_id}_r_{self._current_round}_checkpoint.pth.tar"
            # save_path = os.path.join(save_path, f"{self.conf.task_id}_r_{self._current_round}_checkpoint.pth.tar")

            is_best = self._current_loss < self._best_loss
            self._best_loss = min(self._current_loss, self._best_loss)

            try:
                checkpoint = {
                    'round': self._current_round,
                    'info': {'machine': self.conf.distributed.init_method, 'GPUS': self.conf.gpu},
                    'args': self.conf,
                    'arch': self.conf.arch,
                    'state_dict': self._model.cpu().state_dict(),
                    'best_loss': self._best_loss,
                    'progress_table': self._progress_table,
                    'stats': self._stats,
                    'loss_history': self._loss_history,
                    'code_archive': self.get_code_archive(),
                    'client_models': [m.cpu().state_dict() for m in self._client_models]
                }
                self.save_checkpoint(checkpoint, False, save_path, save_filename)

                if is_best:
                    logger.info(f"Best validation loss at round {self._current_round}: {self._best_loss}")
                    self._best_model = copy.deepcopy(self._model)
                    self.save_checkpoint(None, True, save_path, save_filename)
                self.print_("Checkpoint saved at {}".format(save_path))
            except:
                self.print_('Save checkpoint failed...')


    def save_checkpoint(self, state, is_best, directory='', filename='checkpoint.pth.tar'):
        path = os.path.join(directory, filename)
        if is_best:
            best_path = os.path.join(directory, f"best_{self.conf.task_id}_checkpoint.pth.tar")
            shutil.copyfile(path, best_path)
        else:
            torch.save(state, path)

    def get_code_archive(self):
        file_contents = {}
        for i in os.listdir('.'):
            if i[-3:] == '.py':
                with open(i, 'r') as file:
                    file_contents[i] = file.read()
        return file_contents
