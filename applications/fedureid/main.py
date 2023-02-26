import argparse
import os
import time

import torch._utils

import easyfl
from dataset import prepare_train_data, prepare_test_data
from reid.bottomup import *
from reid.models.model import BUCModel
from easyfl.client.base import BaseClient
from easyfl.distributed import slurm
from easyfl.distributed.distributed import CPU
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.tracking import metric

logger = logging.getLogger(__name__)


LOCAL_TEST = "local_test"
GLOBAL_TEST = "global_test"

RELABEL_LOCAL = "local"
RELABEL_GLOBAL = "global"


class FedUReIDClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0,
                 is_remote=False, local_port=23000, server_addr="localhost:22999", tracker_addr="localhost:12666"):
        super(FedUReIDClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time,
                                                 is_remote, local_port, server_addr, tracker_addr)
        logger.info(conf)
        self.conf = conf
        self.current_step = -1

        self._local_model = None  # for caching local model in testing
        self.gallery_cam = None
        self.gallery_label = None
        self.query_cam = None
        self.query_label = None
        self.test_gallery_loader = None
        self.test_query_loader = None

        self.train_data = train_data
        self.test_data = test_data

        self.labeled_ys = self.train_data.data[self.cid]['y']
        self.unlabeled_ys = [i for i in range(len(self.labeled_ys))]
        # initialize unlabeled transform
        self.train_data.data[self.cid]['y'] = self.unlabeled_ys

        num_classes = len(np.unique(np.array(self.unlabeled_ys)))

        merge_percent = conf.buc.merge_percent
        self.nums_to_merge = int(num_classes * conf.buc.merge_percent)
        self.steps = int(1 / merge_percent) - 1

        self.buc = BottomUp(cid=self.cid,
                            model=self.model,  # model is None
                            batch_size=conf.buc.batch_size,
                            eval_batch_size=conf.buc.eval_batch_size,
                            num_classes=num_classes,
                            train_data=self.train_data,
                            test_data=self.test_data,
                            device=device,
                            initial_epochs=conf.buc.initial_epochs,
                            local_epochs=conf.buc.local_epochs,
                            embedding_feature_size=conf.buc.feature_size,
                            seed=conf.seed)

    def train(self, conf, device=CPU):
        logger.info("--------- training -------- cid: {}, on {}".format(self.cid, device))

        start_time = time.time()

        step_to_upload = self.current_step + conf.buc.upload_frequency
        total_steps = self.steps

        while self.current_step < step_to_upload:
            self.current_step += 1
            logger.info("current step: {}".format(self.current_step))
            logger.info("training transform amount: {}".format(len(self.unlabeled_ys)))

            if conf.buc.relabel == RELABEL_GLOBAL:
                if self.current_step > 0:
                    logger.info("-------- bottom-up clustering: relabel train transform with global aggregated model")
                    self.unlabeled_ys = self.buc.relabel_train_data(device,
                                                                    self.unlabeled_ys,
                                                                    self.labeled_ys,
                                                                    self.nums_to_merge,
                                                                    size_penalty=conf.buc.size_penalty)
                    self.train_data.data[self.cid]['y'] = self.unlabeled_ys

            self.buc.set_model(self.model, self.current_step)
            model = self.buc.train(self.current_step, conf.buc.dynamic_epoch)
            self._local_model = copy.deepcopy(self.model)
            self.model.load_state_dict(model.state_dict())

            rank1, rank5, rank10, mAP = self.buc.evaluate(self.cid)
            logger.info("Local test {}, step {}, mAP: {:4.2%}, Rank@1: {:4.2%}, Rank@5: {:4.2%}, Rank@10: {:4.2%}"
                        .format(self.cid, self.current_step, mAP, rank1, rank5, rank10))

            if self.current_step == total_steps:
                logger.info("Total steps just reached, force global update")
                break

            # get new train transform for the next iteration
            if self.current_step > total_steps:
                logger.info("Total steps reached, skip relabeling")
                continue

            if conf.buc.relabel == RELABEL_LOCAL:
                logger.info("-------- bottom-up clustering: relabel train transform with local trained model")
                self.unlabeled_ys = self.buc.relabel_train_data(device,
                                                                self.unlabeled_ys,
                                                                self.labeled_ys,
                                                                self.nums_to_merge,
                                                                size_penalty=conf.buc.size_penalty)

                self.train_data.data[self.cid]['y'] = self.unlabeled_ys

        self.save_model(LOCAL_TEST, device)
        self.current_round_time = time.time() - start_time
        logger.info("Local training time {}".format(self.current_round_time))
        self.track(metric.TRAIN_TIME, self.current_round_time)

        self.model = self.model.to(device)

    def test(self, conf, device=CPU):
        rank1 = 0
        if conf.buc.global_evaluation:
            logger.info("-------- evaluation -------- {}: {}".format(GLOBAL_TEST, self.cid))
            rank1, rank5, rank10, mAP = self.buc.evaluate(self.cid, self.model)
            logger.info("Global test {}, step {}, mAP: {:4.2%}, Rank@1: {:4.2%}, Rank@5: {:4.2%}, Rank@10: {:4.2%}"
                        .format(self.cid, self.current_step, mAP, rank1, rank5, rank10))
            self.save_model(GLOBAL_TEST, device)

        self._upload_holder = server_pb.UploadContent(
            data=codec.marshal(server_pb.Performance(accuracy=rank1, loss=0)),  # loss not applicable
            type=common_pb.DATA_TYPE_PERFORMANCE,
            data_size=len(self.train_data.data[self.cid]['x']),
        )

    def save_model(self, typ=LOCAL_TEST, device=CPU):
        path = os.path.join(os.getcwd(), "saved_models")
        if not os.path.exists(path):
            os.makedirs(path)
        if typ == GLOBAL_TEST:
            save_path = os.path.join(path, "{}_global_model_{}.pth".format(self.current_step, time.time()))
            if device == 0 or device == CPU:
                torch.save(self.model.cpu().state_dict(), save_path)
        else:
            save_path = os.path.join(path, "{}_{}_local_model_{}.pth".format(self.current_step, self.cid, time.time()))
            torch.save(self.model.cpu().state_dict(), save_path)
        logger.info("save model {}".format(save_path))


def get_merge_percent(num_images, num_identities, rounds):
    nums_to_merge = int((num_images - num_identities) / rounds)
    merge_percent = nums_to_merge / num_images
    return merge_percent, nums_to_merge


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default="datasets/fedreid_data")
    parser.add_argument("--datasets", nargs="+", default=["ilids"])
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--upload_frequency', type=int, default=1, help='frequency of upload for aggregation')
    parser.add_argument('--merge_percent', type=float, default=0.05, help='merge percentage of each step')
    parser.add_argument('--steps', type=int, default=0, help='steps to decide merge percent')
    parser.add_argument('--initial_epochs', type=int, default=20, help='local epochs for first step/round')
    parser.add_argument('--local_epochs', type=int, default=1, help='local epochs after first step/round')
    parser.add_argument('--dynamic_epoch', default=False, action='store_true', help='dynamic local epochs')
    parser.add_argument('--relabel', type=str, default='local', help='use "local" or "global" model to relabel')
    parser.add_argument('--merge', default=False, action='store_true')
    args = parser.parse_args()

    print("args:", args)

    # MAIN
    train_data = prepare_train_data(args.datasets, args.data_dir)
    test_data = prepare_test_data(args.datasets, args.data_dir)
    easyfl.register_dataset(train_data, test_data)
    easyfl.register_model(BUCModel)
    easyfl.register_client(FedUReIDClient)

    # configurations
    global_evaluation = False
    if args.steps:
        rounds = args.steps
    else:
        rounds = int(1 / args.merge_percent)

    config = {
        "server": {
            "rounds": rounds,
        },
        "client": {
            "buc": {
                "global_evaluation": global_evaluation,
                "relabel": args.relabel,
                "initial_epochs": args.initial_epochs,
                "local_epochs": args.local_epochs,
                "dynamic_epoch": args.dynamic_epoch,
                "batch_size": args.batch_size,
                "upload_frequency": args.upload_frequency,
                "merge_percent": args.merge_percent,
                "steps": args.steps,
            },
            "datasets": args.datasets,
        }
    }

    # For distributed training over multiple GPUs only
    try:
        rank, local_rank, world_size, host_addr = slurm.setup()
        global_evaluation = True if world_size > 1 else False
        config["client"]["buc"]["global_evaluation"] = global_evaluation
        distributed_config = {
            "gpu": world_size,
            "distributed": {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "init_method": host_addr,
                "backend": "nccl",
            },
        }
        config.update(distributed_config)
    except KeyError:
        pass
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
    config = easyfl.load_config(config_file, config)

    print("config:", config)
    easyfl.init(config, init_all=True)
    easyfl.run()
