import json
import logging
import random
import string
import time

import numpy as np

from easyfl.pb import common_pb2 as common_pb
from easyfl.utils.float import rounding

PREFIX_TASK_ID = "task"

CONFIGURATION = "configuration"

# clients
SELECTED_CLIENTS = 'selected_clients'
GROUPED_CLIENTS = 'grouped_clients'

# communication cost
DOWNLOAD_SIZE = 'download_size'
TRAIN_DOWNLOAD_SIZE = 'train_download_size'
TRAIN_UPLOAD_SIZE = 'train_upload_size'
TEST_DOWNLOAD_SIZE = 'test_download_size'
TEST_UPLOAD_SIZE = 'test_upload_size'

# distribute time
UPLOAD_TIME = "upload_time"
TRAIN_UPLOAD_TIME = "train_upload_time"
TEST_UPLOAD_TIME = "test_upload_time"
TRAIN_DISTRIBUTE_TIME = "train_distribute_time"
TEST_DISTRIBUTE_TIME = "test_distribute_time"

# time
ROUND_TIME = "round_time"
TRAIN_TIME = 'train_time'
TEST_TIME = 'test_time'
TRAIN_EPOCH_TIME = 'train_epoch_time'

# performance
TRAIN_ACCURACY = 'train_accuracy'
TRAIN_LOSS = 'train_loss'
AVG_TRAIN_LOSS = 'avg_train_loss'

TEST_ACCURACY = 'test_accuracy'
TEST_LOSS = 'test_loss'

TEST_LOCAL_ACCURACY = 'test_local_accuracy'
TEST_LOCAL_LOSS = 'test_local_loss'

# general
EXTRA = "extra"  # for not specifically defined metrics
DEFAULT_FOLDER = "metrics"
PREFIX_METRIC_ID = "metric"

logger = logging.getLogger(__name__)


class Metric(object):
    def __init__(self):
        self.metrics = {
            EXTRA: {}
        }

    def add(self, metric_name, metric_value, convert=True):
        """Add metrics. Add to "extra" if the metric is not predefined.
        """
        if self.predefined_metrics() and metric_name in self.predefined_metrics():
            if convert:
                metric_value = self._value_conversion(metric_value)
            self.metrics[metric_name] = metric_value
        elif metric_name == EXTRA:
            self.metrics[EXTRA].update(metric_value)
        else:
            self.metrics[EXTRA][metric_name] = metric_value

    def get(self, metric_name, default=0):
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        return default

    @classmethod
    def predefined_metrics(cls):
        return []

    @property
    def extra(self):
        """Retrieve extra information, not specifically defined metric, stored in the metrics.
        :return dictionary of metrics, return {} if extra stored.
        """
        return self.metrics[EXTRA]

    @staticmethod
    def _value_conversion(value):
        """Convert float to keep only 4 decimal points
        """
        if isinstance(value, float):
            value = np.around(value, 4)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
            value = rounding(value, 4)
        return value


class TaskMetric(object):
    def __init__(self, task_id, conf=None):
        self._task_id = task_id
        self._conf = conf

    def add(self, name, value):
        if name == CONFIGURATION:
            self.add_configuration(value)

    def add_configuration(self, conf):
        self._conf = conf

    @classmethod
    def from_sql(cls, sql_result):
        task_id, conf = sql_result
        conf = {} if conf == "" else json.loads(conf)
        return cls(task_id, conf)

    def to_sql_param(self):
        conf = json.dumps(self.configuration) if self.configuration is not None else ""
        return self.task_id, conf

    @property
    def task_id(self):
        return self._task_id

    @property
    def configuration(self):
        return self._conf

    def to_proto(self):
        return common_pb.TaskMetric(
            task_id=self.task_id,
            configuration=json.dumps(self.configuration)
        )

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.task_id, json.loads(proto.configuration))


class RoundMetric(Metric):
    """Metrics of a training round
    Note: testing related metrics may not be available in every round.
    """

    def __init__(self, task_id, round_id):
        super().__init__()
        self.task_id = task_id
        self.round_id = round_id

    @property
    def test_accuracy(self):
        return self.get(TEST_ACCURACY)

    @property
    def test_loss(self):
        return self.get(TEST_LOSS)

    @property
    def train_time(self):
        return self.get(TRAIN_TIME)

    @property
    def test_time(self):
        return self.get(TEST_TIME)

    @property
    def round_time(self):
        return self.get(ROUND_TIME)

    @property
    def train_distribute_time(self):
        return self.get(TRAIN_DISTRIBUTE_TIME, 0)

    @property
    def test_distribute_time(self):
        return self.get(TEST_DISTRIBUTE_TIME, 0)

    @property
    def train_upload_size(self):
        """Communication cost of uploading content from client to server
        """
        return self.get(TRAIN_UPLOAD_SIZE)

    @property
    def train_download_size(self):
        """Communication cost of distributing content from server to client
        """
        return self.get(TRAIN_DOWNLOAD_SIZE)

    @property
    def test_upload_size(self):
        """Communication cost of uploading content from client to server
        """
        return self.get(TEST_UPLOAD_SIZE)

    @property
    def test_download_size(self):
        """Communication cost of distributing content from server to client
        """
        return self.get(TEST_DOWNLOAD_SIZE)

    @property
    def communication_cost(self):
        return self.train_upload_size + self.train_download_size + self.test_upload_size + self.test_download_size

    @classmethod
    def predefined_metrics(cls):
        return [TEST_ACCURACY,
                TEST_LOSS,
                ROUND_TIME,
                TRAIN_TIME,
                TEST_TIME,
                TRAIN_DISTRIBUTE_TIME,
                TEST_DISTRIBUTE_TIME,
                TRAIN_UPLOAD_SIZE,
                TRAIN_DOWNLOAD_SIZE,
                TEST_UPLOAD_SIZE,
                TEST_DOWNLOAD_SIZE]

    @classmethod
    def from_sql(cls, sql_result):
        task_id = sql_result[0]
        round_id = sql_result[1]
        m = cls(task_id, round_id)
        metrics = cls.predefined_metrics()
        for name, value in zip(metrics, sql_result[2:-1]):
            m.add(name, value)
        m.add(EXTRA, json.loads(sql_result[-1]))
        return m

    def to_sql_param(self):
        return (self.task_id,
                self.round_id,
                self.test_accuracy,
                self.test_loss,
                self.round_time,
                self.train_time,
                self.test_time,
                self.train_distribute_time,
                self.test_distribute_time,
                self.train_upload_size,
                self.train_download_size,
                self.test_upload_size,
                self.test_download_size,
                json.dumps(self.extra))

    def to_proto(self):
        return common_pb.RoundMetric(
            task_id=self.task_id,
            round_id=self.round_id,
            test_accuracy=self.test_accuracy,
            test_loss=self.test_loss,
            round_time=self.round_time,
            train_time=self.train_time,
            test_time=self.test_time,
            train_distribute_time=self.train_distribute_time,
            test_distribute_time=self.test_distribute_time,
            train_upload_size=self.train_upload_size,
            train_download_size=self.train_download_size,
            test_upload_size=self.test_upload_size,
            test_download_size=self.test_download_size,
            extra=json.dumps(self.extra)
        )

    @classmethod
    def from_proto(cls, proto):
        m = cls(proto.task_id, proto.round_id)
        metrics = cls.predefined_metrics()
        values = [proto.test_accuracy,
                  proto.test_loss,
                  proto.round_time,
                  proto.train_time,
                  proto.test_time,
                  proto.train_distribute_time,
                  proto.test_distribute_time,
                  proto.train_upload_size,
                  proto.train_download_size,
                  proto.test_upload_size,
                  proto.test_download_size]
        for name, value in zip(metrics, values):
            m.add(name, value)

        if proto.extra:
            m.add(EXTRA, json.loads(proto.extra))

        return m


class ClientMetric(Metric):
    """Metrics for a client in a round of training.
    """

    def __init__(self, task_id, round_id, client_id):
        super().__init__()
        self.task_id = task_id
        self.round_id = round_id
        self.client_id = client_id

    @property
    def train_accuracy(self):
        return self.get(TRAIN_ACCURACY)

    @property
    def test_accuracy(self):
        return self.get(TEST_ACCURACY)

    @property
    def train_loss(self):
        return self.get(TRAIN_LOSS)

    @property
    def test_loss(self):
        return self.get(TEST_LOSS)

    @property
    def train_time(self):
        return self.get(TRAIN_TIME)

    @property
    def test_time(self):
        return self.get(TEST_TIME)

    @property
    def train_upload_time(self):
        return self.get(TRAIN_UPLOAD_TIME)

    @property
    def test_upload_time(self):
        return self.get(TEST_UPLOAD_TIME)

    @property
    def train_upload_size(self):
        return self.get(TRAIN_UPLOAD_SIZE)

    @property
    def train_download_size(self):
        return self.get(TRAIN_DOWNLOAD_SIZE)

    @property
    def test_upload_size(self):
        return self.get(TEST_UPLOAD_SIZE)

    @property
    def test_download_size(self):
        return self.get(TEST_DOWNLOAD_SIZE)

    @property
    def communication_cost(self):
        return self.train_upload_size + self.train_download_size + self.test_upload_size + self.test_download_size

    @classmethod
    def predefined_metrics(cls):
        return [TRAIN_ACCURACY,
                TRAIN_LOSS,
                TEST_ACCURACY,
                TEST_LOSS,
                TRAIN_TIME,
                TEST_TIME,
                TRAIN_UPLOAD_TIME,
                TEST_UPLOAD_TIME,
                TRAIN_UPLOAD_SIZE,
                TRAIN_DOWNLOAD_SIZE,
                TEST_UPLOAD_SIZE,
                TEST_DOWNLOAD_SIZE]

    @classmethod
    def from_sql(cls, sql_result):
        task_id = sql_result[0]
        round_id = sql_result[1]
        client_id = sql_result[2]
        m = cls(task_id, round_id, client_id)
        metrics = cls.predefined_metrics() + [EXTRA]
        for name, value in zip(metrics, sql_result[3:]):
            if name in [TRAIN_ACCURACY, TRAIN_LOSS, EXTRA]:
                value = json.loads(value)
            m.add(name, value)
        return m

    def to_sql_param(self):
        return (self.task_id,
                self.round_id,
                self.client_id,
                json.dumps(self.train_accuracy),
                json.dumps(self.train_loss),
                self.test_accuracy,
                self.test_loss,
                self.train_time,
                self.test_time,
                self.train_upload_time,
                self.test_upload_time,
                self.train_upload_size,
                self.train_download_size,
                self.test_upload_size,
                self.test_download_size,
                json.dumps(self.extra))

    def to_proto(self):
        return common_pb.ClientMetric(
            task_id=self.task_id,
            round_id=self.round_id,
            client_id=self.client_id,
            train_accuracy=self.train_accuracy,
            train_loss=self.train_loss,
            test_accuracy=self.test_accuracy,
            test_loss=self.test_loss,
            train_time=self.train_time,
            test_time=self.test_time,
            train_upload_time=self.train_upload_time,
            test_upload_time=self.test_upload_time,
            train_upload_size=self.train_upload_size,
            train_download_size=self.train_download_size,
            test_upload_size=self.test_upload_size,
            test_download_size=self.test_download_size,
            extra=json.dumps(self.extra)
        )

    @classmethod
    def from_proto(cls, proto):
        m = cls(proto.task_id, proto.round_id, proto.client_id)
        train_accuracy = [x for x in proto.train_accuracy]
        train_loss = [x for x in proto.train_loss]
        metrics = cls.predefined_metrics()
        values = [train_accuracy,
                  train_loss,
                  proto.test_accuracy,
                  proto.test_loss,
                  proto.train_time,
                  proto.test_time,
                  proto.train_upload_time,
                  proto.test_upload_time,
                  proto.train_upload_size,
                  proto.train_download_size,
                  proto.test_upload_size,
                  proto.test_download_size]
        for name, value in zip(metrics, values):
            m.add(name, value)

        if proto.extra:
            m.add(EXTRA, json.loads(proto.extra))

        return m

    def set_train_metrics(self, m):
        if self.is_same_metric(m):
            self.metrics[TRAIN_ACCURACY] = m.train_accuracy
            self.metrics[TRAIN_LOSS] = m.train_loss
            self.metrics[TRAIN_TIME] = m.train_time
            self.metrics[TRAIN_UPLOAD_TIME] = m.train_upload_time
            self.metrics[TRAIN_UPLOAD_SIZE] = m.train_upload_size
            self.metrics[TRAIN_DOWNLOAD_SIZE] = m.train_download_size

    def set_test_metrics(self, m):
        if self.is_same_metric(m):
            self.metrics[TEST_ACCURACY] = m.test_accuracy
            self.metrics[TEST_LOSS] = m.test_loss
            self.metrics[TEST_TIME] = m.test_time
            self.metrics[TEST_UPLOAD_TIME] = m.test_upload_time
            self.metrics[TEST_UPLOAD_SIZE] = m.test_upload_size
            self.metrics[TEST_DOWNLOAD_SIZE] = m.test_download_size

    def is_same_metric(self, m):
        return self.task_id == m.task_id and self.round_id == m.round_id and self.client_id == m.client_id

    @classmethod
    def merge_train_to_test_metrics(cls, train_metrics, test_metrics):
        """Merge train metrics to test_metrics
        """
        train_metrics_ = {m.client_id: m for m in train_metrics}
        for test_metric in test_metrics:
            client_id = test_metric.client_id
            if client_id in train_metrics_:
                test_metric.set_train_metrics(train_metrics_[client_id])
        return test_metrics


def generate_tid():
    length = 6
    letters = string.ascii_lowercase
    random.seed(time.time())
    result = "".join(random.choice(letters) for i in range(length))
    return "{}_{}".format(PREFIX_TASK_ID, result)
