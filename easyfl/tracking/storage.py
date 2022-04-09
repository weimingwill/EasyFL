import logging
import os
import random
import sqlite3
import time

from easyfl.communication import grpc_wrapper
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import tracking_service_pb2 as tracking_pb
from easyfl.protocol.codec import marshal

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_DB = "easyfl.db"

STORAGE_SQLITE = "sqlite"
STORAGE_REMOTE = "remote"

TYPE_ROUND = "round"
TYPE_CLIENT = "client"

DEFAULT_TIMEOUT = 10

CREATE_TASK_METRIC_SQL = '''
CREATE TABLE IF NOT EXISTS task_metric 
(task_id       CHAR(50)    NOT NULL PRIMARY KEY,
config         TEXT);'''

CREATE_ROUND_METRIC_SQL = '''
CREATE TABLE IF NOT EXISTS round_metric 
(task_id              CHAR(50)    NOT NULL,
round_id              INT         NOT NULL,
accuracy              REAL        NOT NULL,
loss                  REAL        NOT NULL,
round_time            REAL        NOT NULL,
train_time            REAL        NOT NULL,
test_time             REAL        NOT NULL,
train_distribute_time REAL,
test_distribute_time  REAL,
train_upload_size     REAL,
train_download_size   REAL,
test_upload_size      REAL,
test_download_size    REAL,
extra                 TEXT,
PRIMARY KEY (task_id, round_id));'''

CREATE_CLIENT_METRIC_SQL = '''
CREATE TABLE IF NOT EXISTS client_metric 
(task_id            CHAR(50)    NOT NULL,
round_id            INT         NOT NULL,
client_id           CHAR(20)    NOT NULL,
train_accuracy      TEXT        ,
train_loss          TEXT        ,
test_accuracy       REAL        ,
test_loss           REAL        ,
train_time          REAL        ,
test_time           REAL        ,
train_upload_time   REAL        ,
test_upload_time    REAL        ,
train_upload_size   REAL        ,
train_download_size REAL        ,
test_upload_size    REAL        ,
test_download_size  REAL        ,
extra               TEXT        ,
PRIMARY KEY (task_id, round_id, client_id));'''


def get_store(path=None, address=None):
    if address:
        return RemoteStorage(address)
    else:
        return SqliteStorage(path)


def get_storage_type(is_remote=True):
    if is_remote:
        return STORAGE_REMOTE
    else:
        return STORAGE_SQLITE


class SqliteStorage(object):
    """SqliteStorage uses sqlite to save tracking metrics

    """

    def __init__(self, database=None):
        if database is None:
            database = os.path.join(os.getcwd(), "tracker", DEFAULT_SQLITE_DB)
        self._conn = sqlite3.connect(database, check_same_thread=False)
        self.setup()

    def __del__(self):
        self._conn.close()

    def setup(self):
        with self._conn:
            try:
                self._retry_execute(CREATE_TASK_METRIC_SQL)
                logger.info("Setup task metric table")
                self._retry_execute(CREATE_ROUND_METRIC_SQL)
                logger.info("Setup round metric table")
                self._retry_execute(CREATE_CLIENT_METRIC_SQL)
                logger.info("Setup client metric table")
            except sqlite3.OperationalError as e:
                logger.error(f"Failed to setup table, error: {e}")

    # ------------------ store metrics ------------------

    def store_task_metric(self, metric):
        sql = "INSERT INTO task_metric(task_id, config) VALUES (?, ?)"
        try:
            self._retry_execute(sql, metric.to_sql_param())
            logger.debug("Task metric saved successfully")
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to store round metric, error: {e}")

    def store_round_metric(self, metric):
        sql = '''
        INSERT INTO round_metric (
        task_id,
        round_id,
        accuracy,
        loss,
        round_time,
        train_time,
        test_time,
        train_distribute_time,
        test_distribute_time,
        train_upload_size,
        train_download_size,
        test_upload_size,
        test_download_size,
        extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''

        try:
            self._retry_execute(sql, metric.to_sql_param())
            logger.debug("Round metric saved successfully")
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to store round metric {metric.task_id} {metric.round_id}, error: {e}")

    def store_client_metrics(self, metrics):
        """Store a list of client metrics. If the client exists, replace the values.
        :param metrics, list of client metrics to store, [].
        """
        sql = '''
        INSERT INTO client_metric (
        task_id,
        round_id,
        client_id,
        train_accuracy,
        train_loss,
        test_accuracy,
        test_loss,
        train_time,
        test_time,
        train_upload_time,   
        test_upload_time,    
        train_upload_size,   
        train_download_size, 
        test_upload_size,    
        test_download_size,  
        extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''

        params = [metric.to_sql_param() for metric in metrics]

        try:
            with self._conn:
                self._conn.executemany(sql, params)
            logger.debug("Client metrics saved successfully")
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to store client metrics, error: {e}")

    def store_client_train_metric(self, tid, rid, cid, train_loss, train_time, train_upload_time,
                                  train_download_size, train_upload_size):

        sql = "INSERT INTO client_metric (task_id, round_id, client_id, train_loss, train_time, " \
              "train_upload_size, train_download_size, train_upload_size) VALUES (?, ? ,? ,?, ?, ?, ?, ?);"

        param = (tid, rid, cid, train_loss, train_time, train_upload_time, train_download_size, train_upload_size)
        try:
            self._retry_execute(sql, param)
        except sqlite3.OperationalError as e:
            logger.error("Failed to store client train metric, error: {}".format(e))

    def store_client_test_metric(self, tid, rid, cid, test_acc, test_loss, test_time,
                                 test_upload_time, test_download_size):
        sql = "UPDATE client_metric SET test_accuracy=?, test_loss=?, test_time=? ,test_upload_size=?, " \
              "test_download_size=? WHERE task_id=? AND round_id=? AND client_id=?;"
        param = (test_acc, test_loss, test_time, test_upload_time, test_download_size, tid, rid, cid)
        try:
            self._retry_execute(sql, param)
        except sqlite3.OperationalError as e:
            logger.error("Failed to store client test metric, error: {}".format(e))

    # ------------------ get metrics ------------------

    def get_task_metric(self, task_id):
        sql = "SELECT * FROM task_metric WHERE task_id=?"
        with self._conn:
            result = self._conn.execute(sql, (task_id,))
            for r in result:
                return r

    def get_round_metrics(self, task_id, rounds):
        if rounds:
            sql = "SELECT * FROM round_metric WHERE task_id=? AND round_id IN (%s)" % ("?," * len(rounds))[:-1]
            param = [task_id] + rounds
        else:
            sql = "SELECT * FROM round_metric WHERE task_id=?"
            param = (task_id,)
        with self._conn:
            result = self._conn.execute(sql, param)
        return result

    def get_client_metrics(self, task_id, round_id, client_ids=None):
        if client_ids:
            sql = "SELECT * FROM client_metric WHERE task_id=? AND round_id=?  \
                   AND client_id IN (%s)" % ("?," * len(client_ids))[:-1]
            param = [task_id, round_id] + client_ids
        else:
            sql = "SELECT * FROM client_metric WHERE task_id=? AND round_id=?"
            param = (task_id, round_id)
        with self._conn:
            result = self._conn.execute(sql, param)
        return result

    def get_round_train_test_time(self, tid, rounds, interval=1):
        sql = "SELECT SUM(train_time+test_time) FROM round_metric WHERE task_id=? AND round_id<?"
        result = []
        for r in range(interval, rounds + interval, interval):
            param = (tid, r)
            with self._conn:
                res = self._conn.execute(sql, param)
            for i in res:
                result.append((r, i[0]))
        return result

    # ------------------ delete metrics ------------------
    def truncate_task_metric(self):
        sql = "DELETE FROM task_metric"
        try:
            self._retry_execute(sql)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to truncate task metric, error: {e}")

    def truncate_round_metric(self):
        sql = "DELETE FROM round_metric"
        try:
            self._retry_execute(sql)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to truncate round metric, error: {e}")

    def truncate_client_metric(self):
        sql = "DELETE FROM client_metric"
        try:
            self._retry_execute(sql)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to truncate round metric, error: {e}")

    def delete_round_metric(self, task_id, round_id):
        sql = "DELETE FROM round_metric WHERE task_id=? AND round_id=?"
        try:
            self._retry_execute(sql, (task_id, round_id))
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Failed to delete round metric {task_id} {round_id}, error: {e}")

    def _retry_execute(self, sql, param=(), timeout=DEFAULT_TIMEOUT):
        for t in range(0, timeout + 1):
            try:
                with self._conn:
                    self._conn.execute(sql, param)
                break
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                logger.info("retry tracking, error: {}".format(e))
                if t == timeout:
                    raise e
                sleep_time = random.uniform(0, 0.2)
                time.sleep(sleep_time)
                continue


class RemoteStorage(object):
    """RemoteStorage sends request to remote service to store tracking metrics
    """

    def __init__(self, address="localhost:12666"):
        # TODO: put the remote address in config
        self.tracking_stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_TRACKING, address)

    def store_task_metric(self, metric):
        response = self.tracking_stub.TrackTaskMetric(tracking_pb.TrackTaskMetricRequest(task_metric=metric.to_proto()))
        if response.status == common_pb.SC_UNKNOWN:
            logger.error("Failed to store task metric.")
        return response.status

    def store_round_metric(self, metric):
        req = tracking_pb.TrackRoundMetricRequest(round_metric=metric.to_proto())
        response = self.tracking_stub.TrackRoundMetric(req)
        if response.status == common_pb.SC_UNKNOWN:
            logger.error(f"Failed to store round metric, task_id: {metric.task_id} round_id: {metric.round_id}.")
        return response.status

    def store_client_metrics(self, metrics):
        client_metrics = [m.to_proto() for m in metrics]
        req = tracking_pb.TrackClientMetricRequest(client_metrics=client_metrics)
        response = self.tracking_stub.TrackClientMetric(req)
        if response.status == common_pb.SC_UNKNOWN:
            logger.error(f"Failed to store client metrics.")
        return response.status

    def store_client_train_metric(self, tid, rid, cid, train_loss, train_time, train_upload_time,
                                  train_download_size, train_upload_size):
        req = tracking_pb.TrackClientTrainMetricRequest(task_id=tid,
                                                        round_id=rid,
                                                        client_id=cid,
                                                        train_loss=train_loss,
                                                        train_time=train_time,
                                                        train_upload_time=train_upload_time,
                                                        train_download_size=train_download_size,
                                                        train_upload_size=train_upload_size)
        response = self.tracking_stub.TrackClientTrainMetric(req)
        if response.status == common_pb.SC_UNKNOWN:
            logger.error("Failed to store client metric, task id: {} round id: {} client id: {}.".format(tid, rid, cid))
        return response.status

    def store_client_test_metric(self, tid, rid, cid, test_acc, test_loss, test_time,
                                 test_upload_time, test_download_size):
        req = tracking_pb.TrackClientTestMetricRequest(task_id=tid,
                                                       round_id=rid,
                                                       client_id=cid,
                                                       test_accuracy=test_acc,
                                                       test_loss=test_loss,
                                                       test_time=test_time,
                                                       test_upload_time=test_upload_time,
                                                       test_download_size=test_download_size)
        response = self.tracking_stub.TrackClientTestMetric(req)
        if response.status == common_pb.SC_UNKNOWN:
            logger.error("Failed to store client metric, task id: {} round id: {} client id: {}.".format(tid, rid, cid))
        return response.status
