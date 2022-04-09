import argparse
import logging

from easyfl.communication import grpc_wrapper
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import tracking_service_pb2 as tracking_pb
from easyfl.pb import tracking_service_pb2_grpc as tracking_grpc
from easyfl.tracking import metric
from easyfl.tracking.storage import SqliteStorage

logger = logging.getLogger(__name__)


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Federated Tracker')
    parser.add_argument('--local-port',
                        type=int,
                        default=12666,
                        help='Listen port of the client')
    return parser


class TrackingService(tracking_grpc.TrackingServiceServicer):
    def __init__(self, storage=SqliteStorage):
        self._storage = storage()
        logger.info("Tracking service is online")
        self._storage.setup()

    def TrackTaskMetric(self, request, context):
        response = tracking_pb.TrackTaskMetricResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )

        try:
            self._storage.store_task_metric(metric.TaskMetric.from_proto(request.task_metric))
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = f"Failed to track task metric, err: {e}"
            logger.error(response.status.message)

        return response

    def TrackRoundMetric(self, request, context):
        response = tracking_pb.TrackRoundMetricResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        try:
            self._storage.store_round_metric(metric.RoundMetric.from_proto(request.round_metric))
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = f"Failed to track round metric, err: {e}"
            logger.error(response.status.message)

        return response

    def TrackClientMetric(self, request, context):
        response = tracking_pb.TrackClientMetricResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        try:
            metrics = [metric.ClientMetric.from_proto(m) for m in request.client_metrics]
            self._storage.store_client_metrics(metrics)
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = f"Failed to track client metric, err: {e}"
            logger.error(response.status.message)

        return response

    def TrackClientTrainMetric(self, request, context):
        response = tracking_pb.TrackClientTrainMetricResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        try:
            self._storage.store_client_train_metric(request.task_id,
                                                    request.round_id,
                                                    request.client_id,
                                                    request.train_loss,
                                                    request.train_time,
                                                    request.train_upload_time,
                                                    request.train_download_size,
                                                    request.train_upload_size)
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = "Tracking client train failed, err: {}".format(e)
            logger.error("Tracking client train failed, err: {}".format(e))

        return response

    def TrackClientTestMetric(self, request, context):
        response = tracking_pb.TrackClientTestMetricResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )

        try:
            self._storage.store_client_test_metric(request.task_id,
                                                   request.round_id,
                                                   request.client_id,
                                                   request.test_accuracy,
                                                   request.test_loss,
                                                   request.test_time,
                                                   request.test_upload_time,
                                                   request.test_download_size)
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = "Tracking client test failed, err: {}".format(e)
            logger.error("Tracking client test failed, err: {}".format(e))

        return response

    def GetRoundTrainTestTime(self, request, context):
        response = tracking_pb.GetRoundTrainTestTimeResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        try:
            resp = self._storage.get_round_train_test_time(request.task_id,
                                                           request.rounds,
                                                           request.interval)
            for i in resp:
                train_test_time = tracking_pb.TrainTestTime(round_id=i[0], time=i[1])
                response.train_test_times.append(train_test_time)
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = "get round train_test time failed, err: {}".format(e)
            logger.error("get round train_test time failed, err: {}".format(e))
        return response

    def GetRoundMetrics(self, request, context):
        response = tracking_pb.GetRoundMetricsResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        try:
            resp = self._storage.get_round_metrics(request.task_id, request.rounds)
            response.metrics = [metric.RoundMetric.from_sql(r) for r in resp]
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = f"Failed to get round metrics, err: {e}"
            logger.error(response.status.message)
        return response

    def GetClientMetrics(self, request, context):
        response = tracking_pb.GetClientMetricsResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        try:
            resp = self._storage.get_client_metrics(request.task_id, request.round_id, request.client_ids)
            response.metrics = [metric.ClientMetric.from_sql(r) for r in resp]
        except Exception as e:
            response.status.code = common_pb.SC_UNKNOWN
            response.status.message = f"Failed to get client metrics failed, err: {e}"
            logger.error(response.status.message)
        return response


def start_tracking_service(local_port=12666):
    logger.info("Tracking GRPC server started at :{}".format(local_port))
    grpc_wrapper.start_service(grpc_wrapper.TYPE_TRACKING, TrackingService(), local_port)
