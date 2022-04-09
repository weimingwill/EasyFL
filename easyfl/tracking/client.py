from easyfl.tracking.metric import TaskMetric, RoundMetric, ClientMetric
from easyfl.tracking.storage import get_store


class TrackingClient(object):
    """Client for tracking task metrics, round metrics, and client metrics.
    Task Tracking:
    client.create_task(task_id, conf)

    Round Tracking:
    client.track_round(name, value)
    client.save_round() # auto increment to next round
    client.track_round(name, value)

    Client Tracking:
    client.set_client_context(task_id, round_id, client_id)
    client.track_client(name, value)
    """

    def __init__(self, db_path=None, db_address=None, init_store=True):
        """If storage is not initialized, the tracking client can only collect metrics but not save them.
        """
        self._task_id = None
        self._round_id = None
        self._client_id = None
        self._current_task = None
        self._current_round = None
        self._current_client = None
        self._cached_task_metrics = {}
        self._cached_round_metrics = {}

        if init_store:
            self._storage = get_store(db_path, db_address)

    def get_task_metric(self, task_id):
        """Get task from storage
        """
        task_metric = self._storage.get_task_metric(task_id)
        if task_metric is None:
            return
        return TaskMetric.from_sql(task_metric)

    def get_round_metric(self, round_id, task_id):
        if task_id == self._task_id and round_id == self._round_id:
            return self._current_round
        return self._storage.get_round_metrics(task_id, [round_id])

    def get_client_metric(self, client_id=None, round_id=None, task_id=None):
        if (task_id == self._task_id and round_id == self._round_id and client_id == self._client_id) or \
                (client_id is None and round_id is None and task_id is None):
            return self._current_client
        return self._storage.get_client_metrics(task_id, round_id, [client_id])

    def get_client_metrics(self, client_ids, round_id, task_id):
        """Get list of client metrics.
        :param client_ids: list of client ids.
        :param round_id: round id.
        :param task_id: task id.
        """
        return self._storage.get_client_metrics(task_id, round_id, client_ids)

    def create_task(self, task_id, conf=None, save=True):
        if task_id is None:
            raise ValueError("task_id cannot be None to create task")
        self._task_id = task_id
        self._current_task = TaskMetric(task_id, conf)
        if save:
            self._storage.store_task_metric(self._current_task)

    def create_round(self, round_id, task_id=None):
        if task_id is None:
            task_id = self._task_id

        if round_id is None:
            raise ValueError("round_id cannot be None to create round")

        if round_id != self._round_id:
            self._round_id = round_id
            self._current_round = RoundMetric(task_id, self._round_id)

    def create_client(self, client_id, reset=True):
        """Create client under current round of task.
        Current implementation requires round and task exist to create client.
        """
        self._check_context()

        if client_id is None:
            raise ValueError("client_id cannot be None to create client.")

        if reset or not self._current_client or client_id != self._client_id:
            self._current_client = ClientMetric(self._task_id, self._round_id, client_id)
            self._client_id = client_id
            return

        self._current_client.task_id = self._task_id
        self._current_client.round_id = self._round_id

    def track_task(self, name, value, task_id=None):
        if self._diff_task_id(task_id):
            self._cached_task_metrics[self._task_id] = self._current_task
            self._task_id = task_id
            self._current_task = TaskMetric(task_id)

        self._current_task.add(name, value)
        self._storage.store_task_metric(self._current_task)

    def track_round(self, name, value, round_id=None, task_id=None):
        if self._diff_task_id(task_id):
            create_task(task_id)

        if self._diff_round_id(round_id):
            # self._cached_round_metrics[self.unique_round_id] = self._current_round
            self.create_round(round_id)

        if self._current_round is None:
            self.create_round(0)

        self._current_round.add(name, value)

    def track_client(self, name, value, client_id=None):
        """Track client under current round and task.
        Current implementation requires round and task exist to track client.
        """
        self._check_context()

        if self._diff_client_id(client_id) or self._current_client is None:
            self.create_client(client_id)

        self._current_client.add(name, value)

    def save_round(self, increment=True, cache=False):
        if self._current_round is None:
            raise ValueError("Round metric is not initialized")
        self._storage.store_round_metric(self._current_round)
        if cache:
            self._cached_round_metrics[self.unique_round_id] = self._current_round
        if increment:
            self.create_round(self._round_id + 1)

    def save_client(self):
        if self._current_client is None:
            raise ValueError("Client metric is not initialized")

        self._storage.store_client_metrics([self._current_client])

    def save_clients(self, client_metrics):
        self._storage.store_client_metrics(client_metrics)

    def set_task(self, task_id):
        if self._current_task is None:
            self.create_task(task_id, save=False)

    def set_round(self, round_id):
        self.create_round(round_id)

    def set_client_context(self, task_id, round_id, client_id, reset_client=True):
        """Set the client context for tracking.
        :param task_id: task id, indicating current the training task
        :param round_id: round id, indicating current round of training/testing
        :param client_id: client id
        :param reset_client: resets and creates a new client.
        """
        self.set_task(task_id)
        self.set_round(round_id)
        self.create_client(client_id, reset=reset_client)

    @property
    def unique_round_id(self):
        return f"{self._task_id}_{self._round_id}"

    def _diff_task_id(self, task_id):
        return task_id is not None and task_id != self._task_id

    def _diff_round_id(self, round_id):
        return round_id is not None and round_id != self._round_id

    def _diff_client_id(self, client_id):
        return client_id is not None and client_id != self._client_id

    def _check_context(self):
        if self._task_id is None or self._round_id is None:
            raise LookupError("task_id or round_id of the client is not set")


_client = TrackingClient(init_store=False)
"""easyfl.tracking.TrackingClient: The global tracking client object"""


def init_tracking(path=None, address=None, init_store=True):
    """Initialize tracking client. This tracking client is isolated from the global tracking client.
    This is useful when an application need to run multiple tasks.
    :param path: database path
    :param address: remote address of tracking service to connect to
    :param init_store: whether initialize storage
    """
    return TrackingClient(path, address, init_store)


# ------ following methods are not well tested yet ------


def setup_tracking(path=None, address=None):
    """Setup tracking with global tracking client.
    """
    global _client
    _client = init_tracking(path, address)


def get_task(task_id):
    return _client.get_task_metric(task_id)


def get_round(round_id, task_id):
    return _client.get_round_metric(round_id, task_id)


def create_task(task_id, conf=None):
    _client.create_task(task_id, conf)


def track_task(name, value, task_id=None):
    _client.track_task(name, value, task_id)


def track_round(name, value, round_id=None, task_id=None):
    _client.track_round(name, value, round_id, task_id)


def track_client(name, value, client_id=None):
    _client.track_client(name, value, client_id)


def set_task(task_id):
    _client.set_task(task_id)


def set_round(round_id):
    _client.set_round(round_id)


def save_round():
    _client.save_round()
