import random
import threading
from contextlib import contextmanager

import etcd3

from easyfl.registry import mock_etcd
from easyfl.registry.vclient import VirtualClient


class EtcdClient(object):
    """Etcd client to connect and communicate with etcd service.
    Etcd is the serves as the registry for remote training.
    Clients register themselves in etcd and server queries etcd to get client addresses.

    Args:
        name (str): The name of etcd.
        addrs (str): Etcd addresses, format: "<ip>:<port>,<ip>:<port>".
        base_dir (str): The prefix of all etcd requests, default to "backends".
        use_mock_etcd (bool): Whether use mocked etcd for testing.
    """

    ETCD_CLIENT_POOL_LOCK = threading.Lock()
    ETCD_CLIENT_POOL = {}
    ETCD_CLIENT_POOL_DESTROY = False

    class Event(object):
        def __init__(self, event, base_dir):
            self._event = event
            self._base_dir = base_dir

        def __getattr__(self, attr):
            return getattr(self._event, attr)

        @property
        def key(self):
            return EtcdClient.normalize_output_key(self._event.key, self._base_dir)

    def __init__(self, name, addrs, base_dir, use_mock_etcd=False):
        self._name = name
        self._base_dir = '/' + EtcdClient._normalize_input_key(base_dir)
        self._addrs = self._normalize_addr(addrs)
        if len(self._addrs) == 0:
            raise ValueError('Empty hosts EtcdClient')
        self._cur_addr_idx = random.randint(0, len(self._addrs) - 1)
        self._use_mock_etcd = use_mock_etcd

    def get_data(self, key):
        addr = self._get_next_addr()
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            return clnt.get(self._generate_path(key))[0]

    def set_data(self, key, data):
        addr = self._get_next_addr()
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            clnt.put(self._generate_path(key), data)

    def delete(self, key):
        addr = self._get_next_addr()
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            return clnt.delete(self._generate_path(key))

    def delete_prefix(self, key):
        addr = self._get_next_addr()
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            return clnt.delete_prefix(self._generate_path(key))

    def cas(self, key, old_data, new_data):
        addr = self._get_next_addr()
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            etcd_path = self._generate_path(key)
            if old_data is None:
                return clnt.put_if_not_exists(etcd_path, new_data)
            return clnt.replace(etcd_path, old_data, new_data)

    def watch_key(self, key):
        addr = self._get_next_addr()
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            notifier, cancel = clnt.watch(self._generate_path(key))

            def prefix_extractor(notifier, base_dir):
                while True:
                    try:
                        yield EtcdClient.Event(next(notifier), base_dir)
                    except StopIteration:
                        break

            return prefix_extractor(notifier, self._base_dir), cancel

    def get_prefix_kvs(self, prefix, ignore_prefix=False):
        addr = self._get_next_addr()
        kvs = []
        path = self._generate_path(prefix)
        with EtcdClient.closing(self._name, addr, self._use_mock_etcd) as clnt:
            for (data, key) in clnt.get_prefix(path, sort_order='ascend'):
                if ignore_prefix and key.key == path.encode():
                    continue
                nkey = EtcdClient.normalize_output_key(key.key, self._base_dir)
                kvs.append((nkey, data))
        return kvs

    def get_clients(self, prefix):
        """Retrieve client addresses from etcd using prefix.

        Args:
            prefix (str): the prefix of clients addresses; default is the docker image name "easyfl-client"

        Returns:
            list[:obj:`VirtualClient`]: A list of clients.
        """
        key_value_tuples = self.get_prefix_kvs(prefix)
        clients = []
        index = 0
        for (key_byte, value_byte) in key_value_tuples:
            key, value = key_byte.decode("utf-8"), value_byte.decode("utf-8")
            parts = key.split("/")
            if len(parts) <= 1:
                continue

            addr = parts[1]
            if not self._is_addr(addr):
                continue

            clients.append(VirtualClient(value, addr, index))
            index += 1
        return clients

    def _is_addr(self, address):
        return len(address.split(":")) > 1

    def _generate_path(self, key):
        return '/'.join([self._base_dir, self._normalize_input_key(key)])

    def _get_next_addr(self):
        return self._addrs[random.randint(0, len(self._addrs) - 1)]

    @staticmethod
    def _normalize_addr(addrs):
        naddrs = []
        for raw_addr in addrs.split(','):
            (host, port_str) = raw_addr.split(':')
            try:
                port = int(port_str)
                if port < 0 or port > 65535:
                    raise ValueError('port {} is out of range')
            except ValueError:
                raise ValueError('{} is not a valid port'.format(port_str))
            naddrs.append((host, port))
        return naddrs

    @staticmethod
    def _normalize_input_key(key):
        skip_cnt = 0
        while key[skip_cnt] == '.' or key[skip_cnt] == '/':
            skip_cnt += 1
        if skip_cnt > 0:
            return key[skip_cnt:]
        return key

    @staticmethod
    def normalize_output_key(key, base_dir):
        if isinstance(base_dir, str):
            assert key.startswith(base_dir.encode())
        else:
            assert key.startswith(base_dir)
        return key[len(base_dir) + 1:]

    @classmethod
    @contextmanager
    def closing(cls, name, addr, use_mock_etcd):
        clnt = None
        with cls.ETCD_CLIENT_POOL_LOCK:
            if (name in cls.ETCD_CLIENT_POOL and
                    len(cls.ETCD_CLIENT_POOL[name]) > 0):
                clnt = cls.ETCD_CLIENT_POOL[name][0]
                cls.ETCD_CLIENT_POOL[name] = cls.ETCD_CLIENT_POOL[name][1:]
        if clnt is None:
            try:
                if use_mock_etcd:
                    clnt = mock_etcd.MockEtcdClient(addr[0], addr[1])
                else:
                    clnt = etcd3.client(host=addr[0], port=addr[1])
            except Exception as e:
                clnt.close()
                raise e
        try:
            yield clnt
        except Exception as e:
            clnt.close()
            raise e
        else:
            with cls.ETCD_CLIENT_POOL_LOCK:
                if cls.ETCD_CLIENT_POOL_DESTROY:
                    clnt.close()
                else:
                    if name not in cls.ETCD_CLIENT_POOL:
                        cls.ETCD_CLIENT_POOL[name] = [clnt]
                    else:
                        cls.ETCD_CLIENT_POOL[name].append(clnt)

    @classmethod
    def destory_client_pool(cls):
        with cls.ETCD_CLIENT_POOL_LOCK:
            cls.ETCD_CLIENT_POOL_DESTROY = True
            for _, clnts in cls.ETCD_CLIENT_POOL.items():
                for clnt in clnts:
                    clnt.close()
