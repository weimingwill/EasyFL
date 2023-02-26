import os

from reid.utils.transform.transforms import TRANSFORM_TRAIN_LIST, TRANSFORM_VAL_LIST
from easyfl.datasets import FederatedImageDataset


def prepare_train_data(db_names, data_dir):
    client_ids = []
    roots = []
    for d in db_names:
        client_ids.append(d)
        data_path = os.path.join(data_dir, d, 'pytorch')
        roots.append(os.path.join(data_path, 'train_all'))
    data = FederatedImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_TRAIN_LIST,
                                 client_ids=client_ids)
    return data


def prepare_test_data(db_names, data_dir):
    roots = []
    client_ids = []
    for d in db_names:
        test_gallery = os.path.join(data_dir, d, 'pytorch', 'gallery')
        test_query = os.path.join(data_dir, d, 'pytorch', 'query')
        roots.extend([test_gallery, test_query])
        client_ids.extend(["{}_{}".format(d, "gallery"), "{}_{}".format(d, "query")])
    data = FederatedImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_VAL_LIST,
                                 client_ids=client_ids)
    return data
