import os

from torchvision import transforms

from easyfl.datasets import FederatedImageDataset

DB_NAMES = ["MSMT17", "Duke", "Market", "cuhk03", "prid", "cuhk01", "viper", "3dpes", "ilids"]

TRANSFORM_TRAIN_LIST = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM_VAL_LIST = transforms.Compose([
    transforms.Resize(size=(256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def prepare_train_data(data_dir, db_names=None):
    if db_names is None:
        db_names = DB_NAMES
    client_ids = []
    roots = []
    for db in db_names:
        client_ids.append(db)
        data_path = os.path.join(data_dir, db, 'pytorch')
        roots.append(os.path.join(data_path, 'train_all'))
    data = FederatedImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_TRAIN_LIST,
                                 client_ids=client_ids)
    return data


def prepare_test_data(data_dir, db_names=None):
    if db_names is None:
        db_names = DB_NAMES
    roots = []
    client_ids = []
    for db in db_names:
        test_gallery = os.path.join(data_dir, db, 'pytorch', 'gallery')
        test_query = os.path.join(data_dir, db, 'pytorch', 'query')
        roots.extend([test_gallery, test_query])
        client_ids.extend(["{}_{}".format(db, "gallery"), "{}_{}".format(db, "query")])
    data = FederatedImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_VAL_LIST,
                                 client_ids=client_ids)
    return data
