import easyfl
from client import FedReIDClient
from dataset import prepare_train_data, prepare_test_data, DB_NAMES
from model import Model

train_data = prepare_train_data(DB_NAMES)
test_data = prepare_test_data(DB_NAMES)

easyfl.start_remote_client(train_data=train_data, test_data=test_data, client=FedReIDClient, model=Model)
