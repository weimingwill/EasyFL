import importlib
import logging
from os import path

from torch import nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()


def load_model(model_name: str):
    dir_path = path.dirname(path.realpath(__file__))
    model_file = path.join(dir_path, "{}.py".format(model_name))
    if not path.exists(model_file):
        logger.error("Please specify a valid model.")
    model_path = "easyfl.models.{}".format(model_name)
    model_lib = importlib.import_module(model_path)
    model = getattr(model_lib, "Model")
    # TODO: maybe return the model class initiator
    return model
