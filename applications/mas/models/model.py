from mas import models
from easyfl.tracking.evaluation import count_model_params


def get_model(arch, tasks, pretrained=False):
    model = models.__dict__[arch](pretrained=pretrained, tasks=tasks)
    print(f"Model has {count_model_params(model)} parameters")
    try:
        print(f"Encoder has {count_model_params(model.encoder)} parameters")
    except:
        print(f"Each encoder has {count_model_params(model.encoders[0])} parameters")
    for decoder in model.task_to_decoder.values():
        print(f"Decoder has {count_model_params(decoder)} parameters")
    return model
