import pickle


def marshal(raw_data):
    return pickle.dumps(raw_data)


def unmarshal(data):
    return pickle.loads(data)
