import easyfl

config = {
    "data": {"dataset": "cifar10"},
    "model": "simple_cnn",
    "test_mode": "test_in_server"
}
easyfl.init(config)
easyfl.run()
