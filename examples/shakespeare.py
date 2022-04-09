import easyfl

config = {
    "data": {"dataset": "shakespeare"},
    "model": "rnn", 
    "test_mode": "test_in_client"
}
easyfl.init(config)
easyfl.run()
