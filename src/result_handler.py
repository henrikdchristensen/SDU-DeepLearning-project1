from convolutionalNetwork import ConvolutionalNetwork
from residualNetwork import ResidualNetwork
from train_model import train_model

def result_handler(configs, device):
    results = {}
    for config in configs:
        if config["net_config"]["type"] == "CNN":
            model = ConvolutionalNetwork(config["net_config"])
        elif config["net_config"]["type"] == "ResNet":
            model = ResidualNetwork(config["net_config"])
        result = train_model(model, device, config)
        label = config["label"]
        results[label] = result
    return results