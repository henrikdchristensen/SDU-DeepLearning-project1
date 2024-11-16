from convolutionalNetwork import ConvolutionalNetwork
from train_model import train_model

def result_handler(configs, device):
    results = {}
    for config in configs:
        model = ConvolutionalNetwork(config["net_config"])
        result = train_model(model, device, config)
        label = config["label"]
        results[label] = result
    return results