import torch
from default_config import normalize_mean, normalize_std


def denormalize_image(image):
    mean = torch.tensor(normalize_mean).view(-1, 1, 1)
    std = torch.tensor(normalize_std).view(-1, 1, 1)
    return image * std + mean
