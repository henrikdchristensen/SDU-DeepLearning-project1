image_size = 200
batch_size = 32
num_workers = 8
pin_memory = True
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
data_dir = "data"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/validation"
test_dir = f"{data_dir}/test"
label_map={0: "cat", 1: "dog"}
# Set seed
#seed = 42
#random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

default_transform_config = {
    "flip": True,
    "rotation": 10,
    "crop_scale": 0.9,
    "brightness_jitter": 0.15,
    "contrast_jitter": 0.15,
    "saturation_jitter": 0.15,
    "hue_jitter": 0.1,
    "blur": 0,
    "affine": None,
}

no_transform_config = {
    "flip": False,
    "rotation": 0,
    "crop_scale": 1.0,
    "brightness_jitter": 0.0,
    "contrast_jitter": 0.0,
    "saturation_jitter": 0.0,
    "hue_jitter": 0.0,
    "blur": 0.0,
}

default_train_config = {
    "optimizer_type": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "momentum": None,
    "reg_type": "None",
    "reg_lambda": 0.001,
    "step_size": None,
    "gamma": None,
}

default_net_config = {
    "in_channels": 3,
    "num_classes": 2,
    "type": "CNN",
    "cv_layers": [
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "batchNorm": False, "maxPool": 0, "maxPool_stride": 1},
        {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "batchNorm": False, "maxPool": 0, "maxPool_stride": 1},
    ],
    "fc_layers": [{"out_features": 64, "batchNorm": False, "dropout_rate": 0.0}],
    "blocks": [2, 2, 2, 2]
}

default_config = {
    "label": "Default Experiment",
    "n_epochs": 20,
    "transform_config": default_transform_config,
    "train_config": default_train_config,
    "net_config": default_net_config
}