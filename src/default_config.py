image_size = 224
batch_size = 64
num_workers = 8
pin_memory = True
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
data_dir = "data"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/validation"
test_dir = f"{data_dir}/test"
label_map = {0: "cat", 1: "dog"}

default_transform_config = {
    "hflip": True,
    
    "brightness_jitter": 0.2,
    "contrast_jitter": 0.2,
    "saturation_jitter": 0.2,
    "hue_jitter": 0.0,
    
    "rotation": 25, # rotation is in degrees
    "crop_scale": 0.3, # crop_scale is like zooming in. 0.3 means zoom in by 30%
    "translate": 0.1, # translate is like shifting the image. 0.1 means shift by 10%
    "shear": 10, # shear is like skewing the image. 10 means shear by 10 degrees
}

no_transform_config = {
    "hflip": False,
    
    "brightness_jitter": 0,
    "contrast_jitter": 0,
    "saturation_jitter": 0,
    "hue_jitter": 0,
    
    "rotation": 0,
    "crop_scale": 1,
    "translate": 0,
    "shear": 0,
}

default_train_config = {
    "optimizer_type": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0,
    "momentum": None,
    "reg_type": "None",
    "reg_lambda": 0.001,
    "step_size": None,
    "gamma": None,
}

default_net_config = {
    "in_channels": 3,
    "num_classes": 2,
    "cv_layers": [
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": False, "max_pool": 0, "max_pool_stride": 1},
        {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": False, "max_pool": 0, "max_pool_stride": 1},
    ],
    "fc_layers": [{"out_features": 64, "batch_norm": False, "dropout_rate": 0}]
}

default_config = {
    "label": "Default Experiment",
    "n_epochs": 100,
    "transform_config": default_transform_config,
    "train_config": default_train_config,
    "net_config": default_net_config
}