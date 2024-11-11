from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from default_config import image_size, batch_size, num_workers, pin_memory, normalize_mean, normalize_std, default_transform_config, train_dir, val_dir

def get_train_loader(transform_config=default_transform_config):
    flip = transform_config["flip"]
    rotation = transform_config["rotation"]
    crop_scale = transform_config["crop_scale"]
    brightness_jitter = transform_config["brightness_jitter"]
    contrast_jitter = transform_config["contrast_jitter"]   
    saturation_jitter = transform_config["saturation_jitter"]
    hue_jitter = transform_config["hue_jitter"]
    blur = transform_config["blur"]
    affine = transform_config.get("affine", None)
    
    transform_list = [transforms.Resize((image_size, image_size))] # always resizing

    # Add transformations
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    if crop_scale < 1.0:
        transform_list.append(transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)))
    color_jitter_params = {}
    if brightness_jitter > 0:
        color_jitter_params["brightness"] = brightness_jitter
    if contrast_jitter > 0:
        color_jitter_params["contrast"] = contrast_jitter
    if saturation_jitter > 0:
        color_jitter_params["saturation"] = saturation_jitter
    if hue_jitter > 0:
        color_jitter_params["hue"] = hue_jitter
    if color_jitter_params:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))
    if blur > 0:
        transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=blur))
    if affine is not None:
        transform_list.append(transforms.RandomAffine(**affine))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    transform_list.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

    # Compose all transformations into a single pipeline
    train_transform = transforms.Compose(transform_list)
    
    # Load data
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Create data loader
    train_loader = DataLoader(train_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    return train_loader

def get_val_loader():
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    
    val_loader = DataLoader(val_data, 
                                batch_size=batch_size,
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    
    return val_loader