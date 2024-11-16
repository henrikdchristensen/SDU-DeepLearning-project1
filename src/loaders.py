from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from default_config import image_size, batch_size, num_workers, pin_memory, normalize_mean, normalize_std, default_transform_config, train_dir, val_dir

def get_train_loader(transform_config=default_transform_config):
    hflip = transform_config["hflip"]
    
    brightness_jitter = transform_config["brightness_jitter"]
    contrast_jitter = transform_config["contrast_jitter"]   
    saturation_jitter = transform_config["saturation_jitter"]
    hue_jitter = transform_config["hue_jitter"]
    
    rotation = transform_config["rotation"]
    crop_scale = transform_config["crop_scale"]
    translate = transform_config["translate"]
    shear = transform_config["shear"]
    
    transform_list = [transforms.Resize((image_size, image_size))] # always resizing

    # Add transformations
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    color_jitter_params = {}
    if brightness_jitter > 0:
        color_jitter_params["brightness"] = brightness_jitter
    if contrast_jitter > 0:
        color_jitter_params["contrast"] = contrast_jitter
    if saturation_jitter > 0:
        color_jitter_params["saturation"] = saturation_jitter
    if hue_jitter > 0:
        color_jitter_params["hue"] = hue_jitter
    if brightness_jitter > 0 or contrast_jitter > 0 or saturation_jitter > 0 or hue_jitter > 0:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))
    
    affine_params = {}
    if rotation > 0:
        affine_params["degrees"] = rotation
    if crop_scale < 1 and rotation > 0:
        affine_params["scale"] = (1-crop_scale, 1+crop_scale)
    if translate > 0 and rotation > 0:
        affine_params["translate"] = (translate, translate)
    if shear > 0 and rotation > 0:
        affine_params["shear"] = (-shear, shear)
    if rotation > 0: # rotation must be set for RandomAffine to work
        transform_list.append(transforms.RandomAffine(**affine_params))
    
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

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

def get_test_loader(dir):
    test_transform = get_test_transform()
    
    test_data = datasets.ImageFolder(dir, transform=test_transform)
    
    test_loader = DataLoader(test_data, 
                                batch_size=batch_size,
                                shuffle=False, 
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    
    return test_loader