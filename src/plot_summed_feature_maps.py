import matplotlib.pyplot as plt
import torch
from PIL import Image
from loaders import test_transform

def plot_summed_feature_maps(model, device, img_path, output_dir):
    # Preprocess the image
    img = Image.open(img_path).convert("RGB")
    transform = test_transform()
    img = transform(img).unsqueeze(0).to(device) # apply transforms and add batch dimension

    # Set model to evaluation mode and move to correct device
    model.eval()
    model.to(device)

    # Collect all convolutional layers
    conv_layers = []
    model_children = list(model.children())
    
    for layer in model_children:
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append(layer)
        elif isinstance(layer, torch.nn.Sequential):
            for child in layer.children():
                if isinstance(child, torch.nn.Conv2d):
                    conv_layers.append(child)

    print(f"Total convolutional layers: {len(conv_layers)}")

    # Forward pass through model and collect feature maps
    outputs = []
    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)

    # Visualize summed feature maps for each layer
    for layer_idx, feature_map in enumerate(outputs):
        feature_map = feature_map.squeeze(0).cpu() # remove batch dimension and move to CPU
        
        # Sum feature maps across filter dimension (channels)
        summed_map = torch.sum(feature_map, dim=0).numpy()

        # Plot summed feature map
        plt.figure(figsize=(10, 10))
        plt.imshow(summed_map, cmap="gray")
        plt.axis("off")
        plt.title(f"Conv Layer {layer_idx + 1}", fontsize=16)

        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}_fmap_conv_layer_{layer_idx + 1}.png")
        plt.close()