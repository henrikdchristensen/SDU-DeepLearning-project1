import matplotlib.pyplot as plt
import torch
from PIL import Image
from loaders import get_test_transform

def plot_summed_feature_maps(model, device, img_path, output_file="feature_maps.png"):
    # Preprocess the image
    img = Image.open(img_path).convert("RGB")
    test_transform = get_test_transform()
    img = test_transform(img).unsqueeze(0).to(device) # apply transforms and add batch dimension

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
        img = layer(img)
        outputs.append(img)

    # Determine grid layout for plotting all layers
    num_layers = len(outputs)
    col_size = 4 # number of columns
    row_size = (num_layers + col_size - 1) // col_size

    # Create the figure
    fig, axes = plt.subplots(row_size, col_size, figsize=(col_size * 2, row_size * 2))
    axes = axes.flatten()

    # Plot each layer's summed feature map
    for layer_idx, feature_map in enumerate(outputs):
        feature_map = feature_map.squeeze(0).cpu() # remove batch dimension and move to CPU
        
        # Sum feature maps across filter dimension (channels)
        summed_map = torch.sum(feature_map, dim=0).detach().numpy() # detach before converting to NumPy

        # Plot on corresponding subplot
        axes[layer_idx].imshow(summed_map, cmap="gray")
        axes[layer_idx].axis("off")
        axes[layer_idx].set_title(f"Layer {layer_idx + 1}", fontsize=10)

    # Turn off any unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis("off")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()