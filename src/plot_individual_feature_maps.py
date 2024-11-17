import matplotlib.pyplot as plt
import torch
from PIL import Image
from loaders import get_test_transform


def plot_individual_feature_maps(
    model, device, img_path, output_file="individual_feature_maps.png"
):
    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")
    test_transform = get_test_transform()
    img_tensor = (
        test_transform(img).unsqueeze(0).to(device)
    )  # apply transforms and add batch dimension

    # Plot and save the original image
    _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Original Image", fontsize=14)
    original_image_file = f"{output_file}_original.png"
    plt.savefig(original_image_file)
    plt.show()

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

    # Forward pass through model and collect feature maps
    outputs = []
    for layer in conv_layers:
        img_tensor = layer(img_tensor)
        outputs.append(img_tensor)

    # Create the figure for feature maps
    for layer_idx, feature_map in enumerate(outputs):
        feature_map = feature_map.squeeze(0).cpu()  # remove batch dimension and move to CPU
        num_filters = feature_map.size(0)  # number of filters in this layer

        # Determine grid layout for plotting all filters
        col_size = 8  # number of columns
        row_size = (num_filters + col_size - 1) // col_size

        _, axes = plt.subplots(row_size, col_size, figsize=(col_size * 2, row_size * 2))
        axes = axes.flatten()

        # Plot each filter's feature map
        for filter_idx in range(num_filters):
            single_filter_map = feature_map[filter_idx].detach().numpy()
            axes[filter_idx].imshow(single_filter_map, cmap="gray")
            axes[filter_idx].axis("off")
            axes[filter_idx].set_title(f"Filter {filter_idx + 1}", fontsize=8)

        # Turn off any unused subplots
        for idx in range(num_filters, len(axes)):
            axes[idx].axis("off")

        # Save figure for each layer
        plt.tight_layout()
        layer_output_file = f"{output_file}_layer_{layer_idx + 1}.png"
        plt.savefig(layer_output_file)
        plt.show()
