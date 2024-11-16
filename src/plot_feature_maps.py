import matplotlib.pyplot as plt
import torch
from PIL import Image
from loaders import test_transform

def plot_feature_maps(model, device, img_path, output_dir):
    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")
    transform = test_transform()
    img = transform(img).unsqueeze(0).to(device) # apply the transforms and add batch dimension

    # Set model to evaluation mode and move to correct device
    model.eval()
    model.to(device)

    # List to store all feature maps
    feature_maps = []
    layer_names = []

    # Pass image through each convolutional layer and collect feature maps
    with torch.no_grad():
        x = img
        cv_layer = 0
        for i, layer in enumerate(model.cv_layers):
            x = layer(x) # forward pass through the layer
            if isinstance(layer, torch.nn.Conv2d): # check if it's a Conv2d layer
                feature_maps.append(x.clone()) # save feature map
                layer_names.append(f"Conv Layer {cv_layer+1}") # save layer name
                cv_layer += 1

    # Print a concetaenated list of feature map sizes
    print(f"Feature map sizes: {[f.shape for f in feature_maps]}")

    # Plot feature maps
    for idx, (fmap, name) in enumerate(zip(feature_maps, layer_names)):
        num_filters = fmap.size(1) # number of filters
        col_size = int(num_filters ** 0.5)  # grid width
        row_size = (num_filters + col_size - 1) // col_size # grid height

        # Create a subfigure for this layer
        fig, axes_layer = plt.subplots(row_size, col_size, figsize=(col_size * 2, row_size * 2))
        axes_layer = axes_layer.flatten()

        for i, feature_map in enumerate(fmap.squeeze(0)):
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            axes_layer[i].imshow(gray_scale.data.cpu().numpy())
            axes_layer[i].axis("off")
        for j in range(len(fmap.squeeze(0)), len(axes_layer)):
            axes_layer[j].axis("off") # turn off extra subplots for this layer

        # Add title and layout
        fig.suptitle(name, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}_feature_maps_conv_{idx + 1}.png")
        plt.show()