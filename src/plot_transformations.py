import matplotlib.pyplot as plt
import math

num_images_in_row = 3

def plot_transformations(transformed_images, transforms):
    num_images = len(transformed_images)
    rows = math.ceil(num_images / num_images_in_row)
    
    fig, axes = plt.subplots(rows, min(num_images_in_row, num_images), figsize=(5 * min(num_images_in_row, num_images), 5 * rows))

    if rows == 1:
        axes = [axes]
    elif num_images <= num_images_in_row:
        axes = [axes]

    # Plot each image in the corresponding subplot
    for i, img in enumerate(transformed_images):
        row, col = divmod(i, num_images_in_row)
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{transforms[i]}")
    
    # Turn off axis for all subplots
    for row in axes:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    plt.show()