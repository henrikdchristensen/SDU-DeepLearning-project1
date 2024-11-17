import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_scores(results):
    for label, data in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

        epochs = range(1, data["n_epochs"] + 1)

        # Plot loss
        axes[0].plot(epochs, data["train_losses"], marker="o", markersize=3, label=f"{label} train")
        axes[0].plot(
            epochs,
            data["val_losses"],
            marker="o",
            markersize=3,
            linestyle="--",
            label=f"{label} val",
        )
        axes[0].set_title("loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")

        # Plot accuracy
        axes[1].plot(
            epochs, data["train_accuracies"], marker="o", markersize=3, label=f"{label} train"
        )
        axes[1].plot(
            epochs,
            data["val_accuracies"],
            marker="o",
            markersize=3,
            linestyle="--",
            label=f"{label} val",
        )
        axes[1].set_title("accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy (%)")

        # Set 4 ticks on the x and y-axis
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))

        # Create a single legend below the plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2)

        plt.tight_layout(rect=[0, 0.1, 1, 0.9])
        plt.show()
