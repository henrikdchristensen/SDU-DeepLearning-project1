import matplotlib.pyplot as plt

def plot_scores(results):
    for label, data in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
        
        epochs = range(1, data["n_epochs"] + 1)
        
        # Plot loss
        axes[0].plot(epochs, data["train_losses"], marker='o', label=f"{label} Train Loss")
        axes[0].plot(epochs, data["val_losses"], marker='o', linestyle='--', label=f"{label} Val Loss")
        
        # Plot accuracy
        axes[1].plot(epochs, data["train_accuracies"], marker='o', label=f"{label} Train Accuracy")
        axes[1].plot(epochs, data["val_accuracies"], marker='o', linestyle='--', label=f"{label} Val Accuracy")
    
        # Customize Loss plot
        axes[0].set_title("loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        
        # Customize Accuracy plot
        axes[1].set_title("accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy (%)")
        
        # Create a single legend below the plots
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)
                
        fig.legend(handles, labels, loc="lower center", ncol=2)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])
        plt.show()