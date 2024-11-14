import torch
from loaders import get_val_loader

def save_image_probabilities(model, model_path, device, output_file="image_probabilities.txt"):
    # Load model
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Get validation loader
    val_loader = get_val_loader()
    
    # Access image paths directly from the dataset within the loader
    image_paths = [sample[0] for sample in val_loader.dataset.samples]
    
    with open(output_file, "w") as f:
        # Update header to include class names (e.g., "cat" and "dog")
        f.write("Image Name\tCorrect Prediction\tTrue Label\tPredicted Label\tCat Probability\tDog Probability\n")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Get predicted labels
                probabilities = torch.softmax(outputs, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                
                # Save each image's probabilities and labels to file
                for idx in range(images.size(0)):
                    image_index = batch_idx * val_loader.batch_size + idx
                    image_name = image_paths[image_index]  # Access the original image file name
                    true_label = labels[idx].cpu().item()
                    pred_label = predicted_labels[idx].cpu().item()
                    correct_prediction = true_label == pred_label
                    prob_values = probabilities[idx].cpu().tolist()
                    
                    # Format probabilities for printing
                    prob_str = "\t".join([f"{p:.4f}" for p in prob_values])
                    
                    # Write to file with image name
                    f.write(f"{image_name}\t{correct_prediction}\t{true_label}\t{pred_label}\t{prob_str}\n")

    print(f"Image probabilities have been saved to {output_file}")