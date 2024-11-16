import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from loaders import get_test_loader
from denormalize_image import denormalize_image
from default_config import label_map
import csv

def predict(model, model_path, device, num_images, output_file="image_probabilities.csv"):
    # Load model
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Get validation loader
    val_loader = get_test_loader()
    
    misclassified = []
    correctly_classified = []
    correct_class_counts = defaultdict(int)
    misclass_class_counts = defaultdict(int)

    # Access image paths directly from the dataset within the loader
    image_paths = [sample[0] for sample in val_loader.dataset.samples]

    # Open CSV file for writing
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Correct Prediction", "True Label", "Predicted Label", "Probabilities"])
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                probs = torch.softmax(outputs, dim=1)
                predicted_labels = torch.argmax(probs, dim=1)
                
                for idx in range(images.size(0)):
                    image_index = batch_idx * val_loader.batch_size + idx
                    img = images[idx].cpu()
                    image_name = image_paths[image_index]
                    
                    true_label_num = labels[idx].cpu().item()
                    pred_label_num = predicted_labels[idx].cpu().item()
                    true_label = label_map[true_label_num]
                    pred_label = label_map[pred_label_num]
                    
                    prob_values = probs[idx].cpu().tolist()
                    correct_prediction = true_label == pred_label
                    
                    if correct_prediction:
                        correctly_classified.append((img, max(prob_values), true_label, pred_label))
                        correct_class_counts[true_label] += 1
                    else:
                        misclassified.append((img, max(prob_values), true_label, pred_label))
                        misclass_class_counts[true_label] += 1
                    
                    # Write to CSV
                    writer.writerow([
                        image_name,
                        correct_prediction,
                        true_label,
                        pred_label,
                        [f"{p:.2f}" for p in prob_values]
                    ])

    # Print total counts and accuracy
    total_correct = len(correctly_classified)
    total_misclassified = len(misclassified)
    total = total_correct + total_misclassified
    accuracy = (total_correct / total) * 100
    print(f"Total Correctly Classified: {total_correct} | Total Misclassified: {total_misclassified} | Accuracy: {accuracy:.2f}%")
    
    # Print class-wise statistics
    print("\nClass-wise Correctly Classified Counts:")
    for label, count in correct_class_counts.items():
        print(f"Class {label_map[label]}: {count}")

    print("\nClass-wise Misclassified Counts:")
    for label, count in misclass_class_counts.items():
        print(f"Class {label_map[label]}: {count}")

    # Sort by confidence
    correctly_classified.sort(key=lambda x: x[1], reverse=True)
    misclassified.sort(key=lambda x: x[1], reverse=True)

    # Get top `num_images` for each class
    top_correct = []
    top_incorrect = []
    for label in set(x[2] for x in correctly_classified):
        top_correct.extend([x for x in correctly_classified if x[2] == label][:num_images])
        top_incorrect.extend([x for x in misclassified if x[2] == label][:num_images])

    # Plot correctly and misclassified images
    fig, axs = plt.subplots(2, num_images * 2 + 1, figsize=(15, 6))
    
    # Add row labels
    axs[0, 0].text(0.5, 0.5, 'Correctly Classified', fontsize=12, ha='center', va='center', rotation=90)
    axs[0, 0].axis('off')
    axs[1, 0].text(0.5, 0.5, 'Misclassified', fontsize=12, ha='center', va='center', rotation=90)
    axs[1, 0].axis('off')

    # Show correctly classified images
    for i, (img, prob, true_label, pred_label) in enumerate(top_correct):
        img = denormalize_image(img)
        axs[0, i + 1].imshow(img.permute(1, 2, 0))
        axs[0, i + 1].set_title(f"True: {label_map[true_label]}, Pred: {label_map[pred_label]}, Prob: {prob:.2f}")
        axs[0, i + 1].axis('off')

    # Show misclassified images
    for i, (img, prob, true_label, pred_label) in enumerate(top_incorrect):
        img = denormalize_image(img)
        axs[1, i + 1].imshow(img.permute(1, 2, 0))
        axs[1, i + 1].set_title(f"True: {label_map[true_label]}, Pred: {label_map[pred_label]}, Prob: {prob:.2f}")
        axs[1, i + 1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Predictions saved to {output_file}")
