from default_config import default_config, batch_size, image_size
from loaders import get_train_loader, get_test_loader
from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(model, device, config=default_config):
    label = config["label"]
    n_epochs = config["n_epochs"]
    
    train_config = config["train_config"]
    transform_config = config["transform_config"]
    
    model = model.to(device) # move model to device
    
    criterion = nn.CrossEntropyLoss() # loss function
    
    optimizer_type = train_config["optimizer_type"]
    learning_rate = train_config["learning_rate"]
    weight_decay = train_config.get("weight_decay", 0.0)
    momentum = train_config.get("momentum", 0.0)
    reg_type = train_config["reg_type"]
    reg_lambda = train_config["reg_lambda"]
    step_size = train_config["step_size"]
    gamma = train_config["gamma"]
    
    # Get transformations
    train_loader = get_train_loader(transform_config)
    val_loader = get_test_loader()

    # Select optimizer
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        # If weight decay is specified, apply AdamW instead
        if weight_decay > 0:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler
    if step_size is not None and gamma is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(f"\nExperiment: {label}")
    
    # Track total training time
    total_start_time = time.time()
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train() # set model to training mode
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training batches
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # move data to device
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels) # calculate loss
            
            # Apply regularization if specified
            if reg_type == 'L1':
                l1_norm = sum(param.abs().sum() for param in model.parameters())
                loss += reg_lambda * l1_norm
            elif reg_type == 'L2':
                l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                loss += reg_lambda * l2_norm
            
            loss.backward() # backpropagation
            optimizer.step() # update weights
            train_loss += loss.item() # add the loss to the training set loss
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1) # get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # count correct predictions
        
        if step_size is not None and gamma is not None:
            scheduler.step() # update learning rate
        
        # Training set statistics
        train_losses.append(round(train_loss / len(train_loader), 4))
        train_accuracies.append(round(100 * correct / total, 2) if total > 0 else 0)

        # Evaluation on validation set
        model.eval()  # set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # no need to calculate gradients for validation set
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # move data to device
                outputs = model(images)
                loss = criterion(outputs, labels)  # calculate loss
                val_loss += loss.item()  # add the loss to the validation set loss
                _, predicted = torch.max(outputs, 1)  # get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Validation set statistics
        val_losses.append(round(val_loss / len(val_loader), 4))
        val_accuracies.append(round(100 * correct / total, 2) if total > 0 else 0)

        # Print epoch summary
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_losses[-1]} (acc. {train_accuracies[-1]}%) | "
            f"Val Loss: {val_losses[-1]} (acc. {val_accuracies[-1]}%) | Time: {epoch_duration:.2f}s")
    
    # Calculate and print total training time
    total_training_time = time.time() - total_start_time
    print(f"Training Time: {total_training_time:.2f}s")

    # Save model and metrics to file
    with open(f"models/{label}.txt", "w") as f:
        model_summary = summary(model, input_size=(batch_size, 3, image_size, image_size), verbose=0)
        f.write(str(model_summary))
        f.write("\nTraining and Validation Metrics:\n")
        f.write(f"Train Losses: {train_losses}\n")
        f.write(f"Train Accuracies: {train_accuracies}\n")
        f.write(f"Val Losses: {val_losses}\n")
        f.write(f"Val Accuracies: {val_accuracies}\n")
    
    # Save model to file
    torch.save(model.state_dict(), f"models/{label}.pth")