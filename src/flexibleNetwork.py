from default_config import default_net_config, image_size
import torch
import torch.nn as nn

class FlexibleNetwork(nn.Module):
    def __init__(self, net_config=default_net_config):
        super(FlexibleNetwork, self).__init__()
        
        in_channels = net_config["in_channels"]
        num_classes = net_config["num_classes"]
        
        layers = []
        self.relu = nn.ReLU()
        
        tmp_channels = in_channels
        for config in net_config["cv_layers"]:
            out_channels = config["out_channels"]
            kernel_size = config.get("kernel_size", 3)
            stride = config.get("stride", 1)
            padding = config.get("padding", 0)
            batchnorm = config.get("batchNorm", False)
            maxpool = config.get("maxPool", 0)
            dropout_rate = config.get("dropout_rate", 0.0)
            
            # Convolutional layer
            # - Output channels are the number of filters in the convolutional layer
            # - Kernel size is the size of the filter
            # - Stride is the step size for the filter
            # - Padding is the number of pixels to add around the input image
            layers.append(nn.Conv2d(tmp_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            
            # Batch Normalization
            # - Batch normalization is used to normalize the input layer by adjusting and scaling the activations.
            #   It is used to make the model faster and more stable.
            if batchnorm:
                layers.append(nn.BatchNorm2d(num_features=out_channels))
            
            # ReLU activation
            # - An activation function is used to introduce non-linearity to the output of a neuron
            layers.append(self.relu)
            
            # Maxpooling layer
            # - Max pooling is a downsampling operation that reduces the dimensionality of the input
            # - Kernel size is the size of the filter
            # - Stride is the step size for the filter
            if maxpool > 1:
                maxpool_stride = config.get("maxPool_stride", 1)
                layers.append(nn.MaxPool2d(kernel_size=maxpool, stride=maxpool_stride))
            
            # Dropout layer
            # - Dropout is a regularization technique to prevent overfitting by randomly setting some output features to zero
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            
            # Update input channels for next iteration
            tmp_channels = out_channels
        
        # Sequential container for convolutional layers
        self.cv_layers = nn.Sequential(*layers)
        
        # Calculate input size of tensor after convolutional layers
        with torch.no_grad():
            sample_input = torch.randn(1, in_channels, image_size, image_size)  # create a random sample input tensor
            conv_output = self.cv_layers(sample_input)
            fc_input_features = conv_output.numel() # calculate the number of elements in the tensor
        
        # Fully connected layers
        fc_layers_list = []
        for config in net_config["fc_layers"]:
            out_features = config["out_features"]
            dropout_rate = config.get("dropout_rate", 0.0)
            batchnorm = config.get("batchNorm", False)
            
            # Fully connected layer
            # - Input features are the number of neurons in previous layer
            # - Output features are the number of neurons to create in current layer
            fc_layers_list.append(nn.Linear(in_features=fc_input_features, out_features=out_features))
            
            # Batch Normalization
            # - Batch normalization is used to normalize the input layer by adjusting and scaling the activations.
            #   It is used to make the model faster and more stable.
            if batchnorm:
                fc_layers_list.append(nn.BatchNorm1d(out_features))
            
            # ReLU activation
            # - An activation function is used to introduce non-linearity to the output of a neuron
            fc_layers_list.append(self.relu)
            
            # Dropout
            # - Dropout is a regularization technique to prevent overfitting by randomly setting some output features to zero
            if dropout_rate > 0:
                fc_layers_list.append(nn.Dropout(p=dropout_rate))
            
            # Update input features for next iteration
            fc_input_features = out_features
        
        # Output layer (no dropout or activation)
        fc_layers_list.append(nn.Linear(fc_input_features, num_classes))
        
        # Sequential container for fully connected layers
        self.fc_layers = nn.Sequential(*fc_layers_list)
    
    def forward(self, x):
        x = self.cv_layers(x) # pass through convolutional layers
        x = x.flatten(start_dim=1, end_dim=-1) # flatten tensor from convolutional layers for the linear fully connected layers
        x = self.fc_layers(x) # pass through fully connected layers
        return x