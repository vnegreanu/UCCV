## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 kernel, SAME padding=>224,224,32
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 1)
        #second conv layer 32 input channes, 4 output channels, 5 kernel size ->conv 1x1 => 224,224,4
        self.conv2 = nn.Conv2d(32, 4, 5, 1, 1)
        #third conv layer 4 input channels, 32 output channels, 3 kernel size, SAME padding => 112,112,32 after pooling 2x2
        self.conv3 = nn.Conv2d(4, 32, 3, 1, 1)
        #fourth conv layer 32 input channels, 4 output channels, 3 kernel size ->conv 1x1 => 112,112,4
        self.conv4 = nn.Conv2d(32, 4, 3, 1, 1)
        #fifth conv layer 4 input channels, 32 output channels, 2 kernel size, SAME padding => 56,56,32 after pooling 2x2
        self.conv5 = nn.Conv2d(4, 32, 2, 1, 1)
        #fourth conv layer 32 input channels, 4 output channels, 2 kernel size ->conv 1x1 => 28,28,4 => 3136 for first linear layer
        self.conv6 = nn.Conv2d(32, 4, 2, 1, 1)
        
        
        #max pooling layer # kernel size = 2 stride = 2
        self.pool = nn.MaxPool2d(2 ,2) 
        
        #first fc layer as taken from ouput of x.shape after flattening
        self.fc1 = nn.Linear(3136, 1000)
        #second fc layer keep it steady
        self.fc2 = nn.Linear(1000, 1000)
        #third fc layer - 136 ouputs 2 for each 68 keypoint pairs
        self.fc3 = nn.Linear(1000, 136)
        
        #prepare batch norm instead of dropout 1 after 1 conv layer and 1 conv 1 x 1 layer
        self.bn1 = nn.BatchNorm2d(num_features=4, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(num_features=4, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=4, eps=1e-05)
        
        #dropouts for first 2 FC layers, won't need for the third since it's the readout layer 
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.6)

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #conv layers
        x = F.elu(self.conv1(x))
        x = self.bn1(self.pool(F.elu(self.conv2(x))))
        
        x = F.elu(self.conv3(x))
        x = self.bn2(self.pool(F.elu(self.conv4(x))))
        
        x = F.elu(self.conv5(x))
        x = self.bn3(self.pool(F.elu(self.conv6(x))))
        
        #flatten
        x = x.view(x.size(0), -1)
        
        #start dense layers
        x = self.dp1(F.elu(self.fc1(x)))
        
        x = self.dp2(F.elu(self.fc2(x)))
        
        #output readout layer 
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
