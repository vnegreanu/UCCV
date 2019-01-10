### Facial Keipoint Detection

A link to the source code can be found here:
[P1_Facial_Keypoints](https://github.com/vnegreanu/UCCV/tree/master/01_Facial_Keypoints)

#### Description

Use image processing techniques and Convolutional Neural Networks to detect keypoints on faces images such as eyes, nose, mouth, teeth and chin.
Keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face.

An example is shown in the picture below:
![Example image](local_images/key_pts_example.png)

#### Files
* `Notebook 1`: Loading and visualizing the facial keypoint data
* `Notebook 2`: Training of a Convolutional Neural Network (CNN) to predict facial keypoints
* `Notebook 3`: Facial keypoint detection using HAAR cascades and the trained CNN
* `Notebook 4`: Fun filters and keypoints uses
* `models.py`: Define the neural network architecture
* `data_load.py`: Dataset manipulation class
* `workspace_utils.py`: Utilities for the project


#### CNN architecture

First I've tried NaimishNet but get rid of the idea since it was an overkill for the tasks. I have chosen my custom implementation described below with Conv 1x1 layers and Batch Norm between the convolutional layers. Finally after flattening I was able to get only 3136 input features into the first FC layer.

| Layer             | Size and  features   | 
| -------------     |:--------------------:| 
| Input             | size : (224, 224, 1) | 
| Conv1             | size (224,224,32); filters : 32; kernel size : (5 x 5); stride : (1 x 1); padding : SAME; activation : ELU |
| Conv2             | size (224,224,4); filters : 4; kernel size : (5 x 5); stride : (1 x 1); padding : SAME; activation : ELU | 
| Max Pooling       | kernel size: (2x2); stride: (2x2) |
| BatchNorm1	    | features: 4; eps: 1e-05 |
| Conv3             | size (112,112,32); filters : 32; kernel size : (3 x 3); stride : (1 x 1); padding : SAME; activation : ELU |
| Conv4             | size (112, 112, 4); filters : 4; kernel size : (3 x 3); stride : (1 x 1); padding : SAME; activation : ELU |   
| Max Pooling       | kernel size: (2x2); stride: (2x2) |
| BatchNorm2        | features: 4; eps: 1e-05 |
| Conv5             | size (56,56,32); filters : 32; kernel size : (2 x 2); stride : (1 x 1); padding : SAME; activation : ELU |
| Conv6             | size (28, 28, 4); filters : 4; kernel size : (3 x 3); stride : (1 x 1); padding : SAME; activation : ELU |   
| Max Pooling       | kernel size: (2x2); stride: (2x2) |
| BatchNorm3        | features: 4; eps: 1e-05 |
| Flatten 	    | (28 x 28 x 4) => 3136 |
| Fully Connected 1 | neurons : 1000; activation : ELU |
| Dropout1 	    | probability : 0.5 |
| Fully Connected 2 | neurons : 1000; activation : ELU |
| Dropout2 	    | probability : 0.6|
| Output 	    | size : (136 x 1) |
