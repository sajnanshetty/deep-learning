# Goal :
1. Apply Albumentaion
2. Implement GradCam function as a module. 

## Albumentaion applied images for all cifa10 target classes
![albumentaion1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/0.PNG)
![albumentaion2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/1.PNG)
![albumentaion3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/2.png)
![albumentaion4](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/3.PNG)
![albumentaion5](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/4.PNG)
![albumentaion6](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/5.PNG)
![albumentaion7](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/6.PNG)
![albumentaion8](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/7.PNG)
![albumentaion9](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/8.PNG)
![albumentaion10](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/9.PNG)

## GradCam and heatmap visualization of an image in layer1, layer2, layer3, layer4
![layer1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer1.png)
![layer2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer2.png)
![layer3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer3.png)
![layer4](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer4.png)

## GradCam and heatmap visualization for few correctly classified images in layer4
![correctclassified1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_correctclassified.png)
![correctclassified2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_correctclassified1.png)
![correctclassified3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_correctclassified2.png)

## GradCam and heatmap visualization for few misclassified classified images in layer4
![correctclassified1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_misclassified.png)
![correctclassified2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_misclassified1.png)
![correctclassified3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_misclassified2.png)

## Validation Accuracy:

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/validation_accuracy_graph.png)

## Validation Loss:

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/validation_loss_graph.png)

## Train Accuracy:

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/train_accuracy_graph.png)

## Train Loss:

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/train_loss_graph.png)

## 25 Misclassified Images:
 
![misclassified](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_misclssified.png) 

## 25 Correctly Classified Images:
 
![correctclassified](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_correctlyified.png)


### Summary:
1. Used L2 Regularization, schedular OneCycleLR
2. Batch Size 64 and Epochs=20
3. Parameters = 11,173,962
4. Applied albumentations for train set such as Normalize, HorizontalFlip, Cutout and ToTensor
5. Applied albumentations for test set such as Normalize and ToTensor
6. Implement GradCam function as a module. 
7. Maximum Train accuracy:   92.97
8. Test accuracy:  92.11




