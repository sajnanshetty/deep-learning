# Goal :
1. Apply Albumentaion
2. Implement GradCam function as a module. 


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

## Albumentaion applied images for all cifa10 classes
![albumentaion1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion.png)
![albumentaion2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion1.png)
![albumentaion3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion2.png)
![albumentaion4](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion3.png)
![albumentaion5](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion4.png)
![albumentaion6](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion5.png)
![albumentaion7](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion6.png)
![albumentaion8](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion7.png)
![albumentaion9](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion8.png)
![albumentaion10](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/albumentaion/albumentaion9.png)

## GradCam and Heatmap visualization for few correctly classified images in Layer4
![correctclassified1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_correctclassified.png)
![correctclassified2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_correctclassified1.png)
![correctclassified3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_correctclassified2.png)

## GradCam and Heatmap visualization for few misclassified classified images in Layer4
![correctclassified1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_misclassified.png)
![correctclassified2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_misclassified1.png)
![correctclassified3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_misclassified2.png)

## GradCam and Heatmap visualization of an image in Layer1, Layer2, Layer3, Layer4
![layer1](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer1.png)
![layer2](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer2.png)
![layer3](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer3.png)
![layer4](https://github.com/sajnanshetty/deep-learning/blob/master/s9/images/gradcam_result_images/gradcam_layer4.png)


### Summary:
1. Used L2 Regularization, schedular OneCycleLR
2. Batch Size 64 and Epochs=20
3. Parameters = 11,173,962
4. Applied albumentations for train set such as Normalize, HorizontalFlip, Cutout and ToTensor
5. Applied albumentations for test set such as Normalize and ToTensor
6. Implement GradCam function as a module. 
7. Maximum Train accuracy:   92.97
8. Test accuracy:  92.11




