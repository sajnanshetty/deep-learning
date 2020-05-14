# Goal :
1. Go through this repository: https://github.com/kuangliu/pytorch-cifar
2. Extract the ResNet18 model from this repository and add it to your API/repo. 
3. Use data loader, model loading, train, and test code to train ResNet18 on Cifar10
4. Achieve Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
5. one of the layers must use Dilated Convolution


## Validation Accuracy:

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s8/images/validation_accuracy_graph.png)

## Validation Loss:

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s8/images/validation_loss_graph.png)

## Train Accuracy:

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s8/images/train_accuracy_graph.png)

## Train Loss:

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s8/images/train_loss_graph.png)

## 25 Misclassified Images:
 
![misclassified](https://github.com/sajnanshetty/deep-learning/blob/master/s8/images/cifar_misclassified_images.png) 

### Summary:
1. Used L2 Regularization, schedular OneCycleLR
2. Batch Size 64 and Epochs=20
3. Parameters = 174,336
3. Maximum Train accuracy:  95.69
4. Test accuracy:  92.39




