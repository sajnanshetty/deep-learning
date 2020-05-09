# Goal :
1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (basically 3 MPs)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 
8. upload to Github

## Validation Accuracy:

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s7/images/validation_accuracy_graph.png)

## Validation Loss:

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s7/images/validation_loss_graph.png)

## Train Accuracy:

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s7/images/train_accuracy_graph.png)

## Train Loss:

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s7/images/train_loss_graph.png)

## 25 Misclassified Images:
 
![misclassified](https://github.com/sajnanshetty/deep-learning/blob/master/s7/images/cifar_misclassified_images.png) 

### Summary:
1. Used L2 Regularization, Batch Normalization, Dropout, Dilated and DepthwiseConvolution.
2. Batch Size 128 and Epochs=40
3. Parameters = 174,336
3. Maximum Train accuracy:  76.91
4. Test accuracy:  80.86 
5. Receptive Field=100




