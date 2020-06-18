# Goal :
```
 1. Write a code that draws below sample curve (without the arrows). 
```
 ![sample_triangle](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/triangle_sample.PNG)
``` 
2. Writting Code in below architecture for Cifar10:
 PrepLayer:
    Conv 3x3 s1, p1) >> BN >> RELU [64k]
 Layer1:
     X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
     R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
     Add(X, R1)
     Layer 2:
     Conv 3x3 [256k]
     MaxPooling2D
     BN
     ReLU
 Layer 3:
     X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
     R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
     Add(X, R2)
     MaxPooling with Kernel Size 4
     FC Layer 
     SoftMax
 3. Use One Cycle Policy such that:
     Total Epochs = 24
     Max at Epoch = 5
     LRMIN = FIND
     LRMAX = FIND
     NO Annihilation
 4. Use transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
 5. Use Batch size = 512
 6. Achieve Target Accuracy: 90%. 
```

### Albumentaion applied images 
![albumentaion1](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/albumentation.PNG)

### Cyclic triangle plot using max lr:1, start lr:1e-4 and num of iterations:3395
![clr](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/triangle/clr_triangle.PNG)

### Finding Max LR using range test wit inputs max lr:1, start lr:1e-4 and num of iterations:3395 i.e 35 epochs
![lr_finder](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/lr_accuracy.PNG)

### One Cycle triangle using all trained lr for 24 epochs
![one_cycle](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/triangle/onecycle_triangle.png)

### Train and Validation Accuracy Change:
![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/train_validation_accuracy_change.png)

### Accuracy and Loss for both Train and Validation:
![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/all_graph.png)

### GradCam and heatmap visualization for 25 missclassified images in layer4
![misclassified1](https://github.com/sajnanshetty/deep-learning/blob/master/s11/images/gradcam.png)

### Summary:
1. Used L2 Regularization, scheduler OneCycleLR with max LR value `0.08326219372576217` which
 is found using lr finder range test.
2. Batch Size 512 and Epochs=24
3. Parameters = 6,573,130
4. Applied albumentations for train set such as Normalize, HorizontalFlip, Cutout and ToTensor
5. Applied albumentations for test set such as Normalize and ToTensor
7. Maximum Train accuracy:  `96.56`
8. Test accuracy:  `90.57`

#### Experiment result of OneCycleLR scheduler for different values of div_factor(1 to 5):

``` 
Div Factor 5:
Maximum Training Accuracy =  94.13
Maximum Testing Accuracy =   89.17

Div Factor 6:
Maximum Training Accuracy =  95.23
Maximum Testing Accuracy =  89.53

Div Factor 7:
Maximum Training Accuracy =  95.5
Maximum Testing Accuracy =  89.53

Div factor 8:
Maximum Training Accuracy =  96.1
Maximum Testing Accuracy =  90.52

Div factor 9:
Maximum Training Accuracy =  96.15
Maximum Testing Accuracy =  90.18

Div factor 10:
Maximum Training Accuracy =  96.56
Maximum Testing Accuracy =  90.57
``` 

Conclusion: Div factor 8 to 10 achieves target accuracy but the model is outfitting.




