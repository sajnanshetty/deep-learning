# 25 Misclassified Images for below cases

## without L1/L2 with BN

![BN_misclassified](https://github.com/sajnanshetty/deep-learning/blob/master/s6/images/BN.PNG) 

## without L1/L2 with GBN

![GBN_misclassified](https://github.com/sajnanshetty/deep-learning/blob/master/s6/images/GBN.PNG)

# Validation accuracy and losses are calculated for the below cases
1. without L1/L2 with BN
2. without L1/L2 with GBN
3. with L1 with BN
4. with L1 with GBN
5. with L2 with BN
6. with L2 with GBN
7. with L1 and L2 with BN
8. with L1 and L2 with GBN

## Validation Accuracy Graph of all above 8 models.

![accuracy_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s6/images/accuracy.PNG)

## Validation Loss of all above 8 models

![loss_graph](https://github.com/sajnanshetty/deep-learning/blob/master/s6/images/loss.PNG)

#### Analysis from graph:
1. Model1(with BN without L1 and L2): 
Maximum Train accuracy:  99.43
Test accuracy:  99.46 
gives a good accuracy without overfitting and which is consistent with the last few epochs
2. Model5 (with L2 and BN):
Maximum Train accuracy:  99.37
Test accuracy:  99.53
Applying L2 Regularization is helping to increase the model accuracy.
3. Model3(with L1 and BN)  
Maximum Train accuracy: 98.11
Validation accuracy: 98.94
Applying L1 regularization is not benefitting much as its validation accuracy is less compared to all other models.




