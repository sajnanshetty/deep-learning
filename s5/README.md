## SESSION 5 assignment:

### Code1: Created Skeleton code

Target:
  * Achieve 99.4% accuracy using skeleton code.
  * Set the Transforms
  * Set Basic Working Code
  * Set Basic Training  & Test Loop
  
Results:
  * Parameters: 6.3M
  * Batch size: 128
  * epoch: 20
  * Best Training Accuracy: 100.00
  * Best Test Accuracy: 99.30
  
Analysis:
  * Extremely Heavy Model for such a problem
  * Model is over-fitting, but we are changing our model in the next code
	
### Code2: Created with lighter code with batch normalization

Target:
  * Achieve 99.4% accuracy using lighter code.
  * Set the Transforms
  * Set Basic Working Code
  * Set Basic Training  & Test Loop
  * Applies Batch Normalization
  
Results:
  * Parameters: 10.9k
  * Batch size: 128
  * epoch: 15
  * Best Training Accuracy: 99.77
  * Best Test Accuracy: 99.20
  
Analysis:
  * Model is is overfitting.
  * The training accuracy is meeting the target accuracy.
  * Even the model pushed further the test accuracy will not be able to meet the target.
	
### Code3: Apply Regularization using drop out

Target:
  * Achieve 99.4% accuracy using lighter code.
  * Set the Transforms
  * Set Basic Working Code
  * Set Basic Training  & Test Loop
  * Applied Batch normalization.
  * Applied Drop out with .05%
  
Results:
  * Parameters: 10.9k
  * Batch size: 128
  * epoch: 15
  * Best Training Accuracy: 98.88
  * Best Test Accuracy: 99.06
  
Analysis:
  * Model is not overfitting.
  * Used 15 epochs to train model could not meet target accuracy.
  * If the model pushed further the test accuracy will be able to meet the target.
	
### Code4: Apply GAP

Target:
  * Achieve 99.4% accuracy using lighter code.
  * Set the Transforms
  * Set Basic Working Code
  * Set Basic Training  & Test Loop
  * Applied Batch normalization.
  * Applied Drop out with .05%
  * Applied GAP
  
Results:
  * Parameters: 6k
  * Batch size: 128
  * epoch: 15
  * Best Training Accuracy: 96.12
  * Best Test Accuracy: 98.76
  
Analysis:
  * Model is not overfitting.
  * This works better. But Model did not meet the target accuracy.
  * This model is very well trained even with fewer parameters.
	
### Code5: Apply Image Augumentaion maintaining parameters below 10k

Target:
  * Achieve 99.4% accuracy using lighter code.
  * Set the Transforms
  * Set Basic Working Code
  * Set Basic Training  & Test Loop
  * Applied Batch normalization
  * Applied GAP
  * Added LR scheduler with the step of 1
  * Applied Image Augumentaion with degree 9
  
Results:
  * Parameters: 9.5k
  * Batch size: 64
  * epoch: 14
  * Best Training Accuracy: **99.2**
  * Best Test Accuracy: **99.5** in Epoch 12(Also achieved target accuracy in following Epoch 10: 99.45, Epoch 13: 99.41, Epoch 14: 99.4)
  
Analysis:
  * Model is not overfitting.
  * Best model with consistency in the accuracy.