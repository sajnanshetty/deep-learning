## SESSION5 assignment:

#### Code1: Created Skeleton code

### Target:
	* Achieve 99.4% accuracy using skeleton code.
	* Set the Transforms
	* Set Basic Working Code
	* Set Basic Training  & Test Loop
### Results:
	* Parameters: 6.3M
	* Batch size: 128
	* ephoch: 20
	* Best Training Accuracy: 100.00
	* Best Test Accuracy: 99.30
### Analysis:
	* Extremely Heavy Model for such a problem
	* Model is over-fitting, but we are changing our model in the next code
	
	
#### Code2: Created lighter code

### Target:
	* Achieve 99.4% accuracy using lighter code.
	* Set the Transforms
	* Set Basic Working Code
	* Set Basic Training  & Test Loop
### Results:
	* Parameters: 10.7k
	* Batch size: 32
	* ephoch: 15
	* Best Training Accuracy: 99.29
	* Best Test Accuracy: 98.6
### Analysis:
	* Model is is under fitting.
	* Even model pushed further the test accuracy will not be able to meet the target.
	
#### Code2: Created with lighter code with batch normalization

### Target:
	* Achieve 99.4% accuracy using lighter code.
	* Set the Transforms
	* Set Basic Working Code
	* Set Basic Training  & Test Loop
	* Applied Batch Normalization
### Results:
	* Parameters: 10.9k
	* Batch size: 128
	* ephoch: 15
	* Best Training Accuracy: 99.77
	* Best Test Accuracy: 99.20
### Analysis:
	* Model is is over fitting.
	* The train accuracy is meeting the target accuracy.
	* Even model pushed further the test accuracy will not be able to meet the target.
	
#### Code3: Apply Reguralization using droup out

### Target:
	* Achieve 99.4% accuracy using lighter code.
	* Set the Transforms
	* Set Basic Working Code
	* Set Basic Training  & Test Loop
	* Applied Batch normalization.
	* Applied Drop out with .05%
### Results:
	* Parameters: 10.9k
	* Batch size: 128
	* ephoch: 15
	* Best Training Accuracy: 98.88
	* Best Test Accuracy: 99.06
### Analysis:
	* Model is is not over fitting.
	* Used 15 epochs to train model could not meet target accuracy.
	* If model pushed further the test accuracy will be able to meet the target.
	
#### Code4: Apply GAP

### Target:
	* Achieve 99.4% accuracy using lighter code.
	* Set the Transforms
	* Set Basic Working Code
	* Set Basic Training  & Test Loop
	* Applied Batch normalization.
	* Applied Drop out with .05%
	* Applied GAP
### Results:
	* Parameters: 6k
	* Batch size: 128
	* ephoch: 15
	* Best Training Accuracy: 96.12
	* Best Test Accuracy: 98.76
### Analysis:
	* Model is not over fitting.
	* This works better.But Model did not meet the target accuracy.
	* This model is very well trained even with less parameters.
	
#### Code5: Apply Image Augumentaion maintaining parameters below 10k

### Target:
	* Achieve 99.4% accuracy using lighter code.
	* Set the Transforms
	* Set Basic Working Code
	* Set Basic Training  & Test Loop
	* Applied Batch normalization
	* Applied GAP
	* Added LR scheduler with step of 1
	* Applied Image Augumentaion with degree 9
### Results:
	* Parameters: 9.5k
	* Batch size: 64
	* ephoch: 14
	* Best Training Accuracy: 99.2
	* Best Test Accuracy: 99.5 in Epoch 12(Also achieved target accuracy in following Epoch 10: 99.45, Epoch 13: 99.41, Epoch 14: 99.4)
### Analysis:
	* Model is not over fitting.
	* Best model with consitency in accuracy.