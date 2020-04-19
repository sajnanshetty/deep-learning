SESSION4 assignment:

Target is to achieve 99.4% accuracy using less than 20k parameters using MNSIT data set.

Solution: Using below approachs achieved 9.52% with parameters 18,106
1]Used number of kernels 16 in each layer and also 2 times max pooling.
2]Adding Batch Normalization and Drop out 0.05% in each layer except last convolution block.
3]The target is achieved without using Fully Connected and GAP layer.
