## SESSION4 assignment:

### Target is to achieve 99.4% accuracy using less than 20k parameters using MNSIT data set.

### Solution: Using below approachs achieved 99.52% accuracy with parameters 18,106
   * Each convolution blocks used 16 kernels and also 2 times max pooling.
   * Adding Batch Normalization and Drop out(0.05%) in each layer except last convolution block.
   * Used 20 epochs and 64 as batch size.
   * The target is achieved without using Fully Connected and GAP layer.
