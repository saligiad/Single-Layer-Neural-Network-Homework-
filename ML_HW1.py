import random
import numpy
import csv
import pandas


class Perceptron:
    """ Perceptron Class 
        For learning to identify a digit in a 28x28 gray-scale image.  Has data 
        for a learning rate, bias weight, and vectorized array with one weight 
        for each pixel.
    """
    def __init__(self,rate):
        """ __init__ Method
            Initializes learning rate with an argument and selects random values in
            the range [-0.5,0.5) for the bias and weights
        """
        self.numInputs = 784
        self.rate = rate
        self.bias = random.random() - 0.5
        self.weights = numpy.random.random(self.numInputs) - 0.5
        
    def train(self, input, target):
        """ train Method
            Takes a vectorized array of inputs (pixel intensity values, scaled to 
            the range [0,1]) and a target (1 if the perceptron should give a positive 
            result and 0 otherwise).  
            
            Computes the dot product of input and weights, 
            adds the bias and determines whether the identification is positive or 
            negative, then adjusts the weights if the identification was incorrect.

            Returns the activation (dot product plus bias) of the perceptron
        """
        ret = self.bias + self.weights.dot(input)
        if ret > 0:
            a = target - 1
        else:
            a = target - 0

        if a != 0:
            self.bias += rate * a
            for i in range(self.numInputs):
                self.weights[i] += rate * a * input[i]
        return ret
            
    def test(self, input):
        """ test Method
            Takes an vectorized array of inputs and results the dot product of the
            input and weights. Assumes the dot product can be taken.
        """
        return self.bias + self.weights.dot(input)


# Data Section:
# Learning rate for the Perceptrons
rate = 1
# Perceptron array, index corresponds to identifying digit
pArray = [Perceptron(rate) for i in range(10)]
# 2D Array, one row per image, one column per pixel
trainData = []
# Array of the correct values for each training image
trainTargets = []
# 2D Array, one row per image, one column per pixel
testData = []
# Array of the correct values for each test image
testTargets = []
# Array for training data accuracy results for each epoch 
trainAccuracy = [0 for i in range (51)]
# Array for testing data accuracy results for each epoch 
testAccuracy = [0 for i in range (51)]
# Confusion matrix, used to store what each test image was identified as
confusion = [[0 for j in range(10)]for i in range(10)]


# Code Section
# Read in the data
trainData = pandas.read_csv('mnist_train.csv').values
testData = pandas.read_csv('mnist_test.csv').values

"""
with open('mnist_train.csv') as train:
    reader = csv.reader(train, delimiter=',')
    for entry in reader:
        trainTargets.append(int(entry[0]))
        trainData.append(numpy.array(entry[1:]).astype(float)/255)

with open('mnist_test.csv') as test:
    reader = csv.reader(test, delimiter=',')
    for entry in reader:
        testTargets.append(int(entry[0]))
        testData.append(numpy.array(entry[1:]).astype(float)/255)
"""

# Test then train the data for 51 epochs
for i in range(51):  
    print(i)
    # Test the data for the current epoch
    for j,instance in enumerate(testData): 
        # Initialize variables to track the max perceptron activation
        max = float('-inf')
        result = max
        maxIdx = -1
        # Iterate over each perceptron
        for p,perceptron in enumerate(pArray):
            result = perceptron.test(instance[1:])
            if result > max:
                max = result
                maxIdx = p
        # If the correct perceptron had the max activation, increase accuracy
        if maxIdx == instance[0]:
            testAccuracy[i] += 1
        # Calculate the confusion matrix in the last epoch
        if i == 50:
            confusion[instance[0]][maxIdx] += 1

    # Train the data for the current epoch
    for j, instance in enumerate(trainData):
        # Initialize variables to track the max perceptron activation
        max = float('-inf')
        result = max
        maxIdx = -1
        # Iterate over each perceptron
        for p,perceptron in enumerate(pArray):
            # Set the target for training based on instance label
            if p == instance[0]:
                t = 1
            else:
                t = 0
            # Train the perectron and save the result if its the largest
            result = perceptron.train(instance[1:],t)
            if result > max:
                max = result
                maxIdx = p
        if maxIdx == instance[0]:
            trainAccuracy[i] += 1

    # Computer fractional accuracy for this epoch
    testAccuracy[i] /= len(testData)
    trainAccuracy[i] /= len(trainData) 

# Write the accuracy and confusion results to a csv file
if rate == 0.01:
    rateTxt = '0-01'
elif rate == 0.1:
    rateTxt = '0-1'
elif rate == 1:
    rateTxt = '1'
print('Writing Accuracy')
# Write accuracy for the training data
writeFile = 'trainAccuracy' + rateTxt + '.csv'
with open(writeFile, mode='w') as accWrite:
    writer = csv.writer(accWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i,acc in enumerate(trainAccuracy):
        writer.writerow([i,acc])
# Write accuracy for the testing data
writeFile = 'testAccuracy' + rateTxt + '.csv'
with open(writeFile, mode='w') as accWrite:
    writer = csv.writer(accWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i,acc in enumerate(testAccuracy):
        writer.writerow([i,acc])
# Write the confusion matrix
"""
print('Writing Confusion')
writeFile = 'confusion' + rateTxt + '.csv'
with open(writeFile, mode='w') as confWrite:
    writer = csv.writer(confWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(10):
        writer.writerow([confusion[i]])
"""