"""
Problem:
Update the correction function to have independant weight values to see if it can fix the problem.

Todo:
Making a small adjustment to the weights is never effecting the error, with only 1000 records.

It seems like binary activation is hard to train because it is difficult to determine which way to adjust the weights.

This post addresses this problem:
https://towardsdatascience.com/binarized-neural-networks-an-overview-d065dc3c94ca

I think we can swap out the binary activation with a continious output activation in order to figure out the new weights.

With binary activation and binary weights we should be able to figure out if to images are identical, by converting the images to binary first.

If have an image where each pixel is a integer, than binary weights and a single node, wont be enough.
"""

import numpy

# Generate some data.
numpy.random.seed(99)
data = numpy.random.choice(a=[0,1], size=(100000,3))

# Generate the labels
# The labels is true if the sum of the inputs is 3.
# If the inputs are interpreted as a integer in binary,
labels = data[:,0] & data[:,1] & (data[:,2]-1)*-1


def get_record(data, labels):
    """A generator that gives the next row of data."""
    for i in range(len(data)):
        yield data[i], labels[i]


def boolean_activation(inputs, weights):
    """
    If K=0 learning is not possible, with one node and without a edge bias.
    This is because it is not possible to map a float between 0 and 1,
    to be greater than 0 if the value is greater than .8 just by scaling it.
    Maybe this can be solved with multiple nodes?
    Is it meaningful to have more nodes than inputs on the first layer?
    """
    K = 0.5
    return int(sum(inputs * weights) > K)


def continious_activation(inputs, weights):
    """
    I want to restrict the output to be between 0 and 1.
    I'll just truncate for now.
    """
    MIN = 0
    MAX = 1
    result = min(sum(inputs * weights),1)
    result = max(0, result)
    return result

def simple_correction(weights, error):
    weights[:] = weights + (0.01*error)


def naive_correction(record, label, weights, activation=boolean_activation):
    """
    Taking a guess at how to update weights independantly.
    If there is an error, adjust each weight seperatly, and check the error again.
    One problem is that because there is a binary output, a change in the weight wont always have an effect on the output. Which means that training will be slower.
    Maybe one way to fix this would be to not have a binary output, then a change in the weight should always effect the output.
    For now I will stick with the binary output.
    Also we will need to figure out the direction to adjust the weight in.
    The correction api is different when we have to adjust the weights independantly.
    """
    # Constant to increment or decrement a weight by.
    # Because the output is binary the adjustment to the weight
    # can't be proporital to the error.
    K = 1

    error = label - activation(record, weights)

    if error != 0:
        for i, weight in enumerate(weights):
            temp_weights = weights.copy()
            temp_weights[i] -= K
            new_error = label - activation(record, temp_weights)
            if new_error==0:
                # Even with K=1 the error is never improved.
                breakpoint()
                weights[i] = temp_weights[i]
        for i, weight in enumerate(weights):
            temp_weights = weights.copy()
            temp_weights[i] += K
            new_error = label - activation(record, temp_weights)
            if new_error==0:
                breakpoint()
                weights[i] = temp_weights[i]


def train_record(weights, inputs, label,activation=boolean_activation, correction=naive_correction):
    output = activation(inputs, weights)
    error = label-output
    before = weights.copy()
    correction(inputs, label, weights)
    print({'record': inputs, 'label': label,
           'output': output, 'error': error})


def train_dataset(weights, data_loader):
    total_error = 0
    for inputs, label in data_loader:
        train_record(weights, inputs, label)


def check_accuracy(weights, data_loader, activation=boolean_activation):
    errors = [activation(inputs, weights)==label for inputs, label in data_loader]
    return sum(errors)/len(errors)


def train_example():
    weights = numpy.zeros(len(data[0]))
    for i in range(10):
        data_loader = get_record(data, labels)
        train_dataset(weights, data_loader)
        print(weights)
        data_loader = get_record(data, labels)
        print("accuracy", check_accuracy(weights, data_loader))
