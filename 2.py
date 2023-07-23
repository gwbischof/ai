"""
Problem:
- It seems like a single node NN is limited to approximating a linear function.
- What are the limitations of a single node binary activation NN?
    - Can I check if a binary number between 0-3 is equal to 3 with just a single node? How many nodes do I need to do this?

Results:
- I get accuracy of 100%
- A single node with binary activation is sufficient to check if a number in binary format is greater than or equal to some value. Can it check if it is equal to 3?

To do:
- Use an extra bit to the input, so that the maximum integer value is 7, and then check if it can guess if a value is 3.
"""

import numpy

# Generate some data.
numpy.random.seed(99)
data = numpy.random.choice(a=[0,1], size=(1000,2))

# Generate the labels
# The labels is true if both inputs are true.
# If the inputs are interpreted as a integer in binary,
# Then the label is true if the number is 3.
labels = data[:,0] & data[:,1]


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
    K = 2
    return int(sum(inputs * weights) > K)


# Currently both weights are updated the same way.
# I will probably need to figure out how to update them independantly.
# Hmm, maybe this will work with matching weights, because the inputs are summed
# and it just becomes a linear function.
def simple_correction(weights, error):
    weights[:] = weights + (0.01*error)


def train_record(weights, inputs, label,activation=boolean_activation, correction=simple_correction):
    output = activation(inputs, weights)
    error = label-output
    before = weights.copy()
    simple_correction(weights, error)
    print(output, label, error, before, weights)


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
