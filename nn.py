import numpy

# Generate some data.
numpy.random.seed(99)
data = numpy.random.random((1000, 1))
transform = numpy.vectorize(lambda x: int(x > 0.8))
labels = transform(data)


def get_record(data, labels):
    """A generator that gives the next row of data."""
    for i in range(len(data)):
        yield data[i], labels[i][0]


def boolean_activation(inputs, weights):
    """
    If K=0 learning is not possible, with one node and without a edge bias.
    This is because it is not possible to map a float between 0 and 1,
    to be greater than 0 if the value is greater than .8 just by scaling it.
    Maybe this can be solved with multiple nodes?
    Is it meaningful to have more nodes than inputs on the first layer?
    """
    K = 1
    return int(sum(inputs * weights) > K)


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
