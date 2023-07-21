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
    return int(sum(inputs * weights[:,1]) > 0)


def simple_correction(weights, error):
    return numpy.stack([weights[:,0] + 1, weights[:,1] + error/(weights[0,0]+1)],axis=1)


def train_record(weights, inputs, label,activation=boolean_activation, correction=simple_correction):
    output = activation(inputs, weights)
    error = output - label
    return simple_correction(weights, error)


def train_dataset(weights, data_loader):
    total_error = 0
    local_weights = weights.copy()
    for inputs, label in data_loader:
        local_weights = train_record(local_weights, inputs, label)
    return local_weights


def test_record(weights, inputs, label, activation=boolean_activation):
    output = activation(inputs, weights)
    error = output - label
    return error


def check_accuracy(weights, data_loader, activation=boolean_activation):
    errors = [int(bool(activation(inputs, weights)-label)) for inputs, label in data_loader]
    return sum(errors)/len(errors)


def train_example():
    weights = numpy.zeros((len(data[0]), 2))
    for i in range(10):
        data_loader = get_record(data, labels)
        weights = train_dataset(weights, data_loader)
        print(weights)
        data_loader = get_record(data, labels)
        print("accuracy", check_accuracy(weights, data_loader))
