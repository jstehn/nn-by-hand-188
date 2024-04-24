import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return -1 if nn.as_scalar(self.run(x)) < 0 else 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        trained = False
        while not trained:
            trained = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    trained = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.learning_rate = 0.05
        self.w0 = nn.Parameter(1, 64)
        self.b0 = nn.Parameter(1, 64)
        self.w1 = nn.Parameter(64, 64)
        self.b1 = nn.Parameter(1, 64)
        self.w2 = nn.Parameter(64, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x_w0 = nn.Linear(x, self.w0)
        x_w0_b0 = nn.AddBias(x_w0, self.b0)
        relu_0 = nn.ReLU(x_w0_b0)
        r0_w1 = nn.Linear(relu_0, self.w1)
        r0_w1_b1 = nn.AddBias(r0_w1, self.b1)
        relu_1 = nn.ReLU(r0_w1_b1)
        r1_w2 = nn.Linear(relu_1, self.w2)
        r1_w2_b2 = nn.AddBias(r1_w2, self.b2)
        return r1_w2_b2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(1):
            loss = self.get_loss(x, y)
            grads = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])
            self.w0.update(grads[0], -self.learning_rate)
            self.w1.update(grads[1], -self.learning_rate)
            self.b0.update(grads[2], -self.learning_rate)
            self.b1.update(grads[3], -self.learning_rate)
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """
    def __init__(self):
        self.learning_rate = 0.05
        self.batch_size = 60
        self.w0 = nn.Parameter(784, 512)
        self.b0 = nn.Parameter(1, 512)
        self.w1 = nn.Parameter(512, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        z0 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        a0 = nn.ReLU(z0)
        z1 = nn.AddBias(nn.Linear(a0, self.w1), self.b1)
        a1 = nn.ReLU(z1)
        z2 = nn.AddBias(nn.Linear(a1, self.w2), self.b2)
        a2 = nn.ReLU(z2)
        z3 = nn.AddBias(nn.Linear(a2, self.w3), self.b3)
        return z3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        parameters = [self.w0, self.w1, self.w2, self.w3, self.b0, self.b1, self.b2, self.b3]
        for x,y in dataset.iterate_forever(self.batch_size):
            grads = nn.gradients(self.get_loss(x, y), parameters)
            for i in range(len(parameters)):
                parameters[i].update(grads[i], -self.learning_rate)
            if dataset.get_validation_accuracy() > 0.978:
                return



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.batch_size = 50
        self.dimensions = len(self.languages)
        self.hidden_dimensions = 612
        self.w = nn.Parameter(self.num_chars, self.hidden_dimensions)
        self.b = nn.Parameter(1, self.hidden_dimensions)
        self.w1 = nn.Parameter(self.hidden_dimensions, self.hidden_dimensions)
        self.b1 = nn.Parameter(1, self.hidden_dimensions)

        self.w_hidden = nn.Parameter(self.hidden_dimensions, self.hidden_dimensions)
        self.b_hidden = nn.Parameter(1, self.hidden_dimensions)

        self.w0 = nn.Parameter(self.hidden_dimensions, self.hidden_dimensions)
        self.b0 = nn.Parameter(1, self.hidden_dimensions)
        self.w1 = nn.Parameter(self.hidden_dimensions, self.hidden_dimensions)
        self.b1 = nn.Parameter(1, self.hidden_dimensions)
        self.w_final = nn.Parameter(self.hidden_dimensions, self.dimensions)
        self.b_final = nn.Parameter(1, self.dimensions)
        self.learning_rate = 0.02

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w), self.b))
        for i in range(1, len(xs)):
            h = nn.AddBias(nn.Add(nn.Linear(xs[i], self.w), nn.Linear(h, self.w_hidden)), self.b_hidden)
        h = nn.AddBias(nn.Linear(h, self.w0), self.b0)
        h = nn.ReLU(h)
        h = nn.AddBias(nn.Linear(h, self.w1), self.b1)
        h = nn.ReLU(h)
        h = nn.AddBias(nn.Linear(h, self.w_final), self.b_final)
        return h

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        parameters = [self.w, self.w_hidden, self.w0, self.w1, self.w_final,
                      self.b, self.b_hidden, self.b0, self.b1, self.b_final]
        for x, y in dataset.iterate_forever(self.batch_size):
            grads = nn.gradients(self.get_loss(x, y), parameters)
            for i in range(len(parameters)):
                parameters[i].update(grads[i], -self.learning_rate)
            if dataset.get_validation_accuracy() > 0.85:
                return
