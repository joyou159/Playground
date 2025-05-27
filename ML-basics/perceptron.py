import numpy as np

class Perceptron:
    """
    Parameters
    -------------
    eta (float): Learning rate.
    n_iter/epochs (int): Number of interations through the training dataset.
    random_state (int): Random number generator seed for random weight initianlization.

    Attributes
    -------------
    W_ (1d narray): weights after fitting.
    b_ (Scalar): bias after fitting.
    errors_ (list): Number of misclassifications per epoch.
    """
    def __init__(self, eta, n_iter, random_state):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        fitting training data.

        Parameters
        --------------
        X {array-like}, shape = [n_examples, n_features]: training data matrix.
        y {array-like}, shape = [n_examples]: target values.

        Return
        --------
        self (Object): Perceptron Trained Object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size= X.shape[1])
        self.b_ = np.float(0.)
        self.errors_ = list()

        for _ in range(self.n_iter):
            error = 0
            for xi, y in zip(X, y):
                update = self.eta * (y - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0) # any measure of loss
            
            self.errors_.append(error)
        return self
    
    def net_input(self, X):
        return (X @ self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step thresholding"""
        return self.net_input(X) >= 0 # return 1 

