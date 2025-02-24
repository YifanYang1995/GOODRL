import numpy as np


# Reference:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):

    def __init__(self, theta, epsilon=1e-08):
        self.theta = theta
        self.epsilon = epsilon
        self.dim = len(theta)
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        self.theta += step
        return self.theta

    def _compute_step(self, globalg):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def step2(self, gradients):   # I added
        # Assuming gradients is a flat array of gradients
        step = self._compute_step(gradients)
        self.theta -= step  # Update the parameters
        self.t += 1  # Increment time step