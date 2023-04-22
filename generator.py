import numpy as np
import wandb

class Generator:
    def __init__(self, N, m, d, sparsity, k, noise_dev, seed) -> None:
        self.N = N
        self.m = m
        self.n = int(self.N /self.m)
        self.d = d
        self.sparsity = sparsity
        self.k = k
        self.noise_dev = noise_dev # sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.theta = self._init_ground_truth()
        self.X, self.Y = self.__generate()
        self.max_eig = np.max(np.linalg.eig(self.X.transpose(0,2,1) @ self.X)[0].real)
        self.radius = np.linalg.norm(self.theta, ord=1)
        wandb.run.summary["ground_truth radius"] = self.radius
        wandb.run.summary["sum of X"] = self.X.sum()

    def _init_ground_truth(self):
        theta = self.rng.normal(0, 1, (self.d, 1))
        theta_abs = np.abs(theta)
        threshold = np.quantile(theta_abs, 1 - self.sparsity / self.d)
        mask = theta_abs > threshold
        theta = mask * theta
        return theta

    def __generate(self):
        z = (self.rng.normal(0, 1, (self.N, self.d)) / np.sqrt(1-self.k**2)).reshape(self.m, self.n, self.d)
        X = np.ones((self.m, self.n, self.d))
        X[:, :, 0] = z[:, :, 0]
        for i in range(1, self.d):
            X[:, :, i] = self.k * X[:, :, i - 1] + z[:, :, i]
        epsilon = (self.rng.normal(0, self.noise_dev ** 2, (self.N, 1))).reshape(self.m, self.n, 1)
        Y = X @ self.theta + epsilon
        print(X.shape, Y.shape)
        return X, Y
        
    def __len__(self):
        return len(self.batch_size_generator)
    
    def __call__(self):
        return self.X, self.Y
