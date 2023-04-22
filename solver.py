import numpy as np
from projection import proj_l1ball as proj
import wandb


class NetworkSolver:
    def __init__(self, generator, network, gamma, args):
        self.generator = generator
        self.network = network
        self.gamma = gamma
        self.args = args
        self.theta = np.zeros((self.generator.m, self.generator.d, 1))
        self._round = 0

    def fit(self):
        # initialize parameters we need
        X, Y =self.generator()
        critical_iteration = []
        # iterates!
        while self._round < self.args.max_iter:
            if self.args.solver == 'cta':
                self.theta = self.communicate(self.theta)
            for i in range(self.args.local_computation):
                self.compute(X, Y)
            if self.args.solver == 'atc':
                self.theta = self.communicate(self.theta)
            log_loss = np.log(self.iter_loss())
            self.log(log_loss)
            if len(critical_iteration)==0 and log_loss - (self.args.centerized_loss) < 0.03:
                wandb.run.summary["critical iteration_0.03"] = self._round
                critical_iteration.append(self._round)
            if len(critical_iteration)==1 and log_loss - (self.args.centerized_loss) < 0.02:
                critical_iteration.append(self._round)
                wandb.run.summary["critical iteration_0.02"] = self._round
            if len(critical_iteration)==2 and log_loss - (self.args.centerized_loss) < 0.01:
                critical_iteration.append(self._round)
                wandb.run.summary["critical iteration_0.01"] = self._round
            self._round += 1
    
    def compute(self):
        raise NotImplementedError

    def communicate(self, matrix):
        assert len(matrix.shape) == 3
        matrix = np.expand_dims(np.linalg.matrix_power(self.network, self.args.communication) @ matrix.squeeze(axis=2), axis=2)
        return matrix

    def show_param(self):
        raise NotImplementedError


    def iter_loss(self):
        repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
        loss = np.linalg.norm(self.theta.squeeze(axis=2) - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
        return loss

    def log(self, loss):
        wandb.log({"iter_loss (log scale)": loss}, step=self._round)


class NetworkGD(NetworkSolver):
    def __init__(self, generator, network, gamma, args):
        super().__init__(generator, network, gamma, args)

    def compute(self, x, y):
        m, n, d = x.shape
        r = self.generator.radius * self.args.radius_const
        gamma = self.gamma()
        if self.args.constraint == 'lagrangian':
            beta = m * n * gamma / (self.generator.max_eig * gamma + n)
            self.theta = (self.theta - gamma / n * x.transpose(0,2,1) @ (x @ self.theta - y)) * beta / (m * gamma) + self.theta * (m * gamma - beta) / (m * gamma)
            tmp = (np.sign(self.theta) * np.clip(np.abs(self.theta) - gamma * self.args.lmda, 0, None)).squeeze(axis=2)
            mask = np.greater(np.linalg.norm(tmp, ord=1, axis=1, keepdims=True), r).astype(int)
            self.theta = np.expand_dims(proj(self.theta.squeeze(axis=2), r) * mask + tmp * (1-mask), axis=2)

        elif self.args.constraint == 'projected':
            self.theta -= gamma / n * x.transpose(0,2,1) @ (x @ self.theta - y)
            self.theta = np.expand_dims((proj(self.theta.squeeze(axis=2), self.generator.radius)), axis=2)


