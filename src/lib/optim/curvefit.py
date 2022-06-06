# Copyright (c) 2021-2022 Javad Komijani

"""This module is for curve fitting using pytorch

**NOTE** this package will change....
"""


import torch
import time


class CurveFitter:
    """
    Parameters:
    -----------
    net: torch.nn.Module
    """

    def __init__(self, net):
        self.net = net
        self.train_history = {'loss': []}
        
    def fit(self, x, y, weight_matrix=None,
            n_epochs=1000,
            optimizer_class=torch.optim.Adam,
            learning_rate=0.001,
            weight_decay=1e-2,
            scheduler=None,
            loss_fn=None,
            print_stride=100
           ):
        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        else:
            self.loss_fn = loss_fn
        self.print_dict = dict(stride=print_stride)
        opt_kwargs = dict(lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = optimizer_class(self.net.parameters(), **opt_kwargs)
        self.scheduler = None if scheduler is None else scheduler(self.optimizer)
        self._train(x, y, weight_matrix, n_epochs)
        
    def _train(self, x, y, weight_matrix, n_epochs):
        T1 = time.time()
        last_epoch = len(self.train_history["loss"]) + 1
        for epoch in range(last_epoch, last_epoch + n_epochs):
            self._train_step(x, y, weight_matrix)
            self.print_fit_status(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
        T2 = time.time()
        print("Time = {:.3g} sec.".format(T2 - T1))
        
    def _train_step(self, x, y, weight_matrix):
        if weight_matrix is None:
            loss = self.loss_fn(self.net(x), y)
        else:
            loss = self.loss_fn(self.net(x), y, weight_matrix)
        self.optimizer.zero_grad()  # clears old gradients from the last steps.
        loss.backward()
        self.optimizer.step()
        self.train_history['loss'].append(loss.item())
        
    @staticmethod
    def loss_weighted_mse(y_hat, y, weight_matrix):
        dy = y_hat - y
        return torch.matmul(dy, torch.matmul(weight_matrix, dy))
        
    def print_fit_status(self, epoch):
        stride = self.print_dict['stride']
        if (stride is None) or not (epoch == 1 or epoch % stride == 0):
            return
        print("Epoch {0} | loss = {1}".format(epoch, self.train_history['loss'][epoch-1]))
