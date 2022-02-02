import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime
import torch
import torch.nn as nn



from torch.nn import MSELoss
from torch.optim import Adam
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
"""
This code was refered from the https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
"""

class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 2048)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(2048, 100)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(100, 1)
        xavier_uniform_(self.hidden3.weight)
        #self.double()
        #self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = X.float()
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        #X = self.act3(X)
        return X
