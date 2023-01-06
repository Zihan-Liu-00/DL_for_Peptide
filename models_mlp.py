import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, args, num_layers=1):
        super().__init__()
        self.args = args
        if args.task_type == 'Regression':
            last_layer = 1
        elif args.task_type == 'Classification':
            last_layer = 2
        self.src_emb = nn.Embedding(args.src_vocab_size, args.d_model)
        self.forwardCalculation = nn.Linear(args.src_len*args.d_model, last_layer)

        hidden_layer = [512,256,64,32,last_layer]

        self.e1 = nn.Linear(args.src_len*args.d_model, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e5 = nn.Linear(hidden_layer[3], hidden_layer[4])

        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm0 = nn.BatchNorm1d(hidden_layer[0])
        self.batchnorm1 = nn.BatchNorm1d(hidden_layer[1])
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer[2])
        self.batchnorm3 = nn.BatchNorm1d(hidden_layer[3])

    def forward(self, inputs):
        x = self.src_emb(inputs)
        b, s, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.reshape(b, -1)
        # x = self.forwardCalculation(x)

        h_1 = F.leaky_relu(self.batchnorm0(self.e1(x)), negative_slope=0.05, inplace=True)
        h_1 = self.dropout(h_1)
        h_2 = F.leaky_relu(self.batchnorm1(self.e2(h_1)), negative_slope=0.05, inplace=True)
        h_2 = self.dropout(h_2)
        h_3 = F.leaky_relu(self.e3(h_2), negative_slope=0.1, inplace=True)
        h_3 = self.dropout(h_3)
        h_4 = F.leaky_relu(self.e4(h_3), negative_slope=0.1, inplace=True)
        y = self.e5(h_4)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1)
        elif self.args.task_type == 'Regression':
            return y