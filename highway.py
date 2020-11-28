#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, e_word):
        """ Highway Network Model (Srivastava et al)
        @param e_word (int): the embedding size for a word
        """
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(in_features=e_word, out_features=e_word, bias=True)
        self.w_gate = nn.Linear(in_features=e_word, out_features=e_word, bias=True)
        
    def forward(self, x_conv_out):
        """ Take a batch of of words and run them through the Highway network.
        @param x_conv_out (Tensor) (batch_size, e_word): 
            the output of the convolution.
        @returns x_highway (Tensor) (batch_size, e_word) :
            the output of the highway network applied to x_conv_out.
        """
        # (batch_size, e_word)
        x_proj = nn.functional.relu(self.w_proj(x_conv_out))
        # (batch_size, e_word)
        x_gate = torch.sigmoid(self.w_gate(x_conv_out))
        
        # Note : * is the Hadamard product, not matrix multiplication
        # (batch_size, e_word)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway
        


### END YOUR CODE 