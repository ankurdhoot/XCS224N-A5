#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, e_char, e_word, kernel_size=5):
        """ Implements the convolutional network in the char level encoder.
        @param e_char (int): the embedding size for chars (in_channel)
        @param e_word (int): the embedding size for words (out_channel)
        @param kernel_size (int): the size of the convolutional kernel
        """
        super(CNN, self).__init__()
        # char embedding dimension
        self.e_char = e_char
        # word embedding dimension
        self.e_word = e_word
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=kernel_size)
        
    def forward(self, x_in):
        """ Run the input through the convolutional network. 
        MaxPool(Relu(Conv1d(x_in)))
        e_char is the dimension of a char encoding.
        m_word is the maximum word length to which all words are padded.
        @param x_in (tensor) (batch_size, e_char, m_word): the input tensor
        @return x_maxpool (tensor) (batch_size, e_word): the output tensor
        """
        # (batch_size, e_char, m_word) --> (batch_size, e_word, m_word - kernel_size + 1)
        x_conv = self.conv1d(x_in)
        # (batch_size, e_word, m_word - kernel_size + 1)
        x_relu = nn.functional.relu(x_conv)
        # Setting the kernel size to x_relu.size()[-1] takes the max over the
        # entire last dimension.
        # (batch_size, e_word, 1)
        x_maxpool = nn.functional.max_pool1d(input=x_relu, kernel_size=x_relu.size()[-1])
        # (batch_size, e_word)
        x_maxpool = torch.squeeze(x_maxpool, dim=-1)
        return x_maxpool

### END YOUR CODE

