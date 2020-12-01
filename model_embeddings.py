#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        # Embedding dimensionality for characters
        self.embed_size = embed_size
        self.e_char = 50
        self.e_word = embed_size
        self.dropout_prob = 0.3
        self.embedding = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim=self.e_char, padding_idx=vocab.char2id['<pad>'])
        self.cnn = CNN(self.e_char, self.e_word)
        self.highway = Highway(self.e_word)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        # (sentence_length, batch_size, m_word) --> (sentence_length, batch_size, m_word, e_char)
        char_embeddings = self.embedding(input_tensor)
        # (sentence_length, batch_size, e_char, m_word)
        x_conv_in = char_embeddings.permute(0, 1, 3, 2)
        # (sentence_length, batch_size, e_word)
        x_conv_out = torch.stack([self.cnn(x_conv_in_batch) for x_conv_in_batch in x_conv_in], dim=0)
        # (sentence_length, batch_size, e_word)
        x_highway_in = x_conv_out
        # (sentence_length, batch_size, e_word) 
        x_highway_out = self.highway(x_highway_in)
        # (sentence_length, batch_size, e_word)
        return self.dropout(x_highway_out)

        ### END YOUR CODE
