#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        
        # TODO(ankur): How many layers do we want for the LSTM?
        self.target_vocab = target_vocab
        self.v_char = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features=self.v_char, bias=True)
        self.char_padding_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(num_embeddings=self.v_char, embedding_dim=char_embedding_size, padding_idx=self.char_padding_idx)
    

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # TODO(ankur): Need to use pack_padded_sequence here?
        # (length, batch, char_embedding_size)
        char_embeddings = self.decoderCharEmb(input)
        # (length, batch, h), ((1, batch, h), (1, batch, h))
        output, (h_n, c_n) = self.charDecoder(char_embeddings, dec_hidden)
        # (length, batch, V_char)
        s = self.char_output_projection(output)
        print("Training forward")
        return s, (h_n, c_n)
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # (length, batch)
        char_sequence = char_sequence.contiguous()
        # (length - 1, batch)
        char_sequence_input = char_sequence[:-1]
        # (length - 1, batch, char_embedding_size)
        char_embeddings_input = self.decoderCharEmb(char_sequence_input)
        # ((length - 1) * batch)
        char_sequence_output = char_sequence[1:].view(-1)      
        
        # (length - 1, batch, h), ((1, batch, h), (1, batch, h))
        output, (h_n, c_n) = self.charDecoder(char_embeddings_input, dec_hidden)
        # (length - 1, batch, V_char)
        s = self.char_output_projection(output)
        
        # ((length - 1) * batch)
        loss = nn.functional.cross_entropy(input = s.view(-1, self.v_char), target = char_sequence_output, reduction='none')


        # TODO(ankur): Use ignore_index instead.        
        # ((length - 1) * batch), contains a 1 in the non-padding entries
        target_masks = (char_sequence_output != self.char_padding_idx).float()
        
        # ((length - 1) * batch)
        loss = loss * target_masks
        
        # scalar
        loss = loss.sum()
        
        return loss
        
        

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        start_char = '{'
        start_char_idx = self.target_vocab.char2id[start_char]
        end_char = '}'
        end_char_idx = self.target_vocab.char2id[end_char]
        
        # (1, batch, hidden_size), (1, batch, hidden_size)
        h_t, c_t = initialStates
        
        batch_size = h_t.size()[1]
        
        # List of lists, each inner list is a list of characters that make up the word.
        output_words = [[] for i in range(batch_size)]
        # (batch_size, )
        last_char_idx = torch.tensor([start_char_idx] * batch_size, device=device)
        # (batch_size, char_embedding_size)
        last_char_embeddings = self.decoderCharEmb(last_char_idx)
        # (1, batch_size, char_embedding_size)
        last_char_embeddings = torch.unsqueeze(last_char_embeddings, dim=0)
        
        
        for t in range(max_length):
            
            # (1, batch, hidden_size), (1, batch, hidden_size), (1, batch, hidden_size)
            output, (h_t, c_t) = self.charDecoder(last_char_embeddings, (h_t, c_t))
            # (1, batch, v_char)
            s_t = self.char_output_projection(h_t)
            # (batch, v_char)
            s_t = torch.squeeze(s_t, dim=0)
            # (batch, )
            predicted_char_idx = torch.argmax(s_t, dim=1)
            # (batch, )
            predicted_chars = [self.target_vocab.id2char[char_id.item()] for char_id in predicted_char_idx]
            
            for index, char in enumerate(predicted_chars):
                output_words[index].append(char)
                
            # (batch_size, )
            last_char_idx = predicted_char_idx
            # (batch_size, char_embedding_size)
            last_char_embeddings = self.decoderCharEmb(last_char_idx)
            # (1, batch_size, char_embedding_size)
            last_char_embeddings = torch.unsqueeze(last_char_embeddings, dim=0)
            
            
        # Join and truncate. 
        output_words_str = []
        for word_chars in output_words:
            if end_char in word_chars:
                end_char_id = word_chars.index(end_char)
                # Don't include anything from the end character onwards.
                word_chars = word_chars[:end_char_id]
            # Concatenate all the characters to form the word
            output_words_str.append(''.join(word_chars))
        return output_words_str
        
        ### END YOUR CODE

