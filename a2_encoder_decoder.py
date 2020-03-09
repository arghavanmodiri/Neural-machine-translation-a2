# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.source_vocab_size,
            embedding_dim=self.word_embedding_size, 
            padding_idx=self.pad_id)

        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size,
                num_layers=self.num_hidden_layers,
                dropout=self.dropout,
                bidirectional =True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size,
                num_layers=self.num_hidden_layers,
                dropout=self.dropout,
                bidirectional =True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size,
                num_layers=self.num_hidden_layers,
                dropout=self.dropout,
                bidirectional =True)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        x = self.embedding(F)
        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        sequence = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        output, hidden = self.rnn(sequence)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=h_pad)
        return h


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
        self.embedding = torch.nn.Embedding(
            num_embeddings = self.target_vocab_size,
            embedding_dim = self.word_embedding_size,
            padding_idx = self.pad_id)
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size = self.word_embedding_size,
                hidden_size = self.hidden_state_size)
        elif self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size =self.word_embedding_size,
                hidden_size = self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size = self.word_embedding_size,
                hidden_size = self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size,
                self.target_vocab_size)
        #self.ff = torch.nn.Linear(self.hidden_state_size,
        #        self.hidden_state_size)

    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat
        #unpadded = h[:F_lens, :]
        #:self.hidden_state_size // 2
        #nn.torch.cat(h[0, :, :unpadded // 2)], 
        #        h[0, :, unpadded // 2:], 1)
        htilde_tm1 = torch.cat((h[F_lens[0]-1,0,:self.hidden_state_size//2],
            h[0, 0,self.hidden_state_size//2:]), 0)
        htilde_tm1 = htilde_tm1.view(1,self.hidden_state_size)

        for idx in range(1, F_lens.shape[0]):
            output =torch.cat((h[F_lens[idx]-1,idx,:self.hidden_state_size//2],
                h[0,idx,self.hidden_state_size//2:]) , 0)
            output = output.view(1,self.hidden_state_size)
            htilde_tm1 = torch.cat((htilde_tm1, output), 0)

        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        xtilde_t = self.embedding(E_tm1)
        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        #c_tm1 = torch.zeros_like(htilde_tm1)
        if self.cell_type == 'lstm':
            #print(htilde_tm1)
            htilde_t = self.cell(xtilde_t, htilde_tm1)
        else:
            htilde_t = self.cell(xtilde_t, htilde_tm1)
        return htilde_t

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        logits_t = self.ff(htilde_t)
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        self.embedding = torch.nn.Embedding(
            num_embeddings = self.target_vocab_size,
            embedding_dim = self.word_embedding_size,
            padding_idx = self.pad_id)
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(
                input_size = self.word_embedding_size+self.hidden_state_size,
                hidden_size = self.hidden_state_size)
        elif self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(
                input_size = self.word_embedding_size+self.hidden_state_size,
                hidden_size = self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(
                input_size = self.word_embedding_size+self.hidden_state_size,
                hidden_size = self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size,
                self.target_vocab_size)
        #self.ff = torch.nn.Linear(self.hidden_state_size,
        #        self.hidden_state_size)


    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        htilde_0 = torch.zeros_like(h[0], device=h.device)
        return htilde_0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        xtilde_t = self.embedding(E_tm1)
        if self.cell_type == 'lstm':
            temp = self.attend(htilde_tm1[0], h, F_lens)
        else:
            temp = self.attend(htilde_tm1, h, F_lens)
        xtilde_t = torch.cat((xtilde_t, temp), 1)
        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        alpha = self.get_attention_weights(htilde_t, h, F_lens)
        c_t = torch.zeros_like(htilde_t, device=htilde_t.device)

        for idx in range(alpha.shape[0]):
            out = alpha[idx].repeat(h.shape[2],1).T*h[idx]
            c_t = c_t + out

        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        e_t = cos(htilde_t, h[0]).view(1,-1)
        c = 1
        for h_idx in h[1:]:
            c = c+1
            e_t = torch.cat((e_t, cos(htilde_t, h_idx).view(1,-1)), 0)
        return e_t


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
            pad_id=self.source_pad_id,
            word_embedding_size=self.word_embedding_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            hidden_state_size=self.encoder_hidden_size,
            dropout=self.encoder_dropout,
            cell_type=self.cell_type)
        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
            pad_id=self.source_pad_id,
            word_embedding_size=self.word_embedding_size,
            hidden_state_size=2*self.encoder_hidden_size,
            cell_type=self.cell_type)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)
        '''
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        print("htilde_tm1 ", htilde_tm1.shape)
        #xtilde_t = self.decoder.get_current_rnn_input(E[0], htilde_tm1, h, F_lens) #(N, Itilde)
        print("h ", h.shape)

        for di in range(E.shape[0]):
            htilde_tm1 = self.decoder.get_current_hidden_state(xtilde_t, htilde_tm1)
            xtilde_t = self.decoder.get_current_rnn_input(E[di], htilde_tm1, h, F_lens)
            decoder_input = torch.stack([decoder_input, xtilde_t[:-1]])
        print("decoder_input ", decoder_input.shape)
        return decoder_input 
        '''
        logits_ls = []
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        for di in range(E.shape[0]-1):
            if self.cell_type == 'lstm':
                xtilde_t = self.decoder.get_current_rnn_input(E[di], 
                    (htilde_tm1, htilde_tm1),h,F_lens)
                h_t = self.decoder.get_current_hidden_state(xtilde_t,
                    (htilde_tm1, htilde_tm1))
            else:
                xtilde_t = self.decoder.get_current_rnn_input(E[di], 
                    htilde_tm1,h,F_lens)
                h_t = self.decoder.get_current_hidden_state(xtilde_t,
                    htilde_tm1)
            logit_t = self.decoder.get_current_logits(2*h_t[0])
            logits_ls.append(logit_t)
        logits = torch.stack([i for i in logits_ls])
        if logits.shape[0] !=E.shape[0]-1 and \
            logits.shape[0] !=E.shape[1] and \
            logits.shape[0] !=self.target_vocab_size:
            print(False)

        return logits

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]
        assert self.beam_width == 1, "Greedy requires beam width of 1"
        extensions_t = (logpb_tm1.unsqueeze(-1) + logpy_t).squeeze(1)  # (N, V)

        logpb_t, v = extensions_t.max(1)  # (N,), (N,)
        logpb_t = logpb_t.unsqueeze(-1)  # (N, 1) == (N, K)
        # v indexes the maximal element in dim=1 of extensions_t that was
        # chosen, which equals the token index v in k -> v
        v = v.unsqueeze(0).unsqueeze(-1)  # (1, N, 1) == (1, N, K)
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        # For greedy search, all paths come from the same prefix, so
        b_t_0 = htilde_t

        return b_t_0, b_t_1, logpb_t
