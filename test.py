import a2_encoder_decoder
import torch


S = 10  # seq_len
N = 3  # batch_size
H = 512
h_encoder = torch.ones(S, N, 2*H, dtype=torch.float)
h_encoder = torch.FloatTensor(h_encoder)
F_lens = torch.ones(N, dtype=torch.long)
F_lens = torch.LongTensor(F_lens)

decoder = a2_encoder_decoder.DecoderWithoutAttention(target_vocab_size=40, hidden_state_size=H, cell_type='lstm', word_embedding_size=8)
htilde_0 = decoder.get_first_hidden_state(h_encoder, F_lens)
x_tilde_t = decoder.get_current_rnn_input(F_lens, htilde_0, h_encoder, F_lens)
print(x_tilde_t.shape)
print(htilde_0.shape)
print(decoder.cell_type)
htilde_t = decoder.get_current_hidden_state(x_tilde_t, htilde_0)
logits_t = decoder.get_current_logits(htilde_t)
print(logits_t.shape)
# print(x_tilde_t.shape)
# print("here")
# output, (hn,cn) = torch.lstm()