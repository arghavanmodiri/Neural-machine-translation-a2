import a2_encoder_decoder
import torch

'''
S = 10  # seq_len
T = 6
N = 3  # batch_size
H = 512
h_encoder = torch.rand(S, N, 2*H, dtype=torch.float)
h_encoder = torch.FloatTensor(h_encoder)
F_lens = torch.ones(N, dtype=torch.long)
F_lens = torch.LongTensor(F_lens)
E = torch.randint(0,3, (T, N), dtype=torch.long)

decoder = a2_encoder_decoder.DecoderWithAttention(target_vocab_size=40, hidden_state_size=H, cell_type='lstm', word_embedding_size=8)
htilde_0 = decoder.get_first_hidden_state(h_encoder, F_lens)

xtilde_t = decoder.get_current_rnn_input(F_lens, htilde_0, h_encoder, F_lens)
'''

torch.manual_seed(1030)
N, K, V, H = 2, 2, 5, 10
beam_width = 1
ed = a2_encoder_decoder.EncoderDecoder(
    a2_encoder_decoder.Encoder, a2_encoder_decoder.DecoderWithAttention,
    V, V,
    encoder_hidden_size=H,
    cell_type='rnn', beam_width=beam_width,
)
logpb_tm1 = torch.arange(beam_width).flip(0).unsqueeze(0).expand(N, -1).float()
logpb_tm1 -= 1.5
logpb_tm1[1] *= 2  # [[-0.5, -1.5], [-1., -3.]]
htilde_t = torch.rand(N, K, 2 * H)
logpy_t = (
    torch.arange(V).unsqueeze(0).unsqueeze(0)
    .expand(N, beam_width, -1).float() * -1.1
)  # [x, y, :] = [0., -1.1, -2.2, ...]
# [0, x, :] = [0, 1]
b_tm1_1 = torch.arange(beam_width).unsqueeze(0).unsqueeze(0).expand(-1, N, -1)

print("htilde_t ", htilde_t)
print("htilde_t ", htilde_t.shape)
print("b_tm1_1 ", b_tm1_1)
print("b_tm1_1 ", b_tm1_1.shape)
print("logpb_tm1 ", logpb_tm1)
print("logpb_tm1 ", logpb_tm1.shape)
print("logpy_t ", logpy_t)
print("logpy_t ", logpy_t.shape)
b_t_0, b_t_1, logpb_t = ed.update_beam(
    htilde_t, b_tm1_1, logpb_tm1, logpy_t)
# batch 0 picks path 0 extended with 0, then path 1 extended with 0
print(logpb_t)
#assert torch.allclose(logpb_t[0], torch.tensor([-0.5, -1.5]))
assert torch.allclose(b_t_0[0, 0], htilde_t[0, 0])
assert torch.allclose(b_t_0[0, 1], htilde_t[0, 1])
assert torch.allclose(b_t_1[:, 0, 0], torch.tensor([0, 0]))
assert torch.allclose(b_t_1[:, 0, 1], torch.tensor([1, 0]))
# batch 0 picks path 0 extended with 0, then path 0 extended with 1
assert torch.allclose(logpb_t[1], torch.tensor([-1., -2.1]))
assert torch.allclose(b_t_0[1, 0], htilde_t[1, 0])
assert torch.allclose(b_t_0[1, 1], htilde_t[1, 0])
assert torch.allclose(b_t_1[:, 1, 0], torch.tensor([0, 0]))
assert torch.allclose(b_t_1[:, 1, 1], torch.tensor([0, 1]))
