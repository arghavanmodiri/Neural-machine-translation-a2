# Copyright 2020 University of Toronto, all rights reserved

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.source_pad_id)
    loss_tot = 0.0
    i = 0
    seq_count = 0 #NEW
    for F, F_lens, E in dataloader:
      seq_count += E.size()[1] #NEW
      if torch.cuda.is_available():
        F = F.to(device)
        F_lens = F_lens.to(device)
        E = E.to(device)
      optimizer.zero_grad()
      logits = model(F, F_lens, E, device)
      pad_mask = model.get_target_padding_mask(E)
      E = E.masked_fill(pad_mask, model.source_pad_id)
      logits = torch.flatten(logits, 0,1)
      #logits = torch.cat((logits, torch.zeros(E.shape[1],logits.shape[1],
      #    device=device)))
      E = torch.flatten(E[1:, :])
      loss = loss_fn(logits, E)
      loss_tot = loss_tot + loss.item()
      loss.backward()
      optimizer.step()
      del F, F_lens, E, logits, loss

    #avg_loss = loss_tot / len(dataloader)
    avg_loss = loss_tot / seq_count #NEW
    print("************************")
    print("avg_loss : ", avg_loss)
    print("************************")
    return avg_loss


def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, N)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', N)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    E_ref_ls = E_ref.T.tolist()
    #E_ref_ls = " ".join(E_ref_ls)
    #E_ref_ls = E_ref_ls.split(target_sos).strip()
    E_cand_ls = E_cand.T.tolist()

    #E_cand_ls = " ".join(E_cand_ls)
    #E_cand_ls = E_cand_ls.split(target_sos).strip()

    total_bleu = 0.0
    for ref, cand in zip(E_ref_ls, E_cand_ls):
      #New:
      while(ref[-1] == target_eos):
        ref.pop()
      while(cand[-1] == target_eos):
        cand.pop()
      #EndNew
      total_bleu += a2_bleu_score.BLEU_score(ref[1:], cand[1:], 4)

    return total_bleu


def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''
    total_bleu = 0.0
    seq_count = 0 #NEW
    for F, F_lens, E_ref in dataloader:
      if torch.cuda.is_available():
        F = F.to(device)
        F_lens = F_lens.to(device)
      b_1 = model(F, F_lens)
      E_cand = b_1[..., 0] #b_1[:, :, 0]
      total_bleu = total_bleu + compute_batch_total_bleu(E_ref, E_cand,
        target_sos, target_eos)
      seq_count += E_ref.size()[1] #NEW

    #avg_bleu = total_bleu / len(dataloader)
    avg_bleu = total_bleu / seq_count #NEW
    print("************************")
    print("avg_bleu : ", avg_bleu)
    print("************************")
    return avg_bleu
