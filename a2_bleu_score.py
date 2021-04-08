# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x
import numpy as np

def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    ngrams=[]
    for i in range(0, len(seq)-n+1):
        try:
            w = seq[i:i+n]
        except:
            w = seq[i:-1]
            continue
        ngrams.append(w)
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''

    #Below code was submitted
    ref_n_grams = grouper(reference, n)
    can_n_grams = grouper(candidate, n)
    C = 0.0
    if len(can_n_grams)==0:
        return 0.0
    for w in can_n_grams:
        if w in ref_n_grams:
            C += 1
    return C*1.0/len(can_n_grams)
    
    #Below code is written after 1 year in Feb2021
    '''
    ref_n_grams = grouper(reference, n)
    can_n_grams = grouper(candidate, n)
    if len(candidate)==0 or len(can_n_grams)==0:
        return 0.0

    C = 0.0
    mp_ref = {}

    for elem in ref_n_grams:
        elem_tuple = tuple(elem)
        if elem_tuple in mp_ref:
            mp_ref[elem_tuple] += 1
        else:
            mp_ref[elem_tuple] = 1

    for w in can_n_grams:
        w_tuple = tuple(w)
        if w_tuple in mp_ref and mp_ref[w_tuple] > 0:
            C += 1
            mp_ref[w_tuple] -= 1
    return C*1.0/len(can_n_grams)
    '''

def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    ref_len = len(reference)
    can_len = len(candidate)
    if can_len == 0:
        return 0
    brevity = ref_len / can_len * 1.0
    if brevity < 1:
        return 1
    else:
        return exp(1-brevity)


def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    B_score = 0
    temp = 1
    for n_i in range(1, n+1):
        temp = temp * n_gram_precision(reference, hypothesis, n_i)
    B_score = brevity_penalty(reference, hypothesis) * pow(temp, 1/n)
    return B_score


"""
reference = '''\
    it is a guide to action that ensures that the military will always heed
    party commands'''.strip().split()
candidate = '''\
    it is a guide to action which ensures that the military always obeys the
    commands of the party'''.strip().split()

# should work with token ids (ints) as well
reference = [hash(word) for word in reference]
candidate = [hash(word) for word in candidate]
print(n_gram_precision(reference, candidate, 1))
assert np.isclose( n_gram_precision(reference, candidate, 1), 15 / 18)
"""