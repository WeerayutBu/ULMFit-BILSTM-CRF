import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import spacy
import numpy as np
USE_GPU = torch.cuda.is_available()

def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x


def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor.
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else: raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a, async=True)
    return a

def load_ner_model(m, p, strict=True):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd, strict=strict)


    
def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length



def minibatches(data, minibatch_size, use_crf=True):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        if use_crf:
            y_batch += [y]
        else:
            if any([x.isdigit() for x in y]):
                y_batch.append([int(x) for x in y if x.isdigit()])
            else:
                y_batch.append([0,0,0,0,0])

    if len(x_batch) != 0:
        yield x_batch, y_batch


def batch_iter(train, batch_size, return_lengths=False, shuffle=False, sorter=False):
        """
        Builds a generator from the given dataloader to be fed into the model

        Args:
            train: DataLoader
            batch_size: size of each batch
            return_lengths: if True, generator returns a list of sequence lengths for each
                            sample in the batch
                            ie. sequence_lengths = [8,7,4,3]
            shuffle: if True, shuffles the data for each epoch
            sorter: if True, uses a sorter to shuffle the data

        Returns:
            nbatches: (int) number of batches
            data_generator: batch generator yielding
                                dict inputs:{'word_ids' : np.array([[padded word_ids in sent1], ...])
                                             'char_ids': np.array([[[padded char_ids in word1_sent1], ...],
                                                                    [padded char_ids in word1_sent2], ...],
                                                                    ...])}
                                labels: np.array([[padded label_ids in sent1], ...])
                                sequence_lengths: list([len(sent1), len(sent2), ...])
        """

        nbatches = (len(train) + batch_size - 1) // batch_size

        def data_generator():
            while True:
                if shuffle: train.shuffle()
                elif sorter==True and train.sorter: train.sort()

                for i, (words, labels) in enumerate(minibatches(train, batch_size)):

                    # perform padding of the given data
                    char_ids, word_ids = zip(*words)
                    word_ids, sequence_lengths = pad_sequences(word_ids, 1)
                    char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                    nlevels=2)


                    if labels:
                        labels, _ = pad_sequences(labels, 0)
                        # if categorical
                        ## labels = [to_categorical(label, num_classes=len(train.tag_itos)) for label in labels]

                    # build dictionary
                    inputs = {
                        "word_ids": np.asarray(word_ids)
                    }


                    inputs["char_ids"] = np.asarray(char_ids)

                    if return_lengths:
                        yield(inputs, np.asarray(labels), sequence_lengths)

                    else:
                        yield (inputs, np.asarray(labels))

        return (nbatches, data_generator())
    
def create_mask(sequence_lengths, targets, cuda, batch_first=False):
    """ Creates binary mask """
    mask = Variable(torch.ones(targets.size()).type(torch.ByteTensor))
    if cuda: mask = mask.cuda()

    for i,l in enumerate(sequence_lengths):
        if batch_first:
            if l < targets.size(1):
                mask.data[i, l:] = 0
        else:
            if l < targets.size(0):
                mask.data[l:, i] = 0

    return mask