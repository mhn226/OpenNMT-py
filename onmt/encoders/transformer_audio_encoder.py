"""Transformer Audio encoder"""
import math

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

#from onmt.utils.rnn_factory import rnn_factory
from onmt.encoders.encoder import EncoderBase

from onmt.encoders.transformer import TransformerEncoderLayer


class TransformerAudioEncoder(EncoderBase):
    """A transformer encoder for audio input.
    General idea is to stack a few layers of CNN before feeding to the 
    Transformer blocks. 
    Ref: https://arxiv.org/pdf/1904.11660.pdf

    Args:
        rnn_type (str): Type of RNN (e.g. GRU, LSTM, etc).
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        brnn (bool): Bidirectional encoder.
        enc_rnn_size (int): Size of hidden states of the rnn.
        dec_rnn_size (int): Size of the decoder hidden states.
        enc_pooling (str): A comma separated list either of length 1
            or of length ``enc_layers`` specifying the pooling amount.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec
    """
    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 max_relative_positions):
        super(TransformerAudioEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.max_relative_positions = max_relative_positions

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3),
                                padding=(0, 10), stride=(2, 2))
        #self.batch_norm1 = nn.BatchNorm2d(1)        

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(0, 0), stride=(2, 2))
        #self.batch_norm2 = nn.BatchNorm2d(16)

        #input_size = 128 * 19
        input_size = 2432
        self.W = nn.Linear(input_size, d_model, bias=False)

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        #self._check_args(src, lengths)

        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        batch_size, _, nfft, t = src.size()

        #src = src.transpose(0, 1).transpose(0, 3).contiguous() \
        #         .view(t, batch_size, nfft)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()

        src = self.conv1(src[:, :, :, :])
        #src = batch_norm1(src)

        src = self.conv2(src)
        #src = self.batch_norm2(src)
        length = src.size(3)
        src = src.view(batch_size, -1, length)
        src = src.transpose(0, 2).transpose(1, 2)

        src = self.W(src)
        tmp = src
        
        for layer in self.transformer:
            src = layer(src, None)
        memory_bank = self.layer_norm(src) 

        return tmp, memory_bank, lengths

    def update_dropout(self, dropout):
        self.dropout = dropout
        self.conv1.update_dropout(dropout)
        self.conv2.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout)
