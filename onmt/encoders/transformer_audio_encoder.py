"""Transformer Audio encoder"""
import math

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

#from onmt.utils.rnn_factory import rnn_factory
from onmt.encoders.encoder import EncoderBase

from onmt.encoders.transformer import TransformerEncoderLayer
from collections import OrderedDict

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


        # Two blocks of conv, 2 layers of conv each block
        # Try to use batchnorm instead of layernorm
        #self.conv1 = nn.Sequential(OrderedDict([
        #             ('conv1', nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(0, 10), stride=(2, 2))),
        #             ('batchnorm1', nn.BatchNorm2d(64)),
        #             ('relu1', nn.ReLU()),
        #             ('conv2', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 10), stride=(2, 2))),
        #             ('batchnorm2', nn.BatchNorm2d(64)),
        #             ('relu2', nn.ReLU())]))


        # Conv block 1
        self.b1_conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.b1_layernorm1 = nn.LayerNorm((80, 64), eps=1e-6)
        self.b1_relu1 = nn.ReLU()
        self.b1_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.b1_layernorm2 = nn.LayerNorm((80, 64), eps=1e-6)
        self.b1_relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, padding=0)
        
        # Conv block 2
        self.b2_conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.b2_layernorm1 = nn.LayerNorm((40, 128), eps=1e-6)
        self.b2_relu1 = nn.ReLU()
        self.b2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.b2_layernorm2 = nn.LayerNorm((40, 128), eps=1e-6)
        self.b2_relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, padding=0)

        # Conv block 3
        self.b3_conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.b3_layernorm1 = nn.LayerNorm((20, 128), eps=1e-6)
        self.b3_relu1 = nn.ReLU()
        self.b3_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.b3_layernorm2 = nn.LayerNorm((20, 128), eps=1e-6)
        self.b3_relu2 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, padding=0)

        #self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3),
        #                        padding=(0, 10), stride=(2, 2))
        #self.batch_norm1 = nn.BatchNorm2d(1)        

        #self.conv3 = nn.Sequential(OrderedDict([
        #             ('conv3', nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(0, 10), stride=(2, 2))),
        #             ('batchnorm3', nn.BatchNorm2d(128)),
        #             ('relu3', nn.ReLU()),
        #             ('conv4', nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(0, 10), stride=(2, 2))),
        #             ('batchnorm4', nn.BatchNorm2d(128)),
        #             ('relu4', nn.ReLU())]))

        #self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
        #                        padding=(0, 0), stride=(2, 2))
        #self.batch_norm2 = nn.BatchNorm2d(16)


        #input_size = 128 * 19 = 2432
        #input_size = 56 * 24 = 1344
        #input_size = 2432 
        #input_size = 256
        #input_size = 896
        #input_size = 1024
        #input_size = 2560
        input_size = 1280
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
        # Conv block 1
        src = self.b1_conv1(src[:, :, :, :])
        src = self.b1_layernorm1(src.transpose(3, 1))
        src = self.b1_relu1(src.transpose(3, 1))
        src = self.b1_conv2(src)
        src = self.b1_layernorm2(src.transpose(3, 1))
        src = self.b1_relu2(src.transpose(3, 1))
        src = self.maxpool1(src)

        # Conv block 2
        src = self.b2_conv1(src)
        src = self.b2_layernorm1(src.transpose(3, 1))
        src = self.b2_relu1(src.transpose(3, 1))
        src = self.b2_conv2(src)
        src = self.b2_layernorm2(src.transpose(3, 1))
        src = self.b2_relu2(src.transpose(3, 1))
        src = self.maxpool2(src)

        # Conv block 3
        src = self.b3_conv1(src)
        src = self.b3_layernorm1(src.transpose(3, 1))
        src = self.b3_relu1(src.transpose(3, 1))
        src = self.b3_conv2(src)
        src = self.b3_layernorm2(src.transpose(3, 1))
        src = self.b3_relu2(src.transpose(3, 1))
        src = self.maxpool3(src)
        
        #src = batch_norm1(src)
        #src = self.conv2(src)
        #src = self.batch_norm2(src)
        length = src.size(3)
        src = src.view(batch_size, -1, length)
        src = src.transpose(0, 2).transpose(1, 2)
        src = self.W(src)
        tmp = src
        
        for layer in self.transformer:
            src = layer(src, None)
        memory_bank = self.layer_norm(src) 

        return tmp, memory_bank, orig_lengths.new_tensor(lengths)

    def update_dropout(self, dropout):
        self.dropout = dropout
        self.conv1.update_dropout(dropout)
        self.conv2.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout)
