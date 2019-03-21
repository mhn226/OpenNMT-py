""" Embeddings module """
import math
import warnings

import torch
import torch.nn as nn

from onmt.modules.util_class import Elementwise


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class AudioEmbeddings(nn.Module):
    """Audio embeddings for audio encoder.

    .. mermaid::

       graph LR
          A[Input]
          A-->B[dim & norm]
          A-->C[positional encoding]
          B-->D[sum]
          C-->D
          output == D

    Args:
        n_feats (int): number of input features (default: 40 (MFCCs))
        embedding_size (int): equal to the size of the model (d_model)
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        dropout (float): dropout probability.
        downsampling (bool): apply down sampling or not (default: False)
    """

    def __init__(self, n_feats,
                 embedding_size,
                 position_encoding=False,
                 dropout=0, downsampling=False):
        super(AudioEmbeddings, self).__init__()
        
        self.n_feats = n_feats
        self.embedding_size = embedding_size
        
        self.dropout = dropout
        self.position_encoding = position_encoding 

        self.downsampling = downsampling        

        if self.position_encoding:
            self.pe = PositionalEncoding(dropout=self.dropout, dim=self.embedding_size)
        else:
            self.pe = None

        # Linear projection to change to dimension of the audio input n_feats -> embedding_size
        self.W = nn.Linear(self.n_feats, self.embedding_size, bias=False)
        self.layer_norm = nn.LayerNorm(self.embedding_size, eps=1e-6)


    def forward(self, src):
        """Computes the embeddings for words and features.

        Args:
            src (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """
        src = self.layer_norm(self.W(src))
        if self.pe:
            src = self.pe(src)

        return src

    def down_sampling(self, src):
        """
        Reshape the src tensor (to reduce to len dimension)
        Idea of "Self-attention Acoustic Models"
        Conditions:
            len % downsample_factor = 0 (this should affect padding and batching audio)
            nfeats * downsample_factor = d_model => not very flexible params choosing
        We should use conv layers instead???
        """
        ## Not implement for the moment
        return src
         
