"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 max_relative_positions, model_type, embeddings=None):
        super(TransformerEncoder, self).__init__()

        ## For speech processing
        # Add a projection layer to change the dimension of the input

        self.embeddings = embeddings
        self.model_type = model_type

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            opt.max_relative_positions,
            opt.model_type,
            embeddings)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        if self.model_type == 'audio':
            #print('###################### audio transformer ######################')
            lengths = lengths.view(-1).tolist()
            batch_size, _, nfft, t = src.size()
            #print('####### src size #######  ' + str(src.size()))
            #print('### t1 ###  ' + str(src.transpose(0, 1).size()))
            #print('### t2 ###  ' + str(src.transpose(0, 1).transpose(0, 3).size()))
            #print('### t3 ###  ' + str(src.transpose(0, 1).transpose(0, 3).contiguous().view(t, batch_size, nfft).size()))
            src = src.transpose(0, 1).transpose(0, 3).contiguous() \
                     .view(t, batch_size, nfft)
            emb = self.embeddings(src)
            out = emb.transpose(0, 1).contiguous()

            # Mask
            #print('### src size ###  ' + str(src.size()))
            padding_idx = 0
            mask = src.transpose(0,1).data.eq(padding_idx).sum(dim=2)
            mask = mask.data.eq(nfft).unsqueeze(1)
            #mask = mask.data.eq(padding_idx).eq(padding_idx).unsqueeze(1)
            #print(mask)
            #mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]

        else:
            #print('#### src size ####  ' + str(src.size()))
            self._check_args(src, lengths)

            emb = self.embeddings(src)

            out = emb.transpose(0, 1).contiguous()
            #print('#### src size ####  ' + str(src.size()))
            words = src[:, :, 0].transpose(0, 1)
            #print('#### word size ####  ' + str(words.size()))
            w_batch, w_len = words.size()
            padding_idx = self.embeddings.word_padding_idx
            #print('######### padding idx ###########:  ' + str(padding_idx))
            mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
            #print(mask.size())

        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
