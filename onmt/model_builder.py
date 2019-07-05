"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


import collections

def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # HN 05-07-19 update more audio encoder
    #enc_type = opt.encoder_type if (opt.model_type == "text" or opt.model_type == "audio") else opt.model_type
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    if opt.model_type == "audio" and opt.encoder_type == "transformer":
        enc_type = "transformer_audio"
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab
    # HN: 01-07-19: feed build_base_model a dict instead of a single checkpoint
    # as modified for pretraining purpose
    checkpoints = {}
    checkpoints['full'] = checkpoint
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoints,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt

def load_pretrained(model, checkpoint, part='full'):
    """
    HN added 27-07-19
    Load only a specific part of the model (encoder/decoder) instead of 
        loading the whole model

    Args:
        model: the model (or generator) whose part we want to initiallized
        checkpoint: the check point contains the part 
            whose weights we want to initialize to the model
        part (string): 'full | encoder | decoder | generator'
    
    """
    assert part in ['full', 'encoder', 'decoder', 'generator'], "part must be 'full | encoder | decoder | generator'"
    print('Load ' + part)
    if part == 'full':
        model.load_state_dict(checkpoint, strict=False)
        return model
    model_dict = model.state_dict()
    pretrained_dict = collections.OrderedDict()
    for key in checkpoint.keys():
        if (part in key) or part == 'generator':
            pretrained_dict[key] = checkpoint[key]
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict, strict=True)
    #getattr(model, part).load_state_dict(tdict, strict=False) 
    return model


def build_base_model(model_opt, fields, gpu, checkpoints=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
      
        HN: change checkpoint --> checkpoints - a dict of checkpoints instead of 1 checkpoint
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    # Build decoder.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    decoder = build_decoder(model_opt, tgt_emb)
    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
    if checkpoints is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        for key in checkpoints.keys():
            checkpoints[key]['model'] = {fix_key(k): v
                               for k, v in checkpoints[key]['model'].items()}
            model = load_pretrained(model, checkpoints[key]['model'], key)
            if key == 'full':
                # For now only load generator if and only if full model is loaded
                generator.load_state_dict(checkpoints['full']['generator'], strict=False)
        
        #checkpoints['full']['model'] = {fix_key(k): v
        #                       for k, v in checkpoints['full']['model'].items()}
        # end of patch for backward compatibility
        #model = load_pretrained(model, checkpoint['model'], 'encoder')
        #model = load_pretrained(model, checkpoint['model'], 'decoder')
        #for name, param in model.named_parameters():
        #    if 'encoder' in name:
        #        print(name)
        #        print(param - checkpoint['model'][name])
        
        #model = load_pretrained(model, checkpoint['model'], 'decoder')
        #model.load_state_dict(checkpoints['full']['model'], strict=False)
        #generator = load_pretrained(generator, checkpoint['generator'], 'generator')
        #generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
