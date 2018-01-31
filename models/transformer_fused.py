#!/usr/bin/env python3
import math
import numpy as np
import tensorflow as tf
from lib.layers import *
from lib.tensor_utils import *
from collections import namedtuple
from . import TranslateModel


from .transformer_other import Transformer


# ============================================================================
#                                  Transformer model

class Model(TranslateModel):

    DecState = namedtuple("transformer_state", ['enc_out', 'enc_attn_mask', 'attnP', 'rdo', 'out_seq', 'offset',
                                                'emb', 'dec_layers', 'dec_enc_kv', 'dec_dec_kv'])

    def __init__(self, name, inp_voc, out_voc, lm, gate_hid_size=None, **hp):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp
        self.debug = hp.get('debug', None)
        self.lm = lm
        self.gate_hid_size = self.transformer.hid_size

        # Parameters
        self.transformer = Transformer(name, inp_voc, out_voc, **hp)

        projection_matrix = None
        if hp.get('dwwt', False):
            projection_matrix = tf.transpose(self.transformer.emb_out.mat)

        with tf.variable_scope(name):
            self.logits = Dense('logits', self.transformer.hid_size, len(out_voc),
                                W=projection_matrix)

            self.gate_hid = Dense('gate_hid',
                                  self.transformer.hid_size,
                                  self.gate_hid_size,
                                  activation=tf.nn.relu)

            self.gate_out = Dense('gate_hid',
                                  self.gate_hid_size, 2,
                                  activation=tf.nn.sigmoid)

    # Train interface
    def symbolic_score(self, inp, out, is_train=False):
        inp_len = infer_length(inp, self.inp_voc.eos, time_major=False)
        out_len = infer_length(out, self.out_voc.eos, time_major=False)

        out_reverse = tf.zeros_like(inp_len)  # batch['out_reverse']

        # rdo: [batch_size * nout * hid_dim]
        enc_out, enc_attn_mask = self.transformer.encode(inp, inp_len, is_train)
        rdo = self.transformer.decode(out, out_len, out_reverse, enc_out, enc_attn_mask, is_train)

        trans_logits = self.logits(rdo)
        lm_logits = self.lm(out, is_train=False)

        gates = self.gate_out(self.gate_hid(rdo))
        trans_gate, lm_gate = tf.unstack(gates[..., None], axis=-2)
        return trans_logits * trans_gate + lm_logits * lm_gate


    def encode(self, batch, is_train=False, **kwargs):
        """
        :param batch: a dict containing 'inp':int32[batch_size * ninp] and optionally inp_len:int32[batch_size]
        :param is_train: if True, enables dropouts
        """
        inp = batch['inp']
        inp_len = batch.get('inp_len', infer_length(inp, self.inp_voc.eos, time_major=False))
        with dropout_scope(is_train), tf.name_scope(self.transformer.name):
            if self.debug:
                inp = tf.Print(inp, [tf.shape(inp), inp], message="encode(): inp", first_n=100, summarize=100)

            # Encode.
            enc_out, enc_attn_mask = self.transformer.encode(inp, inp_len, is_train=False)

            # Decoder dummy input/output
            ninp = tf.shape(inp)[1]
            batch_size = tf.shape(inp)[0]
            hid_size = tf.shape(enc_out)[-1]
            out_seq = tf.zeros([batch_size, 0], dtype=inp.dtype)
            rdo = tf.zeros([batch_size, hid_size], dtype=enc_out.dtype)

            attnP = tf.ones([batch_size, ninp]) / tf.to_float(inp_len)[:, None]

            offset = tf.zeros((batch_size,))
            if self.transformer.dst_rand_offset:
                BIG_LEN = 32000
                random_offset = tf.random_uniform(tf.shape(offset), minval=-BIG_LEN, maxval=BIG_LEN, dtype=tf.int32)
                offset += tf.to_float(random_offset)

            trans = self.transformer
            empty_emb = tf.zeros([batch_size, 0, trans.emb_size])
            empty_dec_layers = [tf.zeros([batch_size, 0, trans.hid_size])] * trans.num_layers_dec
            input_layers = [empty_emb] + empty_dec_layers[:-1]

            #prepare kv parts for all decoder attention layers. Note: we do not preprocess enc_out
            # for each layer because ResidualLayerWrapper only preprocesses first input (query)
            dec_enc_kv = [layer.kv_conv(enc_out)
                          for i, layer in enumerate(trans.dec_enc_attn)]
            dec_dec_kv = [layer.kv_conv(layer.preprocess(input_layers[i]))
                          for i, layer in enumerate(trans.dec_attn)]

            new_state = self.DecState(enc_out, enc_attn_mask, attnP, rdo, out_seq, offset,
                                      empty_emb, empty_dec_layers, dec_enc_kv, dec_dec_kv)

            # perform initial decode (instead of force_bos) with zero embeddings
            new_state = self.decode(new_state, is_train=is_train)
            return new_state

    def decode(self, dec_state, words=None, is_train=False, **kwargs):
        """
        Performs decoding step given words and previous state.
        Returns next state.

        :param words: previous output tokens, int32[batch_size]. if None, uses zero embeddings (first step)
        :param is_train: if True, enables dropouts
        """
        trans = self.transformer
        enc_out, enc_attn_mask, attnP, rdo, out_seq, offset, prev_emb = dec_state[:7]
        prev_dec_layers = dec_state.dec_layers
        dec_enc_kv = dec_state.dec_enc_kv
        dec_dec_kv = dec_state.dec_dec_kv

        batch_size = tf.shape(rdo)[0]
        if words is not None:
            out_seq = tf.concat([out_seq, tf.expand_dims(words, 1)], 1)

        with dropout_scope(is_train), tf.name_scope(trans.name):
            # Embeddings
            if words is None:
                # initial step: words are None
                emb_out = tf.zeros((batch_size, 1, trans.emb_size))
            else:
                emb_out = trans.emb_out(words[:, None])  # [batch_size * 1 * emb_dim]
                if trans.rescale_emb:
                    emb_out *= trans.emb_size ** .5

            # Prepare decoder
            dec_inp_t = trans._add_timing_signal(emb_out, offset=offset)
            # Apply dropouts
            if is_dropout_enabled():
                dec_inp_t = tf.nn.dropout(dec_inp_t, 1.0 - trans.res_dropout)

            # bypass info from Encoder to avoid None gradients for num_layers_dec == 0
            if trans.num_layers_dec == 0:
                inp_mask = tf.squeeze(tf.transpose(enc_attn_mask, perm=[3, 1, 2, 0]), 3)
                dec_inp_t += tf.reduce_mean(enc_out * inp_mask, axis=[0, 1], keep_dims=True)

            # Decoder
            new_emb = tf.concat([prev_emb, dec_inp_t], axis=1)
            _out = tf.pad(out_seq, [(0, 0), (0, 1)])
            dec_attn_mask = trans._make_dec_attn_mask(_out)[:, :, -1:, :]  # [1, 1, n_q=1, n_kv]

            new_dec_layers = []
            new_dec_dec_kv = []

            for layer in range(trans.num_layers_dec):
                # multi-head self-attention: use only the newest time-step as query,
                # but all time-steps up to newest one as keys/values
                next_dec_kv = trans.dec_attn[layer].kv_conv(trans.dec_attn[layer].preprocess(dec_inp_t))
                new_dec_dec_kv.append(tf.concat([dec_dec_kv[layer], next_dec_kv], axis=1))
                dec_inp_t = trans.dec_attn[layer](dec_inp_t, dec_attn_mask, kv=new_dec_dec_kv[layer])

                dec_inp_t = trans.dec_enc_attn[layer](dec_inp_t, enc_attn_mask, kv=dec_enc_kv[layer])
                dec_inp_t = trans.dec_ffn[layer](dec_inp_t)

                new_dec_inp = tf.concat([prev_dec_layers[layer], dec_inp_t], axis=1)
                new_dec_layers.append(new_dec_inp)

            if trans.normalize_out:
                dec_inp_t = trans.dec_out_norm(dec_inp_t)

            rdo = dec_inp_t[:, -1]

            new_state = self.DecState(enc_out, enc_attn_mask, attnP, rdo, out_seq, offset + 1,
                                      new_emb, new_dec_layers, dec_enc_kv, new_dec_dec_kv)
            return new_state

    def get_rdo(self, dec_state, **kwargs):
        return dec_state.rdo, dec_state.out_seq

    def get_attnP(self, dec_state, **kwargs):
        return dec_state.attnP

    def get_logits(self, dec_state, **flags):
        trans_logits = self.logits(dec_state.rdo)
        lm_logits = self.lm(dec_state.out_seq, is_train=False)

        gates = self.gate_out(self.gate_hid(dec_state.rdo))
        trans_gate, lm_gate = tf.unstack(gates[..., None], axis=-2)
        return trans_logits * trans_gate + lm_logits * lm_gate
