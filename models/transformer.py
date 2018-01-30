from lib.layers import *
from lib.tensor_utils import *
from lib.transformer_layers import TransformerEncoder, TransformerDecoder,\
                                   add_timing_signal, make_attn_mask
from collections import namedtuple
from . import TranslateModel


class Transformer(TranslateModel):
    """ The main model, consisting of encoder, decoder and logits """
    def __init__(self, name, inp_voc, out_voc, *_args,
                 hid_size=256, **hp):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp
        hp['force_bos'] = hp.get('force_bos', True)

        with tf.variable_scope(name):
            self.enc = TransformerEncoder("enc", inp_voc, hid_size=hid_size, **hp)
            self.dec = TransformerDecoder("dec", inp_voc, hid_size=hid_size,
                                          allow_lookahead=False,
                                          **hp)

            projection_matrix = None
            if hp.get('dwwt', False):
                projection_matrix = tf.transpose(self.dec.emb_out.mat)

            self.logits = Dense('logits', hid_size, len(out_voc),
                                activation=nop, W=projection_matrix)

    def symbolic_score(self, inp, out, is_train=False):
        """ Computes a sequence of logits aka model prediction. Used for training """

        enc_out, enc_attn_mask = self.enc(inp, is_train=is_train)

        # rdo: [batch_size * nout * hid_dim]
        dec_out, dec_attn_mask = self.dec(enc_out, out, enc_attn_mask, is_train=is_train)

        logits = self.logits(dec_out)
        return tf.nn.log_softmax(logits)

    # Translation code

    DecState = namedtuple("transformer_state", ['enc_out', 'enc_attn_mask', 'attnP', 'rdo', 'out_seq', 'offset',
                                                'emb', 'dec_layers', 'dec_enc_kv', 'dec_dec_kv'])

    def encode(self, batch, is_train=False, **kwargs):
        """
        :param batch: a dict containing 'inp':int32[batch_size * ninp] and optionally inp_len:int32[batch_size]
        :param is_train: if True, enables dropouts
        """
        inp = batch['inp']
        inp_len = batch.get('inp_len', infer_length(inp, self.inp_voc.eos, time_major=False))
        with dropout_scope(is_train), tf.name_scope(self.name):

            # Encode.
            enc_out, enc_attn_mask = self.enc(inp, is_train=is_train)

            # Decoder dummy input/output
            ninp = tf.shape(inp)[1]
            batch_size = tf.shape(inp)[0]
            hid_size = tf.shape(enc_out)[-1]
            out_seq = tf.zeros([batch_size, 0], dtype=inp.dtype)
            rdo = tf.zeros([batch_size, hid_size], dtype=enc_out.dtype)

            attnP = tf.ones([batch_size, ninp]) / tf.to_float(inp_len)[:, None]

            offset = tf.zeros((batch_size,))

            empty_emb = tf.zeros([batch_size, 0, self.enc.emb_size])
            empty_dec_layers = [tf.zeros([batch_size, 0, self.dec.hid_size])] * self.dec.num_layers
            input_layers = [empty_emb] + empty_dec_layers[:-1]

            # prepare kv parts for all decoder attention layers. Note: we do not preprocess enc_out
            # for each layer because ResidualLayerWrapper only preprocesses first input (query)
            dec_enc_kv = [layer.kv_conv(enc_out)
                          for i, layer in enumerate(self.dec.dec_enc_attn)]
            dec_dec_kv = [layer.kv_conv(layer.preprocess(input_layers[i]))
                          for i, layer in enumerate(self.dec.dec_attn)]

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
        enc_out, enc_attn_mask, attnP, rdo, out_seq, offset, prev_emb = dec_state[:7]
        prev_dec_layers = dec_state.dec_layers
        dec_enc_kv = dec_state.dec_enc_kv
        dec_dec_kv = dec_state.dec_dec_kv

        batch_size = tf.shape(rdo)[0]
        if words is not None:
            out_seq = tf.concat([out_seq, tf.expand_dims(words, 1)], 1)

        with dropout_scope(is_train), tf.name_scope(self.name):
            # Embeddings
            if words is None:
                # initial step: words are None
                emb_out = tf.zeros((batch_size, 1, self.dec.emb_size))
            else:
                emb_out = self.dec.emb_out(words[:, None])  # [batch_size * 1 * emb_dim]
                if self.dec.rescale_emb:
                    emb_out *= self.dec.emb_size ** .5

            # Prepare decoder
            dec_inp_t = add_timing_signal(emb_out, offset=offset)
            # Apply dropouts
            if is_dropout_enabled():
                dec_inp_t = tf.nn.dropout(dec_inp_t, 1.0 - self.dec.dropout)

            # bypass info from Encoder to avoid None gradients for num_layers_dec == 0
            if self.dec.num_layers == 0:
                inp_mask = tf.squeeze(tf.transpose(enc_attn_mask, perm=[3, 1, 2, 0]), 3)
                dec_inp_t += tf.reduce_mean(enc_out * inp_mask, axis=[0, 1], keep_dims=True)

            # Decoder
            new_emb = tf.concat([prev_emb, dec_inp_t], axis=1)
            _out = tf.pad(out_seq, [(0, 0), (0, 1)])
            dec_attn_mask = make_attn_mask(_out, self.out_voc.eos, allow_lookahead=False)[:, :, -1:, :]

            new_dec_layers = []
            new_dec_dec_kv = []

            for layer in range(self.dec.num_layers):
                # multi-head self-attention: use only the newest time-step as query,
                # but all time-steps up to newest one as keys/values
                next_dec_kv = self.dec.dec_attn[layer].kv_conv(self.dec.dec_attn[layer].preprocess(dec_inp_t))
                new_dec_dec_kv.append(tf.concat([dec_dec_kv[layer], next_dec_kv], axis=1))
                dec_inp_t = self.dec.dec_attn[layer](dec_inp_t, dec_attn_mask, kv=new_dec_dec_kv[layer])

                dec_inp_t = self.dec.dec_enc_attn[layer](dec_inp_t, enc_attn_mask, kv=dec_enc_kv[layer])
                dec_inp_t = self.dec.dec_ffn[layer](dec_inp_t)

                new_dec_inp = tf.concat([prev_dec_layers[layer], dec_inp_t], axis=1)
                new_dec_layers.append(new_dec_inp)

            if self.dec.normalize_out:
                dec_inp_t = self.dec.dec_out_norm(dec_inp_t)

            rdo = dec_inp_t[:, -1]

            new_state = self.DecState(enc_out, enc_attn_mask, attnP, rdo, out_seq, offset + 1,
                                      new_emb, new_dec_layers, dec_enc_kv, new_dec_dec_kv)
            return new_state

    def get_rdo(self, dec_state):
        return dec_state.rdo, dec_state.out_seq

    def get_logits(self, dec_state, **flags):
        return self.logits(dec_state.rdo)

    def get_attnP(self, dec_state):
        return dec_state.attnP

    def symbolic_translate(self, inp, out, is_train=False):
        raise NotImplementedError("TODO(jheuristic) finish merging")
