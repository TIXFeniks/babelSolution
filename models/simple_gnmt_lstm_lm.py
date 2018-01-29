import tensorflow as tf
import tfnn

from warnings import warn
warn("simple_gnmt_lstm_lm has long been broken and needs to die. Please don't poke it.")

from tfnn.layers.basic import *
from tfnn.layers.attn import *
from tfnn.ops.basic import *
from tfnn.ops.nn_resource import *

from tensorflow.python.framework import ops

ops.NotDifferentiable("LmScoreTrain")
ops.NotDifferentiable("PtScoreTrain")


def scatter_values_train(indices, values, voc_size):
    shp = tf.shape(indices)

    t_size = shp[0]
    b_size = shp[1]
    w_size = shp[2]

    # Build indices for other dimensions
    ti, bi, _ = tf.meshgrid(tf.range(0, t_size), tf.range(0, b_size), tf.range(0, w_size), indexing='ij')

    return tf.scatter_nd(tf.stack([ti, bi, indices], axis=3), values, [t_size, b_size, voc_size])

def scatter_values_predict(indices, values, voc_size):
    shp = tf.shape(indices)

    b_size = shp[0]
    w_size = shp[1]

    # Build indices for other dimensions
    bi, _ = tf.meshgrid(tf.range(0, b_size), tf.range(0, w_size), indexing='ij')

    return tf.scatter_nd(tf.stack([bi, indices], axis=2), values, [b_size, voc_size])


def zero_t0(values):
    shp = tf.shape(values)
    zero = tf.zeros([1, shp[1], shp[2]], dtype=values.dtype)
    return tf.concat([zero, values[1:]], 0)


class LossLogSoftmaxWithLm(SequenceLossBase):
    def __init__(self, name, rdo_size, voc, hp, matrix=None, resource_scope=None):
        """
        Parameters:

          Dense: <name>/logits
        """
        self.name = name
        self.rdo_size = rdo_size
        self.voc_size = voc.size()

        self.bos = voc.bos
        self.prefix_size = 10

        self.log_softmax_pos = hp.get('log_softmax_pos', 'after')
        self.lm_sample_size = hp.get('lm_sample_size', 1000)
        self.lm_score_module = tfnn.ops.load_op_library('lm_score.so')
        self.lm_gate = hp.get('lm_gate', False)
        self.lm_bos_eos_zero = hp.get('lm_bos_eos_zero', True)

        if 'lm_path' in hp:
            self.lm_score_atts = {
                'nn_resource_scope_id': resource_scope.scope_id,
                'dictionary': voc.words(list(range(voc.size()))),
                'lm_key': 'out_lm',
                'lm_voc_key': 'out_lm_voc',
                'bos': voc.bos if self.lm_bos_eos_zero else -1,
                'eos': voc.eos if self.lm_bos_eos_zero else -1,
                'unk_penalty': hp.get('lm_unk_penalty', 2.0),
            }
        else:
            self.lm_score_atts = None

        if 'pt_path' in hp:
            self.pt_score_atts = {
                'dictionary': voc.words(list(range(voc.size()))),
                'pt_path': hp['pt_path'],
                'unk_penalty': hp.get('pt_unk_penalty', 2.0),
            }
        else:
            self.pt_score_atts = None

        with tf.variable_scope(name):
            self.rdo_to_logits = Dense('logits', rdo_size, self.voc_size, activ=nop, matrix=matrix)

            if self.lm_gate:
                lm_gate_hid_size = hp.get('lm_gate_hid', 128)

                self.rdo_to_lm_gate_l0 = Dense('lm_gate_l0', rdo_size, lm_gate_hid_size, activ=tf.nn.relu)
                self.rdo_to_lm_gate_l1 = Dense('lm_gate_l1', lm_gate_hid_size, 3, activ=tf.nn.sigmoid)

            if hp.get('lm_fix_weights', False):
                if not self.lm_gate:
                    self.lm_score_weight = hp.get('lm_weight_init', -2.0)
                    self.pt_score_weight = hp.get('pt_weight_init', -5.1)

                self.lm_score_default = hp.get('lm_default_init', -25.0)
                self.lm_score_min = hp.get('lm_min_score', -20.0)

                self.pt_score_default = hp.get('pt_default_init', -30.0)
                self.pt_score_min = hp.get('pt_min_score', -30.0)
            else:
                if not self.lm_gate:
                    self.lm_score_weight = get_model_variable('lm_score_weight', shape=[], initializer=tf.constant_initializer(hp.get('lm_weight_init', -2.0)))
                    self.pt_score_weight = get_model_variable('pt_score_weight', shape=[], initializer=tf.constant_initializer(hp.get('pt_weight_init', -5.1)))

                self.lm_score_default = get_model_variable('lm_score_default', shape=[], initializer=tf.constant_initializer(hp.get('lm_default_init', -25.0)))
                self.lm_score_min = get_model_variable('lm_score_min', shape=[], initializer=tf.constant_initializer(hp.get('lm_min_score', -20.0)))

                self.pt_score_default = get_model_variable('pt_score_default', shape=[], initializer=tf.constant_initializer(hp.get('pt_default_init', -30.0)))
                self.pt_score_min = get_model_variable('pt_score_min', shape=[], initializer=tf.constant_initializer(hp.get('pt_min_score', -30.0)))

    def get_score_weights(self, rdo):
        if self.lm_gate:
            gate = self.rdo_to_lm_gate_l1(self.rdo_to_lm_gate_l0(rdo))

            return tf.unstack(tf.expand_dims(gate, axis=-1), axis=-2)
        else:
            nn_weight = 1.0
            lm_weight = tf.nn.sigmoid(self.lm_score_weight)
            pt_weight = tf.nn.sigmoid(self.pt_score_weight)

            return nn_weight, lm_weight, pt_weight

    def __call__(self, rdo, out, out_len, inp_words, attn_P_argmax):
        """
        rdo: [ninp, batch_size, rdo_size]
        out: [ninp, batch_size], dtype=int
        out_len: [batch_size]
        inp_words: [ninp, batch_size], dtype=string
        attn_P_argmax: [ninp, batch_size], dtype=int
        --------------------------
        Ret: [batch_size]
        """
        nn_weight, lm_weight, pt_weight = self.get_score_weights(rdo)

        nn_scores = self.rdo_to_logits(rdo) # [ninp, batch_size, voc_size]
        if self.log_softmax_pos == 'before':
            nn_scores = log_softmax(nn_scores, 2)

        top_indices = tf.nn.top_k(nn_scores, k=self.lm_sample_size).indices

        if self.lm_score_atts is not None:
            top_lm_scores = self.lm_score_module.lm_score_train(top_indices, out, inp_words, tf.cast(attn_P_argmax, tf.int32), **self.lm_score_atts)
            top_lm_scores = tf.clip_by_value(top_lm_scores, self.lm_score_min, 0) - self.lm_score_default
        else:
            top_lm_scores = self.lm_score_min * 1e-9

        if self.pt_score_atts is not None:
            top_pt_scores = self.lm_score_module.pt_score_train(top_indices, inp_words, tf.cast(attn_P_argmax, tf.int32), **self.pt_score_atts)
            top_pt_scores = tf.clip_by_value(top_pt_scores, self.pt_score_min, 0) - self.pt_score_default
        else:
            top_pt_scores = self.pt_score_min * 1e-9

        top_additions = top_lm_scores * lm_weight + top_pt_scores * pt_weight
        top_defaults = self.lm_score_default * lm_weight + self.pt_score_default * pt_weight

        logits = nn_scores * nn_weight + zero_t0(scatter_values_train(top_indices, top_additions, self.voc_size)) + top_defaults

        neg_log_P = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=out)
        loss = neg_log_P * sequence_mask(out_len, dtype=logits.dtype)
        return tf.reduce_sum(loss, [0])

    def make_sample_fn(self, rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax, pt_penalty=None):
        """
        rdo: [batch_size, rdo_size]
        prefix: [batch_size, prefix_size]
        inp_words: [batch_size, input_size]
        attn_P_argmax: [batch_size]
        ---------------------------
        Ret: sample_fn(n, base_scores) -> hypo_indices, words, delta_scores
          n: []
          base_scores: [batch_size]
          hypo_indices: [n]
          words: [n]
          delta_scores: [n]
        """
        return LossSampler(self.get_logP(rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax), self.voc_size, self.name, pt_penalty=pt_penalty)

    def get_scores_and_weights(self, rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax):
        nn_weight, lm_weight, pt_weight = self.get_score_weights(rdo)

        nn_scores = self.rdo_to_logits(rdo)
        if self.log_softmax_pos == 'before':
            nn_scores = log_softmax(nn_scores, 1)

        top_indices = tf.nn.top_k(nn_scores, k=self.lm_sample_size, sorted=False).indices

        if self.lm_score_atts is not None:
            top_lm_scores = self.lm_score_module.lm_score_predict(top_indices, prefix, inp_words, tf.cast(attn_P_argmax, tf.int32), prefix_attnP_argmax, **self.lm_score_atts)
            top_lm_scores = tf.clip_by_value(top_lm_scores, self.lm_score_min, 0) - self.lm_score_default
        else:
            top_lm_scores = tf.zeros(tf.shape(top_indices))

        if self.pt_score_atts is not None:
            top_pt_scores = self.lm_score_module.pt_score_predict(top_indices, inp_words, tf.cast(attn_P_argmax, tf.int32), **self.pt_score_atts)
            top_pt_scores = tf.clip_by_value(top_pt_scores, self.pt_score_min, 0) - self.pt_score_default
        else:
            top_pt_scores = tf.zeros(tf.shape(top_indices))

        return nn_weight, lm_weight, pt_weight, nn_scores, top_indices, top_lm_scores, top_pt_scores

    def get_logP(self, rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax):
        nn_weight, lm_weight, pt_weight, nn_scores, top_indices, top_lm_scores, top_pt_scores = self.get_scores_and_weights(rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax)

        # Decompose a*nn + b*(scatter(top_lm) + def_lm) + c*(scatter(top_pt) + def_pt) into
        # a*nn + (b*def_lm + c*def_pt) + scatter(b*top_lm + c*top_pt)

        base = nn_scores * nn_weight + (self.lm_score_default * lm_weight + self.pt_score_default * pt_weight)
        addition = scatter_values_predict(top_indices, top_lm_scores * lm_weight + top_pt_scores * pt_weight, self.voc_size)
        logits = base + addition

        if self.log_softmax_pos == 'after':
            logits = tf.nn.log_softmax(logits, 1)

        return logits

    def get_logP_ext(self, rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax):
        nn_weight, lm_weight, pt_weight, nn_scores, top_indices, top_lm_scores, top_pt_scores = self.get_scores_and_weights(rdo, prefix, inp_words, attn_P_argmax, prefix_attnP_argmax)

        lm_scores = scatter_values_predict(top_indices, top_lm_scores, self.voc_size) + self.lm_score_default
        pt_scores = scatter_values_predict(top_indices, top_pt_scores, self.voc_size) + self.pt_score_default

        logits = nn_scores * nn_weight + lm_scores * lm_weight + pt_scores * pt_weight

        if self.log_softmax_pos == 'after':
            logits = log_softmax(logits, 1)

        return logits, {'raw': log_softmax(nn_scores, 1), 'lm': log_softmax(lm_scores, 1), 'pt': log_softmax(pt_scores, 1)}, {'nn_weight': nn_weight, 'lm_weight': lm_weight, 'pt_weight': pt_weight}


class AttnRNNCellWithArgmax(tf.contrib.rnn.RNNCell):
    def __init__(self, attn_cell, rnn_cell, enc, inp_len):
        """
                    Original cell    This cell
        ----------- ---------------- ------------------------
        Input       <plain>          <plain>
        Output      <any>            (<any>, attn, attn_P_argmax)
        State       <any>            <any>
        -----------------------------------------------------

        Note that attention lags one step back behind outputs.

        No parameters.
        """
        self.attn_cell = attn_cell
        self.rnn_cell = rnn_cell
        self.enc = enc
        self.inp_mask = sequence_mask(inp_len, dtype=self.enc.dtype)

    @property
    def output_size(self):
        return (self.rnn_cell.output_size, self.attn_cell.output_size, tf.TensorShape([]))

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    def __call__(self, inputs, state):
        """
        inputs: [batch_size, inp_size]
        state: [batch_size, hid_size]*
        --------------------------------
        outputs: [batch_size, out_size]*
        """
        # Compute attention.
        #   attn: [batch_size, attn_size]
        attn, attn_P = self.attn_cell.attn_and_P(self.enc, state, self.inp_mask)
        attn_P_argmax = tf.cast(1 + tf.argmax(attn_P[1:-1], axis=0), tf.float32)

        # Append attention to inputs.
        with tf.variable_scope(self.rnn_cell.name):
            inputs = tf.concat([inputs, attn], 1)

        # Run cell
        outputs, new_state = self.rnn_cell(inputs, state)
        return (outputs, attn, attn_P_argmax), new_state


def fullword_dropout(inp, inp_len, dropout, unk):
    inp_shape = tf.shape(inp)

    mask = tf.transpose(tf.sequence_mask(inp_len - 2, inp_shape[0] - 2))
    mask = tf.concat((tf.fill([1, inp_shape[1]], False), mask, tf.fill([1, inp_shape[1]], False)), axis=0)
    mask = tf.logical_and(mask, tf.random_uniform(inp_shape) < dropout)

    return tf.where(mask, tf.fill(inp_shape, tf.cast(unk, inp.dtype)), inp)


class EncDec:
    def __init__(
            self,
            name,
            inp_voc,
            out_voc,
            emb_size,
            hid_size,
            attn_hid_size,
            rdo_size,
            peephole,
            use_orto_init,
            recurrent_dropout,
            input_dropout,
            output_dropout,
            word_dropout,
            enc_fullword_dropout,
            dec_fullword_dropout
            ):
        self.name = name
        self.hid_size = hid_size
        self.emb_size = emb_size
        self.rdo_size = rdo_size
        self.word_dropout = word_dropout

        self.enc_fullword_dropout = enc_fullword_dropout
        self.dec_fullword_dropout = dec_fullword_dropout

        self.inp_voc = inp_voc
        self.out_voc = out_voc

        with tf.variable_scope(name):
            self.emb_inp = Embedding('emb_inp', inp_voc.size(), emb_size)
            self.emb_out = Embedding('emb_out', out_voc.size(), emb_size)
            self.enc0_fwd = LSTMCell('enc0_fwd', emb_size, hid_size, peephole=peephole,
                                     recurrent_dropout=recurrent_dropout, use_orto_init=use_orto_init,
                                     input_dropout=input_dropout, output_dropout=output_dropout)
            self.enc0_rev = LSTMCell('enc0_rev', emb_size, hid_size, peephole=peephole,
                                     recurrent_dropout=recurrent_dropout, use_orto_init=use_orto_init,
                                     input_dropout=input_dropout, output_dropout=output_dropout)
            self.attn = AttnBahdanau('attn', 2 * hid_size, hid_size, attn_hid_size)
            self.dec0 = LSTMCell('dec0', emb_size + 2 * hid_size, hid_size, peephole=peephole,
                                 recurrent_dropout=recurrent_dropout, use_orto_init=use_orto_init,
                                 input_dropout=input_dropout, output_dropout=output_dropout)
            self.rdo = Dense('rdo', emb_size + 3*hid_size, rdo_size, activ=nop)

    def __call__(self, inp, out, inp_len, out_len, is_train):
        with dropout_scope(is_train), tf.variable_scope(self.name):
            if is_train and self.enc_fullword_dropout > 0:
                inp = fullword_dropout(inp, inp_len, self.enc_fullword_dropout, self.inp_voc._unk)

            if is_train and self.dec_fullword_dropout > 0:
                out = fullword_dropout(out, out_len, self.dec_fullword_dropout, self.out_voc._unk)

            # First encoder layer (bidirectional).
            emb_inp = self.emb_inp(inp)
            emb_inp = tf.nn.dropout(emb_inp, 1 - self.word_dropout)
            dtype = emb_inp.dtype

            enc, _ = tf.nn.bidirectional_dynamic_rnn(
                self.enc0_fwd,
                self.enc0_rev,
                emb_inp,
                sequence_length=inp_len,
                time_major=True,
                dtype=dtype
                )
            enc = tf.concat(enc, 2)

            # First decoder layer.
            emb_out = self.emb_out(out)
            (dec, attn, attn_P_argmax), _ = tf.nn.dynamic_rnn(
                AttnRNNCellWithArgmax(self.attn, self.dec0, enc, inp_len),
                emb_out,
                sequence_length=out_len,
                time_major=True,
                dtype=dtype
                )

            # Correct readout.
            nout = tf.shape(out)[0]
            batch_size = tf.shape(out)[1]
            ###
            dec_start = tf.zeros([1, batch_size, self.hid_size], dtype)
            dec_end = tf.slice(dec, [0, 0, 0], [nout-1, -1, -1])
            dec = tf.concat([dec_start, dec_end], 0)

            emb_out_start = tf.zeros([1, batch_size, self.emb_size], dtype)
            emb_out_end = tf.slice(emb_out, [0, 0, 0], [nout-1, -1, -1])
            emb_out = tf.concat([emb_out_start, emb_out_end], 0)

            return self.rdo(tf.concat([emb_out, dec, attn], 2)), attn_P_argmax

## ===============================================================================================
#                                        TranslatorCore

class TranslatorCore:
    def __init__(self, encdec, loss, name=None, pt_penalty=None):
        self.encdec = encdec
        self.loss = loss
        self.attnP_val = None

        with tf.variable_scope(name + '/' + encdec.name if name else encdec.name):
            # Placeholders.
            self.inp = tf.placeholder(tf.int32, [None, None], name='t_inp')
            self.inp_words_ph = tf.placeholder(tf.string, [None, None], name='t_inp_words_ph')
            self.inp_len = tf.placeholder(tf.int32, [None], name='t_inp_len')
            self.hypo_indices = tf.placeholder(tf.int32, [None], name='t_hypo_indices')
            self.words = tf.placeholder(tf.int32, [None], name='t_words')
            self.mask = tf.placeholder(tf.bool, [None], name='t_mask')

            # State.
            N = False
            self.inp_words_var = tf.get_variable('t_inp_words', [], dtype=tf.string, initializer=tf.constant_initializer(''), trainable=N, validate_shape=N,
                                                 collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.inp_mask_var = tf.get_variable('t_inp_mask', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.enc_var = tf.get_variable('t_enc', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.h_dec_var = tf.get_variable('t_h_dec', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.c_dec_var = tf.get_variable('t_c_dec', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.attn_var = tf.get_variable('t_attn', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.attnP_var = tf.get_variable('t_attnP', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.rdo_var = tf.get_variable('t_rdo', [], trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.prefix_var = tf.get_variable('t_prefix', [], dtype=tf.int32, trainable=N, validate_shape=N, collections=[tfnn.train.saveload.DO_NOT_SAVE])
            self.prefix_attnP_argmax_var = tf.get_variable('t_prefix_attnP_argmax', [], dtype=tf.int32,trainable=N, validate_shape=N,
                                                           collections=[tfnn.train.saveload.DO_NOT_SAVE])

            # State with shape.
            self.inp_words = with_shape(self.inp_words_var, [None, None])
            self.inp_mask = with_shape(self.inp_mask_var, [None, None])
            self.enc = with_shape(self.enc_var, [None, None, encdec.hid_size * 2])
            self.h_dec = with_shape(self.h_dec_var, [None, encdec.hid_size])
            self.c_dec = with_shape(self.c_dec_var, [None, encdec.hid_size])
            self.attn = with_shape(self.attn_var, [None, encdec.hid_size * 2])
            self.attnP = with_shape(self.attnP_var, [None, None])
            self.rdo = with_shape(self.rdo_var, [None, encdec.rdo_size])
            self.prefix = with_shape(self.prefix_var, [None, loss.prefix_size])
            self.prefix_attnP_argmax = with_shape(self.prefix_attnP_argmax_var, [None, loss.prefix_size])

            # Operations
            self.encode_op = self._make_encode_op()
            self.shuffle_op = self._make_shuffle_op()
            self.decode_op = self._make_decode_op()

        # Sample function (in global scope).
        self.sample_fn = loss.make_sample_fn(self.rdo, self.prefix, self.inp_words, tf.cast(1+tf.argmax(self.attnP[:, 1:-1], axis=1), tf.int32), self.prefix_attnP_argmax, pt_penalty)

    def encode(self, inp, inp_len, inp_words):
        tf.get_default_session().run(self.encode_op, feed_dict={
            self.inp: inp,
            self.inp_words_ph: inp_words,
            self.inp_len: inp_len
            })
        self.attnP_val = self.attnP_var.eval()

    def shuffle(self, hypo_indices):
        tf.get_default_session().run(self.shuffle_op, feed_dict={
            self.hypo_indices: hypo_indices
            })

    def sample(self, n, base_scores, slices, **kwargs):
        return self.sample_fn(n, base_scores, slices, **kwargs)

    def decode(self, words, mask):
        tf.get_default_session().run(self.decode_op, feed_dict={
            self.words: words,
            self.mask: mask
            })
        self.attnP_val = self.attnP_var.eval()

    def get_attnP(self):
        return self.attnP_val

    def _make_encode_op(self):
        with dropout_scope(False):
            # Encode.
            emb_inp = self.encdec.emb_inp(self.inp)
            dtype = emb_inp.dtype

            # Input mask.
            inp_mask = sequence_mask(self.inp_len, dtype=dtype)

            enc, _ = tf.nn.bidirectional_dynamic_rnn(
                self.encdec.enc0_fwd,
                self.encdec.enc0_rev,
                emb_inp,
                sequence_length=self.inp_len,
                time_major=True,
                dtype=dtype
                )
            enc = tf.concat(enc, 2)

            # Decoder start.
            batch_size = tf.shape(self.inp)[1]
            dec_start = tf.zeros([batch_size, self.encdec.hid_size], dtype)

            # Attention & readout.
            attn, attnP = self.encdec.attn.attn_and_P(enc, dec_start, inp_mask)
            attnP = tf.transpose(attnP)
            emb_out_start = tf.zeros([batch_size, self.encdec.emb_size], dtype)
            rdo = self.encdec.rdo(tf.concat([emb_out_start, dec_start, attn], 1))
            prefix = tf.fill([batch_size, self.loss.prefix_size], self.loss.bos)
            prefix_attnP_argmax = tf.zeros([batch_size, self.loss.prefix_size], dtype=tf.int32)
            inp_words = tf.transpose(self.inp_words_ph)

            return [
                tf.assign(self.inp_mask_var, inp_mask, validate_shape=False),
                tf.assign(self.inp_words_var, inp_words, validate_shape=False),
                tf.assign(self.enc_var, enc, validate_shape=False),
                tf.assign(self.h_dec_var, dec_start, validate_shape=False),
                tf.assign(self.c_dec_var, dec_start, validate_shape=False),
                tf.assign(self.attn_var, attn, validate_shape=False),
                tf.assign(self.attnP_var, attnP, validate_shape=False),
                tf.assign(self.rdo_var, rdo, validate_shape=False),
                tf.assign(self.prefix_var, prefix, validate_shape=False),
                tf.assign(self.prefix_attnP_argmax_var, prefix_attnP_argmax, validate_shape=False)
                ]

    def _make_shuffle_op(self):
        inp_mask = tf.transpose(self.inp_mask)
        inp_mask = tf.gather(inp_mask, self.hypo_indices)
        inp_mask = tf.transpose(inp_mask)
        inp_words = tf.gather(self.inp_words, self.hypo_indices)
        enc = tf.transpose(self.enc, [1, 0, 2])
        enc = tf.gather(enc, self.hypo_indices)
        enc = tf.transpose(enc, [1, 0, 2])
        h_dec = tf.gather(self.h_dec, self.hypo_indices)
        c_dec = tf.gather(self.c_dec, self.hypo_indices)
        attn = tf.gather(self.attn, self.hypo_indices)
        attnP = tf.gather(self.attnP, self.hypo_indices)
        rdo = tf.gather(self.rdo, self.hypo_indices)
        prefix = tf.gather(self.prefix, self.hypo_indices)
        prefix_attnP_argmax = tf.gather(self.prefix_attnP_argmax, self.hypo_indices)

        return [
            tf.assign(self.inp_mask_var, inp_mask, validate_shape=False),
            tf.assign(self.inp_words_var, inp_words, validate_shape=False),
            tf.assign(self.enc_var, enc, validate_shape=False),
            tf.assign(self.h_dec_var, h_dec, validate_shape=False),
            tf.assign(self.c_dec_var, c_dec, validate_shape=False),
            tf.assign(self.attn_var, attn, validate_shape=False),
            tf.assign(self.attnP_var, attnP, validate_shape=False),
            tf.assign(self.rdo_var, rdo, validate_shape=False),
            tf.assign(self.prefix_var, prefix, validate_shape=False),
            tf.assign(self.prefix_attnP_argmax_var, prefix_attnP_argmax, validate_shape=False)
            ]

    def _make_decode_op(self):
        with dropout_scope(False):
            # Decode.
            emb_out = self.encdec.emb_out(self.words)
            inputs = tf.concat([emb_out, self.attn], 1)
            dec, (h_dec_state, c_dec_state) = self.encdec.dec0(inputs, (self.h_dec, self.c_dec))
            attn, attnP = self.encdec.attn.attn_and_P(self.enc, (h_dec_state, c_dec_state), self.inp_mask)
            attnP = tf.transpose(attnP)

            # Attention & readout.
            rdo = self.encdec.rdo(tf.concat([emb_out, dec, attn], 1))

            h_dec_state = tf.where(self.mask, h_dec_state, self.h_dec)
            c_dec_state = tf.where(self.mask, c_dec_state, self.c_dec)
            attn = tf.where(self.mask, attn, self.attn)
            attnP = tf.where(self.mask, attnP, self.attnP)
            rdo = tf.where(self.mask, rdo, self.rdo)
            prefix = tf.where(self.mask, tf.concat([self.prefix[:, 1:], tf.reshape(self.words, [-1, 1])], axis=1), self.prefix)
            prefix_attnP_argmax = tf.where(self.mask, tf.concat([self.prefix_attnP_argmax[:, 1:], tf.reshape(tf.cast(1+tf.argmax(self.attnP[:, 1:-1], axis=1), tf.int32), [-1, 1])], axis=1), self.prefix_attnP_argmax)

            return [
                tf.assign(self.h_dec_var, h_dec_state, validate_shape=False),
                tf.assign(self.c_dec_var, c_dec_state, validate_shape=False),
                tf.assign(self.attn_var, attn, validate_shape=False),
                tf.assign(self.attnP_var, attnP, validate_shape=False),
                tf.assign(self.rdo_var, rdo, validate_shape=False),
                tf.assign(self.prefix_var, prefix, validate_shape=False),
                tf.assign(self.prefix_attnP_argmax_var, prefix_attnP_argmax, validate_shape=False)
                ]


class Model(tfnn.task.seq2seq.Seq2SeqModel):
    def __init__(self, inp_voc, out_voc, hp):
        # Hyperparameters.
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp

        set_precharge(hp.get('lm_precharge', True))
        self.resource_scope = ResourceScope()

        if 'lm_path' in hp:
            self.resource_scope.load_resource("out_lm", NN_RES_LANGUAGE_MODEL, hp["lm_path"])
            self.resource_scope.load_resource("out_lm_voc", NN_RES_VOCABULARY, hp["lm_voc_path"])

        # Parameters
        self.gnmt = EncDec(
            'mod', inp_voc, out_voc, hp['emb_size'], hp['hid_size'],
            hp['attn_hid_size'], hp['rdo_size'], hp.get('peephole', False), hp.get('use_orto_init', False),
             hp.get('recurrent_dropout', 0), hp.get('input_dropout', 0),
            hp.get('output_dropout', 0), hp.get('word_dropout', 0),
            hp.get('enc_fullword_dropout', 0), hp.get('dec_fullword_dropout', 0)
            )

        matrix = tf.transpose(self.gnmt.emb_out.mat) if hp.get('dwwt', False) else None
        self.loss = LossLogSoftmaxWithLm('loss', hp['rdo_size'], out_voc, matrix=matrix, hp=hp, resource_scope=self.resource_scope)

        # Translator.
        self.translator = tfnn.task.seq2seq.translate.Translator(
            core=TranslatorCore(self.gnmt, self.loss, pt_penalty=hp.get('pt_penalty', None)),
            bos=out_voc.bos,
            eos=out_voc.eos,
            beam_size=hp['beam_size'],
            beam_spread=hp['beam_spread'],
            len_alpha=hp['len_alpha'],
            attn_beta=hp['attn_beta'],
            out_voc=out_voc,
            debug=hp.get('debug', False),
            pass_inp_words=True)

        # Predictor .
        self.predictor = tfnn.task.seq2seq.predict.Predictor(
            core=self.translator.core,
            inp_voc=self.inp_voc,
            out_voc=self.out_voc,
            len_alpha=hp['len_alpha'],
            attn_beta=hp['attn_beta'],
            pass_inp_words=True)

    def encode_decode(self, batch, is_train):
        inp = batch['inp']
        out = batch['out']
        inp_len = batch['inp_len']
        out_len = batch['out_len']

        return self.gnmt(inp, out, inp_len, out_len, is_train)

    def translate_many(self, lines, **kwargs):
        lines = tfnn.task.seq2seq.translate.translate_many(
            lines, self.translator, self.inp_voc, self.out_voc,
            self.hp.get('replace'), inp_bos=True, inp_eos=True,
            **kwargs)
        return [line.replace(' `', '') for line in lines]

    def predict(self):
        self.predictor.main()
