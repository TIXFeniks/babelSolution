import keras.layers as L
import tensorflow as tf
from lib.layers import AttentionLayer
from lib.tensor_utils import infer_length, infer_mask
from vocab import Vocab

# NOTE: gnmt_lstm does not actually inherit TranslateModel, please don't use it to test advanced inference.


class AttentiveTranslationModel:
    def __init__(self, name, inp_voc, out_voc,
                 emb_size, hid_size, attn_size):
        """
        :param name: scope for all model variables
        :type inp_voc: Vocab
        :type out_voc: Vocab
        """
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc

        with tf.variable_scope(name):
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            self.enc0 = tf.nn.rnn_cell.LSTMCell(hid_size)

            self.cell_start = L.Dense(hid_size)
            self.hid_start = L.Dense(hid_size)

            self.dec0 = tf.nn.rnn_cell.LSTMCell(hid_size)
            self.attn = AttentionLayer('attn', hid_size, hid_size, attn_size)
            self.logits = L.Dense(len(out_voc))

            # run on dummy output to .build all layers (and therefore create weight variables)
            inp = tf.placeholder('int32', [None, None])
            out = tf.placeholder('int32', [None, None])
            h0 = self.encode(inp)

            h1 = self.decode(h0, out[:, 0])
            # h2 = self.decode(h1,out[:,1]) etc.

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos)
        enc_mask = infer_mask(inp, self.inp_voc.eos)
        inp_emb = self.emb_inp(inp)

        enc_seq, enc_last = tf.nn.dynamic_rnn(
            self.enc0, inp_emb,
            sequence_length=inp_lengths,
            dtype=inp_emb.dtype)

        cell0 = self.cell_start(enc_last[1])
        hid0 = self.hid_start(enc_last[1])

        attn_probas0 = tf.zeros(tf.shape(enc_seq)[:2])
        return [cell0, hid0, enc_seq, enc_mask, attn_probas0]

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """

        [prev_cell, prev_hid, enc_seq, enc_mask, prev_probas] = prev_state

        prev_emb = self.emb_out(prev_tokens[:, None])[:, 0]

        attn, attn_probas = self.attn(enc_seq, prev_hid, enc_mask)

        # if we have multiple recurrent layers, each LSTM/GRU/RNN layer should be placed
        # in a scope with different name to prevent variable reuse! (bug/"feature" in tf 1.3)
        with tf.variable_scope(self.name + "_lstm1"):
            lstm_inp = tf.concat([prev_emb, attn], axis=-1)
            prev_dec = tf.nn.rnn_cell.LSTMStateTuple(prev_cell, prev_hid)
            new_dec_out, new_dec_state = self.dec0(lstm_inp, prev_dec)

        new_cell, new_hid = new_dec_state

        output_logits = self.logits(new_dec_out)

        return [new_cell, new_hid, enc_seq, enc_mask, attn_probas], output_logits

    def symbolic_score(self, inp, out, eps=1e-30, return_state=False, crop_last=True, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [bath,time,n_tokens]
        NOTE: log-probabilities time axis  is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        first_state = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        def step(blob, y_prev):
            h_prev = blob[:-1]
            h_new, logits = self.decode(h_prev, y_prev, **flags)
            return list(h_new) + [logits]

        results = tf.scan(step, initializer=list(first_state) + [first_logits],
                          elems=tf.transpose(out))

        # gather state and logits, each of shape [time,batch,...]
        states_seq, logits_seq = results[:-1], results[-1]

        # add initial state and logits
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
        states_seq = [tf.concat((init[None], states), axis=0)
                      for init, states in zip(first_state, states_seq)]

        # convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        states_seq = [tf.transpose(states, [1, 0] + list(range(2, states.shape.ndims)))
                      for states in states_seq]

        logprobs = tf.nn.log_softmax(logits_seq)

        if crop_last:
            logprobs = logprobs[:, :-1]
            states_seq = [state_seq[:, :-1] for state_seq in states_seq]

        if return_state:
            return logprobs, states_seq
        else:
            return logprobs

    def symbolic_translate(self, inp, greedy=False, return_state=False, max_len=None, eps=1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        first_state = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)
        max_len = tf.reduce_max(tf.shape(inp)[1]) * 2

        def step(blob, t):
            h_prev, y_prev = blob[:-2], blob[-1]
            h_new, logits = self.decode(h_prev, y_prev, **flags)
            y_new = tf.argmax(logits, axis=-1) if greedy else tf.multinomial(logits, 1)[:, 0]
            return list(h_new) + [logits, tf.cast(y_new, y_prev.dtype)]

        results = tf.scan(step, initializer=list(first_state) + [first_logits, bos],
                          elems=[tf.range(max_len)])

        # gather state, logits and outs, each of shape [time,batch,...]
        states_seq, logits_seq, out_seq = results[:-2], results[-2], results[-1]

        # add initial state, logits and out
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)
        states_seq = [tf.concat((init[None], states), axis=0)
                      for init, states in zip(first_state, states_seq)]

        # convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)
        states_seq = [tf.transpose(states, [1, 0] + list(range(2, states.shape.ndims)))
                      for states in states_seq]

        logprobs = tf.nn.log_softmax(logits_seq)

        if return_state:
            return out_seq, logprobs, states_seq
        else:
            return out_seq, logprobs
