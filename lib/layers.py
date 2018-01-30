import tensorflow as tf
import numpy as np
from .tensor_utils import is_tf_tensor, dot, is_dropout_enabled, nop


class Dense:
    def __init__(
        self, name,
        inp_size, out_size, activaction=nop,
        W=None, b=None,
    ):
        """
        Dense layer that actually creates all it's variables on init.
        :param inp_size: number of input features
        :param out_size: number of units
        :param activaction: nonlinearity applied after linear part
        :param W: tf tensor (to be used permanently) or tf initializer (to be used at init)
            shape should be (inp_size, out_size)
        :param b: tf tensor (to be used permanently) or tf initializer (to be used at init)
            shape should be (out_size,)
        Stores two variables: name/W and name/b
        """
        self.name = name
        self.activ = activaction
        self.inp_size = inp_size
        self.out_size = out_size

        with tf.variable_scope(name):
            if is_tf_tensor(W):
                self.W = W
            else:
                self.W = tf.get_variable('W', shape=[inp_size, out_size], initializer=W)

            if is_tf_tensor(b):
                self.b = b
            else:
                self.b = tf.get_variable('b', shape=[out_size], initializer=b)

    def __call__(self, inp):
        """
        inp: [..., inp_size]
        --------------------
        Ret: [..., out_size]
        """
        with tf.variable_scope(self.name):
            out = self.activ(dot(inp, self.W) + self.b)
            out.set_shape([None]*(out.shape.ndims-1) + [self.out_size])
            return out

    @property
    def input_size(self):
        return self.inp_size

    @property
    def output_size(self):
        return self.out_size


class Embedding:
    def __init__(self, name, voc_size, emb_size, matrix=None, initializer=None):
        """
        Embedding layer that actually creates all it's variables on init.

        :param voc_size: maximum index at input
        :param emb_size: number of units
        :param matrix: tf tensor (to be used permanently) or tf initializer (to be used at init)
            shape should be (inp_size, out_size)
        Stores weights as name/mat
        """
        self.name = name
        self.voc_size = voc_size
        self.emb_size = emb_size

        if is_tf_tensor(matrix):
            self.mat = matrix
        else:
            with tf.variable_scope(name):
                self.mat = tf.get_variable('mat', shape=[voc_size, emb_size], initializer=matrix)

    def __call__(self, inp, adv_eps=0, gumbel=False):
        """ input.shape: [...], output.shape: [..., emb_size] """
        mat = self.mat + adv_eps if adv_eps is not None else self.mat
        with tf.variable_scope(self.name):
            return tf.gather(mat, inp) if not gumbel else dot(inp, mat)


class AttentionLayer:
    def __init__(self, name, enc_size, dec_size, hid_size, activ=tf.tanh,):
        """
        A basic layer that computes attention weights and response
        """
        self.name = name
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.hid_size = hid_size
        self.activ = activ

        with tf.variable_scope(name):
            self.dec = tf.get_variable('dec', shape=[dec_size, hid_size])
            self.enc = tf.get_variable('enc', shape=[enc_size, hid_size])
            self.vec = tf.get_variable('vec', shape=[hid_size, 1])[:, 0]

    def __call__(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        Input shapes:
        enc: [batch_size, ninp, enc_size]
        dec: [batch_size, dec_size]
        inp_mask: [batch_size, ninp]
        Output shapes:
        attn: [batch_size, enc_size]
        probs: [batch_size, ninp]
        """
        with tf.variable_scope(self.name):
            # Compute hiddens.
            # in case of LSTM it is tuple (hid_state, cell_state)
            if isinstance(dec, tuple):
                dec = dec[0]

            # Bahdanau attention
            hid = self.activ(  # [batch_size, ninp, hid_size]
                dot(enc, self.enc) +
                tf.expand_dims(dot(dec, self.dec), 1)
                )
            scores = dot(hid, self.vec)  # [batch_size, ninp]

            # Compute scores.
            scores -= tf.reduce_max(scores, axis=1, keep_dims=True)
            scores -= (1 - inp_mask) * 1000

            # Compute probabilities
            scores_exp = tf.exp(scores)
            z = tf.reduce_sum(scores_exp, axis=1, keep_dims=True)
            probs = scores_exp / z

            # Compose attention.
            attn = tf.reduce_sum(tf.expand_dims(probs, 2) * enc, axis=1)

            return attn, probs


class FeedforwardBlock:
    """
    Feed-forward layer with expanding bottleneck. Used for transformer.
    inp -> dense -> relu -> dense -> out
    """
    def __init__(self, name,
                 inp_size, hid_size, out_size,
                 relu_dropout):
        self.name = name
        self.relu_dropout = relu_dropout

        with tf.variable_scope(name):
            self.first_conv = Dense(
                'conv1',
                inp_size, hid_size,
                activaction=tf.nn.relu,
                b=tf.zeros_initializer())

            self.second_conv = Dense(
                'conv2',
                hid_size, out_size,
                activaction=lambda x: x,
                b=tf.zeros_initializer())

    def __call__(self, inputs, params_summary=None):
        """
        inp: [batch_size * ninp * inp_dim]
        ---------------------------------
        out: [batch_size * ninp * out_dim]
        """
        with tf.variable_scope(self.name):
            hidden = self.first_conv(inputs)

            if is_dropout_enabled():
                hidden = tf.nn.dropout(hidden, 1.0 - self.relu_dropout)

            outputs = self.second_conv(hidden)

            return outputs


class MultiHeadAttention:
    """
    Multihead scaled-dot-product attention with input/output transformations. Used for transformer.
    """
    MASK_LOGITS = -1e9

    def __init__(
        self, name, inp_size,
        key_depth, value_depth, output_depth,
        num_heads=8, attn_dropout=0, attn_value_dropout=0
    ):
        self.name = name
        self.key_depth = key_depth
        self.value_depth = value_depth
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.attn_value_dropout = attn_value_dropout

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope()

            self.combined_conv = Dense(
                'mem_conv',  # old name for compatibility
                inp_size, key_depth * 2 + value_depth,
                activaction=lambda x: x,
                b=tf.zeros_initializer())

            self.query_conv = Dense(
                'query_conv',
                inp_size, key_depth,
                activaction=lambda x: x,
                W=self.combined_conv.W[:, :key_depth],
                b=self.combined_conv.b[:key_depth],
            )

            self.kv_conv = Dense(
                'kv_conv',
                inp_size, key_depth + value_depth,
                activaction=lambda x: x,
                W=self.combined_conv.W[:, key_depth:],
                b=self.combined_conv.b[key_depth:],
            )

            self.out_conv = Dense(
                'out_conv',
                value_depth, output_depth,
                activaction=lambda x: x,
                b=tf.zeros_initializer())

    def __call__(self, query_inp, attn_mask, kv_inp=None, kv=None):
        """
        query_inp: [batch_size * n_q * inp_dim]
        attn_mask: [batch_size * 1 * n_q * n_kv]
        kv_inp: [batch_size * n_kv * inp_dim]
        -----------------------------------------------
        results: [batch_size * n_q * output_depth]
        """
        assert kv is None or kv_inp is None, "please only feed one of kv or kv_inp"
        with tf.name_scope(self.name):
            if kv_inp is not None or kv is not None:
                q = self.query_conv(query_inp)
                if kv is None:
                    kv = self.kv_conv(kv_inp)
                k, v = tf.split(kv, [self.key_depth, self.value_depth], axis=2)
            else:
                combined = self.combined_conv(query_inp)
                q, k, v = tf.split(combined, [self.key_depth, self.key_depth, self.value_depth], axis=2)
            q = self._split_heads(q)  # [batch_size * n_heads * n_q * (k_dim/n_heads)]
            k = self._split_heads(k)  # [batch_size * n_heads * n_kv * (k_dim/n_heads)]
            v = self._split_heads(v)  # [batch_size * n_heads * n_kv * (v_dim/n_heads)]

            key_depth_per_head = self.key_depth / self.num_heads
            q = q / np.sqrt(key_depth_per_head)

            # Dot-product attention
            # logits: (batch_size * n_heads * n_q * n_kv)
            attn_bias = MultiHeadAttention.MASK_LOGITS * (1 - attn_mask)
            logits = tf.matmul(
                tf.transpose(q, perm=[0, 1, 2, 3]),
                tf.transpose(k, perm=[0, 1, 3, 2])) + attn_bias
            weights = tf.nn.softmax(logits)

            if is_dropout_enabled():
                weights = tf.nn.dropout(weights, 1.0 - self.attn_dropout)
            x = tf.matmul(
                weights,                         # [batch_size * n_heads * n_q * n_kv]
                tf.transpose(v, perm=[0, 1, 2, 3])  # [batch_size * n_heads * n_kv * (v_deph/n_heads)]
            )
            combined_x = self._combine_heads(x)

            if is_dropout_enabled():
                combined_x = tf.nn.dropout(combined_x, 1.0 - self.attn_value_dropout)

            outputs = self.out_conv(combined_x)

            return outputs

    def _split_heads(self, x):
        """
        Split channels (dimension 3) into multiple heads (dimension 1)
        input: (batch_size * ninp * inp_dim)
        output: (batch_size * n_heads * ninp * (inp_dim/n_heads))
        """
        old_shape = x.get_shape().dims
        dim_size = old_shape[-1]
        new_shape = old_shape[:-1] + [self.num_heads] + [dim_size // self.num_heads if dim_size else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [self.num_heads, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])  # [batch_size * n_heads * ninp * (hid_dim//n_heads)]

    def _combine_heads(self, x):
        """
        Inverse of split heads
        input: (batch_size * n_heads * ninp * (inp_dim/n_heads))
        out: (batch_size * ninp * inp_dim)
        """
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        ret.set_shape(new_shape)
        ret = tf.transpose(ret, perm=[0, 1, 2])
        return ret


class LayerNorm:
    """
    Performs Layer Normalization
    """
    def __init__(self, name, inp_size, epsilon=1e-6):
        self.name = name
        self.epsilon = epsilon

        with tf.variable_scope(name):
            self.scale = tf.get_variable('scale', shape=[inp_size], initializer=tf.ones_initializer())
            self.bias = tf.get_variable('bias', shape=[inp_size], initializer=tf.zeros_initializer())

    def __call__(self, inp):
        with tf.variable_scope(self.name):
            mean = tf.reduce_mean(inp, axis=[-1], keep_dims=True)
            variance = tf.reduce_mean(tf.square(inp - mean), axis=[-1], keep_dims=True)
            norm_x = (inp - mean) * tf.rsqrt(variance + self.epsilon)
            return norm_x * self.scale + self.bias


class Wrapper:
    """ Reflection-style wrapper, code from http://code.activestate.com/recipes/577555-object-wrapper-class/ """
    def __init__(self, wrapped_layer):
        self.wrapped_layer = wrapped_layer

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_layer, attr)


class ResidualLayerWrapper(Wrapper):
    def __init__(self, name, wrapped_layer, inp_size, out_size, steps='ldan', dropout=0, dropout_seed=None):
        """
        Applies any number of residual connection, dropout and/or layer normalization before or after wrapped layer
        :param steps: a sequence of operations to perform, containing any combination of:
            - 'l' - call wrapped [l]ayer, this operation should be used exactly once
            - 'd' - apply [d]ropout with p = dropout and seed = dropout_seed
            - 'a' - [a]dd inputs to output (residual connection)
            - 'n' - apply layer [n]ormalization here, can only be done once
        """
        assert steps.count('l') == 1, "residual wrapper must call wrapped layer exactly once"
        assert steps.count('n') <= 1, "in current implementaion, there can be at most one layer normalization step"
        assert inp_size == out_size or 'a' not in steps, "residual step only works if inp_size == out_size"
        self.name = name
        super(self.__class__, self).__init__(wrapped_layer)

        if 'n' in steps:
            ln_size = inp_size if steps.index('n') < steps.index('l') else out_size
            with tf.variable_scope(name):
                self.norm_layer = LayerNorm("layer_norm", ln_size)

        self.preprocess_steps = steps[:steps.index('l')]
        self.postprocess_steps = steps[steps.index('l') + 1:]
        self.dropout = dropout
        self.dropout_seed = dropout_seed

    def __call__(self, inp, *args, **kwargs):
        out = self.preprocess(inp)
        out = self.wrapped_layer(out, *args, **kwargs)
        out = self.postprocess(out, inp)
        return out

    def preprocess(self, inp):
        return self._perform(self.preprocess_steps, inp)

    def postprocess(self, out, inp=None):
        return self._perform(self.postprocess_steps, out, inp=inp)

    def _perform(self, steps, out, inp=None):
        if inp is None:
            inp = out
        for s in steps:
            if s == 'd':
                if is_dropout_enabled():
                    out = tf.nn.dropout(out, 1.0 - self.dropout, seed=self.dropout_seed)
            elif s == 'a':
                out += inp
            elif s == 'n':
                out = self.norm_layer(out)
            else:
                raise RuntimeError("Unknown process step: %s" % s)
        return out
