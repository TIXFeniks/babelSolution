import numpy
import tensorflow as tf
from lib.layers import *
from lib.tensor_utils import *


class TransformerEncoder:
    """
    Implements encoder-style self-attention for transformer network.
    Encodes inp[batch, time], enc_attn_mask[batch,time,units] -> enc_out[batch,time,units]
    """
    def __init__(
            self,
            name,
            inp_voc=None,
            hid_size=128,
            ff_size=None,
            num_heads=8,
            num_layers_enc=1,
            res_steps='nlda',
            normalize_out=False,
            rescale_emb=False,
            dropout=0.,
            **_kwargs
    ):

        if hid_size % num_heads != 0:
            raise Exception("hid_size must be divisible by num_heads")

        assert num_layers_enc > 0

        self.name = name
        self.normalize_out = normalize_out
        self.rescale_emb = rescale_emb
        self.dropout = dropout

        self.inp_voc = inp_voc
        self.emb_size = self.hid_size = hid_size
        self.ff_size = ff_size = ff_size or hid_size

        with tf.variable_scope(name):
            if self.inp_voc is not None:
                self.emb_inp = Embedding(
                    'emb_inp', len(inp_voc), hid_size,
                    initializer=tf.random_normal_initializer(0, hid_size ** -.5))

            self.enc_attn = [ResidualLayerWrapper(
                'enc_attn-%i' % i,
                MultiHeadAttention(
                    'enc_attn-%i' % i,
                    inp_size=hid_size,
                    key_depth=hid_size,
                    value_depth=hid_size,
                    output_depth=hid_size,
                    num_heads=num_heads,
                    attn_dropout=0,
                    attn_value_dropout=0),

                inp_size=hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=self.dropout)
                for i in range(num_layers_enc)]

            self.enc_ffn = [ResidualLayerWrapper(
                'enc_ffn-%i' % i,
                FeedforwardBlock(
                    'enc_ffn-%i' % i,
                    inp_size=hid_size,
                    hid_size=ff_size or hid_size,
                    out_size=hid_size,
                    relu_dropout=0),
                inp_size=hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=self.dropout)
                for i in range(num_layers_enc)]

            if self.normalize_out:
                self.enc_out_norm = LayerNorm('enc_out_norm', inp_size=hid_size)

    def __call__(self, enc_inp, attn_mask=None, is_train=False):
        if enc_inp.shape.ndims == 2 and enc_inp.dtype.is_integer:
            assert self.inp_voc is not None, "TransformerEncoder must have inp_voc to process token indices"
            attn_mask = make_attn_mask(enc_inp, self.inp_voc.eos,
                                       allow_lookahead=True,
                                       allow_past_eos=False)
            enc_inp = self.emb_inp(enc_inp)  # [batch_size * ninp * emb_dim]
            if self.rescale_emb:
                enc_inp *= self.emb_size ** .5

            assert enc_inp.shape.ndims == 3, "input must have shape [batch, time, units]"
        else:
            assert attn_mask is not None, "attn_mask must be specified if inp is not int32"
            assert enc_inp.shape.ndims == 3, "input must have shape [batch, time, units]"
            assert attn_mask.shape.ndims == 4, "attn_mask must have shape [batch, 1, from_time, to_time]"

        with dropout_scope(is_train), tf.name_scope(self.name + '_enc'):

            # Embeddings
            enc_inp = add_timing_signal(enc_inp)

            # Apply dropouts
            if is_dropout_enabled():
                enc_inp = tf.nn.dropout(enc_inp, 1.0 - self.dropout)

            # Encoder
            for layer in range(len(self.enc_attn)):
                enc_inp = self.enc_attn[layer](enc_inp, attn_mask)
                enc_inp = self.enc_ffn[layer](enc_inp)

            if self.normalize_out:
                enc_inp = self.enc_out_norm(enc_inp)

            return enc_inp, attn_mask


class TransformerDecoder:
    """
    Implements full pass of transformer decoder.
    Takes tokens[batch,time], enc_out[batch, time, units], enc_attn_mask[batch,time,units]
        -> enc_out[batch,time,units]

    Note: it assumes that hid_size in encoder and decoder are equal. (easy2fix)
    """
    def __init__(
            self,
            name,
            out_voc=None,
            hid_size=128,
            ff_size=None,
            num_heads=8,
            num_layers_dec=1,
            res_steps='nlda',
            normalize_out=False,
            rescale_emb=False,
            dropout=0.,
            **_kwargs
    ):

        if hid_size % num_heads != 0:
            raise Exception("hid_size must be divisible by num_heads")

        assert num_layers_dec > 0

        self.name = name
        self.normalize_out = normalize_out
        self.rescale_emb = rescale_emb
        self.dropout = dropout
        self.out_voc = out_voc
        self.emb_size = self.hid_size = hid_size
        self.ff_size = ff_size = ff_size or hid_size

        with tf.variable_scope(name):
            if self.out_voc is not None:
                self.emb_out = Embedding(
                    'emb_out', len(out_voc), hid_size,
                    initializer=tf.random_normal_initializer(0, hid_size ** -.5))

            self.dec_attn = [ResidualLayerWrapper(
                'dec_attn-%i' % i,
                MultiHeadAttention(
                    'dec_attn-%i' % i,
                    inp_size=hid_size,
                    key_depth=hid_size,
                    value_depth=hid_size,
                    output_depth=hid_size,
                    num_heads=num_heads,
                    attn_dropout=0,
                    attn_value_dropout=0),
                inp_size=hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=self.dropout)
                for i in range(num_layers_dec)]

            self.dec_enc_attn = [ResidualLayerWrapper(
                'dec_enc_attn-%i' % i,
                MultiHeadAttention(
                    'dec_enc_attn-%i' % i,
                    inp_size=hid_size,
                    key_depth=hid_size,
                    value_depth=hid_size,
                    output_depth=hid_size,
                    num_heads=num_heads,
                    attn_dropout=0,
                    attn_value_dropout=0,
                ),
                inp_size=hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=self.dropout)
                for i in range(num_layers_dec)]

            self.dec_ffn = [ResidualLayerWrapper(
                'dec_ffn-%i' % i,
                FeedforwardBlock(
                    'dec_ffn-%i' % i,
                    inp_size=hid_size,
                    hid_size=ff_size,
                    out_size=hid_size,
                    relu_dropout=0),
                inp_size=hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=self.dropout)
                for i in range(num_layers_dec)]

            if self.normalize_out:
                self.dec_out_norm = LayerNorm('dec_out_norm', inp_size=hid_size)

    def __call__(self, enc_out, dec_inp, enc_attn_mask, dec_attn_mask=None, is_train=True):

        if dec_inp.shape.ndims == 2 and dec_inp.dtype.is_integer:
            assert self.out_voc is not None, "TransformerEncoder must have inp_voc to process token indices"
            if dec_attn_mask is None:
                dec_attn_mask = make_attn_mask(dec_inp, self.out_voc.eos,
                                               allow_lookahead=False,
                                               allow_past_eos=False)
            dec_inp = self.emb_out(dec_inp)  # [batch_size * ninp * emb_dim]
            if self.rescale_emb:
                dec_inp *= self.emb_size ** .5
        else:
            assert dec_attn_mask is not None, "attn_mask must be specified if inp is not int32"

        assert dec_inp.shape.ndims == 3, "input must have shape [batch, time, units]"
        assert dec_attn_mask.shape.ndims == 4, "attn_mask must have shape [batch, 1, from_time, to_time]"

        with dropout_scope(is_train), tf.name_scope(self.name + '_dec'):
            dec_inp = add_timing_signal(dec_inp)

            for layer in range(len(self.dec_attn)):
                dec_inp = self.dec_attn[layer](dec_inp, dec_attn_mask)
                dec_inp = self.dec_enc_attn[layer](dec_inp, enc_attn_mask, enc_out)
                dec_inp = self.dec_ffn[layer](dec_inp)

            if self.normalize_out:
                dec_inp = self.dec_out_norm(dec_inp)

        return dec_inp, dec_attn_mask


def make_attn_mask(seq, eos, allow_lookahead=True, allow_past_eos=False, dtype=tf.float32):
    """
    Computes attention mask.
    :param seq: token indices, int32[batch_size * ninp]
    :param eos: end-of-sequence token index
    :returns: attention mask with restrictions as specified:
        The default behavior is like transformer encoder: allows reading any element before first EOS (inclusive)
        if allow_lookahead = False, prohibits i-th element from reading j-th element if j > i
        if allow_past_eos = True, allows reading elements after first EOS
        The mask shape follows this pattern: [batch, 1, from_index, to_index]
    """
    assert seq.shape.ndims == 2 and seq.dtype.is_integer, "seq must be int[batch, ninp]"

    with tf.variable_scope("make_attn_mask"):
        attn_mask = tf.ones([1, 1, 1, 1], dtype=dtype)

        if not allow_past_eos:
            attn_mask *= infer_mask(seq, eos, dtype=tf.float32)[:, None, None, :]

        if not allow_lookahead:
            length = tf.shape(seq)[1]
            lower_triangle = tf.matrix_band_part(tf.ones([length, length], dtype=dtype), -1, 0)
            attn_mask *= tf.reshape(lower_triangle, [1, 1, length, length])

        return attn_mask


def add_timing_signal(inp, min_timescale=1.0, max_timescale=1.0e4, offset=0, inp_reverse=None):
    """
    :param inp: input vector sequence of shape [batch_size, ninp, hid_dim]
    :param offset: add this number to all character positions.
        if offset == 'random', picks this number uniformly from [-32000,32000] integers
    :type offset: number, tf.Tensor or 'random'
    """
    assert inp.shape.ndims == 3, "inp must be float[batch, ninp, num_units]"

    with tf.variable_scope("add_timing_signal"):
        ninp = tf.shape(inp)[1]
        hid_size = tf.shape(inp)[2]

        position = tf.to_float(tf.range(ninp))[None, :, None]

        if offset == 'random':
            BIG_LEN = 32000
            offset = tf.random_uniform(tf.shape(position), minval=-BIG_LEN, maxval=BIG_LEN, dtype=tf.int32)

        # force broadcasting over batch axis
        if isinstance(offset * 1, tf.Tensor):  # multiply by 1 to also select variables, special generators, etc.
            assert offset.shape.ndims in (0, 1, 2)
            new_shape = [tf.shape(offset)[i] for i in range(offset.shape.ndims)]
            new_shape += [1] * (3 - len(new_shape))
            offset = tf.reshape(offset, new_shape)

        position += tf.to_float(offset)

        if inp_reverse is not None:
            position = tf.multiply(
                position,
                tf.where(
                    tf.equal(inp_reverse, 0),
                    tf.ones_like(inp_reverse, dtype=tf.float32),
                    -1.0 * tf.ones_like(inp_reverse, dtype=tf.float32)
                )[:, None, None]  # (batch_size * ninp * dim)
            )
        num_timescales = hid_size // 2
        log_timescale_increment = (
            np.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

        # scaled_time: [ninp * hid_dim]
        scaled_time = position * inv_timescales[None, None, :]
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
        signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(hid_size, 2)]])
        return inp + signal
