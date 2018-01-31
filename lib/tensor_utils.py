"""
assorted aux functions dealing with tensors
"""
import tensorflow as tf
import numpy as np
from lib.smx import sliced_argmax


def infer_length(seq, eos=1, time_major=False):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos: integer index of end-of-sentence token
    :returns: lengths, int32 vector of [batch_size]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos), 'int32')
    count_eos = tf.cumsum(is_eos, axis=axis, exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), 'int32'), axis=axis)
    return lengths


def infer_mask(seq, eos=1, time_major=False, dtype=tf.bool):
    """
    compute mask
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos: integer index of end-of-sentence token
    :returns: mask, matrix of same shape as seq and of given dtype (bool by default)
    """
    lengths = infer_length(seq, eos=eos, time_major=time_major)
    maxlen = tf.shape(seq)[0 if time_major else 1]
    mask = tf.sequence_mask(lengths, dtype=dtype, maxlen=maxlen)
    if time_major:
        mask = tf.transpose(mask)
    return mask


def initialize_uninitialized_variables(sess=None):
    """
    runs operation that initializes uninitialized variables
    """
    sess = sess or tf.get_default_session()
    uninitialized_names = set(sess.run(tf.report_uninitialized_variables()))
    uninitialized_vars = []
    for var in tf.global_variables():
        if var.name[:-2].encode() in uninitialized_names:
            uninitialized_vars.append(var)

    sess.run(tf.variables_initializer(uninitialized_vars))


def dot(x, y):
    """
    exactly same as theano.dot; taken from yandex translate codebase.
    """
    x_ndim = x.get_shape().ndims
    y_ndim = y.get_shape().ndims
    etc_x = tf.slice(tf.shape(x), [0], [x_ndim-1])
    etc_y = tf.slice(tf.shape(y), [1], [-1])
    a = tf.shape(y)[0]

    # Reshape forth.
    if x_ndim != 2:
        x = tf.reshape(x, [-1, a])
    if y_ndim != 2:
        y = tf.reshape(y, [a, -1])

    # Compute
    ret = tf.matmul(x, y)

    # Reshape back.
    if x_ndim != 2 or y_ndim != 2:
        ret = tf.reshape(ret, tf.concat([etc_x, etc_y], 0))

    return ret


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0, batch_size)[:, None], [1, seq_len])
    time_i = tf.tile(tf.range(0, seq_len)[None, :], [batch_size, 1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values, indices_nd)


# Dropout scope

import threading
from contextlib import contextmanager
_tls = threading.local()


def is_dropout_enabled():
    """
    Usage:
    enable / disable:
    with tf.dropout_scope(True):
        y = model(x)

    inside model(x):
    if is_dropout_enabled():
        x = tf.nn.dropout(x, 0.5)
    """
    if not hasattr(_tls, 'dropout_enabled'):
        _tls.dropout_enabled = True
    return _tls.dropout_enabled


@contextmanager
def dropout_scope(enabled):
    """
    Usage:
    enable / disable:
    with tf.dropout_scope(True):
        y = model(x)

    inside model(x):
    if is_dropout_enabled():
        x = tf.nn.dropout(x, 0.5)
    """
    was_enabled = is_dropout_enabled()
    _tls.dropout_enabled = enabled
    try:
        yield
    finally:
        _tls.dropout_enabled = was_enabled


def log_sigmoid(x): return -tf.nn.softplus(-x)

# miscellaneous stuff

def nop(x): return x

def is_tf_tensor(x):
    """
    Checks if x is some kind of TF object (variable, tensor, sparsetensor).
    This code is dark and full of terrors
    """
    try:
        if isinstance(x + 0, tf.Tensor):  # + 0 used to convert variable to a tensor
            return True
        elif isinstance(x + 0, tf.SparseTensor):
            return True
    except:
        return False


def is_scalar(var):
    """ checks if var is not scalar. Works for list, np.array, tf.tensor and many similar classes """
    return len(np.shape(var)) == 0

def stupidmax(x,w=5,axis=1,mask=None):
    """Computes smooth version of softmax inspired by https://arxiv.org/abs/1612.05628"""
    exp = tf.exp(w*x)
    mean_exp = tf.reduce_sum(exp,axis) if mask is None else tf.reduce_sum(exp*mask[:,:,None],axis)
    return tf.log(mean_exp)/w

